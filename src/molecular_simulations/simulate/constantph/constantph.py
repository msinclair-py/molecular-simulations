# ruff: noqa: RUF012
"""
ConstantPH - Constant pH simulation using AMBER prmtop/inpcrd files directly.

This implementation preserves all force field parameters from the AMBER topology,
including custom ligand parameters and Lipid21 modular lipids.

Key features:
- Uses AmberPrmtopFile for explicit solvent simulation (preserves all parameters)
- Uses ParmEd to create implicit solvent system WITH lipids and ligands
- MC energy evaluations include protein-lipid and protein-ligand interactions
- Only water and ions are stripped from the implicit system
"""

import contextlib
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import parmed as pmd
from openmm import Context, CustomGBForce, GBSAOBCForce, NonbondedForce, System
from openmm.app import (
    OBC2,
    AmberInpcrdFile,
    AmberPrmtopFile,
    ForceField,
    GBn2,
    HBonds,
    Modeller,
    NoCutoff,
    Simulation,
    element,
)
from openmm.app.forcefield import NonbondedGenerator
from openmm.app.internal import compiled  # ty: ignore[unresolved-import]
from openmm.unit import (
    MOLAR_GAS_CONSTANT_R,
    elementary_charge,
    is_quantity,
    kelvin,
    kilojoules_per_mole,
    nanometers,  # ty: ignore[unresolved-import]
)


class ResidueState:
    """Stores parameters for a particular protonation state of a residue."""

    def __init__(
        self,
        residueIndex,
        atomIndices,
        particleParameters,
        exceptionParameters,
        numHydrogens,
    ):
        self.residueIndex = residueIndex
        self.atomIndices = atomIndices  # {atom_name: atom_index}
        self.particleParameters = (
            particleParameters  # {force_index: {atom_name: params}}
        )
        self.exceptionParameters = (
            exceptionParameters  # {force_index: {(res, a1, a2): params}}
        )
        self.numHydrogens = numHydrogens


class ResidueTitration:
    """Manages titration states for a single residue."""

    def __init__(self, variants, referenceEnergies):
        self.variants = variants
        self.referenceEnergies = referenceEnergies
        self.explicitStates = []
        self.implicitStates = []
        self.explicitHydrogenIndices = []
        self.protonatedIndex = -1
        self.currentIndex = -1


class ConstantPH:
    """Constant pH simulation using AMBER topology files directly.

    This class enables constant pH molecular dynamics while preserving all force
    field parameters from AMBER prmtop files, including custom ligand parameters
    and Lipid21 modular lipids.

    The approach:
    1. Use AmberPrmtopFile.createSystem() for the explicit solvent simulation
    2. Use ParmEd to create implicit solvent system WITH lipids and ligands
    3. MC energy evaluations include protein-lipid and protein-ligand interactions
    4. Only water and ions are stripped from the implicit system
    5. Use OpenMM ForceField only for building protonation state parameters

    Args:
        prmtop_file: Path (str or Path) to AMBER prmtop file.
        inpcrd_file: Path (str or Path) to AMBER inpcrd file.
        pH: The pH value(s) for simulation. If a list, simulated tempering is
            used.
        residueVariants: Maps residue indices to lists of variant names.
            Example: {10: ['ASP', 'ASH'], 15: ['GLU', 'GLH']}.
        referenceEnergies: Maps residue indices to lists of reference energies
            (kJ/mol). Example: {10: [0.0, 5.2], 15: [0.0, 4.8]}.
        relaxationSteps: Steps to relax solvent after accepting a protonation
            state change.
        explicitArgs: Arguments for createSystem() for explicit solvent.
        implicitArgs: Arguments for ParmEd createSystem() for implicit solvent.
            Supports: implicitSolvent (OBC2, GBn2), solventDielectric,
            soluteDielectric.
        integrator: Integrator for the main simulation.
        relaxationIntegrator: Integrator for solvent relaxation (frozen solute).
        implicitForceField: ForceField for building protonation state
            parameters (protein only). Defaults to amber14 + GBn2.
        excludeResidues: Deprecated and no longer used; lipids and ligands are
            now always included in MC evaluations and only water and ions are
            stripped. Defaults to None.
        gbModel: GB model for implicit solvent: 'OBC2' or 'GBn2'. Defaults to
            'GBn2'.
        weights: Simulated tempering weights. None = auto-determine via
            Wang-Landau. Defaults to None.
        platform: Platform for simulation. None = auto-select. Defaults to None.
        properties: Platform-specific properties. Defaults to None.
    """

    # Standard residues that can be parameterized by OpenMM ForceField
    PROTEIN_RESIDUES = {
        'ALA',
        'ARG',
        'ASN',
        'ASP',
        'ASH',
        'CYS',
        'CYM',
        'CYX',
        'GLN',
        'GLU',
        'GLH',
        'GLY',
        'HIS',
        'HID',
        'HIE',
        'HIP',
        'ILE',
        'LEU',
        'LYS',
        'LYN',
        'MET',
        'PHE',
        'PRO',
        'SER',
        'THR',
        'TRP',
        'TYR',
        'VAL',
        'ACE',
        'NME',
        'NHE',
    }

    # Ion elements to exclude from implicit system
    ION_ELEMENTS = (
        element.cesium,
        element.potassium,
        element.lithium,
        element.sodium,
        element.rubidium,
        element.chlorine,
        element.bromine,
        element.fluorine,
        element.iodine,
    )

    # Water/ion residue names to strip (lipids and ligands are KEPT)
    WATER_ION_NAMES = {
        'HOH',
        'WAT',
        'Na+',
        'Cl-',
        'NA',
        'CL',
        'K+',
        'K',
        'SOD',
        'CLA',
        'POT',
        'OPC',
        'TIP3',
        'SPC',
    }

    # Titratable hydrogen differences: maps (deprotonated, protonated) variant pairs
    # to the number of hydrogens LOST when going from protonated to deprotonated
    TITRATION_H_DIFF = {
        # Aspartate: ASH (protonated) -> ASP (deprotonated), loses 1 H
        ('ASP', 'ASH'): -1,
        ('ASH', 'ASP'): 1,
        # Glutamate: GLH (protonated) -> GLU (deprotonated), loses 1 H
        ('GLU', 'GLH'): -1,
        ('GLH', 'GLU'): 1,
        # Histidine: HIP (doubly protonated) -> HID/HIE (singly protonated)
        ('HID', 'HIP'): -1,
        ('HIP', 'HID'): 1,
        ('HIE', 'HIP'): -1,
        ('HIP', 'HIE'): 1,
        ('HID', 'HIE'): 0,
        ('HIE', 'HID'): 0,  # Same H count, different position
        # Lysine: LYS (protonated) -> LYN (neutral), loses 1 H
        ('LYN', 'LYS'): -1,
        ('LYS', 'LYN'): 1,
        # Cysteine: CYS (protonated thiol) -> CYM (deprotonated thiolate)
        ('CYM', 'CYS'): -1,
        ('CYS', 'CYM'): 1,
        # Tyrosine: TYR (protonated) -> TYD (deprotonated phenolate)
        ('TYD', 'TYR'): -1,
        ('TYR', 'TYD'): 1,
    }

    def __init__(
        self,
        prmtop_file,
        inpcrd_file,
        pH,
        residueVariants,
        referenceEnergies,
        relaxationSteps,
        explicitArgs,
        implicitArgs,
        integrator,
        relaxationIntegrator,
        implicitForceField=None,
        excludeResidues=None,
        gbModel='GBn2',
        weights=None,
        platform=None,
        properties=None,
    ):

        # Store file paths for ParmEd
        self.prmtop_file = str(prmtop_file)
        self.inpcrd_file = str(inpcrd_file)

        # Load AMBER topology and coordinates
        print('Loading AMBER topology...')
        self.prmtop = AmberPrmtopFile(self.prmtop_file)
        self.inpcrd = AmberInpcrdFile(self.inpcrd_file)

        # Store parameters
        self._explicitArgs = explicitArgs
        self._implicitArgs = implicitArgs
        self.relaxationSteps = relaxationSteps
        self.gbModel = gbModel

        # Set up pH
        if not isinstance(pH, Sequence):
            pH = [pH]
        self.setPH(pH, weights)
        self.currentPHIndex = 0

        # excludeResidues is no longer used - we keep lipids and ligands!
        # Only water and ions are stripped
        if excludeResidues is not None:
            print('  Note: excludeResidues parameter is deprecated.')
            print('        Lipids and ligands are now INCLUDED in MC evaluations.')

        # Set up implicit ForceField (for building protonation state params only)
        if implicitForceField is None:
            self.implicitForceField = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        else:
            self.implicitForceField = implicitForceField

        # Initialize titration tracking
        self.titrations = {}
        for resIndex, variants in residueVariants.items():
            energies = list(referenceEnergies[resIndex])
            self.titrations[resIndex] = ResidueTitration(variants, energies)

        # Build implicit system with lipids/ligands using ParmEd
        print('Building implicit solvent system (includes lipids and ligands)...')
        self._buildImplicitSystemWithParmEd()

        # Build protein-only topology for protonation state parameters
        print('Building protein-only topology for protonation states...')
        self._buildProteinOnlyTopology(residueVariants)

        # Build protonation states for each titratable residue
        print('Computing protonation state parameters...')
        self._buildProtonationStates(residueVariants)

        # Map protonation states to implicit system (fix force indices)
        print('Mapping protonation states to implicit system...')
        self._mapStatesToImplicitSystem()

        # Create the explicit system from AMBER topology
        print('Creating explicit solvent system from AMBER topology...')
        self._buildExplicitSystem(
            integrator, relaxationIntegrator, platform, properties
        )

        # Map protonation states to explicit system
        print('Mapping protonation states to explicit system...')
        self._mapStatesToExplicitSystem()

        print('ConstantPHAmber initialization complete.')

    def _buildImplicitSystemWithParmEd(self):
        """
        Build implicit solvent system using ParmEd, keeping lipids and ligands.

        This method:
        1. Loads the AMBER topology with ParmEd
        2. Strips only water and ions (keeps protein, lipids, ligands)
        3. Creates an implicit solvent System with GB

        The resulting system preserves all AMBER parameters for lipids and ligands,
        enabling accurate MC energy evaluations that include protein-membrane and
        protein-ligand interactions.
        """
        # Load structure with ParmEd
        parm = pmd.load_file(self.prmtop_file, self.inpcrd_file)

        # Select atoms to keep (negate water/ion selection)
        # Use ParmEd's slice syntax to create a subset
        keep_indices = []
        for residue in parm.residues:
            # Check if residue is water or ion
            is_water_ion = residue.name in self.WATER_ION_NAMES
            if not is_water_ion:  # noqa: SIM102
                # Also check for single-atom ions by element
                if len(residue.atoms) == 1 and residue.atoms[0].element in [
                    11,
                    17,
                    19,
                    35,
                    37,
                    55,
                ]:  # Na, Cl, K, Br, Rb, Cs
                    is_water_ion = True

            if not is_water_ion:
                for atom in residue.atoms:
                    keep_indices.append(atom.idx)

        # Create stripped structure by selecting atoms
        stripped_parm = parm[keep_indices]

        # Build residue index mapping (implicit <-> explicit)
        self.implicitToExplicitResidueMap = []
        self.explicitToImplicitResidueMap = {}

        # Track which explicit residues were kept
        explicit_residues = list(self.prmtop.topology.residues())
        implicit_idx = 0

        for explicit_idx, res in enumerate(explicit_residues):
            # Check if this residue was stripped
            if res.name not in self.WATER_ION_NAMES:
                # Check for ions by element
                atoms = list(res.atoms())
                if len(atoms) == 1 and atoms[0].element in self.ION_ELEMENTS:
                    continue  # Skip ions

                self.implicitToExplicitResidueMap.append(explicit_idx)
                self.explicitToImplicitResidueMap[explicit_idx] = implicit_idx
                implicit_idx += 1

        # Store the stripped ParmEd structure
        self._strippedParm = stripped_parm

        # Determine GB model
        if self.gbModel == 'GBn2':
            implicitSolvent = GBn2
        elif self.gbModel == 'OBC2':
            implicitSolvent = OBC2
        else:
            raise ValueError(f'Unknown GB model: {self.gbModel}. Use GBn2 or OBC2.')

        # Create implicit system with GB using ParmEd
        # This preserves all AMBER bonded parameters
        solventDielectric = self._implicitArgs.get('solventDielectric', 78.5)
        soluteDielectric = self._implicitArgs.get('soluteDielectric', 1.0)

        self.implicitSystem = stripped_parm.createSystem(
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
            implicitSolvent=implicitSolvent,
            soluteDielectric=soluteDielectric,
            solventDielectric=solventDielectric,
            removeCMMotion=False,
        )

        # Store topology and positions
        self.implicitTopology = stripped_parm.topology
        self.implicitPositions = stripped_parm.positions

        # Count what we kept
        n_protein = sum(
            1 for r in stripped_parm.residues if r.name in self.PROTEIN_RESIDUES
        )
        n_lipid = sum(
            1
            for r in stripped_parm.residues
            if r.name in {'PA', 'PC', 'PE', 'OL', 'GL'}
        )
        n_other = len(stripped_parm.residues) - n_protein - n_lipid

        print('  Stripped water and ions only')
        print(
            f'  Implicit system: {len(stripped_parm.residues)} residues, '
            f'{len(stripped_parm.atoms)} atoms'
        )
        print(f'    Protein: {n_protein} residues')
        print(f'    Lipids: {n_lipid} residues (PA/PC/PE/OL/GL)')
        print(f'    Other (ligands, etc.): {n_other} residues')
        print(f'    GB model: {self.gbModel}')

    def _buildProteinOnlyTopology(self, residueVariants):
        """
        Build a protein-only topology for computing protonation state parameters.

        This is separate from the implicit system (which includes lipids/ligands).
        We need protein-only to use OpenMM ForceField for variant parameterization.
        """
        topology = self.prmtop.topology
        positions = self.inpcrd.positions

        # Identify non-protein residues to remove
        residuesToRemove = []
        self.proteinToExplicitResidueMap = []
        self.explicitToProteinResidueMap = {}

        removedCount = 0
        for residue in topology.residues():
            isProtein = residue.name in self.PROTEIN_RESIDUES

            if not isProtein:
                residuesToRemove.append(residue)
                removedCount += 1
            else:
                proteinIndex = residue.index - removedCount
                self.proteinToExplicitResidueMap.append(residue.index)
                self.explicitToProteinResidueMap[residue.index] = proteinIndex

        # Create protein-only topology using Modeller
        modeller = Modeller(topology, positions)
        modeller.delete(residuesToRemove)

        self.proteinTopology = modeller.topology
        self.proteinPositions = modeller.positions

        print(
            f'  Protein-only topology: {self.proteinTopology.getNumResidues()} residues, '
            f'{self.proteinTopology.getNumAtoms()} atoms'
        )

    def _buildProtonationStates(self, residueVariants):
        """
        Build ResidueState objects for each protonation state.

        Uses the protein-only topology with OpenMM ForceField to build
        protonation state parameters. These parameters (charges, etc.) will
        be applied to both the implicit system (with lipids/ligands) and
        the explicit system.
        """
        # We need to iterate through each variant index
        variantIndex = 0
        finished = False

        # Use protein-only topology for ForceField parameterization
        proteinVariants = [None] * self.proteinTopology.getNumResidues()

        while not finished:
            finished = True

            # Set variants for this iteration
            for proteinIndex, explicitIndex in enumerate(
                self.proteinToExplicitResidueMap
            ):
                if explicitIndex in residueVariants:
                    variants = residueVariants[explicitIndex]
                    if variantIndex < len(variants):
                        finished = False
                        proteinVariants[proteinIndex] = variants[variantIndex]

            if finished:
                break

            # Build states using protein-only topology
            proteinStates = self._findResidueStates(
                self.proteinTopology,
                self.proteinPositions,
                self.implicitForceField,
                proteinVariants,
                self._implicitArgs,
            )

            # Add to ResidueTitration objects
            for proteinState in proteinStates:
                # Map protein residue index to explicit
                proteinResIndex = proteinState.residueIndex
                explicitResIndex = self.proteinToExplicitResidueMap[proteinResIndex]

                if explicitResIndex in self.titrations:
                    titration = self.titrations[explicitResIndex]
                    if variantIndex < len(titration.variants):
                        # Update residue index to explicit system index
                        proteinState.residueIndex = explicitResIndex
                        titration.implicitStates.append(proteinState)

            variantIndex += 1

        # Identify the fully protonated state and fix numHydrogens for each titration
        # This is necessary because AMBER prmtops already have hydrogens, so addHydrogens
        # may not correctly adjust the topology for different variants
        for resIndex, titration in self.titrations.items():
            variants = titration.variants

            # Find the protonated state based on variant name conventions
            protonatedIdx = self._identifyProtonatedState(variants)
            titration.protonatedIndex = protonatedIdx

            # Get baseline hydrogen count from the protonated state
            baselineH = titration.implicitStates[protonatedIdx].numHydrogens

            # Update numHydrogens for each state based on variant differences
            protonatedVariant = variants[protonatedIdx]
            for i, (state, variant) in enumerate(
                zip(titration.implicitStates, variants, strict=True)
            ):
                if i == protonatedIdx:
                    continue
                # Calculate H difference from protonated state
                key = (variant, protonatedVariant)
                if key in self.TITRATION_H_DIFF:
                    hDiff = self.TITRATION_H_DIFF[key]
                    state.numHydrogens = baselineH + hDiff
                    if (
                        state.numHydrogens
                        != titration.implicitStates[protonatedIdx].numHydrogens
                    ):
                        print(
                            f'  Res {resIndex}: {variant} has {state.numHydrogens} H '
                            f'(vs {baselineH} for {protonatedVariant})'
                        )

            titration.currentIndex = titration.protonatedIndex

    def _identifyProtonatedState(self, variants):
        """Identify which variant index corresponds to the fully protonated state.

        For standard titratable residues:
        - ASH > ASP (ASH is protonated aspartate)
        - GLH > GLU (GLH is protonated glutamate)
        - HIP > HID, HIE (HIP is doubly protonated histidine)
        - LYS > LYN (LYS is protonated lysine)
        - CYS > CYM (CYS has the thiol proton)
        - TYR > TYD (TYR has the phenolic proton)
        """
        # Protonated forms (higher proton count)
        PROTONATED_FORMS = {'ASH', 'GLH', 'HIP', 'LYS', 'CYS', 'TYR'}

        # Find the variant with the most protons
        for i, variant in enumerate(variants):
            if variant in PROTONATED_FORMS:
                return i

        # If no protonated form found, check for intermediate forms
        # For histidine: HID and HIE are equally protonated (singly)
        for i, variant in enumerate(variants):
            if variant in {'HID', 'HIE'}:
                return i

        # Fallback: assume first variant is protonated
        # (This handles custom residue types)
        print(
            f'  Warning: Could not identify protonated state for variants {variants}, '
            f'assuming index 0'
        )
        return 0

    def _mapStatesToImplicitSystem(self):
        """Map protonation state force indices from ForceField system to ParmEd implicit system.

        This method:
        1. Remaps force indices from ForceField-created system to ParmEd implicit system
        2. Updates atom indices to use implicit system coordinates
        3. Ensures all states have consistent atoms with ghost hydrogens
        """
        # Find NonbondedForce and GBSAOBCForce indices in the ParmEd implicit system
        implicitNBForceIdx = None
        implicitGBForceIdx = None
        for fi, force in enumerate(self.implicitSystem.getForces()):
            if isinstance(force, NonbondedForce):
                implicitNBForceIdx = fi
            elif isinstance(force, (GBSAOBCForce, CustomGBForce)):
                # ParmEd builds every GB model (OBC2, GBn2) as a CustomGBForce,
                # so we must recognize both classes here. Only GBSAOBCForce was
                # matched previously, which left the GB charges frozen across
                # protonation states and removed the desolvation term from every
                # MC energy difference.
                implicitGBForceIdx = fi

        if implicitNBForceIdx is None:
            raise RuntimeError('No NonbondedForce found in implicit system')

        implicitNBForce = self.implicitSystem.getForce(implicitNBForceIdx)
        implicitGBForce = (
            self.implicitSystem.getForce(implicitGBForceIdx)
            if implicitGBForceIdx is not None
            else None
        )

        # Build a reference system to identify force types from ForceField
        # We'll use the force field to create a temporary system and check force types
        tempSystem = self.implicitForceField.createSystem(
            self.proteinTopology, **self._implicitArgs
        )
        ffForceTypes = {}
        for fi, force in enumerate(tempSystem.getForces()):
            if isinstance(force, NonbondedForce):
                ffForceTypes[fi] = 'NB'
            elif isinstance(force, GBSAOBCForce):
                ffForceTypes[fi] = 'GB'

        for resIndex, titration in self.titrations.items():
            for state in titration.implicitStates:
                # Map ForceField force indices to implicit system force indices
                ffNBForceIdx = None
                ffGBForceIdx = None
                for fi in state.particleParameters:
                    if fi in ffForceTypes:
                        if ffForceTypes[fi] == 'NB':
                            ffNBForceIdx = fi
                        elif ffForceTypes[fi] == 'GB':
                            ffGBForceIdx = fi

                # Remap particle parameters to use implicit system force indices
                newParticleParams = {}
                if (
                    ffNBForceIdx is not None
                    and ffNBForceIdx in state.particleParameters
                ):
                    newParticleParams[implicitNBForceIdx] = state.particleParameters[
                        ffNBForceIdx
                    ]
                if (
                    ffGBForceIdx is not None
                    and implicitGBForceIdx is not None
                    and ffGBForceIdx in state.particleParameters
                ):
                    newParticleParams[implicitGBForceIdx] = state.particleParameters[
                        ffGBForceIdx
                    ]

                # Remap exception parameters
                newExceptionParams = {}
                if (
                    ffNBForceIdx is not None
                    and ffNBForceIdx in state.exceptionParameters
                ):
                    newExceptionParams[implicitNBForceIdx] = state.exceptionParameters[
                        ffNBForceIdx
                    ]

                # Update the state
                state.particleParameters = newParticleParams
                state.exceptionParameters = newExceptionParams

                # Update atom indices to use implicit system indices
                implicitResIdx = self.explicitToImplicitResidueMap.get(resIndex)
                if implicitResIdx is not None:
                    implicitResidues = list(self.implicitTopology.residues())
                    if implicitResIdx < len(implicitResidues):
                        state.atomIndices = {
                            atom.name: atom.index
                            for atom in implicitResidues[implicitResIdx].atoms()
                        }
                        state.residueIndex = implicitResIdx

            # Second pass: ensure all implicit states have consistent atoms with ghost hydrogens
            protonated = titration.protonatedIndex
            protonatedState = titration.implicitStates[protonated]
            protonatedNBParams = protonatedState.particleParameters.get(
                implicitNBForceIdx, {}
            )
            protonatedExceptionParams = protonatedState.exceptionParameters.get(
                implicitNBForceIdx, {}
            )

            for i, state in enumerate(titration.implicitStates):
                if i == protonated:
                    continue

                # Ensure NB parameters include all atoms from protonated state
                stateNBParams = state.particleParameters.get(implicitNBForceIdx, {})
                for atomName in protonatedNBParams:
                    if atomName not in stateNBParams:
                        originalParams = protonatedNBParams[atomName]
                        zeroParams = self._get_zero_parameters(
                            originalParams, implicitNBForce
                        )
                        stateNBParams[atomName] = zeroParams
                state.particleParameters[implicitNBForceIdx] = stateNBParams

                # Handle exceptions for ghost atoms
                stateExceptionParams = state.exceptionParameters.get(
                    implicitNBForceIdx, {}
                )
                for key in protonatedExceptionParams:
                    if key not in stateExceptionParams:
                        originalParams = protonatedExceptionParams[key]
                        stateExceptionParams[key] = (
                            0.0 * elementary_charge**2,
                            *originalParams[1:],
                        )
                state.exceptionParameters[implicitNBForceIdx] = stateExceptionParams

            # Third pass: derive per-state GB (implicit-solvent) parameters.
            #
            # The GB per-particle charge is the same partial charge as the
            # NonbondedForce; only the charge changes between protonation states,
            # never the GB radii. So for every state we take the GB force's own
            # base parameter tuple (correct radii, straight from ParmEd) and
            # substitute the charge (parameter index 0) with this state's NB
            # charge. Ghost hydrogens carry the zeroed NB charge here too, so
            # they contribute no GB solvation in deprotonated states.
            #
            # Without this, the GB charges stay frozen at the initial (fully
            # protonated) values, the GB term cancels out of every MC energy
            # difference, and titration is driven by unscreened electrostatics.
            if implicitGBForceIdx is not None and implicitGBForce is not None:
                gbIsCustom = isinstance(implicitGBForce, CustomGBForce)
                for state in titration.implicitStates:
                    stateNBParams = state.particleParameters.get(implicitNBForceIdx, {})
                    stateGBParams = {}
                    for atomName, nbParams in stateNBParams.items():
                        atomIdx = state.atomIndices.get(atomName)
                        if atomIdx is None:
                            continue
                        base = list(implicitGBForce.getParticleParameters(atomIdx))
                        charge = nbParams[0]
                        chargeVal = (
                            charge.value_in_unit(elementary_charge)
                            if hasattr(charge, 'value_in_unit')
                            else float(charge)
                        )
                        # CustomGBForce takes bare floats; GBSAOBCForce Quantities.
                        base[0] = (
                            chargeVal if gbIsCustom else chargeVal * elementary_charge
                        )
                        stateGBParams[atomName] = tuple(base)
                    state.particleParameters[implicitGBForceIdx] = stateGBParams

    def _findResidueStates(self, topology, positions, forcefield, variants, ffargs):
        """Build ResidueState objects for residues with specified variants."""
        modeller = Modeller(topology, positions)
        modeller.addHydrogens(forcefield=forcefield, variants=variants)
        system = forcefield.createSystem(modeller.topology, **ffargs)

        atoms = list(modeller.topology.atoms())
        residues = list(modeller.topology.residues())
        states = []

        for residue, variant in zip(residues, variants, strict=True):
            if variant is not None:
                atomIndices = {atom.name: atom.index for atom in residue.atoms()}
                particleParameters = {}
                exceptionParameters = {}

                for i, force in enumerate(system.getForces()):
                    with contextlib.suppress(Exception):
                        particleParameters[i] = {
                            atom.name: force.getParticleParameters(atom.index)
                            for atom in residue.atoms()
                        }

                    if isinstance(force, NonbondedForce):
                        exceptionParameters[i] = {}
                        for j in range(force.getNumExceptions()):
                            p1, p2, chargeProd, sigma, epsilon = (
                                force.getExceptionParameters(j)
                            )
                            atom1 = atoms[p1]
                            atom2 = atoms[p2]
                            if atom1.residue == residue and atom2.residue == residue:
                                exceptionParameters[i][
                                    (residue.index, atom1.name, atom2.name)
                                ] = (chargeProd, sigma, epsilon)

                numHydrogens = sum(
                    1 for atom in residue.atoms() if atom.element == element.hydrogen
                )
                states.append(
                    ResidueState(
                        residue.index,
                        atomIndices,
                        particleParameters,
                        exceptionParameters,
                        numHydrogens,
                    )
                )

        return states

    def _buildExplicitSystem(
        self, integrator, relaxationIntegrator, platform, properties
    ):
        """Create the explicit solvent system from AMBER topology."""
        # Create system preserving all AMBER parameters
        self.explicitSystem = self.prmtop.createSystem(**self._explicitArgs)
        self.explicitTopology = self.prmtop.topology
        explicitPositions = self.inpcrd.positions

        # Create relaxation system (frozen non-solvent)
        relaxationSystem = deepcopy(self.explicitSystem)
        for residue in self.explicitTopology.residues():
            isWater = residue.name == 'HOH'
            isIon = (
                len(residue) == 1
                and next(iter(residue.atoms())).element in self.ION_ELEMENTS
            )
            if not isWater and not isIon:
                for atom in residue.atoms():
                    relaxationSystem.setParticleMass(atom.index, 0.0)

        # Note: implicitSystem is already created by _buildImplicitSystemWithParmEd()
        # It includes lipids and ligands with all AMBER parameters preserved

        # Create simulation and contexts
        self.simulation = Simulation(
            self.explicitTopology,
            self.explicitSystem,
            deepcopy(integrator),
            platform,
            properties,
        )

        actualPlatform = self.simulation.context.getPlatform()
        if properties is None:
            self.implicitContext = Context(
                self.implicitSystem, deepcopy(integrator), actualPlatform
            )
            self.relaxationContext = Context(
                relaxationSystem, deepcopy(relaxationIntegrator), actualPlatform
            )
        else:
            self.implicitContext = Context(
                self.implicitSystem, deepcopy(integrator), actualPlatform, properties
            )
            self.relaxationContext = Context(
                relaxationSystem,
                deepcopy(relaxationIntegrator),
                actualPlatform,
                properties,
            )

        # Set positions
        self.simulation.context.setPositions(explicitPositions)
        self.relaxationContext.setPositions(explicitPositions)

        # Set implicit positions (stripped system)
        self.implicitContext.setPositions(self.implicitPositions)

        # Record atom index mapping for copying positions
        self._buildAtomIndexMapping()

        # Record exception indices
        self.explicitExceptionIndex = self._findExceptionIndices(
            self.explicitSystem, self.explicitTopology
        )
        self.implicitExceptionIndex = self._findExceptionIndices(
            self.implicitSystem, self.implicitTopology
        )
        self.explicitInterResidue14 = self._findInterResidue14(
            self.explicitSystem, self.explicitTopology
        )
        self.implicitInterResidue14 = self._findInterResidue14(
            self.implicitSystem, self.implicitTopology
        )

        # Record 1-4 scale factors (AMBER uses 1/1.2 = 0.8333 for Coulomb)
        self.explicit14Scale = self._find14Scale(self.explicitSystem)
        # For implicit system from ParmEd, use AMBER default
        self.implicit14Scale = 1.0 / 1.2  # AMBER default Coulomb 1-4 scale

    def _buildAtomIndexMapping(self):
        """Build mapping from implicit atom indices to explicit atom indices.

        This maps each atom in the implicit (stripped) system to its corresponding
        atom in the explicit system, enabling position copying between systems.
        """
        numImplicitAtoms = self.implicitSystem.getNumParticles()
        implicitAtomIndex = np.zeros(numImplicitAtoms, dtype=np.int64)

        explicitResidues = list(self.explicitTopology.residues())

        # ParmEd stripped structure
        implicitResidues = self._strippedParm.residues

        # Track atom offset for implicit system
        implicitAtomOffset = 0
        for implicitIndex, explicitIndex in enumerate(
            self.implicitToExplicitResidueMap
        ):
            explicitRes = explicitResidues[explicitIndex]
            implicitRes = implicitResidues[implicitIndex]

            # Build atom name -> index map for explicit residue
            explicitAtoms = {atom.name: atom.index for atom in explicitRes.atoms()}

            # Map implicit atoms to explicit atoms using enumeration
            # This is more robust than using ParmEd's global atom.idx
            for localIdx, atom in enumerate(implicitRes.atoms):
                implicitIdx = implicitAtomOffset + localIdx
                if atom.name in explicitAtoms:
                    implicitAtomIndex[implicitIdx] = explicitAtoms[atom.name]
                else:
                    # Fallback: try to match by position in residue
                    atomList = list(explicitRes.atoms())
                    if localIdx < len(atomList):
                        implicitAtomIndex[implicitIdx] = atomList[localIdx].index

            implicitAtomOffset += len(implicitRes.atoms)

        self.implicitAtomIndex = implicitAtomIndex

    def _mapStatesToExplicitSystem(self):
        """Map protonation state parameters from implicit to explicit system.

        This method:
        1. Creates explicit states for each protonation state
        2. Ensures all states have the same atom indices (using ghost atoms with zero charge)
        3. Tracks hydrogen indices for multi-site titrations
        """
        explicitResidues = list(self.explicitTopology.residues())

        # Find force indices in explicit system
        explicitNBForceIdx = None
        for fi, force in enumerate(self.explicitSystem.getForces()):
            if isinstance(force, NonbondedForce):
                explicitNBForceIdx = fi
                break

        if explicitNBForceIdx is None:
            raise RuntimeError('No NonbondedForce found in explicit system')

        explicitNBForce = self.explicitSystem.getForce(explicitNBForceIdx)

        for resIndex, titration in self.titrations.items():
            protonated = titration.protonatedIndex

            # Get atom indices from the explicit topology
            explicitAtomIndices = {
                atom.name: atom.index for atom in explicitResidues[resIndex].atoms()
            }

            # First pass: build all explicit states with their original parameters
            for implicitState in titration.implicitStates:
                # Get implicit NonbondedForce index
                implicitNBForceIdx = None
                for fi in implicitState.particleParameters:
                    force = self.implicitSystem.getForce(fi)
                    if isinstance(force, NonbondedForce):
                        implicitNBForceIdx = fi
                        break

                # Build explicit state by mapping parameters
                explicitParticleParams = {explicitNBForceIdx: {}}
                explicitExceptionParams = {explicitNBForceIdx: {}}

                # Map particle parameters
                if (
                    implicitNBForceIdx is not None
                    and implicitNBForceIdx in implicitState.particleParameters
                ):
                    for atomName, params in implicitState.particleParameters[
                        implicitNBForceIdx
                    ].items():
                        if atomName in explicitAtomIndices:
                            explicitParticleParams[explicitNBForceIdx][atomName] = (
                                params
                            )

                # Map exception parameters
                if (
                    implicitNBForceIdx is not None
                    and implicitNBForceIdx in implicitState.exceptionParameters
                ):
                    for key, params in implicitState.exceptionParameters[
                        implicitNBForceIdx
                    ].items():
                        # Convert key from implicit to explicit residue index
                        newKey = (resIndex, key[1], key[2])
                        explicitExceptionParams[explicitNBForceIdx][newKey] = params

                explicitState = ResidueState(
                    resIndex,
                    explicitAtomIndices,
                    explicitParticleParams,
                    explicitExceptionParams,
                    implicitState.numHydrogens,
                )
                titration.explicitStates.append(explicitState)

            # Second pass: ensure all states have consistent atoms with ghost hydrogens
            # Get parameters from the fully protonated state (which has all atoms)
            protonatedState = titration.explicitStates[protonated]
            protonatedParams = protonatedState.particleParameters.get(
                explicitNBForceIdx, {}
            )
            protonatedExceptionParams = protonatedState.exceptionParameters.get(
                explicitNBForceIdx, {}
            )

            for i, state in enumerate(titration.explicitStates):
                if i == protonated:
                    continue

                # For each atom in the protonated state, ensure this state has it too
                stateParams = state.particleParameters.get(explicitNBForceIdx, {})
                stateExceptionParams = state.exceptionParameters.get(
                    explicitNBForceIdx, {}
                )

                for atomName in protonatedParams:
                    if atomName not in stateParams:
                        # This atom doesn't exist in this protonation state
                        # Use zero parameters (ghost atom)
                        originalParams = protonatedParams[atomName]
                        zeroParams = self._get_zero_parameters(
                            originalParams, explicitNBForce
                        )
                        stateParams[atomName] = zeroParams

                        # Track this as a titratable hydrogen for multi-site moves
                        atomIdx = explicitAtomIndices.get(atomName)
                        if atomIdx is not None:
                            # Check if it's a hydrogen
                            for atom in explicitResidues[resIndex].atoms():
                                if (
                                    atom.name == atomName
                                    and atom.element == element.hydrogen
                                ):
                                    if atomIdx not in titration.explicitHydrogenIndices:
                                        titration.explicitHydrogenIndices.append(
                                            atomIdx
                                        )
                                    break

                # Handle exceptions for ghost atoms
                for key in protonatedExceptionParams:
                    if key not in stateExceptionParams:
                        # Zero out the charge product for this exception
                        originalParams = protonatedExceptionParams[key]
                        stateExceptionParams[key] = (
                            0.0 * elementary_charge**2,
                            *originalParams[1:],
                        )

                # Update the state's parameters
                state.particleParameters[explicitNBForceIdx] = stateParams
                state.exceptionParameters[explicitNBForceIdx] = stateExceptionParams

    def _findExceptionIndices(self, system, topology):
        """Map (residue, atom1, atom2) -> exception index in NonbondedForce."""
        indices = {}
        atoms = list(topology.atoms())

        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(force.getNumExceptions()):
                    p1, p2, _chargeProd, _sigma, _epsilon = (
                        force.getExceptionParameters(i)
                    )
                    atom1 = atoms[p1]
                    atom2 = atoms[p2]
                    if atom1.residue == atom2.residue:
                        indices[(atom1.residue.index, atom1.name, atom2.name)] = i
                        indices[(atom1.residue.index, atom2.name, atom1.name)] = i

        return indices

    def _findInterResidue14(self, system, topology):
        """Find 1-4 exceptions that span residues for each titratable residue."""
        indices = defaultdict(list)
        atoms = list(topology.atoms())

        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(force.getNumExceptions()):
                    p1, p2, chargeProd, _sigma, _epsilon = force.getExceptionParameters(
                        i
                    )
                    atom1 = atoms[p1]
                    atom2 = atoms[p2]
                    if (
                        atom1.residue != atom2.residue
                        and chargeProd.value_in_unit(elementary_charge**2) != 0.0
                    ):
                        indices[atom1.residue.index].append(i)
                        indices[atom2.residue.index].append(i)

        return indices

    def _find14Scale(self, obj):
        """Find 1-4 Coulomb scale factor from ForceField or System."""
        if isinstance(obj, ForceField):
            for generator in obj.getGenerators():
                if isinstance(generator, NonbondedGenerator):
                    return generator.coulomb14scale
        elif isinstance(obj, System):
            # AMBER default is 1/1.2 = 0.8333
            return 1.0 / 1.2
        return 1.0

    def setPH(self, pH, weights=None):
        """Set the pH value(s) for simulation."""
        self.pH = pH
        if weights is None:
            self._weights = [0.0] * len(pH)
            self._updateWeights = True
            self._weightUpdateFactor = 1.0
            self._histogram = [0] * len(pH)
            self._hasMadeTransition = False
        else:
            self._weights = weights
            self._updateWeights = False

    @property
    def weights(self):
        """Current simulated tempering weights."""
        return [x - self._weights[0] for x in self._weights]

    def printTitrationState(self):
        """Print current state of all titrations for debugging."""
        print(f'Current pH: {self.pH[self.currentPHIndex]:.2f}')
        print(f'Titratable residues: {len(self.titrations)}')
        for resIndex, titration in self.titrations.items():
            current = titration.currentIndex
            protonated = titration.protonatedIndex
            print(
                f'  Res {resIndex}: currentState={current}, protonatedIdx={protonated}'
            )
            print(f'    variants={titration.variants}')
            print(f'    refEnergies={titration.referenceEnergies}')
            for i, (impl, expl) in enumerate(
                zip(titration.implicitStates, titration.explicitStates, strict=True)
            ):
                print(
                    f'    state[{i}]: implicitH={impl.numHydrogens}, explicitH={expl.numHydrogens}, '
                    f'implAtoms={len(impl.atomIndices)}, explAtoms={len(expl.atomIndices)}'
                )

    def validateStates(self):
        """Validate that titration states are properly set up."""
        issues = []
        for resIndex, titration in self.titrations.items():
            # Check we have the right number of states
            if len(titration.implicitStates) != len(titration.variants):
                issues.append(
                    f'Res {resIndex}: {len(titration.implicitStates)} implicit states but {len(titration.variants)} variants'
                )
            if len(titration.explicitStates) != len(titration.variants):
                issues.append(
                    f'Res {resIndex}: {len(titration.explicitStates)} explicit states but {len(titration.variants)} variants'
                )

            # Check numHydrogens differs between states
            implicitHydrogens = [s.numHydrogens for s in titration.implicitStates]
            explicitHydrogens = [s.numHydrogens for s in titration.explicitStates]
            if len(set(implicitHydrogens)) == 1:
                issues.append(
                    f'Res {resIndex}: all implicit states have same numHydrogens={implicitHydrogens[0]}'
                )
            if len(set(explicitHydrogens)) == 1:
                issues.append(
                    f'Res {resIndex}: all explicit states have same numHydrogens={explicitHydrogens[0]}'
                )

            # Check protonated index makes sense
            if titration.protonatedIndex >= len(titration.implicitStates):
                issues.append(
                    f'Res {resIndex}: protonatedIndex={titration.protonatedIndex} out of range'
                )

        if issues:
            print('State validation FAILED:')
            for issue in issues:
                print(f'  - {issue}')
            return False
        else:
            print('State validation PASSED')
            return True

    def attemptMCStep(self, temperature, debug=False):
        """Attempt Monte Carlo moves to change protonation states.

        Args:
            temperature: Simulation temperature (float or Quantity).
            debug: If True, print debugging information about MC moves.
        """
        # Copy positions to implicit context
        state = self.simulation.context.getState(getPositions=True, getParameters=True)
        explicitPositions = state.getPositions(asNumpy=True).value_in_unit(nanometers)

        # Map positions to implicit system
        implicitPositions = explicitPositions[self.implicitAtomIndex]
        self.implicitContext.setPositions(implicitPositions)

        if debug:
            if np.any(np.isnan(implicitPositions)):
                print('WARNING: NaN in implicit positions!')
            testEnergy = self.implicitContext.getState(
                getEnergy=True
            ).getPotentialEnergy()
            if np.isnan(testEnergy.value_in_unit(kilojoules_per_mole)):
                print(
                    'WARNING: Implicit context energy is NaN after setting positions!'
                )
                print(f'  Explicit pos has NaN: {np.any(np.isnan(explicitPositions))}')
                print(f'  Implicit pos has NaN: {np.any(np.isnan(implicitPositions))}')
                print(
                    f'  implicitAtomIndex range: {self.implicitAtomIndex.min()}-{self.implicitAtomIndex.max()}'
                )
                print(f'  explicitPositions shape: {explicitPositions.shape}')

        periodicDistance = compiled.periodicDistance(
            state.getPeriodicBoxVectors().value_in_unit(nanometers)
        )

        # Attempt pH change if using simulated tempering
        if len(self.pH) > 1:
            self._attemptPHChange()

        # Process residues in random order
        anyChange = False
        for resIndex in np.random.permutation(list(self.titrations)):
            titrations = [self.titrations[resIndex]]

            # Select new state
            stateIndex = [self._selectNewState(titrations[0])]

            # Occasionally attempt multi-site titration
            if np.random.random() < 0.25:
                neighbors = self._findNeighbors(
                    resIndex, explicitPositions, periodicDistance
                )
                if len(neighbors) > 0:
                    i = np.random.choice(neighbors)
                    titrations.append(self.titrations[i])
                    stateIndex.append(self._selectNewState(titrations[-1]))

            # Compute implicit energy change
            currentEnergy = self.implicitContext.getState(
                getEnergy=True
            ).getPotentialEnergy()
            for i, t in zip(stateIndex, titrations, strict=True):
                self._applyStateToContext(
                    t.implicitStates[i],
                    self.implicitContext,
                    self.implicitExceptionIndex,
                    self.implicitInterResidue14,
                    self.implicit14Scale,
                )
            newEnergy = self.implicitContext.getState(
                getEnergy=True
            ).getPotentialEnergy()

            # Metropolis criterion
            if not is_quantity(temperature):
                temperature = temperature * kelvin
            kT = MOLAR_GAS_CONSTANT_R * temperature

            deltaRefEnergy = sum(
                [
                    t.referenceEnergies[i] - t.referenceEnergies[t.currentIndex]
                    for i, t in zip(stateIndex, titrations, strict=True)
                ],
                0.0 * kilojoules_per_mole,
            )
            deltaN = sum(
                [
                    t.implicitStates[i].numHydrogens
                    - t.implicitStates[t.currentIndex].numHydrogens
                    for i, t in zip(stateIndex, titrations, strict=True)
                ]
            )

            w = (newEnergy - currentEnergy - deltaRefEnergy) / kT + deltaN * np.log(
                10.0
            ) * self.pH[self.currentPHIndex]

            if debug:
                currE = currentEnergy.value_in_unit(kilojoules_per_mole)
                newE = newEnergy.value_in_unit(kilojoules_per_mole)
                dE = newE - currE
                dRef = deltaRefEnergy.value_in_unit(kilojoules_per_mole)

                if np.isnan(dE) or np.isnan(dRef):
                    print(
                        f'  Residue {titrations[0].implicitStates[0].residueIndex}: '
                        f'NaN DETECTED! currE={currE}, newE={newE}, '
                        f'refs=[{titrations[0].referenceEnergies}]'
                    )
                else:
                    print(
                        f'  Residue {titrations[0].implicitStates[0].residueIndex}: '
                        f'state {titrations[0].currentIndex}->{stateIndex[0]}, '
                        f'deltaN={deltaN}, pH={self.pH[self.currentPHIndex]:.2f}, '
                        f'dE={dE:.2f} kJ/mol, '
                        f'dRef={dRef:.2f} kJ/mol, '
                        f'w={float(w):.3f}, '
                        f'accept={"yes" if w <= 0 else f"prob={np.exp(-float(w)):.4f}"}'
                    )

            if w > 0.0 and np.exp(-w) < np.random.random():
                # Reject: restore previous state
                for t in titrations:
                    self._applyStateToContext(
                        t.implicitStates[t.currentIndex],
                        self.implicitContext,
                        self.implicitExceptionIndex,
                        self.implicitInterResidue14,
                        self.implicit14Scale,
                    )
                continue

            # Accept the move
            anyChange = True
            for i, t in zip(stateIndex, titrations, strict=True):
                t.currentIndex = i
                self._applyStateToContext(
                    t.explicitStates[i],
                    self.simulation.context,
                    self.explicitExceptionIndex,
                    self.explicitInterResidue14,
                    self.explicit14Scale,
                )
                self._applyStateToContext(
                    t.explicitStates[i],
                    self.relaxationContext,
                    self.explicitExceptionIndex,
                    self.explicitInterResidue14,
                    self.explicit14Scale,
                )

        # Relax solvent if any state changed
        if anyChange:
            self.relaxationContext.setPositions(explicitPositions)
            self.relaxationContext.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
            for param in self.relaxationContext.getParameters():
                self.relaxationContext.setParameter(param, state.getParameters()[param])
            self.relaxationContext.getIntegrator().step(self.relaxationSteps)
            relaxedPositions = self.relaxationContext.getState(
                getPositions=True
            ).getPositions(asNumpy=True)
            self.simulation.context.setPositions(relaxedPositions)

    def setResidueState(self, residueIndex, stateIndex, relax=False):
        """Manually set a residue to a specific protonation state."""
        titration = self.titrations[residueIndex]

        self._applyStateToContext(
            titration.explicitStates[stateIndex],
            self.simulation.context,
            self.explicitExceptionIndex,
            self.explicitInterResidue14,
            self.explicit14Scale,
        )
        self._applyStateToContext(
            titration.explicitStates[stateIndex],
            self.relaxationContext,
            self.explicitExceptionIndex,
            self.explicitInterResidue14,
            self.explicit14Scale,
        )
        self._applyStateToContext(
            titration.implicitStates[stateIndex],
            self.implicitContext,
            self.implicitExceptionIndex,
            self.implicitInterResidue14,
            self.implicit14Scale,
        )

        titration.currentIndex = stateIndex

        if relax:
            positions = self.simulation.context.getState(
                getPositions=True
            ).getPositions(asNumpy=True)
            self.relaxationContext.setPositions(positions)
            self.relaxationContext.getIntegrator().step(self.relaxationSteps)
            self.simulation.context.setPositions(
                self.relaxationContext.getState(getPositions=True).getPositions(
                    asNumpy=True
                )
            )

    def _get_zero_parameters(self, original_parameters, force):
        """Get per-particle parameter values with charge set to 0.

        This is used for "ghost" hydrogens that exist in the fully protonated
        state but not in deprotonated states.
        """
        p = list(original_parameters)
        if isinstance(force, (NonbondedForce, GBSAOBCForce)):
            # First parameter is charge for both NonbondedForce and GBSAOBCForce
            p[0] = 0.0
        else:
            # For custom forces, find the charge parameter by name
            for i in range(force.getNumPerParticleParameters()):
                if force.getPerParticleParameterName(i) == 'charge':
                    p[i] = 0.0
        return tuple(p)

    def _applyStateToContext(
        self, state, context, exceptionIndex, interResidue14, coulomb14Scale
    ):
        """Update context parameters to match a protonation state.

        This modifies Force parameters in the System and then calls
        updateParametersInContext() to push changes to the GPU/CPU context.
        """
        for forceIndex, params in state.particleParameters.items():
            force = context.getSystem().getForce(forceIndex)

            # Only NonbondedForce, GBSAOBCForce and CustomGBForce expose
            # per-particle setParticleParameters. CustomGBForce is how ParmEd
            # builds every GB model, so it must be updated too or the implicit
            # solvation term stays frozen across protonation states.
            if not isinstance(force, (NonbondedForce, GBSAOBCForce, CustomGBForce)):
                continue

            for atomName, atomParams in params.items():
                if atomName not in state.atomIndices:
                    continue
                atomIndex = state.atomIndices[atomName]
                try:
                    force.setParticleParameters(atomIndex, atomParams)
                except TypeError:
                    force.setParticleParameters(atomIndex, *atomParams)

            if isinstance(force, NonbondedForce):
                # Update intra-residue exceptions
                for key, exceptionParams in state.exceptionParameters.get(
                    forceIndex, {}
                ).items():
                    if key in exceptionIndex:
                        p = force.getExceptionParameters(exceptionIndex[key])
                        force.setExceptionParameters(
                            exceptionIndex[key], p[0], p[1], *exceptionParams
                        )

                # Update inter-residue 1-4 interactions
                for index in interResidue14.get(state.residueIndex, []):
                    p1, p2, _, sigma, epsilon = force.getExceptionParameters(index)
                    q1, _, _ = force.getParticleParameters(p1)
                    q2, _, _ = force.getParticleParameters(p2)
                    q1_val = (
                        q1.value_in_unit(elementary_charge)
                        if hasattr(q1, 'value_in_unit')
                        else float(q1)
                    )
                    q2_val = (
                        q2.value_in_unit(elementary_charge)
                        if hasattr(q2, 'value_in_unit')
                        else float(q2)
                    )
                    chargeProd = coulomb14Scale * q1_val * q2_val * elementary_charge**2
                    force.setExceptionParameters(
                        index, p1, p2, chargeProd, sigma, epsilon
                    )

            # Update parameters in context for this force
            # Note: no need to call context.reinitialize() - updateParametersInContext is sufficient
            force.updateParametersInContext(context)

    def _selectNewState(self, titration):
        """Randomly select a new protonation state."""
        numStates = len(titration.implicitStates)
        if numStates == 2:
            return 1 - titration.currentIndex
        stateIndex = titration.currentIndex
        while stateIndex == titration.currentIndex:
            stateIndex = np.random.randint(numStates)
        return stateIndex

    def _findNeighbors(self, resIndex, explicitPositions, periodicDistance):
        """Find nearby titratable residues for multi-site moves."""
        neighbors = []
        titration1 = self.titrations[resIndex]

        for resIndex2 in self.titrations:
            if resIndex2 > resIndex:
                titration2 = self.titrations[resIndex2]
                isNeighbor = False

                for i in titration1.explicitHydrogenIndices:
                    for j in titration2.explicitHydrogenIndices:
                        if (
                            i < len(explicitPositions)
                            and j < len(explicitPositions)
                            and periodicDistance(
                                explicitPositions[i], explicitPositions[j]
                            )
                            < 0.2
                        ):
                            isNeighbor = True

                if isNeighbor:
                    neighbors.append(resIndex2)

        return neighbors

    def _attemptPHChange(self):
        """Attempt to change pH (simulated tempering)."""
        hydrogens = sum(
            t.explicitStates[t.currentIndex].numHydrogens
            for t in self.titrations.values()
        )

        logProbability = [
            self._weights[i] - hydrogens * np.log(10.0) * self.pH[i]
            for i in range(len(self._weights))
        ]
        maxLogProb = max(logProbability)
        offset = maxLogProb + np.log(
            sum(np.exp(x - maxLogProb) for x in logProbability)
        )
        probability = [np.exp(x - offset) for x in logProbability]

        r = np.random.random_sample()
        for j in range(len(probability)):
            if r < probability[j]:
                if j != self.currentPHIndex:
                    self._hasMadeTransition = True
                self.currentPHIndex = j

                if self._updateWeights:
                    self._weights[j] -= self._weightUpdateFactor
                    self._histogram[j] += 1
                    minCounts = min(self._histogram)

                    if minCounts > 20 and minCounts >= 0.2 * sum(self._histogram) / len(
                        self._histogram
                    ):
                        self._weightUpdateFactor *= 0.5
                        self._histogram = [0] * len(self.pH)
                        self._weights = [x - self._weights[0] for x in self._weights]
                    elif (
                        not self._hasMadeTransition
                        and probability[self.currentPHIndex] > 0.99
                        and self._weightUpdateFactor < 1024.0
                    ):
                        self._weightUpdateFactor *= 2.0
                        self._histogram = [0] * len(self.pH)
                return
            r -= probability[j]
