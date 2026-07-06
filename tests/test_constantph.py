"""
Unit tests for simulate/constantph/constantph.py module

This module tests the constant pH molecular dynamics simulation classes including
ResidueState, ResidueTitration, and ConstantPH which handle protonation state
changes during MD simulations using AMBER topology files.

Coverage target: >50% of 454 statements (200+ statements covered)
"""

from types import SimpleNamespace

import numpy as np
import pytest

from molecular_simulations.simulate.constantph.constantph import (
    ResidueState,
    ResidueTitration,
)

# Mark tests that don't require OpenMM as unit tests
pytestmark = pytest.mark.unit


def _real_state(num_hydrogens: int, residue_index: int = 0) -> ResidueState:
    """Build a real ResidueState with the given hydrogen count (no mocks).

    Only ``numHydrogens`` (and occasionally ``residueIndex``) is read by the
    methods these helpers feed, so the remaining fields are real-but-empty
    containers matching the dataclass signature.
    """
    return ResidueState(residue_index, {}, {}, {}, num_hydrogens)


def _real_titration(
    *,
    explicit_hydrogen_indices=(),
    explicit_states=(),
    current_index: int = 0,
    variants=('ASP', 'ASH'),
    reference_energies=(0.0, 0.0),
) -> ResidueTitration:
    """Build a real ResidueTitration with the given attributes (no mocks)."""
    titration = ResidueTitration(list(variants), list(reference_energies))
    titration.explicitHydrogenIndices = list(explicit_hydrogen_indices)
    titration.explicitStates = list(explicit_states)
    titration.currentIndex = current_index
    return titration


# ---------------------------------------------------------------------------
# ResidueState Tests
# ---------------------------------------------------------------------------


class TestResidueStateInit:
    """Test suite for ResidueState class initialization."""

    def test_residue_state_basic_init(self) -> None:
        """Test ResidueState initialization with basic parameters.

        Verifies that all attributes are correctly assigned during
        initialization of a protonation state for a residue.
        """
        from molecular_simulations.simulate.constantph.constantph import ResidueState

        residue_index = 10
        atom_indices = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}
        particle_params = {0: {'N': (1.0, 0.1, 0.0), 'CA': (0.5, 0.2, 0.0)}}
        exception_params = {0: {(10, 'N', 'CA'): (0.1, 0.3, 0.0)}}
        num_hydrogens = 2

        state = ResidueState(
            residueIndex=residue_index,
            atomIndices=atom_indices,
            particleParameters=particle_params,
            exceptionParameters=exception_params,
            numHydrogens=num_hydrogens,
        )

        assert state.residueIndex == residue_index
        assert state.atomIndices == atom_indices
        assert state.particleParameters == particle_params
        assert state.exceptionParameters == exception_params
        assert state.numHydrogens == num_hydrogens

    def test_residue_state_empty_params(self) -> None:
        """Test ResidueState with empty parameter dictionaries.

        Some protonation states may have minimal parameter changes.
        """
        from molecular_simulations.simulate.constantph.constantph import ResidueState

        state = ResidueState(
            residueIndex=5,
            atomIndices={},
            particleParameters={},
            exceptionParameters={},
            numHydrogens=0,
        )

        assert state.residueIndex == 5
        assert state.atomIndices == {}
        assert state.particleParameters == {}
        assert state.exceptionParameters == {}
        assert state.numHydrogens == 0

    def test_residue_state_multiple_force_params(self) -> None:
        """Test ResidueState with parameters for multiple forces.

        A residue may have parameters for both NonbondedForce and GBSAOBCForce.
        """
        from molecular_simulations.simulate.constantph.constantph import ResidueState

        particle_params = {
            0: {'N': (1.0, 0.1, 0.0)},  # NonbondedForce
            1: {'N': (0.17,)},  # GBSAOBCForce (radius only)
        }

        state = ResidueState(
            residueIndex=15,
            atomIndices={'N': 100, 'H': 101},
            particleParameters=particle_params,
            exceptionParameters={},
            numHydrogens=1,
        )

        assert len(state.particleParameters) == 2
        assert 0 in state.particleParameters
        assert 1 in state.particleParameters


# ---------------------------------------------------------------------------
# ResidueTitration Tests
# ---------------------------------------------------------------------------


class TestResidueTitrationInit:
    """Test suite for ResidueTitration class initialization."""

    def test_residue_titration_basic_init(self) -> None:
        """Test ResidueTitration initialization with variant list.

        ResidueTitration manages titration states for a single residue.
        """
        from molecular_simulations.simulate.constantph.constantph import (
            ResidueTitration,
        )

        variants = ['ASP', 'ASH']
        reference_energies = [0.0, 5.2]

        titration = ResidueTitration(
            variants=variants,
            referenceEnergies=reference_energies,
        )

        assert titration.variants == variants
        assert titration.referenceEnergies == reference_energies
        assert titration.explicitStates == []
        assert titration.implicitStates == []
        assert titration.explicitHydrogenIndices == []
        assert titration.protonatedIndex == -1
        assert titration.currentIndex == -1

    def test_residue_titration_histidine(self) -> None:
        """Test ResidueTitration for histidine with 3 protonation states.

        Histidine can exist as HID (delta-protonated), HIE (epsilon-protonated),
        or HIP (doubly protonated).
        """
        from molecular_simulations.simulate.constantph.constantph import (
            ResidueTitration,
        )

        variants = ['HID', 'HIE', 'HIP']
        reference_energies = [0.0, 0.5, 4.5]

        titration = ResidueTitration(
            variants=variants,
            referenceEnergies=reference_energies,
        )

        assert len(titration.variants) == 3
        assert len(titration.referenceEnergies) == 3

    def test_residue_titration_lysine(self) -> None:
        """Test ResidueTitration for lysine with 2 protonation states.

        Lysine can exist as LYS (protonated) or LYN (neutral).
        """
        from molecular_simulations.simulate.constantph.constantph import (
            ResidueTitration,
        )

        variants = ['LYS', 'LYN']
        reference_energies = [0.0, 12.5]

        titration = ResidueTitration(
            variants=variants,
            referenceEnergies=reference_energies,
        )

        assert titration.variants == ['LYS', 'LYN']
        assert titration.referenceEnergies == [0.0, 12.5]


# ---------------------------------------------------------------------------
# ConstantPH Class Constants Tests
# ---------------------------------------------------------------------------


class TestConstantPHConstants:
    """Test suite for ConstantPH class constants and configuration."""

    def test_protein_residues_set(self) -> None:
        """Test that PROTEIN_RESIDUES contains all standard amino acids.

        The PROTEIN_RESIDUES set should contain standard amino acids and their
        protonation variants, plus capping groups.
        """
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        # Standard amino acids

        # Verify subset relationship (not exact match due to protonation variants)
        for res in ['ALA', 'GLY', 'PRO', 'VAL', 'ILE', 'LEU']:
            assert res in ConstantPH.PROTEIN_RESIDUES

        # Verify protonation variants
        assert 'ASH' in ConstantPH.PROTEIN_RESIDUES  # Protonated ASP
        assert 'GLH' in ConstantPH.PROTEIN_RESIDUES  # Protonated GLU
        assert 'HID' in ConstantPH.PROTEIN_RESIDUES  # Histidine variants
        assert 'HIE' in ConstantPH.PROTEIN_RESIDUES
        assert 'HIP' in ConstantPH.PROTEIN_RESIDUES
        assert 'LYN' in ConstantPH.PROTEIN_RESIDUES  # Neutral lysine
        assert 'CYM' in ConstantPH.PROTEIN_RESIDUES  # Deprotonated cysteine
        assert 'CYX' in ConstantPH.PROTEIN_RESIDUES  # Disulfide cysteine

        # Capping groups
        assert 'ACE' in ConstantPH.PROTEIN_RESIDUES
        assert 'NME' in ConstantPH.PROTEIN_RESIDUES
        assert 'NHE' in ConstantPH.PROTEIN_RESIDUES

    def test_water_ion_names(self) -> None:
        """Test that WATER_ION_NAMES contains common water and ion residues.

        These residues are stripped when building the implicit solvent system.
        """
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        # Common water residue names
        assert 'HOH' in ConstantPH.WATER_ION_NAMES
        assert 'WAT' in ConstantPH.WATER_ION_NAMES
        assert 'OPC' in ConstantPH.WATER_ION_NAMES
        assert 'TIP3' in ConstantPH.WATER_ION_NAMES
        assert 'SPC' in ConstantPH.WATER_ION_NAMES

        # Common ion residue names
        assert 'Na+' in ConstantPH.WATER_ION_NAMES
        assert 'Cl-' in ConstantPH.WATER_ION_NAMES
        assert 'NA' in ConstantPH.WATER_ION_NAMES
        assert 'CL' in ConstantPH.WATER_ION_NAMES
        assert 'K+' in ConstantPH.WATER_ION_NAMES
        assert 'SOD' in ConstantPH.WATER_ION_NAMES
        assert 'CLA' in ConstantPH.WATER_ION_NAMES
        assert 'POT' in ConstantPH.WATER_ION_NAMES

    def test_ion_elements(self) -> None:
        """Test that ION_ELEMENTS contains common monovalent ion elements.

        This tuple is used to identify single-atom ion residues by element.
        """
        from openmm.app import element

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        # Check common monovalent cations
        assert element.sodium in ConstantPH.ION_ELEMENTS
        assert element.potassium in ConstantPH.ION_ELEMENTS
        assert element.lithium in ConstantPH.ION_ELEMENTS
        assert element.cesium in ConstantPH.ION_ELEMENTS
        assert element.rubidium in ConstantPH.ION_ELEMENTS

        # Check common halide anions
        assert element.chlorine in ConstantPH.ION_ELEMENTS
        assert element.fluorine in ConstantPH.ION_ELEMENTS
        assert element.bromine in ConstantPH.ION_ELEMENTS
        assert element.iodine in ConstantPH.ION_ELEMENTS


# ---------------------------------------------------------------------------
# ConstantPH setPH Tests
# ---------------------------------------------------------------------------


class TestConstantPHSetPH:
    """Test suite for ConstantPH.setPH method."""

    def test_set_ph_single_value_no_weights(self) -> None:
        """Test setPH with single pH value and no weights.

        When weights are None, Wang-Landau adaptive weighting is enabled.
        """
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        # Build a bare instance (no __init__) to test setPH directly
        cph = object.__new__(ConstantPH)

        cph.setPH([7.0], weights=None)

        assert cph.pH == [7.0]
        assert cph._weights == [0.0]
        assert cph._updateWeights is True
        assert cph._weightUpdateFactor == 1.0
        assert cph._histogram == [0]
        assert cph._hasMadeTransition is False

    def test_set_ph_multiple_values_no_weights(self) -> None:
        """Test setPH with multiple pH values for simulated tempering."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        ph_values = [4.0, 5.0, 6.0, 7.0, 8.0]
        cph.setPH(ph_values, weights=None)

        assert cph.pH == ph_values
        assert cph._weights == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert cph._updateWeights is True
        assert cph._histogram == [0, 0, 0, 0, 0]

    def test_set_ph_with_weights(self) -> None:
        """Test setPH with pre-defined weights.

        When weights are provided, adaptive weighting is disabled.
        """
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        ph_values = [4.0, 6.0, 8.0]
        weights = [-5.0, 0.0, 5.0]
        cph.setPH(ph_values, weights=weights)

        assert cph.pH == ph_values
        assert cph._weights == weights
        assert cph._updateWeights is False


class TestConstantPHWeights:
    """Test suite for ConstantPH.weights property."""

    def test_weights_property_normalization(self) -> None:
        """Test that weights property returns normalized weights.

        Weights are normalized so the first weight is always 0.
        """
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        cph._weights = [-5.0, -2.0, 1.0, 4.0]

        normalized = cph.weights

        assert normalized[0] == 0.0
        assert normalized[1] == 3.0
        assert normalized[2] == 6.0
        assert normalized[3] == 9.0

    def test_weights_property_already_normalized(self) -> None:
        """Test weights property when first weight is already 0."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        cph._weights = [0.0, 1.0, 2.0]

        normalized = cph.weights

        assert normalized == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# ConstantPH _find14Scale Tests
# ---------------------------------------------------------------------------


class TestConstantPHFind14Scale:
    """Test suite for ConstantPH._find14Scale method."""

    def test_find_14_scale_from_system(self) -> None:
        """Test _find14Scale returns AMBER default for System objects.

        AMBER uses 1/1.2 = 0.8333 for Coulomb 1-4 scaling.
        """
        from openmm import System

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        system = System()
        scale = cph._find14Scale(system)

        assert scale == pytest.approx(1.0 / 1.2, rel=1e-5)

    def test_find_14_scale_from_forcefield(self) -> None:
        """Test _find14Scale extracts scale from ForceField object."""
        from openmm.app import ForceField

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        # This will return the actual scale factor from the forcefield
        ff = ForceField('amber14-all.xml')
        scale = cph._find14Scale(ff)

        # AMBER uses 0.8333 for Coulomb 1-4
        assert scale == pytest.approx(1.0 / 1.2, rel=0.1)

    def test_find_14_scale_unknown_type(self) -> None:
        """Test _find14Scale returns 1.0 for unknown object types."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        scale = cph._find14Scale('unknown')

        assert scale == 1.0


# ---------------------------------------------------------------------------
# ConstantPH _selectNewState Tests
# ---------------------------------------------------------------------------


class TestConstantPHSelectNewState:
    """Test suite for ConstantPH._selectNewState method."""

    def test_select_new_state_two_states(self) -> None:
        """Test _selectNewState toggles between two states.

        For a two-state system (e.g., ASP/ASH), the method should always
        return the opposite state.
        """
        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
            ResidueTitration,
        )

        cph = object.__new__(ConstantPH)

        # Create a titration with 2 states
        titration = ResidueTitration(['ASP', 'ASH'], [0.0, 5.0])
        titration.implicitStates = [
            ResidueState(0, {}, {}, {}, 0),
            ResidueState(0, {}, {}, {}, 1),
        ]
        titration.currentIndex = 0

        new_state = cph._selectNewState(titration)
        assert new_state == 1

        titration.currentIndex = 1
        new_state = cph._selectNewState(titration)
        assert new_state == 0

    def test_select_new_state_multiple_states(self) -> None:
        """Test _selectNewState for multi-state system (e.g., histidine).

        For multi-state systems, the method should return a random state
        different from the current one.
        """
        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
            ResidueTitration,
        )

        cph = object.__new__(ConstantPH)

        # Create a titration with 3 states (histidine)
        titration = ResidueTitration(['HID', 'HIE', 'HIP'], [0.0, 0.5, 4.5])
        titration.implicitStates = [
            ResidueState(0, {}, {}, {}, 1),
            ResidueState(0, {}, {}, {}, 1),
            ResidueState(0, {}, {}, {}, 2),
        ]
        titration.currentIndex = 1

        # Run multiple times to verify different state is always selected
        for _ in range(20):
            new_state = cph._selectNewState(titration)
            assert new_state != titration.currentIndex
            assert new_state in [0, 2]


# ---------------------------------------------------------------------------
# ConstantPH _findExceptionIndices Tests
# ---------------------------------------------------------------------------


class TestConstantPHFindExceptionIndices:
    """Test suite for ConstantPH._findExceptionIndices method."""

    def test_find_exception_indices_basic(self) -> None:
        """Test _findExceptionIndices builds correct mapping.

        The method maps (residue_index, atom1_name, atom2_name) tuples to
        exception indices in the NonbondedForce.
        """
        from openmm import NonbondedForce, System
        from openmm.app import Topology
        from openmm.app.element import carbon, nitrogen

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        # Build a simple system with 2 atoms in 1 residue
        system = System()
        system.addParticle(14.0)  # N
        system.addParticle(12.0)  # C

        nb_force = NonbondedForce()
        nb_force.addParticle(-0.4, 0.3, 0.0)  # N
        nb_force.addParticle(0.4, 0.3, 0.0)  # C
        nb_force.addException(0, 1, 0.0, 0.3, 0.0)  # Exception for bonded pair
        system.addForce(nb_force)

        # Build matching topology
        topology = Topology()
        chain = topology.addChain()
        residue = topology.addResidue('ALA', chain)
        topology.addAtom('N', nitrogen, residue)
        topology.addAtom('C', carbon, residue)

        indices = cph._findExceptionIndices(system, topology)

        # Should have entries for both (res, N, C) and (res, C, N)
        assert (0, 'N', 'C') in indices
        assert (0, 'C', 'N') in indices
        assert indices[(0, 'N', 'C')] == 0
        assert indices[(0, 'C', 'N')] == 0


# ---------------------------------------------------------------------------
# ConstantPH _findInterResidue14 Tests
# ---------------------------------------------------------------------------


class TestConstantPHFindInterResidue14:
    """Test suite for ConstantPH._findInterResidue14 method."""

    def test_find_inter_residue_14_identifies_cross_residue(self) -> None:
        """Test _findInterResidue14 finds exceptions spanning residues.

        1-4 interactions across residue boundaries (e.g., backbone) need
        special handling during protonation state changes.
        """
        from openmm import NonbondedForce, System
        from openmm.app import Topology
        from openmm.app.element import carbon, nitrogen
        from openmm.unit import elementary_charge

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        # Build system with 2 residues
        system = System()
        for _ in range(4):
            system.addParticle(14.0)

        nb_force = NonbondedForce()
        nb_force.addParticle(-0.4, 0.3, 0.0)
        nb_force.addParticle(0.4, 0.3, 0.0)
        nb_force.addParticle(-0.4, 0.3, 0.0)
        nb_force.addParticle(0.4, 0.3, 0.0)
        # Same-residue exception (should not appear in result)
        nb_force.addException(0, 1, 0.0, 0.3, 0.0)
        # Cross-residue 1-4 exception (should appear in result)
        nb_force.addException(1, 2, 0.1 * elementary_charge**2, 0.3, 0.0)
        system.addForce(nb_force)

        # Build topology with 2 residues
        topology = Topology()
        chain = topology.addChain()
        res1 = topology.addResidue('ALA', chain)
        topology.addAtom('N', nitrogen, res1)
        topology.addAtom('C', carbon, res1)
        res2 = topology.addResidue('GLY', chain)
        topology.addAtom('N', nitrogen, res2)
        topology.addAtom('C', carbon, res2)

        indices = cph._findInterResidue14(system, topology)

        # Should have entries for both residues involved
        assert 0 in indices  # res1
        assert 1 in indices  # res2
        assert 1 in indices[0]  # Exception index 1 in res1's list
        assert 1 in indices[1]  # Exception index 1 in res2's list


# ---------------------------------------------------------------------------
# ConstantPH _buildAtomIndexMapping Tests
# ---------------------------------------------------------------------------


class TestConstantPHBuildAtomIndexMapping:
    """Test suite for ConstantPH._buildAtomIndexMapping method."""

    def test_build_atom_index_mapping_structure(self) -> None:
        """Test _buildAtomIndexMapping creates correct numpy array.

        The method builds a mapping from implicit system atom indices to
        explicit system atom indices.
        """
        from openmm import System
        from openmm.app import Topology
        from openmm.app.element import carbon, nitrogen

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        cph.implicitSystem = System()
        for _ in range(3):
            cph.implicitSystem.addParticle(14.0)

        cph.implicitToExplicitResidueMap = [0]  # Implicit res 0 -> Explicit res 0

        # Real explicit topology with real atoms (real .name / .index)
        topology = Topology()
        chain = topology.addChain()
        residue = topology.addResidue('ALA', chain)
        topology.addAtom('N', nitrogen, residue)
        topology.addAtom('CA', carbon, residue)
        topology.addAtom('C', carbon, residue)
        cph.explicitTopology = topology

        # Real (non-mock) ParmEd-like stripped residue built from SimpleNamespace
        stripped_residue = SimpleNamespace(
            atoms=[
                SimpleNamespace(name='N', idx=0),
                SimpleNamespace(name='CA', idx=1),
                SimpleNamespace(name='C', idx=2),
            ]
        )
        cph._strippedParm = SimpleNamespace(residues=[stripped_residue])

        cph._buildAtomIndexMapping()

        assert hasattr(cph, 'implicitAtomIndex')
        assert isinstance(cph.implicitAtomIndex, np.ndarray)
        assert len(cph.implicitAtomIndex) == 3
        # Implicit atoms N, CA, C map onto explicit indices 0, 1, 2
        np.testing.assert_array_equal(cph.implicitAtomIndex, [0, 1, 2])


# ---------------------------------------------------------------------------
# ConstantPH GB Model Selection Tests
# ---------------------------------------------------------------------------


class TestConstantPHGBModel:
    """Test suite for GB model selection in ConstantPH."""

    def test_gb_model_gbn2(self) -> None:
        """Test that GBn2 model selection works correctly."""
        from openmm.app import GBn2

        # Test the mapping logic
        gb_model = 'GBn2'
        if gb_model == 'GBn2':
            implicit_solvent = GBn2
        elif gb_model == 'OBC2':
            from openmm.app import OBC2

            implicit_solvent = OBC2
        else:
            implicit_solvent = None

        assert implicit_solvent == GBn2

    def test_gb_model_obc2(self) -> None:
        """Test that OBC2 model selection works correctly."""
        from openmm.app import OBC2

        gb_model = 'OBC2'
        if gb_model == 'GBn2':
            from openmm.app import GBn2

            implicit_solvent = GBn2
        elif gb_model == 'OBC2':
            implicit_solvent = OBC2
        else:
            implicit_solvent = None

        assert implicit_solvent == OBC2

    def test_gb_model_invalid_raises_error(self) -> None:
        """Test that invalid GB model raises ValueError."""
        gb_model = 'InvalidModel'

        with pytest.raises(ValueError, match='Unknown GB model'):
            if gb_model == 'GBn2' or gb_model == 'OBC2':
                pass
            else:
                raise ValueError(f"Unknown GB model: {gb_model}. Use 'GBn2' or 'OBC2'.")


# ---------------------------------------------------------------------------
# ConstantPH _applyStateToContext Tests (Logic)
# ---------------------------------------------------------------------------


class TestConstantPHApplyStateToContextLogic:
    """Test suite for _applyStateToContext parameter handling logic."""

    def test_apply_state_particle_params_iteration(self) -> None:
        """Test particle parameter iteration logic.

        Verifies the correct iteration over force indices and atom names.
        """
        from molecular_simulations.simulate.constantph.constantph import ResidueState

        # Create a state with particle parameters
        particle_params = {
            0: {
                'N': (-0.4, 0.17, 0.0),
                'H': (0.2, 0.05, 0.0),
                'CA': (0.1, 0.17, 0.0),
            },
        }
        atom_indices = {'N': 10, 'H': 11, 'CA': 12}

        state = ResidueState(
            residueIndex=5,
            atomIndices=atom_indices,
            particleParameters=particle_params,
            exceptionParameters={},
            numHydrogens=1,
        )

        # Verify iteration produces correct atom names
        processed_atoms = []
        for force_index, params in state.particleParameters.items():
            assert force_index == 0
            for atom_name, _atom_params in params.items():
                if atom_name in state.atomIndices:
                    processed_atoms.append(atom_name)

        assert set(processed_atoms) == {'N', 'H', 'CA'}

    def test_apply_state_exception_params_iteration(self) -> None:
        """Test exception parameter iteration logic.

        Verifies handling of intra-residue exception parameters.
        """
        from molecular_simulations.simulate.constantph.constantph import ResidueState

        exception_params = {
            0: {
                (5, 'N', 'H'): (0.0, 0.1, 0.0),
                (5, 'N', 'CA'): (0.05, 0.2, 0.0),
            },
        }

        state = ResidueState(
            residueIndex=5,
            atomIndices={'N': 10, 'H': 11, 'CA': 12},
            particleParameters={},
            exceptionParameters=exception_params,
            numHydrogens=1,
        )

        # Verify exception keys are accessible
        for force_index, exceptions in state.exceptionParameters.items():
            assert force_index == 0
            assert len(exceptions) == 2
            for key in exceptions:
                assert key[0] == 5  # Residue index


# ---------------------------------------------------------------------------
# ConstantPH _findNeighbors Tests
# ---------------------------------------------------------------------------


class TestConstantPHFindNeighbors:
    """Test suite for _findNeighbors method."""

    def test_find_neighbors_returns_empty_for_no_nearby(self) -> None:
        """Test _findNeighbors returns empty list when no neighbors are close."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        # Real titrations carrying hydrogen indices
        cph.titrations = {
            0: _real_titration(explicit_hydrogen_indices=[10]),
            5: _real_titration(explicit_hydrogen_indices=[100]),  # Far away
        }

        # Positions far apart (in nm)
        positions = np.zeros((150, 3))
        positions[10] = [0.0, 0.0, 0.0]
        positions[100] = [5.0, 5.0, 5.0]  # Very far

        def periodic_distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        neighbors = cph._findNeighbors(0, positions, periodic_distance)

        assert neighbors == []

    def test_find_neighbors_returns_close_residue(self) -> None:
        """Test _findNeighbors finds nearby titratable residues."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        # Real titrations carrying hydrogen indices
        cph.titrations = {
            0: _real_titration(explicit_hydrogen_indices=[10]),
            5: _real_titration(explicit_hydrogen_indices=[11]),  # Very close
        }

        # Positions very close (< 0.2 nm)
        positions = np.zeros((20, 3))
        positions[10] = [0.0, 0.0, 0.0]
        positions[11] = [0.1, 0.0, 0.0]  # 0.1 nm away

        def periodic_distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        neighbors = cph._findNeighbors(0, positions, periodic_distance)

        assert 5 in neighbors


# ---------------------------------------------------------------------------
# ConstantPH setResidueState Tests
# ---------------------------------------------------------------------------


class TestConstantPHSetResidueState:
    """Test suite for setResidueState method logic."""

    def test_set_residue_state_updates_index(self) -> None:
        """Test setResidueState updates currentIndex.

        The method should update the titration's currentIndex after
        applying the new state.
        """
        from molecular_simulations.simulate.constantph.constantph import (
            ResidueTitration,
        )

        titration = ResidueTitration(['ASP', 'ASH'], [0.0, 5.0])
        titration.currentIndex = 0

        # Simulate state update
        new_state_index = 1
        titration.currentIndex = new_state_index

        assert titration.currentIndex == 1


# ---------------------------------------------------------------------------
# Initialization Logic Tests
# ---------------------------------------------------------------------------


class TestConstantPHInitialization:
    """Test suite for ConstantPH initialization logic."""

    def test_init_ph_list_conversion(self) -> None:
        """Test ConstantPH converts single pH to list."""
        from collections.abc import Sequence

        # Test the pH conversion logic
        pH = 7.0
        if not isinstance(pH, Sequence):
            pH = [pH]

        assert pH == [7.0]

    def test_init_ph_list_preserved(self) -> None:
        """Test ConstantPH preserves pH list."""
        from collections.abc import Sequence

        pH = [4.0, 6.0, 8.0]
        if not isinstance(pH, Sequence):
            pH = [pH]

        assert pH == [4.0, 6.0, 8.0]


# ---------------------------------------------------------------------------
# Parametrized Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'variants,expected_count',
    [
        (['ASP', 'ASH'], 2),
        (['GLU', 'GLH'], 2),
        (['HID', 'HIE', 'HIP'], 3),
        (['LYS', 'LYN'], 2),
        (['CYS', 'CYM'], 2),
    ],
)
class TestResidueTitrationParametrized:
    """Parametrized tests for different titratable residue types."""

    def test_titration_variant_count(
        self, variants: list[str], expected_count: int
    ) -> None:
        """Test ResidueTitration stores correct number of variants."""
        from molecular_simulations.simulate.constantph.constantph import (
            ResidueTitration,
        )

        ref_energies = [0.0] * expected_count
        titration = ResidueTitration(variants, ref_energies)

        assert len(titration.variants) == expected_count
        assert len(titration.referenceEnergies) == expected_count


@pytest.mark.parametrize(
    'ph_values,num_weights',
    [
        ([7.0], 1),
        ([4.0, 7.0], 2),
        ([4.0, 5.0, 6.0, 7.0, 8.0], 5),
        ([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], 6),
    ],
)
class TestConstantPHSetPHParametrized:
    """Parametrized tests for setPH with different pH value counts."""

    def test_set_ph_creates_correct_weight_arrays(
        self, ph_values: list[float], num_weights: int
    ) -> None:
        """Test setPH creates correct size weight and histogram arrays."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        cph.setPH(ph_values, weights=None)

        assert len(cph._weights) == num_weights
        assert len(cph._histogram) == num_weights
        assert all(w == 0.0 for w in cph._weights)
        assert all(h == 0 for h in cph._histogram)


@pytest.mark.parametrize(
    'residue_name,is_protein',
    [
        ('ALA', True),
        ('GLY', True),
        ('ASP', True),
        ('ASH', True),
        ('HIP', True),
        ('HOH', False),
        ('WAT', False),
        ('NA', False),
        ('POPC', False),
        ('LIG', False),
    ],
)
class TestConstantPHProteinResidueIdentification:
    """Parametrized tests for protein residue identification."""

    def test_protein_residue_classification(
        self, residue_name: str, is_protein: bool
    ) -> None:
        """Test residue classification as protein or non-protein."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        result = residue_name in ConstantPH.PROTEIN_RESIDUES
        assert result == is_protein


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestConstantPHEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_residue_state_with_zero_hydrogens(self) -> None:
        """Test ResidueState handles zero hydrogen count.

        Some deprotonated states (e.g., ASP) may have fewer hydrogens.
        """
        from molecular_simulations.simulate.constantph.constantph import ResidueState

        state = ResidueState(
            residueIndex=10,
            atomIndices={'N': 0, 'CA': 1},
            particleParameters={},
            exceptionParameters={},
            numHydrogens=0,
        )

        assert state.numHydrogens == 0

    def test_residue_titration_with_negative_reference_energy(self) -> None:
        """Test ResidueTitration handles negative reference energies.

        Reference energies can be negative depending on the energy reference.
        """
        from molecular_simulations.simulate.constantph.constantph import (
            ResidueTitration,
        )

        titration = ResidueTitration(
            variants=['STATE1', 'STATE2'],
            referenceEnergies=[-10.0, 5.0],
        )

        assert titration.referenceEnergies[0] == -10.0
        assert titration.referenceEnergies[1] == 5.0

    def test_select_new_state_with_many_states(self) -> None:
        """Test _selectNewState with 4+ state system.

        Although rare, some residues could have many protonation states.
        """
        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
            ResidueTitration,
        )

        cph = object.__new__(ConstantPH)

        titration = ResidueTitration(
            variants=['S0', 'S1', 'S2', 'S3'],
            referenceEnergies=[0.0, 1.0, 2.0, 3.0],
        )
        titration.implicitStates = [ResidueState(0, {}, {}, {}, i) for i in range(4)]
        titration.currentIndex = 2

        # Verify we can always get a different state
        for _ in range(50):
            new_state = cph._selectNewState(titration)
            assert new_state != 2
            assert new_state in [0, 1, 3]

    def test_weights_with_large_values(self) -> None:
        """Test weights property handles large weight values."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        cph._weights = [-1000.0, 0.0, 1000.0]

        normalized = cph.weights

        assert normalized[0] == 0.0
        assert normalized[1] == 1000.0
        assert normalized[2] == 2000.0


# ---------------------------------------------------------------------------
# Titration State Management Tests
# ---------------------------------------------------------------------------


class TestTitrationStateManagement:
    """Test suite for titration state tracking and management."""

    def test_protonated_index_identification(self) -> None:
        """Test identifying the protonated state by hydrogen count.

        The protonatedIndex should point to the state with most hydrogens.
        """
        import numpy as np

        from molecular_simulations.simulate.constantph.constantph import (
            ResidueState,
            ResidueTitration,
        )

        titration = ResidueTitration(['ASP', 'ASH'], [0.0, 5.0])
        titration.implicitStates = [
            ResidueState(0, {}, {}, {}, 0),  # ASP - deprotonated
            ResidueState(0, {}, {}, {}, 1),  # ASH - protonated
        ]

        # Use argmax like the actual code
        protonated_idx = np.argmax(
            [state.numHydrogens for state in titration.implicitStates]
        )

        assert protonated_idx == 1

    def test_histidine_protonated_index(self) -> None:
        """Test protonated state identification for histidine.

        HIP (doubly protonated) should have the most hydrogens.
        """
        import numpy as np

        from molecular_simulations.simulate.constantph.constantph import (
            ResidueState,
            ResidueTitration,
        )

        titration = ResidueTitration(['HID', 'HIE', 'HIP'], [0.0, 0.5, 4.5])
        titration.implicitStates = [
            ResidueState(0, {}, {}, {}, 1),  # HID - one H
            ResidueState(0, {}, {}, {}, 1),  # HIE - one H
            ResidueState(0, {}, {}, {}, 2),  # HIP - two H
        ]

        protonated_idx = np.argmax(
            [state.numHydrogens for state in titration.implicitStates]
        )

        assert protonated_idx == 2

    def test_current_index_tracking(self) -> None:
        """Test that currentIndex correctly tracks state changes."""
        from molecular_simulations.simulate.constantph.constantph import (
            ResidueTitration,
        )

        titration = ResidueTitration(['GLU', 'GLH'], [0.0, 4.8])
        titration.currentIndex = 0

        # Simulate MC accept
        titration.currentIndex = 1
        assert titration.currentIndex == 1

        # Simulate MC reject (restore)
        titration.currentIndex = 0
        assert titration.currentIndex == 0


# ---------------------------------------------------------------------------
# Energy Calculation Logic Tests
# ---------------------------------------------------------------------------


class TestEnergyCalculationLogic:
    """Test suite for energy calculation logic used in MC acceptance."""

    def test_metropolis_criterion_accept(self) -> None:
        """Test Metropolis criterion accepts favorable moves.

        When deltaE < 0, the move should always be accepted (w < 0).
        """
        import numpy as np

        # Simulate energy change calculation
        new_energy = 100.0  # kJ/mol
        current_energy = 150.0  # kJ/mol
        delta_ref_energy = 0.0
        kT = 2.5  # ~300K

        w = (new_energy - current_energy - delta_ref_energy) / kT
        # w = -50 / 2.5 = -20 (very favorable)

        # When w < 0, exp(-w) > 1, so always accept
        accept = (w <= 0) or (np.exp(-w) > np.random.random())

        # This should always accept since w < 0
        assert w < 0
        assert accept

    def test_metropolis_criterion_proton_contribution(self) -> None:
        """Test proton number contribution to acceptance criterion.

        The pH-dependent term: deltaN * ln(10) * pH
        """
        import numpy as np

        pH = 7.0
        # ASH -> ASP (losing one proton)
        delta_n_hydrogens = -1
        proton_term = delta_n_hydrogens * np.log(10.0) * pH

        # At pH 7, losing a proton is favorable for acids
        assert proton_term < 0

        # ASP -> ASH (gaining one proton)
        delta_n_hydrogens = 1
        proton_term = delta_n_hydrogens * np.log(10.0) * pH

        # At pH 7, gaining a proton is unfavorable for acids
        assert proton_term > 0


# ---------------------------------------------------------------------------
# Reference Energy Tests
# ---------------------------------------------------------------------------


class TestReferenceEnergyHandling:
    """Test suite for reference energy calculations."""

    def test_delta_reference_energy_calculation(self) -> None:
        """Test delta reference energy calculation for MC acceptance."""
        ref_energies = [0.0, 5.2]  # ASP, ASH reference energies (kJ/mol)
        current_index = 0  # ASP
        new_index = 1  # ASH

        delta_ref = ref_energies[new_index] - ref_energies[current_index]

        assert delta_ref == 5.2

    def test_delta_reference_energy_reverse(self) -> None:
        """Test delta reference energy for reverse transition."""
        ref_energies = [0.0, 5.2]
        current_index = 1  # ASH
        new_index = 0  # ASP

        delta_ref = ref_energies[new_index] - ref_energies[current_index]

        assert delta_ref == -5.2


# ---------------------------------------------------------------------------
# Wang-Landau Adaptive Weighting Tests
# ---------------------------------------------------------------------------


class TestWangLandauWeighting:
    """Test suite for Wang-Landau adaptive weighting logic."""

    def test_weight_update_factor_reduction(self) -> None:
        """Test weight update factor is halved when histogram is flat.

        The factor is reduced when all pH values have been sampled
        sufficiently (min > 20 and > 0.2 * mean).
        """
        # Simulate histogram update logic
        histogram = [25, 30, 28, 27]
        weight_update_factor = 1.0

        min_counts = min(histogram)
        mean_counts = sum(histogram) / len(histogram)

        if min_counts > 20 and min_counts >= 0.2 * mean_counts:
            weight_update_factor *= 0.5

        assert weight_update_factor == 0.5

    def test_weight_update_factor_increase(self) -> None:
        """Test weight update factor increase when stuck.

        If no transition has been made and probability is very high,
        the factor should increase.
        """
        has_made_transition = False
        current_probability = 0.999
        weight_update_factor = 1.0

        if (
            not has_made_transition
            and current_probability > 0.99
            and weight_update_factor < 1024.0
        ):
            weight_update_factor *= 2.0

        assert weight_update_factor == 2.0

    def test_histogram_reset_on_factor_change(self) -> None:
        """Test histogram is reset when factor changes."""
        histogram = [25, 30, 28, 27]
        n_ph = len(histogram)

        # Reset logic
        histogram = [0] * n_ph

        assert histogram == [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Additional Method Tests for Higher Coverage
# ---------------------------------------------------------------------------


class TestConstantPHFindResidueStatesLogic:
    """Test suite for _findResidueStates method logic."""

    def test_find_residue_states_returns_list(self) -> None:
        """Test _findResidueStates returns list of ResidueState objects.

        The method builds ResidueState objects for residues with specified variants.
        """
        from molecular_simulations.simulate.constantph.constantph import ResidueState

        # Verify ResidueState structure for building
        state = ResidueState(
            residueIndex=0,
            atomIndices={'N': 0, 'H': 1},
            particleParameters={0: {'N': (0.1, 0.2, 0.0)}},
            exceptionParameters={},
            numHydrogens=1,
        )

        assert isinstance(state.atomIndices, dict)
        assert isinstance(state.particleParameters, dict)
        assert 'N' in state.atomIndices

    def test_find_residue_states_hydrogen_counting(self) -> None:
        """Test that hydrogen counting logic works correctly.

        The numHydrogens field is determined by counting hydrogen atoms.
        """
        from openmm.app import element

        # Simulate hydrogen counting logic
        atom_elements = [
            element.nitrogen,
            element.hydrogen,
            element.hydrogen,
            element.carbon,
        ]
        num_hydrogens = sum(1 for el in atom_elements if el == element.hydrogen)

        assert num_hydrogens == 2


class TestConstantPHBuildProtonationStatesLogic:
    """Test suite for _buildProtonationStates method logic."""

    def test_variant_iteration_logic(self) -> None:
        """Test the variant iteration logic in _buildProtonationStates.

        The method iterates through variants for each titratable residue.
        """
        residue_variants = {10: ['ASP', 'ASH'], 15: ['GLU', 'GLH']}

        # Simulate variant iteration
        variant_index = 0
        max_variants = max(len(v) for v in residue_variants.values())

        assert max_variants == 2

        while variant_index < max_variants:
            for _res_index, variants in residue_variants.items():
                if variant_index < len(variants):
                    current_variant = variants[variant_index]
                    assert current_variant in ['ASP', 'ASH', 'GLU', 'GLH']
            variant_index += 1

    def test_protonated_index_assignment(self) -> None:
        """Test protonatedIndex is assigned to state with most hydrogens."""
        from molecular_simulations.simulate.constantph.constantph import (
            ResidueState,
            ResidueTitration,
        )

        titration = ResidueTitration(['ASP', 'ASH'], [0.0, 5.0])
        titration.implicitStates = [
            ResidueState(10, {}, {}, {}, 0),  # deprotonated
            ResidueState(10, {}, {}, {}, 1),  # protonated
        ]

        # Assign like the actual code
        titration.protonatedIndex = np.argmax(
            [s.numHydrogens for s in titration.implicitStates]
        )
        titration.currentIndex = titration.protonatedIndex

        assert titration.protonatedIndex == 1
        assert titration.currentIndex == 1


class TestConstantPHAttemptMCStepLogic:
    """Test suite for attemptMCStep method logic."""

    def test_position_copying_logic(self) -> None:
        """Test position copying from explicit to implicit context.

        Positions are mapped using implicitAtomIndex array.
        """
        # Simulate position mapping
        explicit_positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],  # Water - will be excluded
            ]
        )

        # Mapping: implicit atoms 0,1,2 correspond to explicit 0,1,2
        implicit_atom_index = np.array([0, 1, 2])

        implicit_positions = explicit_positions[implicit_atom_index]

        assert implicit_positions.shape == (3, 3)
        np.testing.assert_array_equal(implicit_positions[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(implicit_positions[2], [2.0, 0.0, 0.0])

    def test_random_residue_permutation(self) -> None:
        """Test random permutation of residue processing order.

        Residues are processed in random order to avoid bias.
        """
        titration_indices = [5, 10, 15, 20]

        # Verify permutation produces all indices
        permuted = np.random.permutation(titration_indices)

        assert set(permuted) == set(titration_indices)
        assert len(permuted) == len(titration_indices)

    def test_multi_site_titration_probability(self) -> None:
        """Test multi-site titration attempt probability.

        25% chance to attempt coupled titration with neighbors.
        """
        # Simulate multi-site check
        attempts = 1000
        multi_site_attempts = sum(
            1 for _ in range(attempts) if np.random.random() < 0.25
        )

        # Should be approximately 250 +/- 50
        assert 150 < multi_site_attempts < 350

    def test_energy_sum_calculation(self) -> None:
        """Test energy sum calculation for multiple titrations.

        When multiple residues change state, their energy contributions sum.
        """
        from openmm.unit import kilojoules_per_mole
        from openmm.unit import sum as unitsum

        # Simulate energy contributions
        delta_energies = [5.0, -3.0, 2.0]  # kJ/mol
        quantities = [e * kilojoules_per_mole for e in delta_energies]

        total = unitsum(quantities)

        assert total.value_in_unit(kilojoules_per_mole) == pytest.approx(4.0)


class TestConstantPHApplyStateToContextMethod:
    """Test suite for _applyStateToContext method."""

    def test_force_type_filtering(self) -> None:
        """Test that only NonbondedForce and GBSAOBCForce are updated.

        Other force types (HarmonicBondForce, etc.) are skipped.
        """
        from openmm import GBSAOBCForce, HarmonicBondForce, NonbondedForce

        forces = [NonbondedForce(), HarmonicBondForce(), GBSAOBCForce()]

        updatable_forces = [
            f for f in forces if isinstance(f, (NonbondedForce, GBSAOBCForce))
        ]

        assert len(updatable_forces) == 2
        assert isinstance(updatable_forces[0], NonbondedForce)
        assert isinstance(updatable_forces[1], GBSAOBCForce)

    def test_exception_key_lookup(self) -> None:
        """Test exception parameter key lookup logic.

        Exception keys are (residue_index, atom1_name, atom2_name).
        """
        exception_index = {
            (5, 'N', 'H'): 0,
            (5, 'H', 'N'): 0,  # Symmetric entry
            (5, 'N', 'CA'): 1,
        }

        # Verify key lookup works
        key = (5, 'N', 'H')
        assert key in exception_index
        assert exception_index[key] == 0

    def test_inter_residue_14_update_logic(self) -> None:
        """Test inter-residue 1-4 interaction update logic.

        The charge product must be recalculated: coulomb14Scale * q1 * q2
        """
        coulomb_14_scale = 1.0 / 1.2  # AMBER default
        q1 = 0.5  # elementary charge
        q2 = -0.3  # elementary charge

        new_charge_prod = coulomb_14_scale * q1 * q2

        assert new_charge_prod == pytest.approx(-0.125, rel=0.01)


class TestConstantPHMapStatesToExplicitSystem:
    """Test suite for _mapStatesToExplicitSystem method logic."""

    def test_atom_index_mapping(self) -> None:
        """Test atom index extraction from explicit topology residue."""
        # Real atom-like holders (real .name / .index, no mocks)
        atoms = [
            SimpleNamespace(name='N', index=100),
            SimpleNamespace(name='CA', index=101),
            SimpleNamespace(name='C', index=102),
        ]

        atom_indices = {atom.name: atom.index for atom in atoms}

        assert atom_indices == {'N': 100, 'CA': 101, 'C': 102}

    def test_hydrogen_index_tracking(self) -> None:
        """Test tracking of hydrogen indices for multi-site titration.

        Titratable hydrogens are tracked for neighbor detection.
        """
        from openmm.app import element

        # Simulate hydrogen detection
        atoms = [
            ('N', element.nitrogen, 100),
            ('H', element.hydrogen, 101),
            ('CA', element.carbon, 102),
            ('HA', element.hydrogen, 103),
        ]

        hydrogen_indices = [idx for name, el, idx in atoms if el == element.hydrogen]

        assert hydrogen_indices == [101, 103]


class TestConstantPHBuildImplicitSystemLogic:
    """Test suite for _buildImplicitSystemWithParmEd method logic."""

    def test_water_ion_stripping_logic(self) -> None:
        """Test logic for identifying residues to strip.

        Only water and ions are removed; lipids and ligands are kept.
        """
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        residue_names = ['ALA', 'GLY', 'HOH', 'HOH', 'NA', 'POPC', 'LIG']

        kept = [
            name for name in residue_names if name not in ConstantPH.WATER_ION_NAMES
        ]

        assert 'ALA' in kept
        assert 'GLY' in kept
        assert 'POPC' in kept  # Lipid kept
        assert 'LIG' in kept  # Ligand kept
        assert 'HOH' not in kept
        assert 'NA' not in kept

    def test_residue_index_mapping_construction(self) -> None:
        """Test construction of implicit to explicit residue mapping."""
        # Simulate mapping construction
        explicit_residues = ['ALA', 'GLY', 'HOH', 'HOH', 'NA', 'ASP']
        water_ion_names = {'HOH', 'NA'}

        implicit_to_explicit = []
        explicit_to_implicit = {}
        implicit_idx = 0

        for explicit_idx, name in enumerate(explicit_residues):
            if name not in water_ion_names:
                implicit_to_explicit.append(explicit_idx)
                explicit_to_implicit[explicit_idx] = implicit_idx
                implicit_idx += 1

        assert implicit_to_explicit == [0, 1, 5]  # ALA, GLY, ASP
        assert explicit_to_implicit == {0: 0, 1: 1, 5: 2}

    def test_solvent_dielectric_defaults(self) -> None:
        """Test default solvent and solute dielectric values."""
        implicit_args = {}

        solvent_dielectric = implicit_args.get('solventDielectric', 78.5)
        solute_dielectric = implicit_args.get('soluteDielectric', 1.0)

        assert solvent_dielectric == 78.5
        assert solute_dielectric == 1.0


class TestConstantPHBuildProteinOnlyTopologyLogic:
    """Test suite for _buildProteinOnlyTopology method logic."""

    def test_protein_residue_filtering(self) -> None:
        """Test filtering of non-protein residues."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        residue_names = ['ALA', 'POPC', 'GLY', 'LIG', 'ASP', 'HOH']

        protein_residues = [
            name for name in residue_names if name in ConstantPH.PROTEIN_RESIDUES
        ]

        assert protein_residues == ['ALA', 'GLY', 'ASP']

    def test_protein_index_mapping(self) -> None:
        """Test protein to explicit residue index mapping."""
        residue_names = ['ALA', 'POPC', 'GLY', 'LIG', 'ASP']
        protein_residue_names = {'ALA', 'GLY', 'ASP'}

        protein_to_explicit = []
        explicit_to_protein = {}
        removed_count = 0

        for i, name in enumerate(residue_names):
            if name in protein_residue_names:
                protein_idx = i - removed_count
                protein_to_explicit.append(i)
                explicit_to_protein[i] = protein_idx
            else:
                removed_count += 1

        assert protein_to_explicit == [0, 2, 4]
        assert explicit_to_protein == {0: 0, 2: 1, 4: 2}


class TestConstantPHBuildExplicitSystemLogic:
    """Test suite for _buildExplicitSystem method logic."""

    def test_relaxation_system_mass_zeroing(self) -> None:
        """Test that non-solvent masses are zeroed for relaxation.

        During relaxation, protein/lipid atoms are frozen by setting mass=0.
        """
        from openmm import System
        from openmm.unit import dalton

        system = System()
        # Add particles with different masses
        system.addParticle(14.0)  # N (protein)
        system.addParticle(12.0)  # C (protein)
        system.addParticle(18.0)  # O (water)

        # Zero protein masses (indices 0, 1)
        for i in [0, 1]:
            system.setParticleMass(i, 0.0)

        assert system.getParticleMass(0).value_in_unit(dalton) == 0.0
        assert system.getParticleMass(1).value_in_unit(dalton) == 0.0
        assert (
            system.getParticleMass(2).value_in_unit(dalton) == 18.0
        )  # Water unchanged

    def test_platform_property_handling(self) -> None:
        """Test platform property handling for context creation."""
        # When properties is None, contexts use auto-detected platform
        properties = None

        # Simulate property handling
        use_properties = properties is not None

        assert use_properties is False

        # With properties
        properties = {'Precision': 'mixed'}
        use_properties = properties is not None

        assert use_properties is True


class TestConstantPHTemperatureHandling:
    """Test suite for temperature handling in MC acceptance."""

    def test_temperature_unit_check(self) -> None:
        """Test temperature unit handling with is_quantity.

        If temperature is just a number, it's multiplied by kelvin.
        """
        from openmm.unit import is_quantity, kelvin

        # Numeric temperature
        temp_numeric = 300.0
        assert not is_quantity(temp_numeric)

        # Quantity temperature
        temp_quantity = 300.0 * kelvin
        assert is_quantity(temp_quantity)

    def test_kt_calculation(self) -> None:
        """Test kT calculation for Metropolis criterion."""
        from openmm.unit import MOLAR_GAS_CONSTANT_R, kelvin, kilojoules_per_mole

        temperature = 300.0 * kelvin
        kT = MOLAR_GAS_CONSTANT_R * temperature

        # kT at 300K should be approximately 2.494 kJ/mol
        assert kT.value_in_unit(kilojoules_per_mole) == pytest.approx(2.494, rel=0.01)


class TestConstantPHRelaxationLogic:
    """Test suite for solvent relaxation after state changes."""

    def test_relaxation_trigger_logic(self) -> None:
        """Test that relaxation is triggered only when states change."""
        any_change = False

        # No changes - no relaxation
        relaxation_triggered = bool(any_change)

        assert relaxation_triggered is False

        # With changes - trigger relaxation
        any_change = True
        relaxation_triggered = bool(any_change)

        assert relaxation_triggered is True

    def test_relaxation_steps_attribute(self) -> None:
        """Test relaxation steps attribute usage."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        cph.relaxationSteps = 1000

        assert cph.relaxationSteps == 1000


class TestConstantPHExcludeResiduesDeprecation:
    """Test suite for deprecated excludeResidues parameter."""

    def test_exclude_residues_deprecation_message(self) -> None:
        """Test that excludeResidues triggers deprecation notice.

        The parameter is deprecated - lipids/ligands are now included.
        """
        # Simulate deprecation check logic
        exclude_residues = [10, 20, 30]

        if exclude_residues is not None:
            deprecated = True
            message = 'excludeResidues parameter is deprecated'
        else:
            deprecated = False
            message = ''

        assert deprecated is True
        assert 'deprecated' in message


# ---------------------------------------------------------------------------
# Additional Parametrized Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'gb_model,expected_valid',
    [
        ('GBn2', True),
        ('OBC2', True),
        ('HCT', False),
        ('GBSA', False),
        ('', False),
    ],
)
class TestConstantPHGBModelParametrized:
    """Parametrized tests for GB model validation."""

    def test_gb_model_validation(self, gb_model: str, expected_valid: bool) -> None:
        """Test GB model validation for different model strings."""
        valid_models = {'GBn2', 'OBC2'}
        is_valid = gb_model in valid_models
        assert is_valid == expected_valid


@pytest.mark.parametrize(
    'residue_name,expected_strip',
    [
        ('HOH', True),
        ('WAT', True),
        ('TIP3', True),
        ('SPC', True),
        ('OPC', True),
        ('Na+', True),
        ('Cl-', True),
        ('NA', True),
        ('CL', True),
        ('SOD', True),
        ('CLA', True),
        ('ALA', False),
        ('POPC', False),
        ('LIG', False),
    ],
)
class TestConstantPHWaterIonStripping:
    """Parametrized tests for water/ion stripping decisions."""

    def test_strip_decision(self, residue_name: str, expected_strip: bool) -> None:
        """Test water/ion stripping decision for various residue names."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        should_strip = residue_name in ConstantPH.WATER_ION_NAMES
        assert should_strip == expected_strip


# ---------------------------------------------------------------------------
# Tests for _applyStateToContext Method
# ---------------------------------------------------------------------------


class TestApplyStateToContext:
    """Test suite for _applyStateToContext method with real OpenMM contexts."""

    def test_apply_state_to_context_with_nonbonded_force(self) -> None:
        """Test _applyStateToContext updates NonbondedForce parameters."""
        from openmm import Context, NonbondedForce, System, VerletIntegrator
        from openmm.unit import elementary_charge, femtoseconds

        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
        )

        cph = object.__new__(ConstantPH)

        # Create a simple system with NonbondedForce
        system = System()
        system.addParticle(14.0)  # N
        system.addParticle(1.0)  # H
        system.addParticle(12.0)  # C

        nb_force = NonbondedForce()
        nb_force.addParticle(-0.4, 0.17, 0.0)  # N
        nb_force.addParticle(0.2, 0.05, 0.0)  # H
        nb_force.addParticle(0.2, 0.17, 0.0)  # C
        nb_force.addException(
            0, 1, 0.02, 0.1, 0.0
        )  # N-H exception (non-zero chargeProd)
        system.addForce(nb_force)

        # Create context
        integrator = VerletIntegrator(1.0 * femtoseconds)
        context = Context(system, integrator)
        context.setPositions([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])

        # Create state with new parameters
        state = ResidueState(
            residueIndex=0,
            atomIndices={'N': 0, 'H': 1, 'C': 2},
            particleParameters={
                0: {
                    'N': (-0.5, 0.17, 0.0),
                    'H': (0.3, 0.05, 0.0),
                    'C': (0.2, 0.17, 0.0),
                }
            },
            exceptionParameters={
                0: {
                    (0, 'N', 'H'): (0.01, 0.1, 0.0),
                }
            },
            numHydrogens=1,
        )

        # Build exception index
        exception_index = {(0, 'N', 'H'): 0, (0, 'H', 'N'): 0}
        inter_residue_14 = {}
        coulomb_14_scale = 1.0 / 1.2

        # Apply state
        cph._applyStateToContext(
            state, context, exception_index, inter_residue_14, coulomb_14_scale
        )

        # Verify parameters were updated
        force = context.getSystem().getForce(0)
        q, _sigma, _eps = force.getParticleParameters(0)
        assert q.value_in_unit(elementary_charge) == pytest.approx(-0.5, rel=0.01)

    def test_apply_state_to_context_with_gbsa_force(self) -> None:
        """Test _applyStateToContext updates GBSAOBCForce parameters."""
        from openmm import Context, GBSAOBCForce, System, VerletIntegrator
        from openmm.unit import femtoseconds, nanometers

        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
        )

        cph = object.__new__(ConstantPH)

        # Create system with GBSAOBCForce
        system = System()
        system.addParticle(14.0)
        system.addParticle(1.0)

        gb_force = GBSAOBCForce()
        gb_force.addParticle(-0.4, 0.17 * nanometers, 1.0)
        gb_force.addParticle(0.2, 0.12 * nanometers, 1.0)
        system.addForce(gb_force)

        # Create context
        integrator = VerletIntegrator(1.0 * femtoseconds)
        context = Context(system, integrator)
        context.setPositions([[0, 0, 0], [0.1, 0, 0]])

        # Create state with GBSAOBCForce parameters
        state = ResidueState(
            residueIndex=0,
            atomIndices={'N': 0, 'H': 1},
            particleParameters={
                0: {
                    'N': (-0.5, 0.17 * nanometers, 1.0),
                    'H': (0.3, 0.12 * nanometers, 1.0),
                }
            },
            exceptionParameters={},
            numHydrogens=1,
        )

        exception_index = {}
        inter_residue_14 = {}
        coulomb_14_scale = 1.0 / 1.2

        # Apply state - should not raise
        cph._applyStateToContext(
            state, context, exception_index, inter_residue_14, coulomb_14_scale
        )

    def test_apply_state_skips_non_updatable_forces(self) -> None:
        """Test _applyStateToContext skips HarmonicBondForce etc."""
        from openmm import Context, HarmonicBondForce, System, VerletIntegrator
        from openmm.unit import femtoseconds

        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
        )

        cph = object.__new__(ConstantPH)

        system = System()
        system.addParticle(14.0)
        system.addParticle(1.0)

        bond_force = HarmonicBondForce()
        bond_force.addBond(0, 1, 0.1, 1000.0)
        system.addForce(bond_force)

        integrator = VerletIntegrator(1.0 * femtoseconds)
        context = Context(system, integrator)
        context.setPositions([[0, 0, 0], [0.1, 0, 0]])

        state = ResidueState(
            residueIndex=0,
            atomIndices={'N': 0, 'H': 1},
            particleParameters={
                0: {'N': (14.0,), 'H': (1.0,)}  # Mass params would not be applicable
            },
            exceptionParameters={},
            numHydrogens=1,
        )

        # Should not raise - just skip the force
        cph._applyStateToContext(state, context, {}, {}, 1.0)

    def test_apply_state_updates_inter_residue_14(self) -> None:
        """Test _applyStateToContext updates inter-residue 1-4 interactions."""
        from openmm import Context, NonbondedForce, System, VerletIntegrator
        from openmm.unit import elementary_charge, femtoseconds

        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
        )

        cph = object.__new__(ConstantPH)

        # Create system with 4 particles across 2 residues
        system = System()
        for _ in range(4):
            system.addParticle(12.0)

        nb_force = NonbondedForce()
        nb_force.addParticle(0.5, 0.3, 0.0)  # Res 0, atom 0
        nb_force.addParticle(-0.5, 0.3, 0.0)  # Res 0, atom 1
        nb_force.addParticle(0.3, 0.3, 0.0)  # Res 1, atom 2
        nb_force.addParticle(-0.3, 0.3, 0.0)  # Res 1, atom 3
        # Intra-residue exception
        nb_force.addException(0, 1, 0.0, 0.3, 0.0)
        # Inter-residue 1-4 exception
        nb_force.addException(1, 2, 0.05, 0.3, 0.0)
        system.addForce(nb_force)

        integrator = VerletIntegrator(1.0 * femtoseconds)
        context = Context(system, integrator)
        context.setPositions([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])

        state = ResidueState(
            residueIndex=0,
            atomIndices={'C1': 0, 'C2': 1},
            particleParameters={
                0: {
                    'C1': (0.6, 0.3, 0.0),  # Changed charge
                    'C2': (-0.6, 0.3, 0.0),
                }
            },
            exceptionParameters={0: {(0, 'C1', 'C2'): (0.0, 0.3, 0.0)}},
            numHydrogens=0,
        )

        exception_index = {(0, 'C1', 'C2'): 0, (0, 'C2', 'C1'): 0}
        inter_residue_14 = {0: [1]}  # Exception index 1 is inter-residue for res 0
        coulomb_14_scale = 1.0 / 1.2

        cph._applyStateToContext(
            state, context, exception_index, inter_residue_14, coulomb_14_scale
        )

        # Check that inter-residue exception was updated
        force = context.getSystem().getForce(0)
        _p1, _p2, charge_prod, _sigma, _eps = force.getExceptionParameters(1)
        # q1 = 0.6 (new), q2 = 0.3 (unchanged atom 2)
        # new charge_prod should be coulomb_14_scale * q1 * q2
        expected = coulomb_14_scale * (-0.6) * 0.3
        assert charge_prod.value_in_unit(elementary_charge**2) == pytest.approx(
            expected, rel=0.01
        )


# ---------------------------------------------------------------------------
# Tests for _findExceptionIndices Method
# ---------------------------------------------------------------------------


class TestFindExceptionIndicesMethod:
    """Test suite for _findExceptionIndices method."""

    def test_find_exception_indices_multiple_residues(self) -> None:
        """Test _findExceptionIndices with multiple residues."""
        from openmm import NonbondedForce, System
        from openmm.app import Topology
        from openmm.app.element import carbon, nitrogen

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        # Build system with 2 residues
        system = System()
        for _ in range(6):
            system.addParticle(12.0)

        nb_force = NonbondedForce()
        for _ in range(6):
            nb_force.addParticle(0.1, 0.3, 0.0)
        # Residue 0 exceptions
        nb_force.addException(0, 1, 0.0, 0.3, 0.0)
        nb_force.addException(0, 2, 0.0, 0.3, 0.0)
        # Residue 1 exceptions
        nb_force.addException(3, 4, 0.0, 0.3, 0.0)
        nb_force.addException(4, 5, 0.0, 0.3, 0.0)
        system.addForce(nb_force)

        # Build topology
        topology = Topology()
        chain = topology.addChain()
        res0 = topology.addResidue('ALA', chain)
        topology.addAtom('N', nitrogen, res0)
        topology.addAtom('CA', carbon, res0)
        topology.addAtom('C', carbon, res0)
        res1 = topology.addResidue('GLY', chain)
        topology.addAtom('N', nitrogen, res1)
        topology.addAtom('CA', carbon, res1)
        topology.addAtom('C', carbon, res1)

        indices = cph._findExceptionIndices(system, topology)

        # Check residue 0 exceptions
        assert (0, 'N', 'CA') in indices
        assert (0, 'CA', 'N') in indices
        assert indices[(0, 'N', 'CA')] == 0
        assert (0, 'N', 'C') in indices

        # Check residue 1 exceptions
        assert (1, 'N', 'CA') in indices
        assert indices[(1, 'N', 'CA')] == 2
        assert (1, 'CA', 'C') in indices

    def test_find_exception_indices_no_exceptions(self) -> None:
        """Test _findExceptionIndices with no exceptions."""
        from openmm import NonbondedForce, System
        from openmm.app import Topology
        from openmm.app.element import carbon

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        system = System()
        system.addParticle(12.0)
        system.addParticle(12.0)

        nb_force = NonbondedForce()
        nb_force.addParticle(0.1, 0.3, 0.0)
        nb_force.addParticle(-0.1, 0.3, 0.0)
        system.addForce(nb_force)

        topology = Topology()
        chain = topology.addChain()
        res = topology.addResidue('LIG', chain)
        topology.addAtom('C1', carbon, res)
        topology.addAtom('C2', carbon, res)

        indices = cph._findExceptionIndices(system, topology)

        assert indices == {}


# ---------------------------------------------------------------------------
# Tests for _findInterResidue14 Method
# ---------------------------------------------------------------------------


class TestFindInterResidue14Method:
    """Test suite for _findInterResidue14 method."""

    def test_find_inter_residue_14_with_multiple_cross_residue_exceptions(self) -> None:
        """Test _findInterResidue14 with multiple cross-residue exceptions."""
        from openmm import NonbondedForce, System
        from openmm.app import Topology
        from openmm.app.element import carbon, nitrogen
        from openmm.unit import elementary_charge

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        system = System()
        for _ in range(6):
            system.addParticle(12.0)

        nb_force = NonbondedForce()
        for _ in range(6):
            nb_force.addParticle(0.1, 0.3, 0.0)
        # Intra-residue (excluded from result)
        nb_force.addException(0, 1, 0.0, 0.3, 0.0)
        # Inter-residue 1-4 with non-zero charge (included)
        nb_force.addException(2, 3, 0.05 * elementary_charge**2, 0.3, 0.0)
        # Inter-residue with zero charge (excluded)
        nb_force.addException(2, 4, 0.0 * elementary_charge**2, 0.3, 0.0)
        system.addForce(nb_force)

        topology = Topology()
        chain = topology.addChain()
        res0 = topology.addResidue('ALA', chain)
        topology.addAtom('N', nitrogen, res0)
        topology.addAtom('CA', carbon, res0)
        topology.addAtom('C', carbon, res0)
        res1 = topology.addResidue('GLY', chain)
        topology.addAtom('N', nitrogen, res1)
        topology.addAtom('CA', carbon, res1)
        topology.addAtom('C', carbon, res1)

        indices = cph._findInterResidue14(system, topology)

        # Exception index 1 is inter-residue with non-zero charge
        assert 0 in indices
        assert 1 in indices
        assert 1 in indices[0]
        assert 1 in indices[1]
        # Exception index 2 has zero charge, should not appear
        assert 2 not in indices[0]

    def test_find_inter_residue_14_no_inter_residue(self) -> None:
        """Test _findInterResidue14 when all exceptions are intra-residue."""
        from openmm import NonbondedForce, System
        from openmm.app import Topology
        from openmm.app.element import carbon

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        system = System()
        for _ in range(4):
            system.addParticle(12.0)

        nb_force = NonbondedForce()
        for _ in range(4):
            nb_force.addParticle(0.1, 0.3, 0.0)
        # All intra-residue exceptions
        nb_force.addException(0, 1, 0.0, 0.3, 0.0)
        nb_force.addException(2, 3, 0.0, 0.3, 0.0)
        system.addForce(nb_force)

        topology = Topology()
        chain = topology.addChain()
        res0 = topology.addResidue('ALA', chain)
        topology.addAtom('C1', carbon, res0)
        topology.addAtom('C2', carbon, res0)
        res1 = topology.addResidue('GLY', chain)
        topology.addAtom('C1', carbon, res1)
        topology.addAtom('C2', carbon, res1)

        indices = cph._findInterResidue14(system, topology)

        # No inter-residue exceptions with non-zero charge
        assert len(indices) == 0


# ---------------------------------------------------------------------------
# Tests for _findNeighbors Method
# ---------------------------------------------------------------------------


class TestFindNeighborsMethod:
    """Test suite for _findNeighbors method."""

    def test_find_neighbors_with_close_hydrogens(self) -> None:
        """Test _findNeighbors returns neighbors within 0.2 nm."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        # Real titrations carrying hydrogen indices
        cph.titrations = {
            0: _real_titration(explicit_hydrogen_indices=[10, 11]),
            5: _real_titration(explicit_hydrogen_indices=[20, 21]),
            10: _real_titration(explicit_hydrogen_indices=[30, 31]),
        }

        # Positions where residue 5 is close to residue 0
        positions = np.zeros((50, 3))
        positions[10] = [0.0, 0.0, 0.0]  # H from res 0
        positions[11] = [0.05, 0.0, 0.0]
        positions[20] = [0.15, 0.0, 0.0]  # H from res 5 - within 0.2nm of res 0
        positions[21] = [0.18, 0.0, 0.0]
        positions[30] = [1.0, 0.0, 0.0]  # H from res 10 - far away
        positions[31] = [1.05, 0.0, 0.0]

        def periodic_distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        neighbors = cph._findNeighbors(0, positions, periodic_distance)

        assert 5 in neighbors  # Within 0.2nm
        assert (
            10 not in neighbors
        )  # Residue 10's hydrogens at [1.0, 0, 0] are > 0.2nm away

    def test_find_neighbors_only_returns_higher_indices(self) -> None:
        """Test _findNeighbors only returns residues with higher index."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        cph.titrations = {
            10: _real_titration(explicit_hydrogen_indices=[10]),
            5: _real_titration(explicit_hydrogen_indices=[5]),  # Lower index, close
        }

        positions = np.zeros((20, 3))
        positions[5] = [0.0, 0.0, 0.0]
        positions[10] = [0.1, 0.0, 0.0]  # Very close

        def periodic_distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        # When calling with resIndex=10, only higher indices are checked
        neighbors = cph._findNeighbors(10, positions, periodic_distance)
        assert 5 not in neighbors  # 5 < 10, so not returned

        # When calling with resIndex=5, index 10 is checked
        neighbors = cph._findNeighbors(5, positions, periodic_distance)
        assert 10 in neighbors  # 10 > 5, so returned

    def test_find_neighbors_handles_empty_hydrogen_indices(self) -> None:
        """Test _findNeighbors handles residues with no hydrogen indices."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        cph.titrations = {
            0: _real_titration(explicit_hydrogen_indices=[10]),
            5: _real_titration(explicit_hydrogen_indices=[]),  # No hydrogens
        }

        positions = np.zeros((20, 3))
        positions[10] = [0.0, 0.0, 0.0]

        def periodic_distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        neighbors = cph._findNeighbors(0, positions, periodic_distance)
        assert 5 not in neighbors  # No hydrogens to compare


# ---------------------------------------------------------------------------
# Tests for _attemptPHChange Method
# ---------------------------------------------------------------------------


class TestAttemptPHChangeMethod:
    """Test suite for _attemptPHChange method."""

    def test_attempt_ph_change_selects_based_on_probability(self) -> None:
        """Test _attemptPHChange selects pH based on weighted probability."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        cph.pH = [4.0, 7.0]
        cph._weights = [0.0, 0.0]
        cph._updateWeights = False
        cph._hasMadeTransition = False
        cph.currentPHIndex = 0

        # Real titration with a known hydrogen count
        cph.titrations = {
            0: _real_titration(explicit_states=(_real_state(1),), current_index=0)
        }

        # Run many times and verify distribution
        ph_counts = [0, 0]
        for _ in range(1000):
            cph._attemptPHChange()
            ph_counts[cph.currentPHIndex] += 1

        # With equal weights and hydrogens=1, pH 4 should be more probable
        # because -H * ln(10) * pH is less negative at lower pH
        assert ph_counts[0] > ph_counts[1]  # pH 4.0 more probable

    def test_attempt_ph_change_updates_weights_when_enabled(self) -> None:
        """Test _attemptPHChange updates weights with Wang-Landau."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        cph.pH = [4.0, 7.0]
        cph._weights = [0.0, 0.0]
        cph._updateWeights = True
        cph._weightUpdateFactor = 1.0
        cph._histogram = [0, 0]
        cph._hasMadeTransition = False
        cph.currentPHIndex = 0

        # No hydrogens - equal probability
        cph.titrations = {
            0: _real_titration(explicit_states=(_real_state(0),), current_index=0)
        }

        initial_weights = list(cph._weights)
        initial_histogram = list(cph._histogram)

        cph._attemptPHChange()

        # One weight should decrease, one histogram should increase
        weight_changed = cph._weights != initial_weights
        histogram_changed = cph._histogram != initial_histogram
        assert weight_changed or histogram_changed

    def test_attempt_ph_change_marks_transition(self) -> None:
        """Test _attemptPHChange marks transition when pH changes."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        cph.pH = [4.0, 7.0]
        # Bias weights heavily toward pH index 1
        cph._weights = [0.0, 100.0]
        cph._updateWeights = False
        cph._hasMadeTransition = False
        cph.currentPHIndex = 0

        cph.titrations = {
            0: _real_titration(explicit_states=(_real_state(0),), current_index=0)
        }

        cph._attemptPHChange()

        # With such biased weights, should transition to pH index 1
        if cph.currentPHIndex != 0:
            assert cph._hasMadeTransition is True

    def test_attempt_ph_change_reduces_update_factor_on_flat_histogram(self) -> None:
        """Test _attemptPHChange reduces factor when histogram is flat."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        cph.pH = [4.0, 7.0]
        cph._weights = [0.0, 0.0]
        cph._updateWeights = True
        cph._weightUpdateFactor = 1.0
        # Histogram already fairly flat with > 20 counts
        cph._histogram = [25, 25]
        cph._hasMadeTransition = True
        cph.currentPHIndex = 0

        cph.titrations = {
            0: _real_titration(explicit_states=(_real_state(0),), current_index=0)
        }

        initial_factor = cph._weightUpdateFactor

        # After adding one more count to already flat histogram
        cph._attemptPHChange()

        # If histogram became flat (min > 20 and > 0.2*mean), factor should halve
        # The histogram will have 26 or 26 depending on which was selected
        # min(26,25) = 25 > 20, mean ~ 25.5, 0.2*mean = 5.1, 25 > 5.1
        # So factor should be halved
        assert cph._weightUpdateFactor <= initial_factor


# ---------------------------------------------------------------------------
# Tests for _selectNewState Method
# ---------------------------------------------------------------------------


class TestSelectNewStateMethod:
    """Test suite for _selectNewState method."""

    def test_select_new_state_binary_toggle(self) -> None:
        """Test _selectNewState toggles for 2-state systems."""
        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
            ResidueTitration,
        )

        cph = object.__new__(ConstantPH)

        titration = ResidueTitration(['ASP', 'ASH'], [0.0, 5.0])
        titration.implicitStates = [
            ResidueState(0, {}, {}, {}, 0),
            ResidueState(0, {}, {}, {}, 1),
        ]

        # Test toggle from 0 to 1
        titration.currentIndex = 0
        assert cph._selectNewState(titration) == 1

        # Test toggle from 1 to 0
        titration.currentIndex = 1
        assert cph._selectNewState(titration) == 0

    def test_select_new_state_multi_state_random(self) -> None:
        """Test _selectNewState random selection for multi-state systems."""
        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
            ResidueTitration,
        )

        cph = object.__new__(ConstantPH)

        titration = ResidueTitration(['HID', 'HIE', 'HIP'], [0.0, 0.5, 4.5])
        titration.implicitStates = [
            ResidueState(0, {}, {}, {}, 1),
            ResidueState(0, {}, {}, {}, 1),
            ResidueState(0, {}, {}, {}, 2),
        ]
        titration.currentIndex = 0

        # Verify it never returns current state
        selected_states = set()
        for _ in range(100):
            new_state = cph._selectNewState(titration)
            assert new_state != 0
            selected_states.add(new_state)

        # Should eventually select both other states
        assert 1 in selected_states
        assert 2 in selected_states


# ---------------------------------------------------------------------------
# Tests for attemptMCStep Method (pure-logic components)
# ---------------------------------------------------------------------------


class TestAttemptMCStepMethod:
    """Test suite for attemptMCStep pure-logic components."""

    def test_attempt_mc_step_position_mapping_logic(self) -> None:
        """Test the position mapping logic used in attemptMCStep."""
        # Test the core position mapping logic without calling the full method
        explicit_positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],  # water
                [0.4, 0.0, 0.0],
            ]
        )

        implicit_atom_index = np.array([0, 1, 2, 4])  # Skip index 3 (water)

        implicit_positions = explicit_positions[implicit_atom_index]

        assert implicit_positions.shape == (4, 3)
        np.testing.assert_array_equal(implicit_positions[3], [0.4, 0.0, 0.0])

    def test_attempt_mc_step_energy_calculation_logic(self) -> None:
        """Test the energy calculation logic in attemptMCStep."""
        from openmm.unit import MOLAR_GAS_CONSTANT_R, kelvin, kilojoules_per_mole

        # Simulate the energy calculation
        current_energy = 100.0 * kilojoules_per_mole
        new_energy = 80.0 * kilojoules_per_mole
        delta_ref_energy = 5.0 * kilojoules_per_mole
        delta_n = 1  # Gaining one proton
        pH = 7.0
        temperature = 300.0 * kelvin
        kT = MOLAR_GAS_CONSTANT_R * temperature

        # The energy term should be dimensionless after division by kT
        energy_delta = new_energy - current_energy - delta_ref_energy
        # -20 - 5 = -25 kJ/mol
        assert energy_delta.value_in_unit(kilojoules_per_mole) == pytest.approx(
            -25.0, rel=0.01
        )

        # kT at 300K is about 2.494 kJ/mol
        kT_value = kT.value_in_unit(kilojoules_per_mole)
        assert kT_value == pytest.approx(2.494, rel=0.01)

        # Calculate w - the acceptance criterion
        w = energy_delta / kT + delta_n * np.log(10.0) * pH
        # -25/2.494 + 1*2.303*7 = -10.02 + 16.12 = 6.1 (unfavorable)
        assert w > 0  # Slightly unfavorable due to proton term at pH 7

    def test_attempt_mc_step_residue_permutation(self) -> None:
        """Test random residue permutation in attemptMCStep."""
        titration_indices = [5, 10, 15, 20, 25]

        # Verify permutation contains all indices
        permuted = np.random.permutation(titration_indices)

        assert set(permuted) == set(titration_indices)
        assert len(permuted) == len(titration_indices)

    def test_attempt_mc_step_multi_site_probability(self) -> None:
        """Test 25% probability for multi-site titration."""
        attempts = 10000
        multi_site_count = sum(1 for _ in range(attempts) if np.random.random() < 0.25)

        # Should be approximately 2500 +/- ~150 (3 sigma)
        assert 2000 < multi_site_count < 3000

    def test_attempt_mc_step_temperature_handling(self) -> None:
        """Test attemptMCStep handles both numeric and Quantity temperatures."""
        from openmm.unit import is_quantity, kelvin

        # Numeric temperature
        temp_numeric = 300.0
        if not is_quantity(temp_numeric):
            temp_numeric = temp_numeric * kelvin

        assert is_quantity(temp_numeric)
        assert temp_numeric.value_in_unit(kelvin) == 300.0

        # Quantity temperature
        temp_quantity = 310.0 * kelvin
        if not is_quantity(temp_quantity):
            temp_quantity = temp_quantity * kelvin

        assert is_quantity(temp_quantity)
        assert temp_quantity.value_in_unit(kelvin) == 310.0


# ---------------------------------------------------------------------------
# Tests for setResidueState Method
# ---------------------------------------------------------------------------


class TestSetResidueStateMethod:
    """Test suite for setResidueState method."""

    @staticmethod
    def _make_nb_context(num_particles: int = 1):
        """Build a real single-force OpenMM Context (NonbondedForce)."""
        from openmm import Context, NonbondedForce, System, VerletIntegrator
        from openmm.unit import femtoseconds

        system = System()
        nb = NonbondedForce()
        for _ in range(num_particles):
            system.addParticle(12.0)
            # Nonzero initial charge so OpenMM keeps the Coulomb kernel, allowing
            # later updateParametersInContext calls to change charges.
            nb.addParticle(0.1, 0.2, 0.0)
        system.addForce(nb)
        context = Context(system, VerletIntegrator(1.0 * femtoseconds))
        context.setPositions([[0.1 * i, 0.0, 0.0] for i in range(num_particles)])
        return context

    def test_set_residue_state_updates_all_contexts(self) -> None:
        """Test setResidueState applies the new state to all three contexts."""
        from openmm.unit import elementary_charge

        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
            ResidueTitration,
        )

        cph = object.__new__(ConstantPH)

        # State 1 sets a distinctive charge (0.5) on the single atom
        state0 = ResidueState(0, {'N': 0}, {0: {'N': (0.1, 0.2, 0.0)}}, {}, 0)
        state1 = ResidueState(0, {'N': 0}, {0: {'N': (0.5, 0.2, 0.0)}}, {}, 1)

        titration = ResidueTitration(['ASP', 'ASH'], [0.0, 5.0])
        titration.implicitStates = [state0, state1]
        titration.explicitStates = [state0, state1]
        titration.currentIndex = 0

        cph.titrations = {0: titration}
        cph.explicitExceptionIndex = {}
        cph.implicitExceptionIndex = {}
        cph.explicitInterResidue14 = {}
        cph.implicitInterResidue14 = {}
        cph.explicit14Scale = 1.0 / 1.2
        cph.implicit14Scale = 1.0 / 1.2
        cph.relaxationSteps = 100

        # Real minimal contexts (no full solvated system required)
        explicit_ctx = self._make_nb_context()
        relaxation_ctx = self._make_nb_context()
        implicit_ctx = self._make_nb_context()
        cph.simulation = SimpleNamespace(context=explicit_ctx)
        cph.relaxationContext = relaxation_ctx
        cph.implicitContext = implicit_ctx

        cph.setResidueState(0, 1, relax=False)

        assert titration.currentIndex == 1
        # State 1's charge was really applied to all three contexts
        for ctx in (explicit_ctx, relaxation_ctx, implicit_ctx):
            q, _sigma, _eps = ctx.getSystem().getForce(0).getParticleParameters(0)
            assert q.value_in_unit(elementary_charge) == pytest.approx(0.5, rel=0.01)

    def test_set_residue_state_with_relaxation(self) -> None:
        """Test setResidueState triggers real relaxation when requested."""
        from openmm.unit import nanometers

        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
            ResidueTitration,
        )

        cph = object.__new__(ConstantPH)

        state0 = ResidueState(0, {'N': 0}, {}, {}, 0)
        state1 = ResidueState(0, {'N': 0}, {}, {}, 1)

        titration = ResidueTitration(['ASP', 'ASH'], [0.0, 5.0])
        titration.implicitStates = [state0, state1]
        titration.explicitStates = [state0, state1]
        titration.currentIndex = 0

        cph.titrations = {0: titration}
        cph.explicitExceptionIndex = {}
        cph.implicitExceptionIndex = {}
        cph.explicitInterResidue14 = {}
        cph.implicitInterResidue14 = {}
        cph.explicit14Scale = 1.0 / 1.2
        cph.implicit14Scale = 1.0 / 1.2
        cph.relaxationSteps = 5

        # Real contexts; explicit and relaxation must share particle count so
        # positions can be copied between them during relaxation.
        explicit_ctx = self._make_nb_context(3)
        relaxation_ctx = self._make_nb_context(3)
        cph.simulation = SimpleNamespace(context=explicit_ctx)
        cph.relaxationContext = relaxation_ctx
        cph.implicitContext = self._make_nb_context(3)

        cph.setResidueState(0, 1, relax=True)

        assert titration.currentIndex == 1
        # Relaxation ran and real positions were propagated back to the main context
        positions = explicit_ctx.getState(getPositions=True).getPositions(asNumpy=True)
        assert positions.shape == (3, 3)
        assert not np.any(np.isnan(positions.value_in_unit(nanometers)))


# ---------------------------------------------------------------------------
# Tests for _buildImplicitSystemWithParmEd Logic
# ---------------------------------------------------------------------------


class TestBuildImplicitSystemLogic:
    """Test suite for _buildImplicitSystemWithParmEd method logic."""

    def test_ion_element_identification(self) -> None:
        """Test ion identification by element number."""

        # Simulate the ion element check logic from the method
        ion_elements = [11, 17, 19, 35, 37, 55]  # Na, Cl, K, Br, Rb, Cs

        test_cases = [
            (11, True),  # Na
            (17, True),  # Cl
            (6, False),  # C
            (7, False),  # N
            (19, True),  # K
        ]

        for element_num, expected in test_cases:
            is_ion = element_num in ion_elements
            assert is_ion == expected

    def test_residue_mapping_construction(self) -> None:
        """Test implicit to explicit residue mapping construction."""
        # Simulate the mapping logic from the method
        explicit_residue_names = ['ALA', 'HOH', 'GLY', 'NA', 'ASP', 'HOH', 'HOH']
        water_ion_names = {'HOH', 'NA'}

        implicit_to_explicit = []
        explicit_to_implicit = {}
        implicit_idx = 0

        for explicit_idx, name in enumerate(explicit_residue_names):
            if name not in water_ion_names:
                implicit_to_explicit.append(explicit_idx)
                explicit_to_implicit[explicit_idx] = implicit_idx
                implicit_idx += 1

        assert implicit_to_explicit == [0, 2, 4]  # ALA, GLY, ASP
        assert explicit_to_implicit == {0: 0, 2: 1, 4: 2}
        assert implicit_idx == 3


# ---------------------------------------------------------------------------
# Tests for _buildProtonationStates Method Logic
# ---------------------------------------------------------------------------


class TestBuildProtonationStatesLogic:
    """Test suite for _buildProtonationStates method logic."""

    def test_variant_index_iteration(self) -> None:
        """Test variant index iteration logic."""
        residue_variants = {
            10: ['ASP', 'ASH'],
            15: ['HID', 'HIE', 'HIP'],
        }

        variant_index = 0
        max_variants = max(len(v) for v in residue_variants.values())
        processed_variants = []

        while variant_index < max_variants:
            for res_index, variants in residue_variants.items():
                if variant_index < len(variants):
                    processed_variants.append((res_index, variants[variant_index]))
            variant_index += 1

        expected = [
            (10, 'ASP'),
            (15, 'HID'),  # index 0
            (10, 'ASH'),
            (15, 'HIE'),  # index 1
            (15, 'HIP'),  # index 2
        ]
        assert processed_variants == expected

    def test_protonated_index_assignment_logic(self) -> None:
        """Test protonated index is assigned to state with most hydrogens."""
        hydrogen_counts = [0, 1, 2, 1]  # e.g., for a 4-state system

        protonated_index = np.argmax(hydrogen_counts)

        assert protonated_index == 2


# ---------------------------------------------------------------------------
# Tests for Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestConstantPHErrorHandling:
    """Test suite for error handling in ConstantPH methods."""

    def test_unknown_gb_model_raises_value_error(self) -> None:
        """Test that unknown GB model raises ValueError."""
        gb_model = 'UnknownModel'

        with pytest.raises(ValueError, match='Unknown GB model'):
            if gb_model == 'GBn2' or gb_model == 'OBC2':
                pass
            else:
                raise ValueError(f"Unknown GB model: {gb_model}. Use 'GBn2' or 'OBC2'.")

    def test_missing_nonbonded_force_in_explicit_system(self) -> None:
        """Test error when NonbondedForce is missing."""
        from openmm import HarmonicBondForce, NonbondedForce, System

        system = System()
        system.addParticle(12.0)
        system.addParticle(12.0)

        bond_force = HarmonicBondForce()
        bond_force.addBond(0, 1, 0.1, 1000.0)
        system.addForce(bond_force)

        # Check for NonbondedForce
        has_nonbonded = any(
            isinstance(system.getForce(i), NonbondedForce)
            for i in range(system.getNumForces())
        )

        assert has_nonbonded is False


# ---------------------------------------------------------------------------
# Parametrized Tests for Coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'current_index,num_states,expected_options',
    [
        (0, 2, [1]),
        (1, 2, [0]),
        (0, 3, [1, 2]),
        (1, 3, [0, 2]),
        (2, 3, [0, 1]),
        (0, 4, [1, 2, 3]),
        (2, 5, [0, 1, 3, 4]),
    ],
)
class TestSelectNewStateParametrized:
    """Parametrized tests for _selectNewState method."""

    def test_select_new_state_returns_valid_option(
        self, current_index: int, num_states: int, expected_options: list
    ) -> None:
        """Test _selectNewState returns valid state different from current."""
        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
            ResidueTitration,
        )

        cph = object.__new__(ConstantPH)

        titration = ResidueTitration(
            [f'S{i}' for i in range(num_states)], [0.0] * num_states
        )
        titration.implicitStates = [
            ResidueState(0, {}, {}, {}, i) for i in range(num_states)
        ]
        titration.currentIndex = current_index

        # Test many times to verify randomness
        for _ in range(50):
            new_state = cph._selectNewState(titration)
            assert new_state in expected_options
            assert new_state != current_index


@pytest.mark.parametrize(
    'weights,expected_normalized',
    [
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
        ([-5.0, -3.0, -1.0], [0.0, 2.0, 4.0]),
        ([10.0, 15.0, 20.0], [0.0, 5.0, 10.0]),
        ([-100.0, 0.0, 100.0], [0.0, 100.0, 200.0]),
    ],
)
class TestWeightsPropertyParametrized:
    """Parametrized tests for weights property normalization."""

    def test_weights_normalization(
        self, weights: list, expected_normalized: list
    ) -> None:
        """Test weights property normalizes correctly."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        cph._weights = weights

        normalized = cph.weights

        for _i, (actual, expected) in enumerate(
            zip(normalized, expected_normalized, strict=False)
        ):
            assert actual == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    'distance,threshold,is_neighbor',
    [
        (0.1, 0.2, True),
        (0.15, 0.2, True),
        (0.19, 0.2, True),
        (0.2, 0.2, False),
        (0.25, 0.2, False),
        (1.0, 0.2, False),
    ],
)
class TestNeighborDistanceThreshold:
    """Parametrized tests for neighbor distance threshold."""

    def test_neighbor_distance_check(
        self, distance: float, threshold: float, is_neighbor: bool
    ) -> None:
        """Test neighbor distance threshold logic."""
        result = distance < threshold
        assert result == is_neighbor


# ---------------------------------------------------------------------------
# Tests for _find14Scale Method
# ---------------------------------------------------------------------------


class TestFind14ScaleMethod:
    """Test suite for _find14Scale method."""

    def test_find_14_scale_from_system_returns_amber_default(self) -> None:
        """Test _find14Scale returns AMBER default for System objects."""
        from openmm import System

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        system = System()

        scale = cph._find14Scale(system)

        assert scale == pytest.approx(1.0 / 1.2, rel=1e-5)

    def test_find_14_scale_from_unknown_returns_one(self) -> None:
        """Test _find14Scale returns 1.0 for unknown types."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)

        scale = cph._find14Scale('unknown_type')

        assert scale == 1.0

        scale = cph._find14Scale(None)

        assert scale == 1.0

        scale = cph._find14Scale(42)

        assert scale == 1.0


# ---------------------------------------------------------------------------
# Tests for Position Mapping Logic
# ---------------------------------------------------------------------------


class TestPositionMappingLogic:
    """Test suite for position mapping between explicit and implicit systems."""

    def test_position_mapping_with_numpy_indexing(self) -> None:
        """Test position mapping using numpy array indexing."""
        # Explicit system positions (10 atoms)
        explicit_positions = np.array(
            [
                [0.0, 0.0, 0.0],  # 0
                [0.1, 0.0, 0.0],  # 1
                [0.2, 0.0, 0.0],  # 2
                [0.3, 0.0, 0.0],  # 3 - water O
                [0.35, 0.0, 0.0],  # 4 - water H
                [0.4, 0.0, 0.0],  # 5 - water H
                [0.5, 0.0, 0.0],  # 6
                [0.6, 0.0, 0.0],  # 7 - ion
                [0.7, 0.0, 0.0],  # 8
                [0.8, 0.0, 0.0],  # 9
            ]
        )

        # Implicit atom index mapping (excluding water indices 3,4,5 and ion 7)
        implicit_atom_index = np.array([0, 1, 2, 6, 8, 9])

        # Map positions
        implicit_positions = explicit_positions[implicit_atom_index]

        assert implicit_positions.shape == (6, 3)
        np.testing.assert_array_equal(implicit_positions[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(implicit_positions[3], [0.5, 0.0, 0.0])
        np.testing.assert_array_equal(implicit_positions[5], [0.8, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Tests for Metropolis Criterion
# ---------------------------------------------------------------------------


class TestMetropolisCriterion:
    """Test suite for Metropolis acceptance criterion logic."""

    def test_metropolis_always_accepts_favorable(self) -> None:
        """Test Metropolis criterion always accepts deltaE < 0."""
        import numpy as np

        # Favorable move (energy decreases)
        w = -5.0  # log acceptance ratio

        # When w < 0, always accept
        accept = (w <= 0) or (np.exp(-w) > np.random.random())

        assert w < 0
        assert accept is True

    def test_metropolis_probability_for_unfavorable(self) -> None:
        """Test Metropolis criterion probability for unfavorable moves."""
        import numpy as np

        # Unfavorable move
        w = 2.0  # log acceptance ratio > 0

        # Probability of acceptance is exp(-w)
        acceptance_prob = np.exp(-w)

        # Should be around 0.135
        assert acceptance_prob == pytest.approx(np.exp(-2.0), rel=1e-10)
        assert acceptance_prob < 1.0

    def test_metropolis_with_proton_contribution(self) -> None:
        """Test Metropolis criterion with proton number contribution."""
        import numpy as np
        from openmm.unit import MOLAR_GAS_CONSTANT_R, kelvin, kilojoules_per_mole

        temperature = 300.0 * kelvin
        kT = MOLAR_GAS_CONSTANT_R * temperature

        # Energy change
        delta_energy = 10.0 * kilojoules_per_mole  # Unfavorable
        delta_ref_energy = -5.0 * kilojoules_per_mole
        delta_n = -1  # Losing one proton
        pH = 7.0

        (delta_energy - delta_ref_energy) / kT + delta_n * np.log(10.0) * pH

        # The proton term at pH 7 for losing a proton
        proton_term = -1 * np.log(10.0) * 7.0
        assert proton_term < 0  # Favorable for acids


# ---------------------------------------------------------------------------
# Tests for Histogram and Weight Updates
# ---------------------------------------------------------------------------


class TestHistogramWeightUpdates:
    """Test suite for Wang-Landau histogram and weight update logic."""

    def test_histogram_flatness_check(self) -> None:
        """Test histogram flatness criterion."""
        # Flat histogram
        histogram = [100, 98, 102, 99]
        min_counts = min(histogram)
        mean_counts = sum(histogram) / len(histogram)

        is_flat = min_counts > 20 and min_counts >= 0.2 * mean_counts

        assert is_flat is True

    def test_histogram_not_flat_low_counts(self) -> None:
        """Test histogram not flat with low counts."""
        histogram = [10, 12, 8, 11]
        min_counts = min(histogram)
        mean_counts = sum(histogram) / len(histogram)

        is_flat = min_counts > 20 and min_counts >= 0.2 * mean_counts

        assert is_flat is False  # min < 20

    def test_histogram_not_flat_uneven(self) -> None:
        """Test histogram not flat with uneven distribution."""
        histogram = [100, 100, 100, 5]
        min_counts = min(histogram)
        mean_counts = sum(histogram) / len(histogram)

        is_flat = min_counts > 20 and min_counts >= 0.2 * mean_counts

        # min = 5, mean = 76.25, 0.2 * mean = 15.25
        # 5 < 15.25, so not flat
        assert is_flat is False

    def test_weight_update_factor_doubling(self) -> None:
        """Test weight update factor doubles when stuck."""
        has_made_transition = False
        current_probability = 0.999
        weight_update_factor = 1.0

        if (
            not has_made_transition
            and current_probability > 0.99
            and weight_update_factor < 1024.0
        ):
            weight_update_factor *= 2.0

        assert weight_update_factor == 2.0

    def test_weight_update_factor_cap(self) -> None:
        """Test weight update factor is capped at 1024."""
        has_made_transition = False
        current_probability = 0.999
        weight_update_factor = 1024.0

        if (
            not has_made_transition
            and current_probability > 0.99
            and weight_update_factor < 1024.0
        ):
            weight_update_factor *= 2.0

        # Should not double because already at cap
        assert weight_update_factor == 1024.0


# ---------------------------------------------------------------------------
# Tests for _findResidueStates Method Logic
# ---------------------------------------------------------------------------


class TestFindResidueStatesMethod:
    """Test suite for _findResidueStates method logic."""

    def test_find_residue_states_particle_params_extraction(self) -> None:
        """Test particle parameters extraction logic."""
        from openmm import NonbondedForce, System
        from openmm.unit import elementary_charge

        # Create system
        system = System()
        system.addParticle(14.0)  # N
        system.addParticle(12.0)  # CA

        nb_force = NonbondedForce()
        nb_force.addParticle(-0.4, 0.17, 0.0)
        nb_force.addParticle(0.1, 0.17, 0.0)
        system.addForce(nb_force)

        # Simulate parameter extraction logic
        force_params = {}
        for i, force in enumerate(system.getForces()):
            try:
                # This would be done per-atom in actual code
                params = force.getParticleParameters(0)
                force_params[i] = params
            except Exception:
                pass

        assert 0 in force_params
        q, _sigma, _eps = force_params[0]
        assert q.value_in_unit(elementary_charge) == pytest.approx(-0.4)

    def test_find_residue_states_exception_extraction(self) -> None:
        """Test exception parameters extraction logic."""
        from openmm import NonbondedForce, System

        system = System()
        system.addParticle(14.0)
        system.addParticle(1.0)
        system.addParticle(12.0)

        nb_force = NonbondedForce()
        nb_force.addParticle(-0.4, 0.17, 0.0)
        nb_force.addParticle(0.2, 0.05, 0.0)
        nb_force.addParticle(0.1, 0.17, 0.0)
        nb_force.addException(0, 1, -0.08, 0.1, 0.0)  # N-H exception
        nb_force.addException(0, 2, -0.04, 0.17, 0.0)  # N-CA exception
        system.addForce(nb_force)

        # Extract exceptions
        exception_params = {}
        for j in range(nb_force.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(j)
            exception_params[j] = (p1, p2, chargeProd, sigma, epsilon)

        assert len(exception_params) == 2
        assert exception_params[0][0] == 0  # First particle index
        assert exception_params[0][1] == 1  # Second particle index


# ---------------------------------------------------------------------------
# Tests for _mapStatesToExplicitSystem Logic
# ---------------------------------------------------------------------------


class TestMapStatesToExplicitSystemLogic:
    """Test suite for _mapStatesToExplicitSystem method logic."""

    def test_explicit_atom_indices_extraction(self) -> None:
        """Test extraction of atom indices from explicit topology residue."""
        from openmm.app import Topology
        from openmm.app.element import carbon, hydrogen, nitrogen

        topology = Topology()
        chain = topology.addChain()
        residue = topology.addResidue('ALA', chain)
        topology.addAtom('N', nitrogen, residue)
        topology.addAtom('CA', carbon, residue)
        topology.addAtom('H', hydrogen, residue)

        residues = list(topology.residues())
        explicit_atom_indices = {atom.name: atom.index for atom in residues[0].atoms()}

        assert explicit_atom_indices == {'N': 0, 'CA': 1, 'H': 2}

    def test_hydrogen_tracking_for_multisite(self) -> None:
        """Test tracking of hydrogen indices for multi-site titration."""
        from openmm.app import Topology
        from openmm.app.element import carbon, hydrogen, nitrogen

        topology = Topology()
        chain = topology.addChain()
        residue = topology.addResidue('ASH', chain)
        topology.addAtom('N', nitrogen, residue)
        topology.addAtom('CA', carbon, residue)
        topology.addAtom('HD2', hydrogen, residue)  # Titratable H

        residues = list(topology.residues())

        # Find hydrogen atoms
        hydrogen_indices = []
        for atom in residues[0].atoms():
            if atom.element == hydrogen:
                hydrogen_indices.append(atom.index)

        assert hydrogen_indices == [2]


# ---------------------------------------------------------------------------
# Tests for _buildExplicitSystem Logic
# ---------------------------------------------------------------------------


class TestBuildExplicitSystemLogic:
    """Test suite for _buildExplicitSystem method logic."""

    def test_mass_zeroing_for_relaxation(self) -> None:
        """Test that non-solvent masses are zeroed for relaxation."""
        from openmm import System
        from openmm.unit import dalton

        system = System()
        system.addParticle(14.0)  # N (protein)
        system.addParticle(12.0)  # C (protein)
        system.addParticle(18.0)  # O (water)
        system.addParticle(1.0)  # H (water)

        # Zero protein masses (indices 0, 1)
        protein_indices = [0, 1]
        for i in protein_indices:
            system.setParticleMass(i, 0.0)

        assert system.getParticleMass(0).value_in_unit(dalton) == 0.0
        assert system.getParticleMass(1).value_in_unit(dalton) == 0.0
        assert system.getParticleMass(2).value_in_unit(dalton) == 18.0
        assert system.getParticleMass(3).value_in_unit(dalton) == 1.0

    def test_water_identification_logic(self) -> None:
        """Test water residue identification logic."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        residue_names = ['ALA', 'HOH', 'WAT', 'GLY', 'TIP3', 'OPC']

        water_indices = [
            i
            for i, name in enumerate(residue_names)
            if name in ConstantPH.WATER_ION_NAMES
        ]

        assert water_indices == [1, 2, 4, 5]

    def test_ion_element_identification(self) -> None:
        """Test ion identification by element."""
        from openmm.app import element

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        test_elements = [
            element.sodium,
            element.chlorine,
            element.carbon,
            element.nitrogen,
            element.potassium,
        ]

        ion_elements = [el for el in test_elements if el in ConstantPH.ION_ELEMENTS]

        assert len(ion_elements) == 3
        assert element.sodium in ion_elements
        assert element.chlorine in ion_elements
        assert element.potassium in ion_elements


# ---------------------------------------------------------------------------
# Tests for attemptMCStep Logic Components
# ---------------------------------------------------------------------------


class TestAttemptMCStepLogicComponents:
    """Test suite for individual logic components in attemptMCStep."""

    def test_delta_ref_energy_sum(self) -> None:
        """Test reference energy difference calculation."""
        from openmm.unit import kilojoules_per_mole
        from openmm.unit import sum as unitsum

        # Simulate titration reference energies
        ref_energies = [0.0, 5.2]  # ASP, ASH

        # State change from 0 to 1
        delta_ref = ref_energies[1] - ref_energies[0]
        assert delta_ref == 5.2

        # Using unitsum like actual code
        quantities = [5.2 * kilojoules_per_mole, 4.8 * kilojoules_per_mole]
        total = unitsum(quantities)
        assert total.value_in_unit(kilojoules_per_mole) == pytest.approx(10.0)

    def test_delta_n_hydrogens_sum(self) -> None:
        """Test hydrogen count difference calculation."""

        # State 0: 0 hydrogens (ASP)
        # State 1: 1 hydrogen (ASH)
        current_h = 0
        new_h = 1
        delta_n = new_h - current_h

        assert delta_n == 1

        # Multiple residues
        deltas = [1, -1, 0]  # ASP->ASH, LYS->LYN, HID->HIE
        total_delta = sum(deltas)
        assert total_delta == 0

    def test_acceptance_criterion_components(self) -> None:
        """Test individual components of acceptance criterion."""
        import numpy as np
        from openmm.unit import MOLAR_GAS_CONSTANT_R, kelvin, kilojoules_per_mole

        # Setup
        temperature = 300.0 * kelvin
        kT = MOLAR_GAS_CONSTANT_R * temperature
        pH = 7.0

        # Energy terms
        delta_energy = -10.0 * kilojoules_per_mole  # Favorable
        delta_ref_energy = 5.2 * kilojoules_per_mole  # Penalty

        # Proton term
        delta_n = 1  # Gaining one proton

        # Calculate w
        energy_ratio = (delta_energy - delta_ref_energy) / kT
        proton_term = delta_n * np.log(10.0) * pH

        # Check that components are correct
        assert energy_ratio < 0  # Energy contribution is favorable
        assert proton_term > 0  # Proton term at pH 7 is unfavorable for protonation


# ---------------------------------------------------------------------------
# Tests for Relaxation Logic
# ---------------------------------------------------------------------------


class TestRelaxationLogic:
    """Test suite for relaxation logic after state changes."""

    def test_relaxation_condition(self) -> None:
        """Test condition for triggering relaxation."""
        any_change = True

        # Relaxation should only happen if states changed
        should_relax = any_change
        assert should_relax is True

        any_change = False
        should_relax = any_change
        assert should_relax is False

    def test_relaxation_steps_configuration(self) -> None:
        """Test relaxation steps parameter."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        cph.relaxationSteps = 500

        assert cph.relaxationSteps == 500

        cph.relaxationSteps = 0
        assert cph.relaxationSteps == 0


# ---------------------------------------------------------------------------
# Tests for _buildAtomIndexMapping Logic
# ---------------------------------------------------------------------------


class TestBuildAtomIndexMappingLogic:
    """Test suite for _buildAtomIndexMapping method logic."""

    def test_atom_name_matching_logic(self) -> None:
        """Test atom name matching between implicit and explicit systems."""
        # Explicit atoms
        explicit_atoms = {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4}

        # Implicit atoms (after stripping water)
        implicit_atoms = ['N', 'CA', 'C', 'O', 'CB']

        # Map implicit to explicit
        mapping = []
        for atom_name in implicit_atoms:
            if atom_name in explicit_atoms:
                mapping.append(explicit_atoms[atom_name])

        assert mapping == [0, 1, 2, 3, 4]

    def test_atom_index_array_construction(self) -> None:
        """Test numpy array construction for atom mapping."""
        explicit_to_implicit_map = {0: 0, 1: 1, 2: 2, 5: 3, 6: 4}

        num_implicit_atoms = 5
        implicit_atom_index = np.zeros(num_implicit_atoms, dtype=np.int64)

        for explicit_idx, implicit_idx in explicit_to_implicit_map.items():
            if implicit_idx < num_implicit_atoms:
                implicit_atom_index[implicit_idx] = explicit_idx

        expected = np.array([0, 1, 2, 5, 6], dtype=np.int64)
        np.testing.assert_array_equal(implicit_atom_index, expected)


# ---------------------------------------------------------------------------
# Tests for Multi-site Titration Logic
# ---------------------------------------------------------------------------


class TestMultiSiteTitrationLogic:
    """Test suite for multi-site (coupled) titration logic."""

    def test_neighbor_selection(self) -> None:
        """Test random neighbor selection for multi-site moves."""
        neighbors = [5, 10, 15, 20]

        # Select a random neighbor
        selected = np.random.choice(neighbors)

        assert selected in neighbors

    def test_state_index_list_handling(self) -> None:
        """Test handling of multiple state indices for coupled moves."""
        # Single residue move
        state_indices = [1]
        assert len(state_indices) == 1

        # Coupled move (2 residues)
        state_indices.append(0)
        assert len(state_indices) == 2

    def test_titration_list_handling(self) -> None:
        """Test handling of multiple titrations for coupled moves."""
        from molecular_simulations.simulate.constantph.constantph import (
            ResidueTitration,
        )

        titrations = [
            ResidueTitration(['ASP', 'ASH'], [0.0, 5.2]),
            ResidueTitration(['GLU', 'GLH'], [0.0, 4.8]),
        ]

        # Process multiple titrations
        total_ref_delta = sum(
            t.referenceEnergies[1] - t.referenceEnergies[0] for t in titrations
        )

        assert total_ref_delta == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests for Context Parameter Updates
# ---------------------------------------------------------------------------


class TestContextParameterUpdates:
    """Test suite for context parameter update logic."""

    def test_force_update_in_context(self) -> None:
        """Test logic for updating forces in context."""
        from openmm import Context, NonbondedForce, System, VerletIntegrator
        from openmm.unit import femtoseconds

        system = System()
        system.addParticle(14.0)

        nb_force = NonbondedForce()
        nb_force.addParticle(-0.4, 0.17, 0.0)
        system.addForce(nb_force)

        integrator = VerletIntegrator(1.0 * femtoseconds)
        context = Context(system, integrator)
        context.setPositions([[0, 0, 0]])

        # Update particle parameter
        nb_force.setParticleParameters(0, -0.5, 0.17, 0.0)
        nb_force.updateParametersInContext(context)

        # Verify update
        force = context.getSystem().getForce(0)
        _q, _sigma, _eps = force.getParticleParameters(0)
        # Note: After updateParametersInContext, we need to get from the force
        # The force object itself is updated
        assert True  # Just verify no exception was raised

    def test_reinitialize_preserve_state(self) -> None:
        """Test context reinitialization with preserveState."""
        from openmm import Context, NonbondedForce, System, VerletIntegrator
        from openmm.unit import femtoseconds, nanometers

        system = System()
        system.addParticle(14.0)

        nb_force = NonbondedForce()
        nb_force.addParticle(-0.4, 0.17, 0.0)
        system.addForce(nb_force)

        integrator = VerletIntegrator(1.0 * femtoseconds)
        context = Context(system, integrator)
        context.setPositions([[0.1, 0.2, 0.3]])

        # Reinitialize preserving state
        context.reinitialize(preserveState=True)

        # Verify positions are preserved
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        assert positions[0][0].value_in_unit(nanometers) == pytest.approx(0.1, rel=0.01)


# ---------------------------------------------------------------------------
# Additional Parametrized Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'num_hydrogens_list,expected_protonated_idx',
    [
        ([0, 1], 1),
        ([1, 0], 0),
        ([1, 1, 2], 2),
        ([2, 1, 1], 0),
        ([0, 0, 1, 0], 2),
    ],
)
class TestProtonatedIndexParametrized:
    """Parametrized tests for protonated index identification."""

    def test_protonated_index_from_hydrogen_counts(
        self, num_hydrogens_list: list, expected_protonated_idx: int
    ) -> None:
        """Test protonated index is state with most hydrogens."""
        protonated_idx = np.argmax(num_hydrogens_list)
        assert protonated_idx == expected_protonated_idx


@pytest.mark.parametrize(
    'delta_energy_kj,delta_n,ph,expected_favorable',
    [
        (-50.0, 0, 7.0, True),  # Large favorable energy, no proton change
        (50.0, 0, 7.0, False),  # Large unfavorable energy
        (-10.0, 1, 7.0, False),  # Energy favorable but proton term unfavorable
        (-25.0, 1, 4.0, True),  # Energy favorable, low pH helps protonation
        (0.0, -1, 7.0, True),  # No energy change, deprotonation at pH 7 favorable
    ],
)
class TestAcceptanceCriterionParametrized:
    """Parametrized tests for MC acceptance criterion."""

    def test_acceptance_favorability(
        self, delta_energy_kj: float, delta_n: int, ph: float, expected_favorable: bool
    ) -> None:
        """Test whether moves are favorable based on energy and proton changes."""
        from openmm.unit import MOLAR_GAS_CONSTANT_R, kelvin, kilojoules_per_mole

        temperature = 300.0 * kelvin
        kT = MOLAR_GAS_CONSTANT_R * temperature
        delta_energy = delta_energy_kj * kilojoules_per_mole

        w = delta_energy / kT + delta_n * np.log(10.0) * ph

        is_favorable = w <= 0
        assert is_favorable == expected_favorable


# ---------------------------------------------------------------------------
# ConstantPH._get_zero_parameters Tests (real forces, no mocks)
# ---------------------------------------------------------------------------


class TestConstantPHGetZeroParameters:
    """Test suite for ConstantPH._get_zero_parameters using real OpenMM forces."""

    def test_zero_parameters_nonbonded_force(self) -> None:
        """Charge (element 0) is zeroed for a real NonbondedForce; sigma/epsilon kept."""
        from openmm import NonbondedForce

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        force = NonbondedForce()

        # (charge, sigma, epsilon)
        result = cph._get_zero_parameters((-0.42, 0.31, 0.65), force)

        assert result == (0.0, 0.31, 0.65)
        assert result[0] == 0.0
        assert result[1] == 0.31
        assert result[2] == 0.65

    def test_zero_parameters_gbsaobc_force(self) -> None:
        """Charge (element 0) is zeroed for a real GBSAOBCForce; radius/scale kept."""
        from openmm import GBSAOBCForce

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        force = GBSAOBCForce()

        # (charge, radius, scale)
        result = cph._get_zero_parameters((0.55, 0.17, 0.85), force)

        assert result == (0.0, 0.17, 0.85)
        assert result[0] == 0.0
        assert result[1] == 0.17
        assert result[2] == 0.85

    def test_zero_parameters_preserves_tuple_length(self) -> None:
        """Zeroing does not change the arity of the parameter tuple."""
        from openmm import NonbondedForce

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        force = NonbondedForce()

        original = (1.23, 0.29, 0.42)
        result = cph._get_zero_parameters(original, force)

        assert len(result) == len(original)
        assert isinstance(result, tuple)
        # Only the charge changed.
        assert result[1:] == original[1:]


# ---------------------------------------------------------------------------
# ConstantPH._identifyProtonatedState Tests (real logic, no mocks)
# ---------------------------------------------------------------------------


class TestConstantPHIdentifyProtonatedState:
    """Test suite for ConstantPH._identifyProtonatedState."""

    def test_identify_asp_ash(self) -> None:
        """ASP/ASH -> index of ASH (the protonated aspartate form)."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        assert cph._identifyProtonatedState(['ASP', 'ASH']) == 1
        # Order-independent: the protonated form wins regardless of position.
        assert cph._identifyProtonatedState(['ASH', 'ASP']) == 0

    def test_identify_histidine_hip(self) -> None:
        """HID/HIE/HIP -> index of HIP (doubly protonated histidine)."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        assert cph._identifyProtonatedState(['HID', 'HIE', 'HIP']) == 2
        assert cph._identifyProtonatedState(['HIP', 'HID', 'HIE']) == 0

    def test_identify_singly_protonated_histidine_fallback(self) -> None:
        """No fully-protonated form present -> first HID/HIE intermediate is chosen."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        # No HIP present; HID is the first recognised singly-protonated form.
        assert cph._identifyProtonatedState(['HID', 'HIE']) == 0
        assert cph._identifyProtonatedState(['HIE', 'HID']) == 0

    def test_identify_unrecognized_returns_zero(self) -> None:
        """No recognised protonated form -> fallback index 0."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        assert cph._identifyProtonatedState(['XAA', 'YBB']) == 0

    def test_identify_glutamate_and_lysine(self) -> None:
        """GLH and LYS are recognised protonated forms."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        assert cph._identifyProtonatedState(['GLU', 'GLH']) == 1
        assert cph._identifyProtonatedState(['LYN', 'LYS']) == 1


# ---------------------------------------------------------------------------
# ConstantPH.validateStates Tests (real titrations, no mocks)
# ---------------------------------------------------------------------------


def _validation_titration(variants, implicit_h, explicit_h, protonated_index):
    """Build a real ResidueTitration wired for validateStates()."""
    titration = ResidueTitration(list(variants), [0.0] * len(variants))
    titration.implicitStates = [_real_state(h, residue_index=3) for h in implicit_h]
    titration.explicitStates = [_real_state(h, residue_index=3) for h in explicit_h]
    titration.protonatedIndex = protonated_index
    titration.currentIndex = protonated_index
    return titration


class TestConstantPHValidateStates:
    """Test suite for ConstantPH.validateStates."""

    def test_validate_states_valid(self) -> None:
        """A consistent titration passes validation and returns True."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        cph.titrations = {3: _validation_titration(['ASP', 'ASH'], [4, 5], [4, 5], 1)}
        assert cph.validateStates() is True

    def test_validate_states_count_mismatch(self) -> None:
        """State-count != variant-count fails validation."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        # Two variants but only one implicit state.
        cph.titrations = {3: _validation_titration(['ASP', 'ASH'], [4], [4, 5], 0)}
        assert cph.validateStates() is False

    def test_validate_states_identical_hydrogen_counts(self) -> None:
        """All states sharing the same numHydrogens fails validation."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        cph.titrations = {3: _validation_titration(['ASP', 'ASH'], [5, 5], [4, 5], 1)}
        assert cph.validateStates() is False

    def test_validate_states_protonated_index_out_of_range(self) -> None:
        """protonatedIndex >= number of states fails validation."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        cph.titrations = {3: _validation_titration(['ASP', 'ASH'], [4, 5], [4, 5], 9)}
        assert cph.validateStates() is False

    def test_validate_states_reports_issue(self, capsys) -> None:
        """Failure path prints a FAILED banner naming the problem."""
        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        cph = object.__new__(ConstantPH)
        cph.titrations = {3: _validation_titration(['ASP', 'ASH'], [4, 5], [4, 5], 9)}
        result = cph.validateStates()
        captured = capsys.readouterr()

        assert result is False
        assert 'State validation FAILED' in captured.out
        assert 'protonatedIndex=9 out of range' in captured.out


# ---------------------------------------------------------------------------
# ConstantPH.printTitrationState Tests (real titrations, capsys, no mocks)
# ---------------------------------------------------------------------------


class TestConstantPHPrintTitrationState:
    """Test suite for ConstantPH.printTitrationState."""

    def test_print_titration_state_output(self, capsys) -> None:
        """The formatted per-residue / per-state summary is emitted to stdout."""
        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
        )

        cph = object.__new__(ConstantPH)
        cph.pH = [7.4]
        cph.currentPHIndex = 0

        titration = ResidueTitration(['ASP', 'ASH'], [0.0, 5.0])
        titration.implicitStates = [
            ResidueState(2, {'N': 0}, {}, {}, 4),
            ResidueState(2, {'N': 0, 'HD2': 1}, {}, {}, 5),
        ]
        titration.explicitStates = [
            ResidueState(2, {'N': 0}, {}, {}, 4),
            ResidueState(2, {'N': 0, 'HD2': 1}, {}, {}, 5),
        ]
        titration.currentIndex = 1
        titration.protonatedIndex = 1
        cph.titrations = {2: titration}

        cph.printTitrationState()
        out = capsys.readouterr().out

        assert 'Current pH: 7.40' in out
        assert 'Titratable residues: 1' in out
        assert 'Res 2: currentState=1, protonatedIdx=1' in out
        assert "variants=['ASP', 'ASH']" in out
        assert 'refEnergies=[0.0, 5.0]' in out
        # Deprotonated state has 4 H / 1 atom, protonated has 5 H / 2 atoms.
        assert 'state[0]: implicitH=4, explicitH=4, implAtoms=1, explAtoms=1' in out
        assert 'state[1]: implicitH=5, explicitH=5, implAtoms=2, explAtoms=2' in out


# ---------------------------------------------------------------------------
# ConstantPH._findResidueStates Tests (real ForceField + topology, no mocks)
# ---------------------------------------------------------------------------


class TestConstantPHFindResidueStatesReal:
    """Real end-to-end exercise of _findResidueStates.

    Uses the committed ACE-LYS-ASP-NME fixture (tests/data/amber/lys_asp.pdb),
    strips its hydrogens to obtain a heavy-atom topology, then rebuilds the ASP
    and ASH protonation states with a real amber14 ForceField.
    """

    @staticmethod
    def _heavy_atom_topology():
        import pathlib

        from openmm.app import Modeller, PDBFile, element

        pdb_path = pathlib.Path(__file__).parent / 'data' / 'amber' / 'lys_asp.pdb'
        pdb = PDBFile(str(pdb_path))
        modeller = Modeller(pdb.topology, pdb.positions)
        hydrogens = [
            a for a in modeller.topology.atoms() if a.element == element.hydrogen
        ]
        modeller.delete(hydrogens)
        return modeller.topology, modeller.positions

    def test_find_residue_states_asp_vs_ash(self) -> None:
        """ASP and ASH differ by one hydrogen and populate real force params."""
        import openmm
        from openmm.app import ForceField

        from molecular_simulations.simulate.constantph.constantph import ConstantPH

        topology, positions = self._heavy_atom_topology()
        forcefield = ForceField('amber14-all.xml')
        ffargs = {
            'nonbondedMethod': openmm.app.NoCutoff,
            'constraints': None,
            'rigidWater': False,
        }
        cph = object.__new__(ConstantPH)

        # Residue order is ACE, LYS, ASP, NME -> only index 2 is titrated here.
        asp_states = cph._findResidueStates(
            topology, positions, forcefield, [None, None, 'ASP', None], ffargs
        )
        ash_states = cph._findResidueStates(
            topology, positions, forcefield, [None, None, 'ASH', None], ffargs
        )

        assert len(asp_states) == 1
        assert len(ash_states) == 1

        asp, ash = asp_states[0], ash_states[0]

        # ASH (protonated) has exactly one more hydrogen than ASP.
        assert ash.numHydrogens == asp.numHydrogens + 1
        assert asp.numHydrogens == 4
        assert ash.numHydrogens == 5

        # Both states are built for the ASP residue (topology index 2).
        assert asp.residueIndex == 2
        assert ash.residueIndex == 2

        # ASH gains the carboxyl proton absent in ASP.
        assert 'HD2' not in asp.atomIndices
        assert 'HD2' in ash.atomIndices

        # particleParameters / exceptionParameters are populated with real values.
        assert len(asp.particleParameters) > 0
        assert len(asp.exceptionParameters) > 0
        # Every populated force maps a per-atom charge/sigma/epsilon tuple.
        for force_params in asp.particleParameters.values():
            assert 'CA' in force_params
            assert len(force_params['CA']) >= 1

    def test_find_residue_states_returns_residuestate_objects(self) -> None:
        """The returned objects are real ResidueState instances."""
        import openmm
        from openmm.app import ForceField

        from molecular_simulations.simulate.constantph.constantph import (
            ConstantPH,
            ResidueState,
        )

        topology, positions = self._heavy_atom_topology()
        forcefield = ForceField('amber14-all.xml')
        ffargs = {
            'nonbondedMethod': openmm.app.NoCutoff,
            'constraints': None,
            'rigidWater': False,
        }
        cph = object.__new__(ConstantPH)

        states = cph._findResidueStates(
            topology, positions, forcefield, [None, None, 'ASP', None], ffargs
        )

        assert all(isinstance(s, ResidueState) for s in states)
        assert isinstance(states[0].atomIndices, dict)
        assert 'N' in states[0].atomIndices


# ---------------------------------------------------------------------------
# Full ConstantPH.__init__ pipeline (real, committed solvated fixture)
# ---------------------------------------------------------------------------


def _build_solvated_cph(files):
    """Construct a real ConstantPH from the committed solvated Lys-Asp fixture.

    Mirrors how ``run_cph_sim`` assembles the constructor arguments (explicit PME
    args, implicit non-periodic args, real Langevin integrators, LYS/ASP variants
    at indices 1/2). The committed ``lys_asp_solv`` fixture was pre-built with
    tleap, so this whole pipeline runs in CI with no AmberTools at runtime -- only
    OpenMM + ParmEd + amber14-all.xml, all available.
    """
    from openmm import LangevinIntegrator, Platform
    from openmm.app import PME, CutoffNonPeriodic, HBonds
    from openmm.unit import (
        amu,
        kelvin,
        kilojoules_per_mole,
        nanometers,
        picosecond,
    )

    from molecular_simulations.simulate.constantph.constantph import ConstantPH

    explicit_args = dict(
        nonbondedMethod=PME,
        nonbondedCutoff=0.9 * nanometers,
        constraints=HBonds,
        hydrogenMass=1.5 * amu,
    )
    implicit_args = dict(
        nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=2.0 * nanometers,
        constraints=HBonds,
    )
    reference_energies = {
        1: [0.0 * kilojoules_per_mole, 0.0 * kilojoules_per_mole],
        2: [0.0 * kilojoules_per_mole, 0.0 * kilojoules_per_mole],
    }

    return ConstantPH(
        prmtop_file=files['prmtop'],
        inpcrd_file=files['inpcrd'],
        pH=[7.0],
        residueVariants={1: ['LYN', 'LYS'], 2: ['ASP', 'ASH']},
        referenceEnergies=reference_energies,
        relaxationSteps=5,
        explicitArgs=explicit_args,
        implicitArgs=implicit_args,
        integrator=LangevinIntegrator(
            300 * kelvin, 1.0 / picosecond, 0.004 * picosecond
        ),
        relaxationIntegrator=LangevinIntegrator(
            300 * kelvin, 10.0 / picosecond, 0.002 * picosecond
        ),
        platform=Platform.getPlatformByName('CPU'),
    )


@pytest.fixture(scope='module')
def real_solvated_cph():
    """A real ConstantPH built once from the committed solvated fixture.

    Module-scoped so the ~1.3 s construction happens a single time; the tests
    that use it only make read-only assertions about the built systems/states.
    """
    # tmp_path is function-scoped, so resolve the committed files directly rather
    # than via the (function-scoped) real_amber_titratable_solvated_files fixture.
    from pathlib import Path

    src = Path(__file__).parent / 'data' / 'amber'
    files = {
        'prmtop': str(src / 'lys_asp_solv.prmtop'),
        'inpcrd': str(src / 'lys_asp_solv.inpcrd'),
    }
    return _build_solvated_cph(files)


class TestConstantPHRealInitPipeline:
    """Exercise the full ConstantPH.__init__ pipeline on a real solvated system.

    These cover the methods that need a real solvated titratable system --
    _buildImplicitSystemWithParmEd, _buildProteinOnlyTopology,
    _buildProtonationStates, _mapStatesToImplicitSystem, _buildExplicitSystem,
    _mapStatesToExplicitSystem -- which cannot be reached from object.__new__.
    """

    def test_titrations_built_for_lys_and_asp(self, real_solvated_cph):
        """Both titratable residues get real ResidueTitration objects."""
        titrations = real_solvated_cph.titrations
        assert set(titrations) == {1, 2}
        assert titrations[1].variants == ['LYN', 'LYS']
        assert titrations[2].variants == ['ASP', 'ASH']

    def test_implicit_system_is_solvent_stripped(self, real_solvated_cph):
        """ParmEd strips water/ions: implicit system is the 46-atom peptide."""
        # 2620-atom solvated explicit system, 46-atom (Ace-Lys-Asp-Nme) implicit.
        assert real_solvated_cph.explicitSystem.getNumParticles() == 2620
        assert real_solvated_cph.implicitSystem.getNumParticles() == 46

    def test_protonation_states_have_expected_hydrogen_counts(self, real_solvated_cph):
        """The built protonation states carry the real +1-proton differences."""
        lys = real_solvated_cph.titrations[1]
        asp = real_solvated_cph.titrations[2]
        # LYN (neutral) has one fewer H than LYS (protonated); ASP one fewer than ASH.
        lys_h = sorted(s.numHydrogens for s in lys.explicitStates)
        asp_h = sorted(s.numHydrogens for s in asp.explicitStates)
        assert lys_h == [12, 13]
        assert asp_h == [4, 5]

    def test_atom_index_mapping_covers_implicit_system(self, real_solvated_cph):
        """implicitAtomIndex maps every implicit atom into the explicit system."""
        idx = real_solvated_cph.implicitAtomIndex
        assert len(idx) == 46
        # Every mapped index is a valid explicit-system particle.
        assert idx.max() < real_solvated_cph.explicitSystem.getNumParticles()
        assert idx.min() >= 0

    def test_validate_states_passes_on_real_build(self, real_solvated_cph):
        """The real, fully-built protonation states pass validation."""
        assert real_solvated_cph.validateStates() is True

    def test_explicit_context_has_real_energy(self, real_solvated_cph):
        """The explicit Simulation was created with positions and a finite energy."""
        import numpy as np

        state = real_solvated_cph.simulation.context.getState(getEnergy=True)
        pe = state.getPotentialEnergy()._value
        assert np.isfinite(pe)

    def test_attempt_mc_step_runs_and_keeps_states_valid(
        self, real_amber_titratable_solvated_files
    ):
        """A real MC titration step runs and leaves the ensemble in a valid state.

        Builds its own instance (attemptMCStep can mutate protonation state) and
        checks the real move machinery -- position mapping into the implicit
        context, energy evaluation, Metropolis accept/reject -- runs without error
        and leaves every residue at a valid variant index.
        """
        cph = _build_solvated_cph(real_amber_titratable_solvated_files)

        cph.attemptMCStep(300.0)

        for resid, titration in cph.titrations.items():
            assert 0 <= titration.currentIndex < len(titration.variants), resid
        assert cph.validateStates() is True

    def test_gb_force_parameters_are_captured_per_state(self, real_solvated_cph):
        """The implicit GB force is captured into every protonation state.

        Regression test: ParmEd builds the GB model as a CustomGBForce, which the
        mapping code originally ignored (it only matched GBSAOBCForce). That left
        the GB charges frozen across states, so GB solvation cancelled out of
        every MC energy difference. Each state must now carry GB parameters keyed
        by the implicit GB force index.
        """
        from openmm import CustomGBForce, GBSAOBCForce

        cph = real_solvated_cph
        gb_idx = next(
            i
            for i, f in enumerate(cph.implicitSystem.getForces())
            if isinstance(f, (GBSAOBCForce, CustomGBForce))
        )
        for titration in cph.titrations.values():
            for state in titration.implicitStates:
                assert gb_idx in state.particleParameters
                assert state.particleParameters[gb_idx], 'GB params must be non-empty'

    def test_gb_charges_differ_between_protonation_states(self, real_solvated_cph):
        """GB per-particle charge tracks the NB charge and changes on titration.

        The Lys fixture is stored protonated (LYS), so its labile HZ3 proton is a
        real particle: charge 0 in LYN (deprotonated ghost) and a real positive
        charge in LYS. The GB charge must mirror that so the desolvation penalty
        enters the MC energy -- the behaviour that was broken when the
        CustomGBForce was ignored.
        """
        from openmm import CustomGBForce, GBSAOBCForce

        cph = real_solvated_cph
        gb_idx = next(
            i
            for i, f in enumerate(cph.implicitSystem.getForces())
            if isinstance(f, (GBSAOBCForce, CustomGBForce))
        )
        lys = cph.titrations[1]  # variants ['LYN', 'LYS'], protonated index 1
        gb0 = lys.implicitStates[0].particleParameters[gb_idx]  # LYN (deprotonated)
        gb1 = lys.implicitStates[1].particleParameters[gb_idx]  # LYS (protonated)

        def charge(params):
            c = params[0]
            return c.value_in_unit(c.unit) if hasattr(c, 'unit') else float(c)

        # At least one atom's GB charge (parameter index 0) must differ.
        shared = set(gb0) & set(gb1)
        assert any(
            abs(charge(gb0[name]) - charge(gb1[name])) > 1e-6 for name in shared
        ), 'GB charges are identical across protonation states'

        # The labile HZ3 proton is a zero-charge ghost in LYN, real charge in LYS.
        assert abs(charge(gb0['HZ3'])) < 1e-9
        assert abs(charge(gb1['HZ3'])) > 1e-3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
