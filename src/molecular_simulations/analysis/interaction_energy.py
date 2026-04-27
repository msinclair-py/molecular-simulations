"""Interaction energy calculation module.

This module provides tools for computing linear interaction energies
between specified chains and other simulation components using OpenMM.
Supports both static models and dynamic trajectory analysis.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import mdtraj as md
import numpy as np
from openmm import Context, NonbondedForce, Platform, System, VerletIntegrator
from openmm.app import (
    AmberPrmtopFile,
    CutoffNonPeriodic,
    ForceField,
    HBonds,
    PDBFile,
    Topology,
)
from openmm.unit import kilocalories_per_mole, nanometers, picosecond  # type: ignore[attr-defined]
from pdbfixer import PDBFixer
from tqdm import tqdm

PathLike = Path | str

class InteractionEnergy(ABC):
    """Abstract base class for interaction energy calculations.

    Defines the interface for all interaction energy calculation classes.
    """

    @abstractmethod
    def __init__(self):
        """Initialize the InteractionEnergy base class."""
        pass

    @abstractmethod
    def compute(self):
        """Compute the interaction energy.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def energy(self, *args, **kwargs):
        """Get the computed energy.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_selection(self, *args, **kwargs):
        """Get the atom selection for energy calculation.

        Must be implemented by subclasses.
        """
        pass


class StaticInteractionEnergy(InteractionEnergy):
    """Compute linear interaction energy for a static model.

    Computes the linear interaction energy between a specified chain and
    other simulation components. Can specify a range of residues to limit
    the calculation. Works on a static model but can be adapted for
    trajectory data.

    Attributes:
        pdb: Path to the input PDB file.
        chain: Chain identifier for interaction calculations.
        platform: OpenMM platform for computation.
        lj: Lennard-Jones energy after compute().
        coulomb: Coulombic energy after compute().
        selection: Atom indices for the selected chain.

    Args:
        pdb: Path to input PDB file.
        chain: Chain identifier for energy calculation. Computes energy
            between this chain and all other components. Use whitespace
            if there are no chains. Defaults to 'A'.
        platform: OpenMM platform name. Defaults to 'CUDA'.
        first_residue: If set, restricts calculation to residues starting
            from this resid. Defaults to None.
        last_residue: If set, restricts calculation to residues ending
            at this resid. Defaults to None.

    Example:
        >>> ie = StaticInteractionEnergy("complex.pdb", chain="B")
        >>> ie.compute()
        >>> print(f"LJ: {ie.lj}, Coulomb: {ie.coulomb}")
    """

    def __init__(
        self,
        pdb: str,
        chain: str = 'A',
        platform: str = 'CUDA',
        first_residue: int | None = None,
        last_residue: int | None = None,
    ):
        """Initialize the StaticInteractionEnergy calculator.

        Args:
            pdb: Path to input PDB file.
            chain: Chain identifier for calculation.
            platform: OpenMM platform name.
            first_residue: Starting residue for calculation.
            last_residue: Ending residue for calculation.
        """
        self.pdb = pdb
        self.chain = chain
        self.platform = Platform.getPlatformByName(platform)
        self.first = first_residue
        self.last = last_residue

    def get_system(self) -> System:
        """Build an implicit solvent OpenMM system.

        Loads the PDB file and creates an OpenMM system with GB/n2
        implicit solvent model. Automatically fixes the PDB if needed.

        Returns:
            OpenMM System object configured with implicit solvent.
        """
        pdb = PDBFile(self.pdb)
        positions, topology = pdb.positions, pdb.topology
        forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        try:
            system = forcefield.createSystem(
                topology, soluteDielectric=1.0, solventDielectric=80.0
            )
        except ValueError:
            positions, topology = self.fix_pdb()
            system = forcefield.createSystem(
                topology, soluteDielectric=1.0, solventDielectric=80.0
            )

        self.positions = positions
        self.get_selection(topology)

        return system

    def compute(self, positions: np.ndarray | None = None) -> None:
        """Compute interaction energy of the system.

        Computes both Lennard-Jones and Coulombic interaction energies
        between the selected chain and all other atoms.

        Args:
            positions: Optional atomic positions to use instead of
                those from the PDB file. Useful for trajectory analysis.
        """
        self.lj = None
        self.coulomb = None

        system = self.get_system()
        if positions is None:
            positions = self.positions

        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                force.setForceGroup(0)
                force.addGlobalParameter('solute_coulomb_scale', 1)
                force.addGlobalParameter('solute_lj_scale', 1)
                force.addGlobalParameter('solvent_coulomb_scale', 1)
                force.addGlobalParameter('solvent_lj_scale', 1)

                for i in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(i)
                    force.setParticleParameters(i, 0, 0, 0)
                    if i in self.selection:
                        force.addParticleParameterOffset(
                            'solute_coulomb_scale', i, charge, 0, 0
                        )
                        force.addParticleParameterOffset(
                            'solute_lj_scale', i, 0, sigma, epsilon
                        )
                    else:
                        force.addParticleParameterOffset(
                            'solvent_coulomb_scale', i, charge, 0, 0
                        )
                        force.addParticleParameterOffset(
                            'solvent_lj_scale', i, 0, sigma, epsilon
                        )

                for i in range(force.getNumExceptions()):
                    p1, p2, _, _, _ = force.getExceptionParameters(i)
                    force.setExceptionParameters(i, p1, p2, 0, 0, 0)

            else:
                force.setForceGroup(2)

        integrator = VerletIntegrator(0.001 * picosecond)
        context = Context(system, integrator, self.platform)
        context.setPositions(positions)

        total_coulomb = self.energy(context, 1, 0, 1, 0)
        solute_coulomb = self.energy(context, 1, 0, 0, 0)
        solvent_coulomb = self.energy(context, 0, 0, 1, 0)
        total_lj = self.energy(context, 0, 1, 0, 1)
        solute_lj = self.energy(context, 0, 1, 0, 0)
        solvent_lj = self.energy(context, 0, 0, 0, 1)

        coul_final = total_coulomb - solute_coulomb - solvent_coulomb
        lj_final = total_lj - solute_lj - solvent_lj

        self.coulomb = coul_final.value_in_unit(kilocalories_per_mole)
        self.lj = lj_final.value_in_unit(kilocalories_per_mole)

    def get_selection(self, topology: Topology) -> None:
        """Get indices of atoms for pairwise interaction calculation.

        Uses OpenMM's selection capabilities to identify atoms in the
        specified chain and residue range.

        Args:
            topology: OpenMM Topology object.
        """
        if self.first is None and self.last is None:
            selection = [
                a.index for a in topology.atoms() if a.residue.chain.id == self.chain
            ]
        elif self.first is not None and self.last is None:
            selection = [
                a.index
                for a in topology.atoms()
                if a.residue.chain.id == self.chain
                and int(self.first) <= int(a.residue.id)
            ]
        elif self.first is None:
            assert self.last is not None
            selection = [
                a.index
                for a in topology.atoms()
                if a.residue.chain.id == self.chain
                and int(self.last) >= int(a.residue.id)
            ]
        else:
            assert self.last is not None
            selection = [
                a.index
                for a in topology.atoms()
                if a.residue.chain.id == self.chain
                and int(self.first) <= int(a.residue.id) <= int(self.last)
            ]

        self.selection = selection

    def fix_pdb(self) -> tuple:
        """Repair the input PDB using PDBFixer.

        Adds missing residues, atoms, and hydrogens to create a
        complete structure suitable for OpenMM.

        Returns:
            Tuple of (positions, topology) after fixing.
        """
        fixer = PDBFixer(filename=self.pdb)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        return fixer.positions, fixer.topology

    @property
    def interactions(self) -> np.ndarray:
        """Get LJ and Coulombic energies as an array.

        Returns:
            Array of shape (2, 1) containing [lj, coulomb] energies.
        """
        return np.vstack([self.lj, self.coulomb])  # type: ignore[arg-type]

    @staticmethod
    def energy(
        context: Context,
        solute_coulomb_scale: int = 0,
        solute_lj_scale: int = 0,
        solvent_coulomb_scale: int = 0,
        solvent_lj_scale: int = 0,
    ):
        """Compute potential energy for the given context.

        Args:
            context: OpenMM Context object.
            solute_coulomb_scale: Scale for solute Coulombic energy (0 or 1).
            solute_lj_scale: Scale for solute LJ energy (0 or 1).
            solvent_coulomb_scale: Scale for solvent Coulombic energy (0 or 1).
            solvent_lj_scale: Scale for solvent LJ energy (0 or 1).

        Returns:
            Computed energy term with units.
        """
        context.setParameter('solute_coulomb_scale', solute_coulomb_scale)
        context.setParameter('solute_lj_scale', solute_lj_scale)
        context.setParameter('solvent_coulomb_scale', solvent_coulomb_scale)
        context.setParameter('solvent_lj_scale', solvent_lj_scale)
        return context.getState(getEnergy=True, groups={0}).getPotentialEnergy()


class InteractionEnergyFrame(StaticInteractionEnergy):
    """Interaction energy calculator for trajectory frames.

    Inherits from StaticInteractionEnergy and overloads get_system to
    allow for easier trajectory analysis. Requires the OpenMM system
    and topology to be built externally.

    Args:
        system: Pre-built OpenMM System object.
        top: OpenMM Topology object.
        chain: Chain identifier for calculation. Defaults to 'A'.
        platform: OpenMM platform name. Defaults to 'CUDA'.
        first_residue: Starting residue for calculation. Defaults to None.
        last_residue: Ending residue for calculation. Defaults to None.

    Example:
        >>> system = build_system(topology)
        >>> ie = InteractionEnergyFrame(system, topology, chain="A")
        >>> ie.compute(positions)
    """

    def __init__(
        self,
        system: System,
        top: Topology,
        chain: str = 'A',
        platform: str = 'CUDA',
        first_residue: int | None = None,
        last_residue: int | None = None,
    ):
        """Initialize the InteractionEnergyFrame calculator.

        Args:
            system: Pre-built OpenMM System object.
            top: OpenMM Topology object.
            chain: Chain identifier for calculation.
            platform: OpenMM platform name.
            first_residue: Starting residue for calculation.
            last_residue: Ending residue for calculation.
        """
        super().__init__('', chain, platform, first_residue, last_residue)
        self.system = system
        self.top = top

    def get_system(self) -> System:
        """Return the pre-built OpenMM system.

        Sets self.selection via get_selection and returns the existing
        system object.

        Returns:
            The pre-built OpenMM System object.
        """
        self.get_selection(self.top)
        return self.system


class DynamicInteractionEnergy:
    """Compute interaction energies over a trajectory.

    Uses InteractionEnergyFrame to run per-frame energy calculations
    and orchestrates trajectory operations.

    Attributes:
        system: OpenMM System object.
        coordinates: Trajectory coordinate array.
        stride: Frame stride for calculations.
        energies: Energy array after compute_energies().
        IE: InteractionEnergyFrame instance.

    Args:
        top: Path to prmtop topology file.
        traj: Path to DCD trajectory file.
        stride: Stride for moving through trajectory. Defaults to 1.
        chain: Chain identifier for calculation. Defaults to 'A'.
        platform: OpenMM platform name. Defaults to 'CUDA'.
        first_residue: Starting residue for calculation. Defaults to None.
        last_residue: Ending residue for calculation. Defaults to None.
        progress_bar: Whether to display a tqdm progress bar.
            Defaults to False.

    Example:
        >>> die = DynamicInteractionEnergy("system.prmtop", "traj.dcd")
        >>> die.compute_energies()
        >>> print(die.energies.shape)  # (n_frames, 2)
    """

    def __init__(
        self,
        top: PathLike,
        traj: PathLike,
        stride: int = 1,
        chain: str = 'A',
        platform: str = 'CUDA',
        first_residue: int | None = None,
        last_residue: int | None = None,
        progress_bar: bool = False,
    ):
        """Initialize the DynamicInteractionEnergy calculator.

        Args:
            top: Path to topology file.
            traj: Path to trajectory file.
            stride: Frame stride.
            chain: Chain identifier.
            platform: OpenMM platform name.
            first_residue: Starting residue.
            last_residue: Ending residue.
            progress_bar: Whether to show progress.
        """
        top = Path(top)
        traj = Path(traj)
        self.system = self.build_system(top)
        self.coordinates = self.load_traj(top, traj)
        self.stride = stride
        self.progress = progress_bar

        self.IE = InteractionEnergyFrame(
            self.system, self.top, chain, platform, first_residue, last_residue  # type: ignore[arg-type]
        )

    def compute_energies(self) -> None:
        """Compute energies for each frame in the trajectory.

        Stores results in self.energies with shape (n_frames, 2)
        where columns are [LJ, Coulomb].
        """
        n_frames = self.coordinates.shape[0] // self.stride
        self.energies = np.zeros((n_frames, 2))

        if self.progress:
            pbar = tqdm(total=n_frames, position=0, leave=False)

        for i in range(n_frames):
            fr = i * self.stride
            self.IE.compute(self.coordinates[fr, :, :])
            self.energies[i, 0] = self.IE.lj
            self.energies[i, 1] = self.IE.coulomb

            if self.progress:
                pbar.update(1)

        if self.progress:
            pbar.close()

    def build_system(self, top: PathLike) -> System:
        """Build an OpenMM system from the topology file.

        Handles both PDB and prmtop topology files.

        Args:
            top: Path to topology file.

        Returns:
            OpenMM System object.

        Raises:
            NotImplementedError: If topology file type is not supported.
        """
        top_path = Path(top)
        if top_path.suffix == '.pdb':
            pdb_topology = PDBFile(str(top_path)).topology
            self.top = pdb_topology
            forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
            return forcefield.createSystem(
                pdb_topology, soluteDielectric=1.0, solventDielectric=78.5
            )
        elif top_path.suffix == '.prmtop':
            prmtop = AmberPrmtopFile(str(top_path))
            self.top = prmtop
            return prmtop.createSystem(
                nonbondedMethod=CutoffNonPeriodic,
                nonbondedCutoff=2.0 * nanometers,
                constraints=HBonds,
            )
        else:
            raise NotImplementedError(f'Error! Topology type {top} not implemented!')

    def load_traj(self, top: PathLike, traj: PathLike) -> np.ndarray:
        """Load trajectory into mdtraj and extract coordinates.

        Args:
            top: Path to topology file.
            traj: Path to trajectory file.

        Returns:
            Coordinate array with shape (n_frames, n_atoms, 3).
        """
        return md.load(str(traj), top=str(top)).xyz

    def setup_pbar(self) -> None:
        """Build a tqdm progress bar for trajectory iteration."""
        self.pbar = tqdm(total=self.coordinates.shape[0], position=0, leave=False)
