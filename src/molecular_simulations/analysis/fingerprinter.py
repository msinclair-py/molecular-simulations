"""Interaction energy fingerprinting module.

This module calculates interaction energy fingerprints between target and
binder chains in molecular structures, computing both electrostatic and
Lennard-Jones contributions at the residue level.
"""

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import openmm
from numba import njit
from openmm.app import AmberPrmtopFile

OptPath = Path | str | None
PathLike = Path | str


@njit
def unravel_index(n1: int, n2: int) -> tuple[np.ndarray, np.ndarray]:
    """Create unraveled indices for vectorized distance calculations.

    Generates two arrays of indices that, when used together, represent
    all pairwise combinations of indices from ranges [0, n1) and [0, n2).

    Args:
        n1: Size of the first dimension.
        n2: Size of the second dimension.

    Returns:
        Tuple of two arrays, each of shape (n1*n2,), containing the
        row and column indices respectively.
    """
    a, b = np.empty((n1, n2), dtype=np.int32), np.empty((n1, n2), dtype=np.int32)
    for i in range(n1):
        for j in range(n2):
            a[i, j], b[i, j] = i, j
    return a.ravel(), b.ravel()


@njit
def _dist_mat(xyz1: np.ndarray, xyz2: np.ndarray) -> np.ndarray:
    """Compute flattened distance matrix between two coordinate sets.

    Internal function that computes pairwise Euclidean distances between
    all points in xyz1 and xyz2.

    Args:
        xyz1: First coordinate array of shape (n1, ndim).
        xyz2: Second coordinate array of shape (n2, ndim).

    Returns:
        Flattened array of distances with shape (n1 * n2,).
    """
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    ndim = xyz1.shape[1]
    dist_mat = np.zeros(n1 * n2)
    i, j = unravel_index(n1, n2)
    for k in range(n1 * n2):
        dr = xyz1[i[k]] - xyz2[j[k]]
        for ri in range(ndim):
            dist_mat[k] += np.square(dr[ri])
    return np.sqrt(dist_mat)


@njit
def dist_mat(xyz1: np.ndarray, xyz2: np.ndarray) -> np.ndarray:
    """Compute distance matrix between two coordinate sets.

    Args:
        xyz1: First coordinate array of shape (n1, ndim).
        xyz2: Second coordinate array of shape (n2, ndim).

    Returns:
        Distance matrix of shape (n1, n2) where element [i, j]
        is the Euclidean distance between xyz1[i] and xyz2[j].
    """
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    return _dist_mat(xyz1, xyz2).reshape(n1, n2)


@njit
def electrostatic(distance: float, charge_i: float, charge_j: float) -> float:
    """Calculate electrostatic energy between two particles.

    Uses the reaction field method with a 10 Å cutoff and a solvent
    dielectric of 78.5 (water).

    Args:
        distance: Distance between particles i and j in nm.
        charge_i: Charge of particle i in elementary charge units.
        charge_j: Charge of particle j in elementary charge units.

    Returns:
        Electrostatic energy in kJ/mol.

    Note:
        Conversion factors used:
        - Avogadro = 6.022e23 molecules/mol
        - e- to Coulomb = 1.602e-19 C/e-
        - nm to m = 1e-9 m/nm
        - 1/(4*pi*epsilon_0) = 8.988e9 J*m/C^2
    """
    solvent_dielectric = 78.5

    if distance > 1.0:
        energy = 0.0
    else:
        r = distance * 1e-9
        r_cutoff = 1.0 * 1e-9
        k_rf = (
            1 / (r_cutoff**3) * (solvent_dielectric - 1) / (2 * solvent_dielectric + 1)
        )
        c_rf = 1 / r_cutoff * (3 * solvent_dielectric) / (2 * solvent_dielectric + 1)

        outer_term = 8.988e9 * (charge_i * 1.602e-19) * (charge_j * 1.602e-19)
        energy = outer_term * (1 / r + k_rf * r**2 - c_rf) * 6.022e23
    return energy / 1000  # J -> kJ


@njit
def electrostatic_sum(
    distances: np.ndarray, charge_is: np.ndarray, charge_js: np.ndarray
) -> float:
    """Calculate sum of all electrostatic interactions between two groups.

    Args:
        distances: Distance matrix between particles with shape
            (len(charge_is), len(charge_js)).
        charge_is: Array of charges for group i.
        charge_js: Array of charges for group j.

    Returns:
        Total electrostatic interaction energy in kJ/mol.
    """
    n = distances.shape[0]
    m = distances.shape[1]

    energy = 0.0
    for i in range(n):
        for j in range(m):
            energy += electrostatic(distances[i, j], charge_is[i], charge_js[j])
    return energy


@njit
def lennard_jones(
    distance: float, sigma_i: float, sigma_j: float, epsilon_i: float, epsilon_j: float
) -> float:
    """Calculate Lennard-Jones energy between two particles.

    Uses standard Lorentz-Berthelot combining rules with a 12 Å cutoff.

    Args:
        distance: Distance between particles i and j in nm.
        sigma_i: Sigma parameter for particle i in nm.
        sigma_j: Sigma parameter for particle j in nm.
        epsilon_i: Epsilon parameter for particle i in kJ/mol.
        epsilon_j: Epsilon parameter for particle j in kJ/mol.

    Returns:
        Lennard-Jones interaction energy in kJ/mol.
    """
    if distance > 1.2:
        energy = 0.0
    else:
        # Use Lorentz-Berthelot combining rules
        sigma_ij = 0.5 * (sigma_i + sigma_j)
        epsilon_ij = np.sqrt(epsilon_i * epsilon_j)

        # Calculate energy
        sigma_r = sigma_ij / distance
        sigma_r_6 = sigma_r**6
        sigma_r_12 = sigma_r_6**2
        energy = 4.0 * epsilon_ij * (sigma_r_12 - sigma_r_6)
    return energy


@njit
def lennard_jones_sum(
    distances: np.ndarray,
    sigma_is: np.ndarray,
    sigma_js: np.ndarray,
    epsilon_is: np.ndarray,
    epsilon_js: np.ndarray,
) -> float:
    """Calculate sum of all LJ interactions between two groups.

    Args:
        distances: Distance matrix between particles with shape
            (len(sigma_is), len(sigma_js)).
        sigma_is: Array of sigma parameters for group i in nm.
        sigma_js: Array of sigma parameters for group j in nm.
        epsilon_is: Array of epsilon parameters for group i in kJ/mol.
        epsilon_js: Array of epsilon parameters for group j in kJ/mol.

    Returns:
        Total Lennard-Jones interaction energy in kJ/mol.
    """
    n = distances.shape[0]
    m = distances.shape[1]
    energy = 0.0
    for i in range(n):
        for j in range(m):
            energy += lennard_jones(
                distances[i, j], sigma_is[i], sigma_js[j], epsilon_is[i], epsilon_js[j]
            )
    return energy


@njit
def fingerprints(
    xyzs: np.ndarray,
    charges: np.ndarray,
    sigmas: np.ndarray,
    epsilons: np.ndarray,
    target_resmap: list,
    binder_inds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate per-residue interaction fingerprints.

    Computes electrostatic and Lennard-Jones energies between each
    target residue and all binder residues.

    Args:
        xyzs: Coordinate array for all atoms in nm.
        charges: Array of partial charges for all atoms.
        sigmas: Array of sigma parameters for all atoms in nm.
        epsilons: Array of epsilon parameters for all atoms in kJ/mol.
        target_resmap: List of arrays, each containing atom indices
            for a target residue.
        binder_inds: Array of atom indices for the binder.

    Returns:
        Tuple of (lj_fingerprint, es_fingerprint), each of shape
        (n_target_residues,) containing the per-residue interaction
        energies.
    """
    n_target_residues = len(target_resmap)
    es_fingerprint = np.zeros(n_target_residues)
    lj_fingerprint = np.zeros(n_target_residues)
    for i in range(n_target_residues):
        dists = dist_mat(xyzs[target_resmap[i]], xyzs[binder_inds])
        es_fingerprint[i] = electrostatic_sum(
            dists, charges[target_resmap[i]], charges[binder_inds]
        )
        lj_fingerprint[i] = lennard_jones_sum(
            dists,
            sigmas[target_resmap[i]],
            sigmas[binder_inds],
            epsilons[target_resmap[i]],
            epsilons[binder_inds],
        )
    return lj_fingerprint, es_fingerprint


class Fingerprinter:
    """Calculate interaction energy fingerprints between target and binder.

    Computes per-residue Lennard-Jones and electrostatic interaction
    energies between a target selection and a binder selection over
    a molecular dynamics trajectory.

    Attributes:
        topology: Path to the topology file.
        trajectory: Path to the trajectory or coordinate file.
        target_selection: MDAnalysis selection string for the target.
        binder_selection: MDAnalysis selection string for the binder.
        out: Output path for the fingerprint data.
        target_fingerprint: Target fingerprint array after run().
        binder_fingerprint: Binder fingerprint array after run().

    Args:
        topology: Path to topology file (prmtop or PDB).
        trajectory: Path to trajectory or coordinate file. If None,
            will look for inpcrd or rst7 with same stem as topology.
        target_selection: MDAnalysis selection string for target.
            Defaults to 'segid A'.
        binder_selection: MDAnalysis selection string for binder.
            If None, binder is defined as everything not in target.
        out_path: Output directory path. If None, uses topology parent.
        out_name: Output filename. If None, uses 'fingerprint.npz'.

    Example:
        >>> fp = Fingerprinter('complex.prmtop', 'traj.dcd')
        >>> fp.run()
        >>> fp.save()
    """

    def __init__(
        self,
        topology: PathLike,
        trajectory: OptPath = None,
        target_selection: str = 'segid A',
        binder_selection: str | None = None,
        out_path: OptPath = None,
        out_name: str | None = None,
    ):
        """Initialize the Fingerprinter.

        Args:
            topology: Path to topology file.
            trajectory: Path to trajectory or coordinate file.
            target_selection: MDAnalysis selection for target.
            binder_selection: MDAnalysis selection for binder.
            out_path: Output directory path.
            out_name: Output filename.
        """
        self.topology = Path(topology)
        self.trajectory = Path(trajectory) if trajectory is not None else trajectory
        self.target_selection = target_selection

        if binder_selection is not None:
            self.binder_selection = binder_selection
        else:
            self.binder_selection = f'not {target_selection}'

        path = self.topology.parent if out_path is None else Path(out_path)
        self.out = path / 'fingerprint.npz' if out_name is None else path / out_name

    def assign_nonbonded_params(self) -> None:
        """Extract nonbonded parameters from the topology.

        Builds an OpenMM system from the topology and extracts
        charge, sigma, and epsilon parameters for all particles.
        """
        system = AmberPrmtopFile(self.topology).createSystem()

        nonbonded = next(
            iter(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
        )

        self.epsilons = np.zeros(system.getNumParticles())
        self.sigmas = np.zeros(system.getNumParticles())
        self.charges = np.zeros(system.getNumParticles())

        for ind in range(system.getNumParticles()):
            charge, sigma, epsilon = nonbonded.getParticleParameters(ind)
            self.charges[ind] = charge / charge.unit  # elementary charge
            self.sigmas[ind] = sigma / sigma.unit  # nm
            self.epsilons[ind] = epsilon / epsilon.unit  # kJ/mol

    def load_pdb(self) -> None:
        """Load the structure into an MDAnalysis Universe.

        Can handle either PDB or AMBER prmtop files. For prmtop files,
        will look for inpcrd or rst7 coordinates if trajectory was not
        specified.
        """
        if self.topology.suffix == '.pdb':
            self.u = mda.Universe(self.topology)
        else:
            if self.trajectory is not None:
                coordinates = self.trajectory
            elif self.topology.with_suffix('.inpcrd').exists():
                coordinates = self.topology.with_suffix('.inpcrd')
            else:
                coordinates = self.topology.with_suffix('.rst7')

            self.u = mda.Universe(self.topology, coordinates)

    def assign_residue_mapping(self) -> None:
        """Create mappings from residue indices to atom indices.

        Creates residue-to-atom mappings for both target and binder
        selections.
        """
        target = self.u.select_atoms(self.target_selection)
        self.target_resmap = [residue.atoms.ix for residue in target.residues]
        self.target_inds = np.concatenate(self.target_resmap)

        binder = self.u.select_atoms(self.binder_selection)
        self.binder_resmap = [residue.atoms.ix for residue in binder.residues]
        self.binder_inds = np.concatenate(self.binder_resmap)

    def iterate_frames(self) -> None:
        """Run fingerprint calculations over all trajectory frames.

        Initializes fingerprint arrays and iterates through all
        frames, calculating fingerprints for each.
        """
        self.target_fingerprint = np.zeros(
            (len(self.u.trajectory), len(self.target_resmap), 2)
        )

        self.binder_fingerprint = np.zeros(
            (len(self.u.trajectory), len(self.binder_resmap), 2)
        )

        for i, _ts in enumerate(self.u.trajectory):
            self.calculate_fingerprints(i)

    def calculate_fingerprints(self, frame_index: int) -> None:
        """Calculate fingerprints for a single frame.

        Args:
            frame_index: Index of the current frame (may differ from
                frame number if trajectory is discontinuous).
        """
        assert self.u.atoms is not None
        positions = self.u.atoms.positions * 0.1  # convert to nm

        self.target_fingerprint[frame_index] = np.vstack(
            fingerprints(
                positions,
                self.charges,
                self.sigmas,
                self.epsilons,
                self.target_resmap,
                self.binder_inds,
            )
        ).T

        self.binder_fingerprint[frame_index] = np.vstack(
            fingerprints(
                positions,
                self.charges,
                self.sigmas,
                self.epsilons,
                self.binder_resmap,
                self.target_inds,
            )
        ).T

    def run(self) -> None:
        """Execute the complete fingerprinting workflow.

        Obtains parameters, loads the structure, assigns residue
        mappings, and iterates through the trajectory to compute
        fingerprints.
        """
        self.assign_nonbonded_params()
        self.load_pdb()
        self.assign_residue_mapping()
        self.iterate_frames()

    def save(self) -> None:
        """Save fingerprint data to an NPZ file.

        Saves both target and binder fingerprints to the configured
        output path.
        """
        np.savez(
            self.out, target=self.target_fingerprint, binder=self.binder_fingerprint
        )
