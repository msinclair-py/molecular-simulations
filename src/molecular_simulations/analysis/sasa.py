"""Solvent-Accessible Surface Area (SASA) analysis module.

This module computes SASA using the Shrake-Rupley algorithm, implemented
as MDAnalysis AnalysisBase classes for ease of use and built-in parallelism.
Adapted from BioPython and MDTraj implementations.
"""

import warnings

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core import groups
from MDAnalysis.guesser.tables import vdwradii
from scipy.spatial import KDTree

warnings.filterwarnings('ignore')


class SASA(AnalysisBase):
    """Compute solvent-accessible surface area using Shrake-Rupley algorithm.

    Implements SASA calculation as an MDAnalysis AnalysisBase instance,
    supporting ease of deployment and built-in parallelism. Code is adapted
    from BioPython and MDTraj implementations as well as an unmerged PR
    from MDAnalysis.

    Attributes:
        ag: The AtomGroup being analyzed.
        probe_radius: Probe radius for SASA calculation.
        n_points: Number of points on each atomic sphere.
        radii: Array of atomic radii (VDW + probe).
        max_radii: Maximum pairwise radius sum.
        results.sasa: Per-residue SASA values after run().

    Args:
        ag: AtomGroup for which to compute SASA.
        probe_radius: Probe radius in Angstroms. Defaults to 1.4 Å
            (standard water probe).
        n_points: Number of points on each sphere. Higher values are
            more accurate but slower. Defaults to 256.
        **kwargs: Additional arguments passed to AnalysisBase.

    Raises:
        TypeError: If ag is an UpdatingAtomGroup.
        ValueError: If the Universe has no 'elements' property.

    Example:
        >>> u = mda.Universe('system.prmtop', 'traj.dcd')
        >>> sasa = SASA(u.select_atoms('protein'))
        >>> sasa.run()
        >>> print(sasa.results.sasa)
    """

    def __init__(
        self,
        ag: mda.AtomGroup,
        probe_radius: float = 1.4,
        n_points: int = 256,
        **kwargs,
    ):
        """Initialize the SASA analysis.

        Args:
            ag: AtomGroup to analyze.
            probe_radius: Probe radius in Angstroms.
            n_points: Number of sphere points.
            **kwargs: Arguments for AnalysisBase.
        """
        if isinstance(ag, groups.UpdatingAtomGroup):
            raise TypeError('UpdatingAtomGroups are not valid for SASA!')

        super().__init__(ag.universe.trajectory, **kwargs)

        if not hasattr(ag, 'elements'):
            raise ValueError(
                'Cannot assign atomic radii: Universe has no `elements` property!'
            )

        self.ag = ag
        self.probe_radius = probe_radius
        self.n_points = n_points

        self.radii_dict = dict()
        self.radii_dict.update(vdwradii)

        self.radii = np.vectorize(self.radii_dict.get)(self.ag.elements)
        self.radii += self.probe_radius
        self.max_radii = 2 * np.max(self.radii)

        self.sphere = self.get_sphere()

    def get_sphere(self) -> np.ndarray:
        """Generate a Fibonacci unit sphere.

        Creates evenly distributed points on a unit sphere using the
        Fibonacci spiral method.

        Returns:
            Array of shape (n_points, 3) with unit sphere coordinates.
        """
        dl = np.pi * (3 - np.sqrt(5))
        dz = 2.0 / self.n_points
        longitude = 0
        z = 1 - dz / 2

        xyz = np.zeros((self.n_points, 3), dtype=np.float32)
        for i in range(self.n_points):
            r = np.sqrt(1 - z**2)
            xyz[i, :] = [np.cos(longitude) * r, np.sin(longitude) * r, z]

            z -= dz
            longitude += dl

        return xyz

    def measure_sasa(self, ag: mda.AtomGroup) -> np.ndarray:
        """Measure SASA of an AtomGroup in the current frame.

        Args:
            ag: MDAnalysis AtomGroup to measure.

        Returns:
            Array of per-atom SASA values.
        """
        # Radii for THIS AtomGroup, which may be a per-residue sub-selection
        # (e.g. RelativeSASA's tripeptide). self.radii is sized to the full
        # analysis AtomGroup, so indexing it against `ag` mismatches.
        radii = np.vectorize(self.radii_dict.get)(ag.elements) + self.probe_radius
        max_radii = 2 * np.max(radii)

        kdt = KDTree(ag.positions, 10)

        points = np.zeros(ag.n_atoms)
        for i in range(ag.n_atoms):
            sphere = self.sphere.copy() * radii[i]
            sphere += ag.positions[i]
            available = self.points_available.copy()
            kdt_sphere = KDTree(sphere, 10)

            for j in kdt.query_ball_point(ag.positions[i], max_radii, workers=-1):
                if j == i:
                    continue
                if radii[j] < (radii[i] + radii[j]):
                    available -= {
                        n
                        for n in kdt_sphere.query_ball_point(ag.positions[j], radii[j])
                    }

            points[i] = len(available)

        return 4 * np.pi * radii**2 * points / self.n_points

    def _prepare(self):
        """Prepare for analysis by initializing results array.

        Called automatically before trajectory iteration.
        """
        self.results.sasa = np.zeros(self.ag.n_residues)
        self.points_available = set(range(self.n_points))

    def _single_frame(self):
        """Process a single trajectory frame.

        Measures SASA for each atom and sums per-residue.
        """
        area = self.measure_sasa(self.ag)
        result = np.zeros(self.ag.n_residues)
        for i, atom in enumerate(self.ag.atoms):
            result[atom.resid - 1] += area[i]

        self.results.sasa += result

    def _conclude(self):
        """Post-process results by averaging over frames.

        Called automatically after trajectory iteration.
        """
        if self.n_frames != 0:
            self.results.sasa /= self.n_frames


class RelativeSASA(SASA):
    """Compute relative SASA for an AtomGroup.

    Relative SASA is defined as measured SASA divided by maximum
    accessible surface area. For proteins, this is computed in a
    tripeptide context to avoid overestimating SASA of the amide
    linkage and its neighbors.

    Attributes:
        results.sasa: Absolute SASA values.
        results.relative_area: Relative SASA values (0-1 scale).

    Args:
        ag: AtomGroup for SASA calculation.
        probe_radius: Probe radius in Angstroms. Defaults to 1.4 Å.
        n_points: Number of sphere points. Defaults to 256.
        **kwargs: Additional arguments for AnalysisBase.

    Raises:
        ValueError: If the Universe has no 'bonds' property.

    Example:
        >>> u = mda.Universe('system.prmtop', 'traj.dcd')
        >>> rsasa = RelativeSASA(u.select_atoms('protein'))
        >>> rsasa.run()
        >>> print(rsasa.results.relative_area)
    """

    def __init__(
        self,
        ag: mda.AtomGroup,
        probe_radius: float = 1.4,
        n_points: int = 256,
        **kwargs,
    ):
        """Initialize the RelativeSASA analysis.

        Args:
            ag: AtomGroup to analyze.
            probe_radius: Probe radius in Angstroms.
            n_points: Number of sphere points.
            **kwargs: Arguments for AnalysisBase.
        """
        if not hasattr(ag, 'bonds'):
            raise ValueError('Universe has no `bonds` property!')
        super().__init__(ag, probe_radius, n_points, **kwargs)

    def _prepare(self):
        """Prepare for analysis by initializing results arrays.

        Called automatically before trajectory iteration.
        """
        self.results.sasa = np.zeros(self.ag.n_residues)
        self.results.relative_area = np.zeros(self.ag.n_residues)
        self.points_available = set(range(self.n_points))

    def _single_frame(self):
        """Process a single trajectory frame.

        Measures SASA and computes relative SASA using tripeptide
        reference for each residue.
        """
        area = self.measure_sasa(self.ag)
        result = np.zeros(self.ag.n_residues)
        for i, atom in enumerate(self.ag.atoms):
            result[atom.resid - 1] += area[i]

        self.results.sasa += result

        for res_index in self.ag.residues.resindices:
            tri_peptide = self.ag.select_atoms(f'byres (bonded resindex {res_index})')

            if len(tri_peptide) == 0:
                continue

            tri_pep_area = self.measure_sasa(tri_peptide)
            exposed_area = sum(
                [
                    a
                    for a, _id in zip(tri_pep_area, tri_peptide.resindices, strict=True)
                    if _id == res_index
                ]
            )

            if exposed_area != 0.0:
                result[res_index] /= exposed_area

        self.results.relative_area += np.array(
            [result[_id] for _id in self.ag.residues.resindices]
        )

    def _conclude(self):
        """Post-process results by averaging over frames.

        Called automatically after trajectory iteration.
        """
        if self.n_frames != 0:
            self.results.sasa /= self.n_frames
            self.results.relative_area /= self.n_frames
