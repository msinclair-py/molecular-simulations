"""
Unit tests for sasa.py module

This module contains both unit tests (with mocks) and integration tests that use
real MDAnalysis Universe objects when available. Integration tests use actual
spatial calculations to verify SASA behavior.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ============================================================================
# Fixtures for conditional real MDAnalysis usage
# ============================================================================


def _check_mdanalysis_available():
    """Check if MDAnalysis is available."""
    try:
        import MDAnalysis as mda  # noqa: F401

        return True
    except ImportError:
        return False


# Custom marker for tests requiring MDAnalysis
requires_mdanalysis = pytest.mark.skipif(
    not _check_mdanalysis_available(), reason='MDAnalysis not available'
)


@pytest.fixture
def mda_universe():
    """Return a real MDAnalysis Universe from a test PDB file, or skip."""
    try:
        import MDAnalysis as mda

        # Use the test data PDB file
        test_pdb = Path(__file__).parent / 'data' / 'pdb' / 'alanine_dipeptide.pdb'
        if test_pdb.exists():
            return mda.Universe(str(test_pdb))
        else:
            pytest.skip(f'Test PDB file not found: {test_pdb}')
    except ImportError:
        pytest.skip('MDAnalysis not available')


@pytest.fixture
def simple_atomgroup(mda_universe):
    """Return an AtomGroup from the test universe."""
    return mda_universe.select_atoms('all')


# ============================================================================
# Integration tests using real MDAnalysis (when available)
# ============================================================================


class TestSASAIntegration:
    """Integration tests using real MDAnalysis objects.

    These tests verify actual SASA calculations rather than mocked interactions.
    """

    @requires_mdanalysis
    def test_real_universe_loading(self, mda_universe):
        """Test that we can load a real MDAnalysis Universe."""
        assert mda_universe is not None
        assert mda_universe.atoms.n_atoms > 0

    @requires_mdanalysis
    def test_real_atomgroup_has_elements(self, mda_universe):
        """Test that the atomgroup has elements assigned."""
        ag = mda_universe.select_atoms('all')
        # Elements should be available for SASA calculation
        assert hasattr(ag, 'elements')

    @requires_mdanalysis
    def test_real_kdtree_spatial_query(self):
        """Test real KDTree spatial queries work correctly."""
        from scipy.spatial import KDTree

        # Create test coordinates
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 10.0, 10.0],  # Far away point
            ]
        )

        kdt = KDTree(positions)

        # Query ball point - should find 3 atoms within radius 2.0 of origin
        neighbors = kdt.query_ball_point([0.0, 0.0, 0.0], r=2.0)
        assert len(neighbors) == 3  # indices 0, 1, 2
        assert 3 not in neighbors  # far away point should not be included

    @requires_mdanalysis
    def test_fibonacci_sphere_generation(self, mda_universe):
        """Test that Fibonacci sphere points are correctly distributed on unit sphere."""
        from molecular_simulations.analysis import SASA

        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, n_points=100)
        sphere = sasa.get_sphere()

        # Check shape
        assert sphere.shape == (100, 3)

        # Check all points lie on unit sphere (norm should be ~1)
        norms = np.linalg.norm(sphere, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-5)

    @requires_mdanalysis
    def test_sasa_radii_assignment(self, mda_universe):
        """Test that atomic radii are correctly assigned from elements."""
        from molecular_simulations.analysis import SASA

        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, probe_radius=1.4)

        # Radii should be VDW radii + probe radius
        assert len(sasa.radii) == ag.n_atoms

        # All radii should be positive
        assert all(r > 0 for r in sasa.radii)

        # Radii should be larger than probe radius (VDW > 0)
        assert all(r > 1.4 for r in sasa.radii)

    @requires_mdanalysis
    def test_sasa_prepare_initializes_results(self, mda_universe):
        """Test that _prepare correctly initializes results arrays."""
        from molecular_simulations.analysis import SASA

        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag)
        sasa._prepare()

        assert hasattr(sasa.results, 'sasa')
        assert sasa.results.sasa.shape == (ag.n_residues,)
        assert all(s == 0 for s in sasa.results.sasa)

    @requires_mdanalysis
    def test_sasa_values_are_positive(self, mda_universe):
        """Test that SASA calculations produce positive values."""
        from molecular_simulations.analysis import SASA

        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, n_points=64)  # Use fewer points for speed

        # Measure SASA for the atomgroup
        sasa._prepare()
        area = sasa.measure_sasa(ag)

        # All SASA values should be non-negative
        assert all(a >= 0 for a in area)

        # Total SASA should be positive for any molecule
        assert np.sum(area) > 0

    @requires_mdanalysis
    def test_relative_sasa_requires_bonds(self, mda_universe):
        """Test that RelativeSASA raises error without bonds."""
        from molecular_simulations.analysis import RelativeSASA

        ag = mda_universe.select_atoms('all')

        # This test verifies the error handling for missing bonds
        if not hasattr(ag, 'bonds') or ag.bonds is None:
            with pytest.raises(ValueError):
                RelativeSASA(ag)


# ============================================================================
# Original unit tests with mocks
# ============================================================================

# Conditional import with mock fallback
try:
    import MDAnalysis as mda
    from MDAnalysis.core import groups

    from molecular_simulations.analysis import SASA, RelativeSASA

    _MDA_AVAILABLE = True
except ImportError:
    _MDA_AVAILABLE = False
    # Create dummy classes for mocking
    mda = MagicMock()
    groups = MagicMock()
    SASA = MagicMock
    RelativeSASA = MagicMock


class TestSASA:
    """Test suite for SASA class"""

    @patch('molecular_simulations.analysis.sasa.KDTree')
    def test_init(self, mock_kdtree):
        """Test SASA initialization"""
        # Create mock universe and atomgroup
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory

        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H', 'O', 'N'])

        # Create SASA instance
        sasa = SASA(mock_ag, probe_radius=1.4, n_points=256)

        assert sasa.probe_radius == 1.4
        assert sasa.n_points == 256
        assert sasa.ag == mock_ag
        assert hasattr(sasa, 'radii')
        assert hasattr(sasa, 'sphere')

    def test_init_with_updating_atomgroup(self):
        """Test that UpdatingAtomGroup raises TypeError"""
        mock_ag = MagicMock(spec=groups.UpdatingAtomGroup)

        with pytest.raises(TypeError):
            SASA(mock_ag)

    def test_init_without_elements(self):
        """Test that missing elements raises ValueError"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory

        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        del mock_ag.elements  # Remove elements attribute

        with pytest.raises(ValueError):
            SASA(mock_ag)

    def test_get_sphere(self):
        """Test fibonacci sphere generation"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory

        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])

        sasa = SASA(mock_ag, n_points=100)
        sphere = sasa.get_sphere()

        assert sphere.shape == (100, 3)
        # Check that points are on unit sphere
        norms = np.linalg.norm(sphere, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-5)

    @patch('molecular_simulations.analysis.sasa.KDTree')
    def test_measure_sasa(self, mock_kdtree):
        """Test SASA measurement for atomgroup"""
        # Setup mock objects
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory

        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H', 'O'])
        mock_ag.n_atoms = 3
        mock_ag.positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        # Mock KDTree behavior
        mock_kdtree_instance = MagicMock()
        mock_kdtree_instance.query_ball_point.return_value = [0, 1]
        mock_kdtree.return_value = mock_kdtree_instance

        sasa = SASA(mock_ag, n_points=100)
        sasa.radii = np.array([1.7, 1.2, 1.52])
        sasa.points_available = set(range(100))

        result = sasa.measure_sasa(mock_ag)

        assert result.shape == (3,)
        assert all(r >= 0 for r in result)

    def test_prepare(self):
        """Test _prepare method"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory

        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])
        mock_ag.n_residues = 5

        sasa = SASA(mock_ag)
        sasa._prepare()

        assert hasattr(sasa.results, 'sasa')
        assert sasa.results.sasa.shape == (5,)
        assert all(s == 0 for s in sasa.results.sasa)

    @patch.object(SASA, 'measure_sasa')
    def test_single_frame(self, mock_measure):
        """Test _single_frame method"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory

        # Setup mock atoms with residue IDs
        mock_atoms = []
        for i in range(3):
            mock_atom = MagicMock()
            mock_atom.resid = i + 1
            mock_atoms.append(mock_atom)

        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H', 'O'])
        mock_ag.n_residues = 3
        mock_ag.atoms = mock_atoms

        # Mock measure_sasa to return area values
        mock_measure.return_value = np.array([10.0, 20.0, 30.0])

        sasa = SASA(mock_ag)
        sasa.results = MagicMock()
        sasa.results.sasa = np.zeros(3)

        sasa._single_frame()

        mock_measure.assert_called_once_with(mock_ag)
        assert sasa.results.sasa[0] == 10.0
        assert sasa.results.sasa[1] == 20.0
        assert sasa.results.sasa[2] == 30.0

    def test_conclude(self):
        """Test _conclude method"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory

        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])

        sasa = SASA(mock_ag)
        sasa.results = MagicMock()
        sasa.results.sasa = np.array([100.0, 200.0, 300.0])
        sasa.n_frames = 10

        sasa._conclude()

        assert sasa.results.sasa[0] == 10.0
        assert sasa.results.sasa[1] == 20.0
        assert sasa.results.sasa[2] == 30.0


class TestRelativeSASA:
    """Test suite for RelativeSASA class"""

    def test_init_without_bonds(self):
        """Test that missing bonds raises ValueError"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory

        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])
        del mock_ag.bonds  # Remove bonds attribute

        with pytest.raises(ValueError):
            RelativeSASA(mock_ag)

    def test_prepare(self):
        """Test _prepare method for RelativeSASA"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory

        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])
        mock_ag.bonds = MagicMock()
        mock_ag.n_residues = 5

        rel_sasa = RelativeSASA(mock_ag)
        rel_sasa._prepare()

        assert hasattr(rel_sasa.results, 'sasa')
        assert hasattr(rel_sasa.results, 'relative_area')
        assert rel_sasa.results.sasa.shape == (5,)
        assert rel_sasa.results.relative_area.shape == (5,)

    @patch.object(RelativeSASA, 'measure_sasa')
    def test_single_frame_relative(self, mock_measure):
        """Test _single_frame method for RelativeSASA"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory

        # Setup mock residues
        mock_residues = MagicMock()
        mock_residues.resindices = np.array([0, 1, 2])
        mock_residues.resids = np.array([1, 2, 3])

        # Setup mock atoms
        mock_atoms = []
        for i in range(3):
            mock_atom = MagicMock()
            mock_atom.resid = i + 1
            mock_atoms.append(mock_atom)

        # Setup mock atomgroup
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H', 'O'])
        mock_ag.bonds = MagicMock()
        mock_ag.n_residues = 3
        mock_ag.atoms = mock_atoms
        mock_ag.residues = mock_residues

        # Mock select_atoms to return tripeptide
        mock_tripeptide = MagicMock()
        mock_tripeptide.__len__ = MagicMock(return_value=3)
        mock_tripeptide.resindices = np.array([0, 1, 2])
        mock_ag.select_atoms.return_value = mock_tripeptide

        # Mock measure_sasa to return different values for different calls
        mock_measure.side_effect = [
            np.array([10.0, 20.0, 30.0]),  # Initial SASA
            np.array([5.0, 10.0, 15.0]),  # Tripeptide SASA
            np.array([5.0, 10.0, 15.0]),  # Tripeptide SASA
            np.array([5.0, 10.0, 15.0]),  # Tripeptide SASA
        ]

        rel_sasa = RelativeSASA(mock_ag)
        rel_sasa.results = MagicMock()
        rel_sasa.results.sasa = np.zeros(3)
        rel_sasa.results.relative_area = np.zeros(3)

        rel_sasa._single_frame()

        assert mock_measure.called

    def test_conclude_relative(self):
        """Test _conclude method for RelativeSASA"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory

        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])
        mock_ag.bonds = MagicMock()

        rel_sasa = RelativeSASA(mock_ag)
        rel_sasa.results = MagicMock()
        rel_sasa.results.sasa = np.array([100.0, 200.0, 300.0])
        rel_sasa.results.relative_area = np.array([0.5, 0.6, 0.7])
        rel_sasa.n_frames = 10

        rel_sasa._conclude()

        # Use approximate comparison for floating point
        assert np.isclose(rel_sasa.results.sasa[0], 10.0)
        assert np.isclose(rel_sasa.results.sasa[1], 20.0)
        assert np.isclose(rel_sasa.results.sasa[2], 30.0)
        assert np.isclose(rel_sasa.results.relative_area[0], 0.05)
        assert np.isclose(rel_sasa.results.relative_area[1], 0.06)
        assert np.isclose(rel_sasa.results.relative_area[2], 0.07)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
