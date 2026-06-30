from unittest.mock import MagicMock

import MDAnalysis as mda
import numpy as np

from molecular_simulations.analysis.sasa import SASA


def test_get_sphere_points_on_unit_sphere():
    """Test that get_sphere generates points on a unit sphere"""
    # Create a minimal mock setup for SASA
    mock_universe = MagicMock()
    mock_trajectory = MagicMock()
    mock_universe.trajectory = mock_trajectory

    mock_ag = MagicMock(spec=mda.AtomGroup)
    mock_ag.universe = mock_universe
    mock_ag.elements = np.array(['C', 'H'])  # Need at least some elements

    # Create SASA instance
    sasa = SASA(mock_ag, n_points=256)

    # Get the sphere
    s = sasa.get_sphere()

    # Test that we have the right number of points
    assert s.shape == (256, 3)

    # Test that all points are on the unit sphere (radius = 1.0)
    radii = np.linalg.norm(s, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-6)
