import MDAnalysis as mda
import numpy as np

from molecular_simulations.analysis.sasa import SASA


def test_get_sphere_points_on_unit_sphere(real_amber_system_files):
    """Test that get_sphere generates points on a unit sphere.

    Uses a real Universe (the tleap-built Ace-Ala-Nme system) so SASA is
    constructed through its real __init__ -- including the real element-based
    radii assignment -- before exercising the Fibonacci-sphere geometry.
    """
    u = mda.Universe(
        str(real_amber_system_files['prmtop']),
        str(real_amber_system_files['inpcrd']),
    )
    ag = u.select_atoms('all')

    # Create SASA instance from a real AtomGroup
    sasa = SASA(ag, n_points=256)

    # Get the sphere
    s = sasa.get_sphere()

    # Test that we have the right number of points
    assert s.shape == (256, 3)

    # Test that all points are on the unit sphere (radius = 1.0)
    radii = np.linalg.norm(s, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-6)
