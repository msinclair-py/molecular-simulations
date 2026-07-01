"""Tests for the sasa module.

Every test exercises the real Shrake-Rupley implementation: a real MDAnalysis
Universe built from a committed fixture, real scipy ``KDTree`` spatial queries,
and real numpy math. Every assertion checks actual computed SASA values for
finiteness, non-negativity, correct array shape, and physical sanity against
real fixtures.
"""

import MDAnalysis as mda
import numpy as np
import pytest
from scipy.spatial import KDTree

from molecular_simulations.analysis import SASA, RelativeSASA


@pytest.fixture
def mda_universe(alanine_dipeptide_pdb):
    """Real Universe for the Ace-Ala-Nme dipeptide (23 atoms, 3 residues).

    Loaded from a PDB, so it carries ``elements`` (needed by SASA) but no
    ``bonds`` topology (so it is rejected by RelativeSASA).
    """
    return mda.Universe(str(alanine_dipeptide_pdb))


@pytest.fixture
def amber_protein(real_amber_system_files):
    """Real bonded AtomGroup (Ace-Ala-Nme, 22 atoms) from an AMBER prmtop.

    Carries both ``elements`` and ``bonds``, so it is valid for RelativeSASA.
    """
    u = mda.Universe(
        str(real_amber_system_files['prmtop']),
        str(real_amber_system_files['inpcrd']),
    )
    return u.select_atoms('protein')


class TestSASAIntegration:
    """End-to-end checks on real spatial machinery and full runs."""

    def test_real_universe_loading(self, mda_universe):
        """A real Universe loads with atoms present."""
        assert mda_universe is not None
        assert mda_universe.atoms.n_atoms == 23
        assert mda_universe.atoms.n_residues == 3

    def test_real_atomgroup_has_elements(self, mda_universe):
        """The atomgroup exposes elements used for radii assignment."""
        ag = mda_universe.select_atoms('all')
        assert hasattr(ag, 'elements')
        assert len(ag.elements) == ag.n_atoms

    def test_real_kdtree_spatial_query(self):
        """Real scipy KDTree neighbour queries are deterministic and correct."""
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 10.0, 10.0],  # Far away point
            ]
        )

        kdt = KDTree(positions)

        neighbors = kdt.query_ball_point([0.0, 0.0, 0.0], r=2.0)
        assert sorted(neighbors) == [0, 1, 2]
        assert 3 not in neighbors  # far away point excluded

    def test_fibonacci_sphere_generation(self, mda_universe):
        """Fibonacci sphere points lie on the unit sphere."""
        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, n_points=100)
        sphere = sasa.get_sphere()

        assert sphere.shape == (100, 3)
        norms = np.linalg.norm(sphere, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-5)

    def test_sasa_full_run_is_physical(self, mda_universe):
        """A complete SASA.run() yields one finite, non-negative value per residue."""
        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, n_points=64)
        sasa.run()

        result = sasa.results.sasa
        assert result.shape == (ag.n_residues,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
        # An exposed tripeptide must have appreciable total surface area.
        assert np.sum(result) > 0

    def test_relative_sasa_runs_on_bonded_system(self, amber_protein):
        """RelativeSASA computes per-residue relative areas on a real system.

        Regression test for the broadcast bug (issue #9): measure_sasa must use
        the radii of each per-residue sub-selection, not the full AtomGroup.
        """
        rsasa = RelativeSASA(amber_protein, n_points=64)
        rsasa.run()

        area = rsasa.results.relative_area
        assert area.shape == (amber_protein.n_residues,)
        assert np.all(np.isfinite(area))
        # Relative accessibility is a non-negative fraction.
        assert np.all(area >= 0)


class TestSASA:
    """Unit tests for the SASA class against real atomgroups."""

    def test_init(self, mda_universe):
        """Initialization records parameters and derives radii and sphere."""
        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, probe_radius=1.4, n_points=256)

        assert sasa.probe_radius == 1.4
        assert sasa.n_points == 256
        assert sasa.ag is ag
        assert sasa.radii.shape == (ag.n_atoms,)
        assert sasa.sphere.shape == (256, 3)

    def test_init_with_updating_atomgroup(self, mda_universe):
        """An UpdatingAtomGroup is rejected with TypeError."""
        updating = mda_universe.select_atoms('all', updating=True)
        with pytest.raises(TypeError):
            SASA(updating)

    def test_init_without_elements(self):
        """A Universe lacking elements is rejected with ValueError."""
        u = mda.Universe.empty(3, trajectory=True)
        assert not hasattr(u.atoms, 'elements')
        with pytest.raises(ValueError):
            SASA(u.atoms)

    def test_get_sphere(self, mda_universe):
        """get_sphere builds the requested number of unit-sphere points."""
        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, n_points=100)
        sphere = sasa.get_sphere()

        assert sphere.shape == (100, 3)
        norms = np.linalg.norm(sphere, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-5)

    def test_radii_are_physical(self, mda_universe):
        """Radii are VDW radii plus the probe radius: positive and > probe."""
        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, probe_radius=1.4)

        assert len(sasa.radii) == ag.n_atoms
        assert np.all(sasa.radii > 1.4)
        assert sasa.max_radii == pytest.approx(2 * np.max(sasa.radii))

    def test_measure_sasa(self, mda_universe):
        """measure_sasa returns one finite, non-negative area per atom."""
        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, n_points=100)
        sasa._prepare()

        result = sasa.measure_sasa(ag)

        assert result.shape == (ag.n_atoms,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
        assert np.sum(result) > 0

    def test_prepare(self, mda_universe):
        """_prepare zeroes the per-residue results array and seeds the point set."""
        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, n_points=64)
        sasa._prepare()

        assert sasa.results.sasa.shape == (ag.n_residues,)
        assert np.all(sasa.results.sasa == 0)
        assert sasa.points_available == set(range(64))

    def test_single_frame_sums_per_residue(self, mda_universe):
        """_single_frame accumulates per-atom area into per-residue totals."""
        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, n_points=64)
        sasa._prepare()

        area = sasa.measure_sasa(ag)
        sasa._single_frame()

        per_residue = sasa.results.sasa
        assert per_residue.shape == (ag.n_residues,)
        assert np.all(np.isfinite(per_residue))
        assert np.all(per_residue >= 0)
        # Summing per-residue totals must recover the total per-atom area.
        assert np.sum(per_residue) == pytest.approx(np.sum(area))

    def test_conclude_averages_over_frames(self, mda_universe):
        """_conclude divides accumulated SASA by the number of frames."""
        ag = mda_universe.select_atoms('all')
        sasa = SASA(ag, n_points=64)
        sasa.results.sasa = np.array([100.0, 200.0, 300.0])
        sasa.n_frames = 10

        sasa._conclude()

        assert np.allclose(sasa.results.sasa, [10.0, 20.0, 30.0])


class TestRelativeSASA:
    """Unit tests for the RelativeSASA class against real atomgroups."""

    def test_init_without_bonds(self, mda_universe):
        """An AtomGroup without bond topology is rejected with ValueError."""
        ag = mda_universe.select_atoms('all')
        assert not hasattr(ag, 'bonds')
        with pytest.raises(ValueError):
            RelativeSASA(ag)

    def test_prepare(self, amber_protein):
        """_prepare zeroes both the absolute and relative results arrays."""
        rsasa = RelativeSASA(amber_protein, n_points=64)
        rsasa._prepare()

        assert rsasa.results.sasa.shape == (amber_protein.n_residues,)
        assert rsasa.results.relative_area.shape == (amber_protein.n_residues,)
        assert np.all(rsasa.results.sasa == 0)
        assert np.all(rsasa.results.relative_area == 0)

    def test_single_frame_relative(self, amber_protein):
        """_single_frame fills relative_area with finite, non-negative fractions."""
        rsasa = RelativeSASA(amber_protein, n_points=64)
        rsasa._prepare()

        rsasa._single_frame()

        rel = rsasa.results.relative_area
        assert rel.shape == (amber_protein.n_residues,)
        assert np.all(np.isfinite(rel))
        assert np.all(rel >= 0)
        # Absolute SASA was also accumulated and is positive overall.
        assert np.sum(rsasa.results.sasa) > 0

    def test_conclude_relative(self, amber_protein):
        """_conclude averages both absolute and relative arrays over frames."""
        rsasa = RelativeSASA(amber_protein, n_points=64)
        rsasa.results.sasa = np.array([100.0, 200.0, 300.0])
        rsasa.results.relative_area = np.array([0.5, 0.6, 0.7])
        rsasa.n_frames = 10

        rsasa._conclude()

        assert np.allclose(rsasa.results.sasa, [10.0, 20.0, 30.0])
        assert np.allclose(rsasa.results.relative_area, [0.05, 0.06, 0.07])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
