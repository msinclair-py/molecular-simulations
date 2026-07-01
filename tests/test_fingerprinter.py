"""Tests for the fingerprinter module.

The pure-math helpers (unravel_index, dist_mat, electrostatic, lennard_jones and
their vectorized sums) are tested directly. The Fingerprinter class is exercised
end-to-end against real fixtures: real OpenMM AmberPrmtopFile parameter
extraction and real MDAnalysis Universes built from committed test systems.
"""

import os

# Disable numba JIT compilation to avoid path resolution issues during testing.
os.environ['NUMBA_DISABLE_JIT'] = '1'

import MDAnalysis as mda
import numpy as np
import pytest

from molecular_simulations.analysis.fingerprinter import (
    Fingerprinter,
    _dist_mat,
    dist_mat,
    electrostatic,
    electrostatic_sum,
    fingerprints,
    lennard_jones,
    lennard_jones_sum,
    unravel_index,
)


def test_unravel_index_shapes():
    """Test unravel_index produces correct shapes and values"""
    i, j = unravel_index(5, 7)
    assert i.shape == j.shape
    assert i.size == 5 * 7
    assert i.max() == 4 and j.max() == 6


def test_unravel_index_values():
    """Test unravel_index produces correct index pairs"""
    i, j = unravel_index(2, 3)
    # Should produce: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    expected_i = np.array([0, 0, 0, 1, 1, 1])
    expected_j = np.array([0, 1, 2, 0, 1, 2])
    assert np.array_equal(i, expected_i)
    assert np.array_equal(j, expected_j)


def test_dist_mat_basic():
    """Test distance matrix calculation with simple coordinates"""
    xyz1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    xyz2 = np.array([[0.0, 1.0, 0.0]])
    D = dist_mat(xyz1, xyz2)
    assert D.shape == (2, 1)
    assert np.isclose(D[0, 0], 1.0)
    assert np.isclose(D[1, 0], np.sqrt(2.0))


def test_dist_mat_symmetric():
    """Test that distance matrix is symmetric when inputs are swapped"""
    xyz1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    xyz2 = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    D12 = dist_mat(xyz1, xyz2)
    D21 = dist_mat(xyz2, xyz1)

    assert D12.shape == (2, 2)
    assert D21.shape == (2, 2)
    assert np.allclose(D12, D21.T)


def test_electrostatic_symmetry_and_sign():
    """Test electrostatic interaction symmetry and sign"""
    d = 0.5  # distance in nm
    e1, e2 = 0.5, -0.5
    e3, e4 = -0.5, 0.5
    e_ab = electrostatic(d, e1, e2)
    e_ba = electrostatic(d, e2, e1)
    e_cd = electrostatic(d, e3, e4)

    # Symmetric in swapping particles
    assert np.isclose(e_ab, e_ba)
    # Opposite-sign charges should yield negative energy
    assert e_ab < 0
    # Same magnitude pair should match
    assert np.isclose(e_ab, e_cd)


def test_electrostatic_cutoff():
    """Test that electrostatic energy is zero beyond cutoff"""
    # Distance > 1.0 nm should give zero energy
    e_far = electrostatic(1.5, 1.0, 1.0)
    assert e_far == 0.0

    # Distance < 1.0 nm should give non-zero energy
    e_near = electrostatic(0.5, 1.0, 1.0)
    assert e_near != 0.0


def test_electrostatic_sum_vectorization_matches_scalar():
    """Test that vectorized electrostatic_sum matches scalar accumulation"""
    rng = np.random.default_rng(0)
    n, m = 4, 3
    D = rng.random((n, m)) + 0.2
    qi = rng.uniform(-1, 1, size=n)
    qj = rng.uniform(-1, 1, size=m)

    # Scalar accumulation
    scalar = 0.0
    for a in range(n):
        for b in range(m):
            scalar += electrostatic(D[a, b], qi[a], qj[b])

    # Vectorized helper
    vec = electrostatic_sum(D, qi, qj)
    assert np.isclose(vec, scalar)


def test_lj_basic_properties():
    """Test basic Lennard-Jones properties"""
    # Attractive well around sigma, repulsive at very small r
    e_far = lennard_jones(5.0, 1.0, 1.0, 0.2, 0.2)
    e_mid = lennard_jones(1.5, 1.0, 1.0, 0.2, 0.2)
    e_close = lennard_jones(0.5, 1.0, 1.0, 0.2, 0.2)

    # Far distance should be close to zero (cutoff at 1.2 nm)
    assert e_far == 0.0  # Beyond cutoff
    # Close distance should be strongly repulsive
    assert e_close > e_mid


def test_lj_cutoff():
    """Test Lennard-Jones cutoff behavior"""
    # Distance > 1.2 nm should give zero energy
    e_far = lennard_jones(1.5, 1.0, 1.0, 0.2, 0.2)
    assert e_far == 0.0

    # Distance < 1.2 nm should give non-zero energy
    # Use distance=0.8 which is not at the zero-crossing point
    e_near = lennard_jones(0.8, 1.0, 1.0, 0.2, 0.2)
    assert e_near != 0.0


def test_lj_sum_matches_manual_sum():
    """Test that vectorized lennard_jones_sum matches manual accumulation"""
    rng = np.random.default_rng(1)
    n, m = 3, 2
    D = rng.random((n, m)) + 0.3
    si = rng.random(n) + 0.5
    sj = rng.random(m) + 0.5
    ei = rng.random(n) * 0.3 + 0.05
    ej = rng.random(m) * 0.3 + 0.05

    manual = 0.0
    for a in range(n):
        for b in range(m):
            manual += lennard_jones(D[a, b], si[a], sj[b], ei[a], ej[b])

    summed = lennard_jones_sum(D, si, sj, ei, ej)
    assert np.isclose(summed, manual)


def test_lj_combination_rules():
    """Test Lennard-Jones combination rules"""
    # Using Lorentz-Berthelot combining rules:
    # sigma_ij = (sigma_i + sigma_j) / 2
    # epsilon_ij = sqrt(epsilon_i * epsilon_j)

    distance = 1.0
    sigma_i, sigma_j = 0.5, 1.0
    epsilon_i, epsilon_j = 0.1, 0.4

    energy = lennard_jones(distance, sigma_i, sigma_j, epsilon_i, epsilon_j)

    # Should use combined parameters
    sigma_ij = 0.5 * (sigma_i + sigma_j)  # = 0.75
    epsilon_ij = np.sqrt(epsilon_i * epsilon_j)  # = 0.2

    # Verify energy is computed with combined parameters
    sigma_r = sigma_ij / distance
    expected_energy = 4.0 * epsilon_ij * (sigma_r**12 - sigma_r**6)

    assert np.isclose(energy, expected_energy)


class TestDistMatInternal:
    """Test suite for _dist_mat internal function"""

    def test_dist_mat_internal_basic(self):
        """Test _dist_mat returns flattened distances"""
        xyz1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        xyz2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)

        D = _dist_mat(xyz1, xyz2)
        assert D.shape == (2,)  # flattened

    def test_dist_mat_internal_larger(self):
        """Test _dist_mat with larger arrays"""
        n1, n2 = 5, 3
        xyz1 = np.random.rand(n1, 3)
        xyz2 = np.random.rand(n2, 3)

        D = _dist_mat(xyz1, xyz2)
        assert D.shape == (n1 * n2,)


class TestFingerprints:
    """Test suite for fingerprints function"""

    def test_fingerprints_basic(self):
        """Test fingerprints computation"""
        # Create simple test data
        n_atoms = 10
        xyzs = np.random.rand(n_atoms, 3).astype(np.float64)
        charges = np.random.uniform(-1, 1, n_atoms).astype(np.float64)
        sigmas = np.ones(n_atoms, dtype=np.float64) * 0.3
        epsilons = np.ones(n_atoms, dtype=np.float64) * 0.1

        # Target residue mapping (2 residues, 3 atoms each)
        target_resmap = [
            np.array([0, 1, 2], dtype=np.int64),
            np.array([3, 4, 5], dtype=np.int64),
        ]
        # Binder atoms
        binder_inds = np.array([6, 7, 8, 9], dtype=np.int64)

        lj_fp, es_fp = fingerprints(
            xyzs, charges, sigmas, epsilons, target_resmap, binder_inds
        )

        assert len(lj_fp) == 2  # Two target residues
        assert len(es_fp) == 2

    def test_fingerprints_single_residue(self):
        """Test fingerprints with single target residue"""
        n_atoms = 6
        xyzs = np.random.rand(n_atoms, 3).astype(np.float64)
        charges = np.random.uniform(-1, 1, n_atoms).astype(np.float64)
        sigmas = np.ones(n_atoms, dtype=np.float64) * 0.3
        epsilons = np.ones(n_atoms, dtype=np.float64) * 0.1

        target_resmap = [np.array([0, 1, 2], dtype=np.int64)]
        binder_inds = np.array([3, 4, 5], dtype=np.int64)

        lj_fp, es_fp = fingerprints(
            xyzs, charges, sigmas, epsilons, target_resmap, binder_inds
        )

        assert len(lj_fp) == 1
        assert len(es_fp) == 1


class TestFingerprinterClass:
    """Test suite for Fingerprinter construction and configuration."""

    def test_fingerprinter_init(self, real_amber_system_files):
        """Default binder selection is the complement of the target."""
        top = real_amber_system_files['prmtop']
        fp = Fingerprinter(topology=str(top), target_selection='segid A')

        assert fp.topology == top
        assert fp.target_selection == 'segid A'
        assert fp.binder_selection == 'not segid A'

    def test_fingerprinter_init_with_binder_selection(self, real_amber_system_files):
        """An explicit binder selection is honoured."""
        top = real_amber_system_files['prmtop']
        fp = Fingerprinter(
            topology=str(top),
            target_selection='segid A',
            binder_selection='segid B',
        )

        assert fp.binder_selection == 'segid B'

    def test_fingerprinter_output_path(self, real_amber_system_files):
        """Output path and filename are composed from the provided arguments."""
        top = real_amber_system_files['prmtop']
        out_path = real_amber_system_files['path'] / 'output'
        out_path.mkdir()

        fp = Fingerprinter(
            topology=str(top), out_path=str(out_path), out_name='custom.npz'
        )

        assert fp.out == out_path / 'custom.npz'

    def test_assign_nonbonded_params(self, real_amber_system_files):
        """Real OpenMM parameter extraction yields finite arrays of length 22."""
        fp = Fingerprinter(topology=str(real_amber_system_files['prmtop']))
        fp.assign_nonbonded_params()

        for arr in (fp.charges, fp.sigmas, fp.epsilons):
            assert arr.shape == (22,)
            assert np.all(np.isfinite(arr))

        # Sigma and epsilon are non-negative physical Lennard-Jones parameters.
        assert np.all(fp.sigmas >= 0)
        assert np.all(fp.epsilons >= 0)
        # The system is overall neutral, so total charge is ~0.
        assert fp.charges.sum() == pytest.approx(0.0, abs=1e-4)

    def test_save(self, real_amber_system_files):
        """save() writes both fingerprint arrays to a loadable NPZ file."""
        fp = Fingerprinter(topology=str(real_amber_system_files['prmtop']))
        fp.target_fingerprint = np.random.rand(10, 5, 2)
        fp.binder_fingerprint = np.random.rand(10, 3, 2)

        fp.save()

        assert fp.out.exists()
        data = np.load(fp.out)
        assert 'target' in data
        assert 'binder' in data


class TestFingerprinterLoadPdb:
    """Test suite for Fingerprinter.load_pdb against real structures."""

    def test_load_pdb_with_pdb_file(self, real_amber_system_files):
        """A PDB topology loads into a real Universe with the right atom count."""
        fp = Fingerprinter(topology=str(real_amber_system_files['pdb']))
        fp.load_pdb()

        assert isinstance(fp.u, mda.Universe)
        assert fp.u.atoms.n_atoms == 22

    def test_load_pdb_with_prmtop_and_trajectory(self, real_amber_system_files):
        """A prmtop + DCD trajectory loads all frames into a real Universe."""
        fp = Fingerprinter(
            topology=str(real_amber_system_files['prmtop']),
            trajectory=str(real_amber_system_files['dcd']),
        )
        fp.load_pdb()

        assert fp.u.atoms.n_atoms == 22
        assert len(fp.u.trajectory) == 5

    def test_load_pdb_with_prmtop_finds_inpcrd(self, real_amber_system_files):
        """Without a trajectory, load_pdb discovers the sibling .inpcrd file."""
        # real_amber_system_files copies prmtop and inpcrd into the same dir.
        fp = Fingerprinter(topology=str(real_amber_system_files['prmtop']))
        fp.load_pdb()

        assert fp.u.atoms.n_atoms == 22
        assert fp.u.atoms.positions.shape == (22, 3)

    def test_load_pdb_prmtop_without_coordinates_raises(
        self, real_amber_system_files, tmp_path
    ):
        """A prmtop with no trajectory and no sibling .inpcrd raises clearly."""
        import shutil

        # Copy only the prmtop into an otherwise-empty dir (no .inpcrd alongside).
        lonely_prmtop = tmp_path / 'system.prmtop'
        shutil.copy(real_amber_system_files['prmtop'], lonely_prmtop)

        fp = Fingerprinter(topology=str(lonely_prmtop))
        with pytest.raises(FileNotFoundError, match='no sibling'):
            fp.load_pdb()


class TestFingerprinterResidueMapping:
    """Test suite for Fingerprinter.assign_residue_mapping."""

    def test_assign_residue_mapping_two_chains(self, two_chain_pdb):
        """Residue mappings split a real two-chain system by chainID."""
        fp = Fingerprinter(
            topology=str(two_chain_pdb),
            target_selection='chainID A',
            binder_selection='chainID B',
        )
        fp.load_pdb()
        fp.assign_residue_mapping()

        # Each chain is Ace-(residue)-Nme: three residues.
        assert len(fp.target_resmap) == 3
        assert len(fp.binder_resmap) == 3

        # Concatenated atom indices cover every atom exactly once per side.
        target = fp.u.select_atoms('chainID A')
        binder = fp.u.select_atoms('chainID B')
        assert fp.target_inds.shape == (target.n_atoms,)
        assert fp.binder_inds.shape == (binder.n_atoms,)
        assert set(fp.target_inds).isdisjoint(set(fp.binder_inds))


class TestFingerprinterPipeline:
    """End-to-end run of the full fingerprinting workflow on a real system."""

    def test_run_produces_physical_fingerprints(self, real_amber_system_files):
        """run() computes finite per-residue fingerprints over every frame."""
        fp = Fingerprinter(
            topology=str(real_amber_system_files['prmtop']),
            trajectory=str(real_amber_system_files['dcd']),
            target_selection='resid 1',
            binder_selection='resid 2 3',
        )
        fp.run()

        # 5 frames; target = 1 residue, binder = 2 residues; last axis is (LJ, ES).
        assert fp.target_fingerprint.shape == (5, 1, 2)
        assert fp.binder_fingerprint.shape == (5, 2, 2)
        assert np.all(np.isfinite(fp.target_fingerprint))
        assert np.all(np.isfinite(fp.binder_fingerprint))

        # Parameters were extracted for all 22 atoms during the run.
        assert fp.charges.shape == (22,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
