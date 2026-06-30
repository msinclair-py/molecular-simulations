"""
Unit tests for fingerprinter.py module
"""

import os

# Disable numba JIT compilation to avoid path resolution issues during testing
os.environ['NUMBA_DISABLE_JIT'] = '1'

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from molecular_simulations.analysis.fingerprinter import (
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
    """Test suite for Fingerprinter class"""

    def test_fingerprinter_init(self):
        """Test Fingerprinter initialization"""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')

            fp = Fingerprinter(topology=str(top_file), target_selection='segid A')

            assert fp.topology == top_file
            assert fp.target_selection == 'segid A'
            assert fp.binder_selection == 'not segid A'

    def test_fingerprinter_init_with_binder_selection(self):
        """Test Fingerprinter with explicit binder selection"""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')

            fp = Fingerprinter(
                topology=str(top_file),
                target_selection='segid A',
                binder_selection='segid B',
            )

            assert fp.binder_selection == 'segid B'

    def test_fingerprinter_output_path(self):
        """Test Fingerprinter output path configuration"""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')

            out_path = path / 'output'
            out_path.mkdir()

            fp = Fingerprinter(
                topology=str(top_file), out_path=str(out_path), out_name='custom.npz'
            )

            assert fp.out == out_path / 'custom.npz'

    @patch('molecular_simulations.analysis.fingerprinter.AmberPrmtopFile')
    @patch('molecular_simulations.analysis.fingerprinter.openmm')
    def test_assign_nonbonded_params(self, mock_openmm, mock_prmtop):
        """Test assign_nonbonded_params method"""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        # Setup mocks
        mock_system = MagicMock()
        mock_system.getNumParticles.return_value = 10

        mock_nonbonded = MagicMock()
        mock_nonbonded.getParticleParameters.return_value = (
            MagicMock(__truediv__=Mock(return_value=0.5)),  # charge
            MagicMock(__truediv__=Mock(return_value=0.3)),  # sigma
            MagicMock(__truediv__=Mock(return_value=0.1)),  # epsilon
        )

        mock_system.getForces.return_value = [mock_nonbonded]
        mock_openmm.NonbondedForce = type(mock_nonbonded)

        mock_prmtop_inst = MagicMock()
        mock_prmtop_inst.createSystem.return_value = mock_system
        mock_prmtop.return_value = mock_prmtop_inst

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')

            fp = Fingerprinter(topology=str(top_file))
            fp.assign_nonbonded_params()

            assert len(fp.charges) == 10
            assert len(fp.sigmas) == 10
            assert len(fp.epsilons) == 10

    @patch('molecular_simulations.analysis.fingerprinter.mda')
    def test_load_pdb_with_pdb_file(self, mock_mda):
        """Test load_pdb with PDB file"""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        mock_mda.Universe.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.pdb'
            top_file.write_text(
                'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            fp = Fingerprinter(topology=str(top_file))
            fp.load_pdb()

            mock_mda.Universe.assert_called_once_with(top_file)

    @patch('molecular_simulations.analysis.fingerprinter.mda')
    def test_load_pdb_with_prmtop_and_trajectory(self, mock_mda):
        """Test load_pdb with prmtop and trajectory"""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        mock_mda.Universe.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            fp = Fingerprinter(topology=str(top_file), trajectory=str(traj_file))
            fp.load_pdb()

            mock_mda.Universe.assert_called_once_with(top_file, traj_file)

    @patch('molecular_simulations.analysis.fingerprinter.mda')
    def test_assign_residue_mapping(self, mock_mda):
        """Test assign_residue_mapping method"""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        # Create mock residues
        mock_residue1 = MagicMock()
        mock_residue1.atoms.ix = np.array([0, 1, 2])
        mock_residue2 = MagicMock()
        mock_residue2.atoms.ix = np.array([3, 4])

        mock_target = MagicMock()
        mock_target.residues = [mock_residue1, mock_residue2]

        mock_binder_res1 = MagicMock()
        mock_binder_res1.atoms.ix = np.array([5, 6])

        mock_binder = MagicMock()
        mock_binder.residues = [mock_binder_res1]

        mock_universe = MagicMock()
        mock_universe.select_atoms.side_effect = [mock_target, mock_binder]
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.pdb'
            top_file.write_text('mock')

            fp = Fingerprinter(topology=str(top_file))
            fp.u = mock_universe
            fp.assign_residue_mapping()

            assert len(fp.target_resmap) == 2
            assert len(fp.binder_resmap) == 1

    def test_save(self):
        """Test save method"""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')

            fp = Fingerprinter(topology=str(top_file))
            fp.target_fingerprint = np.random.rand(10, 5, 2)
            fp.binder_fingerprint = np.random.rand(10, 3, 2)

            fp.save()

            assert fp.out.exists()

            # Load and verify
            data = np.load(fp.out)
            assert 'target' in data
            assert 'binder' in data


class TestFingerprinterLoadPdb:
    """Test suite for Fingerprinter.load_pdb method."""

    @patch('molecular_simulations.analysis.fingerprinter.mda')
    def test_load_pdb_with_pdb_file(self, mock_mda):
        """Test load_pdb creates Universe from PDB file."""
        mock_universe = MagicMock()
        mock_mda.Universe.return_value = mock_universe

        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.pdb'
            top_file.write_text('mock pdb')

            fp = Fingerprinter(topology=str(top_file))
            fp.load_pdb()

            mock_mda.Universe.assert_called_once_with(top_file)
            assert fp.u is mock_universe

    @patch('molecular_simulations.analysis.fingerprinter.mda')
    def test_load_pdb_with_prmtop_and_trajectory(self, mock_mda):
        """Test load_pdb creates Universe from prmtop + trajectory."""
        mock_universe = MagicMock()
        mock_mda.Universe.return_value = mock_universe

        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock prmtop')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock traj')

            fp = Fingerprinter(topology=str(top_file), trajectory=str(traj_file))
            fp.load_pdb()

            mock_mda.Universe.assert_called_once_with(top_file, Path(str(traj_file)))

    @patch('molecular_simulations.analysis.fingerprinter.mda')
    def test_load_pdb_with_prmtop_finds_inpcrd(self, mock_mda):
        """Test load_pdb finds .inpcrd when no trajectory specified."""
        mock_universe = MagicMock()
        mock_mda.Universe.return_value = mock_universe

        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock prmtop')
            inpcrd_file = path / 'system.inpcrd'
            inpcrd_file.write_text('mock inpcrd')

            fp = Fingerprinter(topology=str(top_file))
            fp.load_pdb()

            mock_mda.Universe.assert_called_once_with(top_file, inpcrd_file)


class TestFingerprinterIterateFrames:
    """Test suite for Fingerprinter.iterate_frames method."""

    def test_iterate_frames_initializes_arrays(self):
        """Test iterate_frames creates correct-shaped fingerprint arrays."""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock')

            fp = Fingerprinter(topology=str(top_file))

            # Mock universe and trajectory
            fp.u = MagicMock()
            fp.u.trajectory = [MagicMock(), MagicMock()]  # 2 frames
            fp.target_resmap = [np.array([0, 1]), np.array([2, 3])]
            fp.binder_resmap = [np.array([4, 5, 6])]

            with patch.object(fp, 'calculate_fingerprints'):
                fp.iterate_frames()

            assert fp.target_fingerprint.shape == (2, 2, 2)
            assert fp.binder_fingerprint.shape == (2, 1, 2)


class TestFingerprinterRun:
    """Test suite for Fingerprinter.run method."""

    def test_run_calls_workflow(self):
        """Test run() calls all workflow steps in order."""
        from molecular_simulations.analysis.fingerprinter import Fingerprinter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock')

            fp = Fingerprinter(topology=str(top_file))

            with (
                patch.object(fp, 'assign_nonbonded_params') as mock_params,
                patch.object(fp, 'load_pdb') as mock_load,
                patch.object(fp, 'assign_residue_mapping') as mock_map,
                patch.object(fp, 'iterate_frames') as mock_iter,
            ):
                fp.run()

            mock_params.assert_called_once()
            mock_load.assert_called_once()
            mock_map.assert_called_once()
            mock_iter.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
