"""
Unit tests for utils/mda_utils.py module.

These tests exercise the real ``trim_trajectory`` against real MDAnalysis
Universes built from on-disk fixtures and the real ``rust_simulation_tools``
Kabsch aligner -- no mocks or patches. Assertions check real outcomes: the
output DCD exists, has the expected frame and atom counts, and (for the align
path) that alignment actually reduced the RMSD to the reference frame.
"""

import warnings

import MDAnalysis as mda
import numpy as np
import pytest

from molecular_simulations.utils.mda_utils import trim_trajectory

# The bundled DCD fixtures trigger a benign MDAnalysis DeprecationWarning about
# DCDReader timestep copying; silence it so test output stays clean.
warnings.filterwarnings(
    'ignore',
    message='DCDReader currently makes independent timesteps',
    category=DeprecationWarning,
)


def _rmsd_to_first(positions: np.ndarray, idx: np.ndarray) -> float:
    """Mean per-frame RMSD of frames 1..N to frame 0 over the given atom subset."""
    ref = positions[0][idx]
    return float(
        np.mean(
            [
                np.sqrt(((frame[idx] - ref) ** 2).sum(axis=1).mean())
                for frame in positions[1:]
            ]
        )
    )


class TestTrimTrajectory:
    """Test suite for trim_trajectory using real Universes and real alignment."""

    def test_trim_trajectory_basic(self, two_chain_trajectory, tmp_path):
        """All atoms, default stride: output has every source frame and atom."""
        top, traj = str(two_chain_trajectory['top']), str(two_chain_trajectory['traj'])
        u = mda.Universe(top, traj)
        n_frames = u.trajectory.n_frames
        n_atoms = u.atoms.n_atoms

        out = tmp_path / 'trimmed.dcd'
        trim_trajectory(u, out)

        assert out.exists()
        result = mda.Universe(top, str(out))
        assert result.trajectory.n_frames == n_frames
        assert result.atoms.n_atoms == n_atoms

    def test_trim_trajectory_with_selection(self, two_chain_trajectory, tmp_path):
        """A name-CA selection writes only the selected atoms for every frame."""
        top, traj = str(two_chain_trajectory['top']), str(two_chain_trajectory['traj'])
        u = mda.Universe(top, traj)
        n_frames = u.trajectory.n_frames
        ca = u.select_atoms('name CA')
        assert ca.n_atoms > 0

        # A topology that matches the trimmed (CA-only) DCD so we can read it back.
        sel_top = tmp_path / 'sel.pdb'
        ca.write(str(sel_top))

        out = tmp_path / 'trimmed.dcd'
        trim_trajectory(u, out, sel='name CA')

        assert out.exists()
        result = mda.Universe(str(sel_top), str(out))
        assert result.atoms.n_atoms == ca.n_atoms
        assert result.trajectory.n_frames == n_frames

    def test_trim_trajectory_with_stride(self, two_chain_trajectory, tmp_path):
        """Stride keeps ceil(n_frames / stride) frames (one per strided step)."""
        top, traj = str(two_chain_trajectory['top']), str(two_chain_trajectory['traj'])
        u = mda.Universe(top, traj)
        n_frames = u.trajectory.n_frames
        stride = 2
        expected = len(range(0, n_frames, stride))

        out = tmp_path / 'trimmed.dcd'
        trim_trajectory(u, out, stride=stride)

        assert out.exists()
        result = mda.Universe(top, str(out))
        assert result.trajectory.n_frames == expected
        assert result.atoms.n_atoms == u.atoms.n_atoms

    def test_trim_trajectory_with_align(self, two_chain_trajectory, tmp_path):
        """Default backbone alignment reduces RMSD-to-reference vs the unaligned trajectory."""
        top, traj = str(two_chain_trajectory['top']), str(two_chain_trajectory['traj'])

        # Reference: unaligned positions and the backbone indices used by the code.
        u_raw = mda.Universe(top, traj)
        backbone_idx = u_raw.atoms.select_atoms('backbone or nucleicbackbone').ix
        assert backbone_idx.size > 0
        raw = np.zeros(
            (u_raw.trajectory.n_frames, u_raw.atoms.n_atoms, 3), dtype=np.float32
        )
        for i, _ in enumerate(u_raw.trajectory):
            raw[i] = u_raw.atoms.positions
        raw_rmsd = _rmsd_to_first(raw, backbone_idx)

        u = mda.Universe(top, traj)
        out = tmp_path / 'aligned.dcd'
        trim_trajectory(u, out, align=True)

        assert out.exists()
        result = mda.Universe(top, str(out))
        assert result.trajectory.n_frames == u.trajectory.n_frames
        assert result.atoms.n_atoms == u.atoms.n_atoms

        aligned = np.array([result.atoms.positions.copy() for _ in result.trajectory])
        aligned_rmsd = _rmsd_to_first(aligned, backbone_idx)

        # Alignment must not increase, and on this drifting trajectory must reduce, RMSD.
        assert aligned_rmsd < raw_rmsd

    def test_trim_trajectory_with_custom_align_selection(
        self, two_chain_trajectory, tmp_path
    ):
        """A custom align selection (name CA) drives a valid, RMSD-reducing alignment."""
        top, traj = str(two_chain_trajectory['top']), str(two_chain_trajectory['traj'])

        u_raw = mda.Universe(top, traj)
        ca_idx = u_raw.atoms.select_atoms('name CA').ix
        assert ca_idx.size > 0
        raw = np.zeros(
            (u_raw.trajectory.n_frames, u_raw.atoms.n_atoms, 3), dtype=np.float32
        )
        for i, _ in enumerate(u_raw.trajectory):
            raw[i] = u_raw.atoms.positions
        raw_rmsd = _rmsd_to_first(raw, ca_idx)

        u = mda.Universe(top, traj)
        out = tmp_path / 'aligned_ca.dcd'
        trim_trajectory(u, out, align=True, align_sel='name CA')

        assert out.exists()
        result = mda.Universe(top, str(out))
        assert result.trajectory.n_frames == u.trajectory.n_frames

        aligned = np.array([result.atoms.positions.copy() for _ in result.trajectory])
        aligned_rmsd = _rmsd_to_first(aligned, ca_idx)
        assert aligned_rmsd <= raw_rmsd

    def test_trim_trajectory_with_rewrap(self, two_chain_trajectory, tmp_path):
        """rewrap=True is a documented no-op and must not change the output."""
        top, traj = str(two_chain_trajectory['top']), str(two_chain_trajectory['traj'])

        out_plain = tmp_path / 'plain.dcd'
        trim_trajectory(mda.Universe(top, traj), out_plain)

        out_rewrap = tmp_path / 'rewrap.dcd'
        trim_trajectory(mda.Universe(top, traj), out_rewrap, rewrap=True)

        assert out_rewrap.exists()

        def load_coords(path):
            u = mda.Universe(top, str(path))
            return np.array([u.atoms.positions.copy() for _ in u.trajectory])

        np.testing.assert_allclose(load_coords(out_plain), load_coords(out_rewrap))

    def test_trim_trajectory_amber_system(self, real_amber_system_files, tmp_path):
        """Works on a real AMBER topology + trajectory (prmtop/DCD), aligning backbone."""
        prmtop, dcd = (
            str(real_amber_system_files['prmtop']),
            str(real_amber_system_files['dcd']),
        )
        u = mda.Universe(prmtop, dcd)
        n_frames = u.trajectory.n_frames

        out = tmp_path / 'amber_trim.dcd'
        trim_trajectory(u, out, align=True)

        assert out.exists()
        result = mda.Universe(prmtop, str(out))
        assert result.trajectory.n_frames == n_frames
        assert result.atoms.n_atoms == u.atoms.n_atoms


class TestModuleImports:
    """Test module-level imports."""

    def test_module_imports(self):
        """The module imports and exposes trim_trajectory."""
        from molecular_simulations.utils import mda_utils

        assert hasattr(mda_utils, 'trim_trajectory')

    def test_trim_trajectory_callable(self):
        """trim_trajectory is callable."""
        assert callable(trim_trajectory)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
