"""
Unit tests for utils/mda_utils.py module
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


# Mock rust_simulation_tools before importing the module
@pytest.fixture(autouse=True)
def mock_rust_tools():
    """Mock rust_simulation_tools before any imports"""
    mock_kabsch = MagicMock(return_value=np.zeros((10, 100, 3), dtype=np.float32))
    mock_unwrap = MagicMock()
    mock_rewrap = MagicMock()

    mock_rust = MagicMock()
    mock_rust.kabsch_align = mock_kabsch
    mock_rust.unwrap_system = mock_unwrap
    mock_rust.rewrap_system = mock_rewrap

    with patch.dict(sys.modules, {'rust_simulation_tools': mock_rust}):
        yield {
            'kabsch_align': mock_kabsch,
            'unwrap_system': mock_unwrap,
            'rewrap_system': mock_rewrap,
        }


class TestTrimTrajectory:
    """Test suite for trim_trajectory function"""

    def test_trim_trajectory_basic(self, mock_rust_tools):
        """Test trim_trajectory with basic parameters"""
        from molecular_simulations.utils.mda_utils import trim_trajectory

        # Create mock universe
        mock_universe = MagicMock()

        # Mock trajectory
        mock_trajectory = MagicMock()
        mock_trajectory.n_frames = 10
        mock_trajectory.__iter__ = Mock(
            return_value=iter([(i, MagicMock()) for i in range(10)])
        )
        mock_trajectory.__getitem__ = Mock(return_value=mock_trajectory)
        mock_universe.trajectory = mock_trajectory

        # Mock atoms
        mock_atoms = MagicMock()
        mock_atoms.n_atoms = 100
        mock_atoms.positions = np.random.rand(100, 3).astype(np.float32)
        mock_universe.atoms = mock_atoms
        mock_universe.select_atoms.return_value = mock_atoms

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'trimmed.dcd'

            with patch('molecular_simulations.utils.mda_utils.mda') as mock_mda:
                mock_writer = MagicMock()
                mock_mda.Writer.return_value.__enter__ = Mock(return_value=mock_writer)
                mock_mda.Writer.return_value.__exit__ = Mock(return_value=None)

                trim_trajectory(mock_universe, out_path)

                mock_mda.Writer.assert_called_once()

    def test_trim_trajectory_with_selection(self, mock_rust_tools):
        """Test trim_trajectory with custom selection"""
        from molecular_simulations.utils.mda_utils import trim_trajectory

        mock_universe = MagicMock()

        mock_trajectory = MagicMock()
        mock_trajectory.n_frames = 5
        mock_trajectory.__iter__ = Mock(
            return_value=iter([(i, MagicMock()) for i in range(5)])
        )
        mock_trajectory.__getitem__ = Mock(return_value=mock_trajectory)
        mock_universe.trajectory = mock_trajectory

        mock_selection = MagicMock()
        mock_selection.n_atoms = 50
        mock_selection.positions = np.random.rand(50, 3).astype(np.float32)
        mock_universe.select_atoms.return_value = mock_selection
        mock_universe.atoms = mock_selection

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'trimmed.dcd'

            with patch('molecular_simulations.utils.mda_utils.mda') as mock_mda:
                mock_writer = MagicMock()
                mock_mda.Writer.return_value.__enter__ = Mock(return_value=mock_writer)
                mock_mda.Writer.return_value.__exit__ = Mock(return_value=None)

                trim_trajectory(mock_universe, out_path, sel='protein')

                mock_universe.select_atoms.assert_called_with('protein')

    def test_trim_trajectory_with_stride(self, mock_rust_tools):
        """Test trim_trajectory with stride parameter"""
        from molecular_simulations.utils.mda_utils import trim_trajectory

        mock_universe = MagicMock()

        mock_trajectory = MagicMock()
        mock_trajectory.n_frames = 100
        mock_trajectory.__iter__ = Mock(
            return_value=iter([(i, MagicMock()) for i in range(50)])
        )
        mock_trajectory.__getitem__ = Mock(return_value=mock_trajectory)
        mock_universe.trajectory = mock_trajectory

        mock_atoms = MagicMock()
        mock_atoms.n_atoms = 100
        mock_atoms.positions = np.random.rand(100, 3).astype(np.float32)
        mock_universe.atoms = mock_atoms
        mock_universe.select_atoms.return_value = mock_atoms

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'trimmed.dcd'

            with patch('molecular_simulations.utils.mda_utils.mda') as mock_mda:
                mock_writer = MagicMock()
                mock_mda.Writer.return_value.__enter__ = Mock(return_value=mock_writer)
                mock_mda.Writer.return_value.__exit__ = Mock(return_value=None)

                trim_trajectory(mock_universe, out_path, stride=2)

    def test_trim_trajectory_with_align(self, mock_rust_tools):
        """Test trim_trajectory with alignment"""
        from molecular_simulations.utils.mda_utils import trim_trajectory

        mock_universe = MagicMock()

        mock_trajectory = MagicMock()
        mock_trajectory.n_frames = 10
        mock_trajectory.__iter__ = Mock(
            return_value=iter([(i, MagicMock()) for i in range(10)])
        )
        mock_trajectory.__getitem__ = Mock(return_value=mock_trajectory)
        mock_universe.trajectory = mock_trajectory

        mock_atoms = MagicMock()
        mock_atoms.n_atoms = 100
        mock_atoms.positions = np.random.rand(100, 3).astype(np.float32)
        mock_universe.atoms = mock_atoms
        mock_universe.select_atoms.return_value = mock_atoms

        # Mock backbone selection for alignment
        mock_backbone = MagicMock()
        mock_backbone.ix = np.array([0, 1, 2, 3, 4])

        def select_side_effect(sel_string):
            if 'backbone' in sel_string:
                return mock_backbone
            return mock_atoms

        mock_atoms.select_atoms = Mock(side_effect=select_side_effect)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'trimmed.dcd'

            with patch('molecular_simulations.utils.mda_utils.mda') as mock_mda:
                mock_writer = MagicMock()
                mock_mda.Writer.return_value.__enter__ = Mock(return_value=mock_writer)
                mock_mda.Writer.return_value.__exit__ = Mock(return_value=None)

                trim_trajectory(mock_universe, out_path, align=True)

                # kabsch_align should have been called
                mock_rust_tools['kabsch_align'].assert_called_once()

    def test_trim_trajectory_with_custom_align_selection(self, mock_rust_tools):
        """Test trim_trajectory with custom alignment selection"""
        from molecular_simulations.utils.mda_utils import trim_trajectory

        mock_universe = MagicMock()

        mock_trajectory = MagicMock()
        mock_trajectory.n_frames = 10
        mock_trajectory.__iter__ = Mock(
            return_value=iter([(i, MagicMock()) for i in range(10)])
        )
        mock_trajectory.__getitem__ = Mock(return_value=mock_trajectory)
        mock_universe.trajectory = mock_trajectory

        mock_atoms = MagicMock()
        mock_atoms.n_atoms = 100
        mock_atoms.positions = np.random.rand(100, 3).astype(np.float32)
        mock_universe.atoms = mock_atoms
        mock_universe.select_atoms.return_value = mock_atoms

        mock_align_sel = MagicMock()
        mock_align_sel.ix = np.array([10, 11, 12, 13, 14])
        mock_atoms.select_atoms = Mock(return_value=mock_align_sel)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'trimmed.dcd'

            with patch('molecular_simulations.utils.mda_utils.mda') as mock_mda:
                mock_writer = MagicMock()
                mock_mda.Writer.return_value.__enter__ = Mock(return_value=mock_writer)
                mock_mda.Writer.return_value.__exit__ = Mock(return_value=None)

                trim_trajectory(
                    mock_universe, out_path, align=True, align_sel='name CA'
                )

                mock_atoms.select_atoms.assert_called_with('name CA')

    def test_trim_trajectory_with_rewrap(self, mock_rust_tools):
        """Test trim_trajectory with rewrap parameter (currently pass)"""
        from molecular_simulations.utils.mda_utils import trim_trajectory

        mock_universe = MagicMock()

        mock_trajectory = MagicMock()
        mock_trajectory.n_frames = 10
        mock_trajectory.__iter__ = Mock(
            return_value=iter([(i, MagicMock()) for i in range(10)])
        )
        mock_trajectory.__getitem__ = Mock(return_value=mock_trajectory)
        mock_universe.trajectory = mock_trajectory

        mock_atoms = MagicMock()
        mock_atoms.n_atoms = 100
        mock_atoms.positions = np.random.rand(100, 3).astype(np.float32)
        mock_universe.atoms = mock_atoms
        mock_universe.select_atoms.return_value = mock_atoms

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'trimmed.dcd'

            with patch('molecular_simulations.utils.mda_utils.mda') as mock_mda:
                mock_writer = MagicMock()
                mock_mda.Writer.return_value.__enter__ = Mock(return_value=mock_writer)
                mock_mda.Writer.return_value.__exit__ = Mock(return_value=None)

                # Should not raise even with rewrap=True (currently a pass)
                trim_trajectory(mock_universe, out_path, rewrap=True)


class TestModuleImports:
    """Test module-level imports"""

    def test_module_imports(self, mock_rust_tools):
        """Test that module can be imported"""
        # This should not raise
        from molecular_simulations.utils import mda_utils

        assert hasattr(mda_utils, 'trim_trajectory')

    def test_pathlike_optional(self, mock_rust_tools):
        """Test Optional type usage"""
        from molecular_simulations.utils.mda_utils import trim_trajectory

        # Check function accepts various inputs
        assert callable(trim_trajectory)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
