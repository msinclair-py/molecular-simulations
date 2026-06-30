"""
Unit tests for simulate/mmpbsa.py module

This module contains both unit tests (with minimal mocks) and integration tests.
Tests that don't require AMBER binaries use the actual classes with mocked
external processes, while pure logic tests avoid mocking entirely.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ============================================================================
# Fixtures and helpers for conditional dependency usage
# ============================================================================


def _check_amberhome():
    """Check if AMBERHOME is set."""
    return 'AMBERHOME' in os.environ


requires_amber = pytest.mark.skipif(not _check_amberhome(), reason='AMBERHOME not set')


@pytest.fixture
def test_data_dir():
    """Return the path to test data directory."""
    return Path(__file__).parent / 'data'


@pytest.fixture
def mock_amberhome(tmp_path):
    """Create a mock AMBERHOME with fake binaries."""
    amber_bin = tmp_path / 'amber' / 'bin'
    amber_bin.mkdir(parents=True)

    # Create dummy executables
    (amber_bin / 'cpptraj').write_text('#!/bin/bash\necho "mock cpptraj"')
    (amber_bin / 'mmpbsa_py_energy').write_text('#!/bin/bash\necho "mock mmpbsa"')

    return tmp_path / 'amber'


# ============================================================================
# Pure logic tests - no mocking needed
# ============================================================================


class TestMMPBSASettings:
    """Test suite for MMPBSASettings dataclass - no mocks needed."""

    def test_mmpbsa_settings_defaults(self):
        """Test MMPBSA_settings with default values"""
        from molecular_simulations.simulate.mmpbsa import MMPBSASettings

        settings = MMPBSASettings(
            top='system.prmtop', dcd='traj.dcd', selections=[':1-100', ':101-150']
        )

        assert settings.first_frame == 0
        assert settings.last_frame == -1
        assert settings.stride == 1
        assert settings.n_cpus == 1
        assert settings.out == 'mmpbsa'
        assert settings.solvent_probe == 1.4
        assert settings.gb_surften == 0.0072
        assert settings.gb_surfoff == 0.0

    def test_mmpbsa_settings_custom(self):
        """Test MMPBSA_settings with custom values"""
        from molecular_simulations.simulate.mmpbsa import MMPBSASettings

        settings = MMPBSASettings(
            top='system.prmtop',
            dcd='traj.dcd',
            selections=[':1-100', ':101-150'],
            first_frame=10,
            last_frame=100,
            stride=5,
            n_cpus=4,
            out='custom_output',
        )

        assert settings.first_frame == 10
        assert settings.last_frame == 100
        assert settings.stride == 5
        assert settings.n_cpus == 4
        assert settings.out == 'custom_output'

    def test_mmpbsa_settings_types(self):
        """Test MMPBSA_settings field types without mocking."""
        from molecular_simulations.simulate.mmpbsa import MMPBSASettings

        settings = MMPBSASettings(
            top='test.prmtop', dcd='test.dcd', selections=[':1-50', ':51-100']
        )

        assert isinstance(settings.top, str)
        assert isinstance(settings.dcd, str)
        assert isinstance(settings.selections, list)
        assert isinstance(settings.first_frame, int)
        assert isinstance(settings.stride, int)
        assert isinstance(settings.solvent_probe, float)


class TestMMPBSAInit:
    """Test suite for MMPBSA class initialization.

    Uses subprocess mocking instead of FileHandler mocking to reduce mock count.
    """

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_mmpbsa_init(self, mock_subprocess, mock_amberhome):
        """Test MMPBSA initialization with mock AMBERHOME."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')
        mock_subprocess.call.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                amberhome=str(mock_amberhome),
            )

            assert mmpbsa.cpptraj == mock_amberhome / 'bin' / 'cpptraj'
            assert (
                mmpbsa.mmpbsa_py_energy == mock_amberhome / 'bin' / 'mmpbsa_py_energy'
            )

    def test_mmpbsa_init_no_amberhome(self):
        """Test MMPBSA initialization without AMBERHOME - no mock needed."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        # Remove AMBERHOME from environment
        env = os.environ.copy()
        env.pop('AMBERHOME', None)

        with patch.dict(os.environ, env, clear=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                top_file = path / 'system.prmtop'
                top_file.write_text('mock topology')
                traj_file = path / 'traj.dcd'
                traj_file.write_text('mock trajectory')

                with pytest.raises(ValueError, match='AMBERHOME not set'):
                    MMPBSA(
                        top=str(top_file),
                        dcd=str(traj_file),
                        selections=[':1-100', ':101-150'],
                    )

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_mmpbsa_init_custom_output(self, mock_subprocess, mock_amberhome):
        """Test MMPBSA initialization with custom output path."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')
        mock_subprocess.call.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            out_dir = path / 'custom_output'
            out_dir.mkdir()

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                out=str(out_dir),
                amberhome=str(mock_amberhome),
            )

            # Use resolve() to handle /private symlink on macOS
            assert mmpbsa.path.resolve() == out_dir.resolve()


class TestMMPBSAWriteMdins:
    """Test suite for MMPBSA write_mdins method.

    Uses real file operations where possible.
    """

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_write_mdins(self, mock_subprocess, mock_amberhome):
        """Test write_mdins method creates actual files."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')
        mock_subprocess.call.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                amberhome=str(mock_amberhome),
            )

            gb_mdin, pb_mdin = mmpbsa.write_mdins()

            # Verify actual files were created
            assert gb_mdin.exists()
            assert pb_mdin.exists()

            # Check file contents
            gb_content = gb_mdin.read_text()
            assert 'igb = 2' in gb_content

            pb_content = pb_mdin.read_text()
            assert 'inp = 2' in pb_content


class TestOutputAnalyzer:
    """Test suite for OutputAnalyzer class - uses real objects, no mocking."""

    def test_output_analyzer_init(self):
        """Test OutputAnalyzer initialization - no mocks needed."""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = OutputAnalyzer(
                path=tmpdir, surface_tension=0.01, sasa_offset=0.5
            )

            assert analyzer.surften == 0.01
            assert analyzer.offset == 0.5
            assert analyzer.free_energy is None
            assert analyzer.systems == ['receptor', 'ligand', 'complex']
            assert analyzer.levels == ['gb', 'pb']

    def test_parse_line_single_value(self):
        """Test parse_line with single key=value - static method, no mocks."""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        line = ' BOND = 123.456'
        result = list(OutputAnalyzer.parse_line(line))

        assert len(result) == 1
        assert result[0][0] == 'BOND'
        assert np.isclose(result[0][1], 123.456)

    def test_parse_line_multiple_values(self):
        """Test parse_line with multiple key=value pairs - static method."""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        line = ' BOND =  123.456  ANGLE =  78.901'
        result = list(OutputAnalyzer.parse_line(line))

        assert len(result) == 2
        keys = [r[0] for r in result]
        assert 'BOND' in keys
        assert 'ANGLE' in keys

    def test_parse_line_energy_format(self):
        """Test parse_line with full energy output format."""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        line = ' VDWAALS = -10.5  EEL = -200.3  EGB = -50.1'
        result = list(OutputAnalyzer.parse_line(line))

        assert len(result) == 3
        result_dict = dict(result)
        assert np.isclose(result_dict['VDWAALS'], -10.5)
        assert np.isclose(result_dict['EEL'], -200.3)
        assert np.isclose(result_dict['EGB'], -50.1)

    def test_read_sasa(self):
        """Test read_sasa method - uses real file, no mocks."""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create real SASA file
            sasa_content = """#Frame SASA
1 1000.0
2 1100.0
3 1050.0
"""
            sasa_file = path / 'test_surf.dat'
            sasa_file.write_text(sasa_content)

            analyzer = OutputAnalyzer(path=str(path), surface_tension=0.0072)
            sasa_series = analyzer.read_sasa(sasa_file)

            assert len(sasa_series) == 3
            # SASA should be scaled by surface tension (0.0072)
            # 1000.0 * 0.0072 = 7.2
            assert np.isclose(sasa_series[0], 7.2)
            assert np.isclose(sasa_series[1], 7.92)
            assert np.isclose(sasa_series[2], 7.56)


class TestFileHandler:
    """Test suite for FileHandler class - uses real files, no mocking."""

    def test_write_file_list(self):
        """Test write_file with list input - static method, no mocks."""
        from molecular_simulations.simulate.mmpbsa import FileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            outfile = path / 'test.txt'

            lines = ['line1', 'line2', 'line3']
            FileHandler.write_file(lines, outfile)

            content = outfile.read_text()
            assert content == 'line1\nline2\nline3'

    def test_write_file_string(self):
        """Test write_file with string input - static method, no mocks."""
        from molecular_simulations.simulate.mmpbsa import FileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            outfile = path / 'test.txt'

            content = 'single string content'
            FileHandler.write_file(content, outfile)

            result = outfile.read_text()
            assert result == content

    def test_write_file_with_path_object(self):
        """Test write_file with Path object."""
        from molecular_simulations.simulate.mmpbsa import FileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            outfile = path / 'path_test.txt'

            lines = ['path', 'object', 'test']
            FileHandler.write_file(lines, outfile)

            assert outfile.exists()
            content = outfile.read_text()
            assert content == 'path\nobject\ntest'


class TestCombineChunks:
    """Test suite for chunk combining methods.

    Uses subprocess mocking only for FileHandler initialization.
    """

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_combine_sasa_chunks(self, mock_subprocess, mock_amberhome):
        """Test _combine_sasa_chunks method with real file operations."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')
        mock_subprocess.call.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create topology and trajectory files
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            # Create chunk files with real data
            for system in ['complex', 'receptor', 'ligand']:
                for i in range(2):
                    chunk_file = path / f'{system}_chunk{i}_surf.dat'
                    if i == 0:
                        chunk_file.write_text('#Frame SASA\n1 1000.0\n2 1050.0\n')
                    else:
                        chunk_file.write_text('#Frame SASA\n3 1100.0\n4 1150.0\n')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                amberhome=str(mock_amberhome),
            )
            mmpbsa.path = path

            mmpbsa._combine_sasa_chunks()

            # Check combined files exist with correct content
            for system in ['complex', 'receptor', 'ligand']:
                combined_file = path / f'{system}_surf.dat'
                assert combined_file.exists()
                content = combined_file.read_text()
                # Should have header and 4 data lines
                assert '1000.0' in content
                assert '1150.0' in content

            # Chunk files should be deleted
            for system in ['complex', 'receptor', 'ligand']:
                for i in range(2):
                    chunk_file = path / f'{system}_chunk{i}_surf.dat'
                    assert not chunk_file.exists()

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_combine_energy_chunks(self, mock_subprocess, mock_amberhome):
        """Test _combine_energy_chunks method with real file operations."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')
        mock_subprocess.call.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            # Create energy chunk files with real data
            for system in ['complex', 'receptor', 'ligand']:
                for level in ['gb', 'pb']:
                    for i in range(2):
                        chunk_file = path / f'{system}_chunk{i}_{level}.mdout'
                        chunk_file.write_text(f' BOND = {100 + i * 10}.0\n')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                amberhome=str(mock_amberhome),
            )
            mmpbsa.path = path

            mmpbsa._combine_energy_chunks()

            # Check combined files exist
            for system in ['complex', 'receptor', 'ligand']:
                for level in ['gb', 'pb']:
                    combined_file = path / f'{system}_{level}.mdout'
                    assert combined_file.exists()


class TestWriteSasaScript:
    """Test suite for _write_sasa_script method.

    Uses subprocess mocking only for FileHandler initialization.
    """

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_write_sasa_script(self, mock_subprocess, mock_amberhome):
        """Test _write_sasa_script method creates real file."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')
        mock_subprocess.call.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                amberhome=str(mock_amberhome),
            )
            mmpbsa.path = path

            prefix = path / 'complex'
            prm = path / 'complex.prmtop'
            trj = path / 'complex_chunk0.crd'

            result = mmpbsa._write_sasa_script(prefix, prm, trj, chunk_idx=0)

            # Check file was created and has correct content
            assert result.exists()
            content = result.read_text()
            assert 'molsurf' in content
            assert 'probe' in content
            assert 'sasa_complex_chunk0.in' in str(result)


class TestOutputAnalyzerMethods:
    """Test suite for OutputAnalyzer methods"""

    def test_output_analyzer_init_full(self):
        """Test OutputAnalyzer initialization with all parameters"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = OutputAnalyzer(
                path=tmpdir, surface_tension=0.01, sasa_offset=0.5
            )

            assert analyzer.surften == 0.01
            assert analyzer.offset == 0.5
            assert analyzer.free_energy is None

    def test_parse_line_standard_format(self):
        """Test parse_line with standard Amber format"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        # Standard format from mmpbsa_py_energy output
        line = ' BOND =    123.456  ANGLE =     78.901'
        result = list(OutputAnalyzer.parse_line(line))

        assert len(result) == 2
        assert result[0][0] == 'BOND'
        assert result[1][0] == 'ANGLE'
        assert np.isclose(result[0][1], 123.456)
        assert np.isclose(result[1][1], 78.901)

    def test_parse_line_single_value(self):
        """Test parse_line with single value"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        line = ' BOND = 123.456'
        result = list(OutputAnalyzer.parse_line(line))

        assert len(result) == 1
        assert result[0][0] == 'BOND'
        assert np.isclose(result[0][1], 123.456)


class TestFileHandlerWriteFile:
    """Test suite for FileHandler.write_file method"""

    def test_write_file_creates_output(self):
        """Test that write_file creates output correctly"""
        from molecular_simulations.simulate.mmpbsa import FileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            outfile = path / 'test_output.txt'

            content = ['line 1', 'line 2', 'line 3']
            FileHandler.write_file(content, outfile)

            assert outfile.exists()
            result = outfile.read_text()
            assert 'line 1' in result
            assert 'line 2' in result
            assert 'line 3' in result


class TestMMPBSARun:
    """Test suite for MMPBSA run method.

    Uses subprocess mocking for external processes.
    """

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_mmpbsa_run_creates_output_dir(self, mock_subprocess, mock_amberhome):
        """Test that run creates output directory."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')
        mock_subprocess.call.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            out_dir = path / 'output' / 'mmpbsa'

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                out=str(out_dir),
                amberhome=str(mock_amberhome),
            )

            # Verify output directory was created
            assert out_dir.exists()
            assert mmpbsa.path.resolve() == out_dir.resolve()


class TestMMPBSARunMethods:
    """Test suite for MMPBSA run methods - uses method-level mocking only."""

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    @patch('molecular_simulations.simulate.mmpbsa.OutputAnalyzer')
    def test_run_serial_mode(self, mock_analyzer, mock_subprocess, mock_amberhome):
        """Test run method in serial mode."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')
        mock_subprocess.call.return_value = 0

        mock_analyzer_inst = MagicMock()
        mock_analyzer_inst.free_energy = -10.5
        mock_analyzer.return_value = mock_analyzer_inst

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                parallel_mode='serial',
                amberhome=str(mock_amberhome),
            )

            with (
                patch.object(mmpbsa, 'write_mdins') as mock_write,
                patch.object(mmpbsa, 'calculate_sasa'),
                patch.object(mmpbsa, 'calculate_energy'),
            ):
                mock_write.return_value = (path / 'gb.mdin', path / 'pb.mdin')

                mmpbsa.run()

                mock_write.assert_called_once()
                mock_analyzer_inst.parse_outputs.assert_called_once()
                assert mmpbsa.free_energy == -10.5

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    @patch('molecular_simulations.simulate.mmpbsa.OutputAnalyzer')
    @patch('molecular_simulations.simulate.mmpbsa._run_sasa_calculation')
    @patch('molecular_simulations.simulate.mmpbsa._run_energy_calculation')
    def test_run_frame_parallel_mode(
        self,
        mock_energy_calc,
        mock_sasa_calc,
        mock_analyzer,
        mock_subprocess,
        mock_amberhome,
    ):
        """Test run method in frame parallel mode."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')
        mock_subprocess.call.return_value = 0

        mock_analyzer_inst = MagicMock()
        mock_analyzer_inst.free_energy = -15.0
        mock_analyzer.return_value = mock_analyzer_inst

        mock_sasa_calc.return_value = ('sasa_script', True, '')
        mock_energy_calc.return_value = ('energy_out', True, '')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                parallel_mode='frame',
                n_cpus=2,
                amberhome=str(mock_amberhome),
            )

            with (
                patch.object(mmpbsa, 'write_mdins') as mock_write,
                patch.object(mmpbsa, '_write_sasa_script') as mock_sasa_script,
                patch.object(mmpbsa, '_combine_sasa_chunks'),
                patch.object(mmpbsa, '_combine_energy_chunks'),
                patch.object(mmpbsa, '_verify_combined_outputs'),
            ):
                mock_write.return_value = (path / 'gb.mdin', path / 'pb.mdin')
                mock_sasa_script.return_value = path / 'sasa.in'

                mmpbsa.run()

                mock_write.assert_called_once()
                mock_analyzer_inst.parse_outputs.assert_called_once()


class TestCalculateEnergy:
    """Test suite for calculate_energy method - uses subprocess mocking only."""

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_calculate_energy(self, mock_subprocess, mock_amberhome):
        """Test calculate_energy method."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stderr='', stdout='')
        mock_subprocess.call.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock')

            # Create mock mdin file
            mdin_file = path / 'gb.mdin'
            mdin_file.write_text('&general\n/')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                amberhome=str(mock_amberhome),
            )
            mmpbsa.path = path

            prefix = path / 'complex'
            prm = path / 'complex.prmtop'
            trj = path / 'complex.crd'
            pdb = path / 'complex.pdb'

            mmpbsa.calculate_energy(prefix, prm, trj, pdb, mdin_file, 'gb')

            # subprocess should have been called for energy calculation
            assert mock_subprocess.run.call_count >= 1


class TestOutputAnalyzerParsing:
    """Test suite for OutputAnalyzer parsing methods"""

    @patch('molecular_simulations.simulate.mmpbsa.OutputAnalyzer.generate_summary')
    def test_parse_outputs(self, mock_summary):
        """Test parse_outputs method"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create GB energy output files
            gb_content = """ BOND =  100.0  ANGLE = 50.0  DIHED = 25.0
 VDWAALS = -10.0  EEL = -200.0  EGB = -50.0
 1-4 VDW = 5.0  1-4 EEL = 10.0  RESTRAINT = 0.0
 ESURF = -5.0
"""
            # Create PB energy output files (different fields)
            pb_content = """ BOND =  100.0  ANGLE = 50.0  DIHED = 25.0
 VDWAALS = -10.0  EEL = -200.0  EPB = -50.0
 1-4 VDW = 5.0  1-4 EEL = 10.0  RESTRAINT = 0.0
 ECAVITY = -5.0  EDISPER = -3.0
"""
            for system in ['complex', 'receptor', 'ligand']:
                # GB file
                gb_file = path / f'{system}_gb.mdout'
                gb_file.write_text(gb_content)

                # PB file
                pb_file = path / f'{system}_pb.mdout'
                pb_file.write_text(pb_content)

                sasa_file = path / f'{system}_surf.dat'
                sasa_file.write_text('#Frame SASA\n1 1000.0\n')

            analyzer = OutputAnalyzer(path=str(path), surface_tension=0.0072)
            analyzer.parse_outputs()

            # Verify parsing occurred
            assert analyzer.gb is not None
            assert analyzer.pb is not None
            mock_summary.assert_called_once()

    def test_parse_line(self):
        """Test parse_line method"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            analyzer = OutputAnalyzer(path=str(path), surface_tension=0.0072)

            # Test parsing energy line
            line = ' BOND =  100.0  ANGLE = 50.0  DIHED = 25.0'
            result = list(analyzer.parse_line(line))

            assert len(result) == 3
            assert ('BOND', 100.0) in result
            assert ('ANGLE', 50.0) in result
            assert ('DIHED', 25.0) in result

    def test_read_sasa_basic(self):
        """Test read_sasa with basic format"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Simple SASA file without comments in the middle
            sasa_content = """#Frame SASA
1 1000.0
2 1100.0
3 1050.0
"""
            sasa_file = path / 'test_surf.dat'
            sasa_file.write_text(sasa_content)

            analyzer = OutputAnalyzer(path=str(path), surface_tension=0.0072)
            sasa_series = analyzer.read_sasa(sasa_file)

            assert len(sasa_series) == 3


class TestFileHandlerMethods:
    """Test suite for FileHandler class methods"""

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_filehandler_init(self, mock_subprocess):
        """Test FileHandler initialization"""
        from molecular_simulations.simulate.mmpbsa import FileHandler

        # Mock subprocess.run to return success with frame count output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'Total frames: 100\n'
        mock_subprocess.run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock')

            fh = FileHandler(
                top=top_file,
                traj=traj_file,
                path=path,
                sels=[':1-100', ':101-150'],
                first=0,
                last=-1,
                stride=1,
                cpptraj_binary='/fake/cpptraj',
                n_chunks=2,
            )

            assert fh.path == path
            mock_subprocess.run.assert_called()

    def test_write_file_pathlike(self):
        """Test write_file with Path object"""
        from molecular_simulations.simulate.mmpbsa import FileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            outfile = path / 'output.txt'

            FileHandler.write_file(['line1', 'line2'], outfile)

            assert outfile.exists()
            content = outfile.read_text()
            assert 'line1' in content
            assert 'line2' in content


class TestRunFrameParallelFailures:
    """Test failure handling in _run_frame_parallel"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    @patch('molecular_simulations.simulate.mmpbsa.OutputAnalyzer')
    @patch('molecular_simulations.simulate.mmpbsa._run_sasa_calculation')
    def test_sasa_failure_raises(self, mock_sasa_calc, mock_analyzer, mock_filehandler):
        """Test that SASA failures raise RuntimeError"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_fh.path = Path('/tmp/test')
        mock_fh.files_chunked = [
            (
                Path('/tmp/test/complex'),
                Path('/tmp/test/complex.prmtop'),
                [Path('/tmp/test/complex_chunk0.crd')],
                Path('/tmp/test/complex.pdb'),
            ),
        ]
        mock_filehandler.return_value = mock_fh

        # SASA calculation fails
        mock_sasa_calc.return_value = ('sasa_script', False, 'SASA error')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                parallel_mode='frame',
            )

            with (
                patch.object(mmpbsa, 'write_mdins') as mock_write,
                patch.object(mmpbsa, '_write_sasa_script') as mock_sasa_script,
            ):
                mock_write.return_value = (path / 'gb.mdin', path / 'pb.mdin')
                mock_sasa_script.return_value = path / 'sasa.in'

                with pytest.raises(RuntimeError, match='SASA calculations failed'):
                    mmpbsa.run()


class TestMMPBSAKwargs:
    """Test MMPBSA with extra kwargs"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_mmpbsa_with_kwargs(self, mock_filehandler):
        """Test MMPBSA stores extra kwargs as attributes"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_filehandler.return_value = mock_fh

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                custom_attr='custom_value',
                another_attr=42,
            )

            assert hasattr(mmpbsa, 'custom_attr')
            assert mmpbsa.custom_attr == 'custom_value'
            assert mmpbsa.another_attr == 42


class TestRunEnergyCalculation:
    """Test suite for _run_energy_calculation worker function."""

    @patch('molecular_simulations.simulate.mmpbsa.time')
    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_successful_energy_calculation(self, mock_subprocess, mock_time):
        """Test worker returns success when output file has energy data."""
        from molecular_simulations.simulate.mmpbsa import _run_energy_calculation

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create output file with energy data
            out_file = Path(tmpdir) / 'test_gb.mdout'
            out_file.write_text(' BOND = 100.0  ANGLE = 50.0\n')

            result = _run_energy_calculation(
                (
                    'mmpbsa_py_energy',
                    'gb.mdin',
                    'sys.prmtop',
                    'sys.pdb',
                    'traj.crd',
                    'test_gb.mdout',
                    tmpdir,
                ),
                max_retries=1,
            )

            assert result[1] is True
            assert result[2] == ''

    @patch('molecular_simulations.simulate.mmpbsa.time')
    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_failed_energy_calculation_missing_output(self, mock_subprocess, mock_time):
        """Test worker returns failure when output file is missing."""
        from molecular_simulations.simulate.mmpbsa import _run_energy_calculation

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_energy_calculation(
                (
                    'mmpbsa_py_energy',
                    'gb.mdin',
                    'sys.prmtop',
                    'sys.pdb',
                    'traj.crd',
                    'missing_output.mdout',
                    tmpdir,
                ),
                max_retries=1,
            )

            assert result[1] is False
            assert 'missing or empty' in result[2]

    @patch('molecular_simulations.simulate.mmpbsa.time')
    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_failed_energy_nonzero_return_code(self, mock_subprocess, mock_time):
        """Test worker captures nonzero return code."""
        from molecular_simulations.simulate.mmpbsa import _run_energy_calculation

        mock_subprocess.run.return_value = MagicMock(
            returncode=1, stderr='segfault', stdout=''
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create output file without BOND data
            out_file = Path(tmpdir) / 'test_gb.mdout'
            out_file.write_text('no energy here\n')

            result = _run_energy_calculation(
                (
                    'mmpbsa_py_energy',
                    'gb.mdin',
                    'sys.prmtop',
                    'sys.pdb',
                    'traj.crd',
                    'test_gb.mdout',
                    tmpdir,
                ),
                max_retries=1,
            )

            assert result[1] is False
            assert 'Return code' in result[2]

    @patch('molecular_simulations.simulate.mmpbsa.time')
    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_timeout_exception(self, mock_subprocess, mock_time):
        """Test worker handles TimeoutExpired."""
        import subprocess as real_subprocess

        from molecular_simulations.simulate.mmpbsa import _run_energy_calculation

        mock_subprocess.run.side_effect = real_subprocess.TimeoutExpired(
            cmd='test', timeout=30
        )
        mock_subprocess.TimeoutExpired = real_subprocess.TimeoutExpired

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_energy_calculation(
                (
                    'mmpbsa_py_energy',
                    'gb.mdin',
                    'sys.prmtop',
                    'sys.pdb',
                    'traj.crd',
                    'test.mdout',
                    tmpdir,
                ),
                max_retries=1,
            )

            assert result[1] is False
            assert 'timed out' in result[2]


class TestRunSasaCalculation:
    """Test suite for _run_sasa_calculation worker function."""

    @patch('molecular_simulations.simulate.mmpbsa.time')
    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_successful_sasa_calculation(self, mock_subprocess, mock_time):
        """Test SASA worker returns success with valid output."""
        from molecular_simulations.simulate.mmpbsa import _run_sasa_calculation

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create SASA input script
            sasa_script = Path(tmpdir) / 'sasa.in'
            sasa_script.write_text(
                'parm sys.prmtop\ntrajin traj.crd\n'
                'molsurf :* out test_surf.dat probe 1.4 offset 0.0\nrun\nquit\n'
            )

            # Create the expected output
            out_file = Path(tmpdir) / 'test_surf.dat'
            out_file.write_text('#Frame SASA\n1 1000.0\n2 1100.0\n')

            result = _run_sasa_calculation(
                ('cpptraj', str(sasa_script), tmpdir), max_retries=1
            )

            assert result[1] is True

    @patch('molecular_simulations.simulate.mmpbsa.time')
    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_sasa_calculation_empty_output(self, mock_subprocess, mock_time):
        """Test SASA worker handles empty output file."""
        from molecular_simulations.simulate.mmpbsa import _run_sasa_calculation

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            sasa_script = Path(tmpdir) / 'sasa.in'
            sasa_script.write_text(
                'molsurf :* out test_surf.dat probe 1.4 offset 0.0\n'
            )

            # Create empty output
            out_file = Path(tmpdir) / 'test_surf.dat'
            out_file.write_text('')

            result = _run_sasa_calculation(
                ('cpptraj', str(sasa_script), tmpdir), max_retries=1
            )

            assert result[1] is False
            assert 'empty' in result[2]


class TestOutputAnalyzerGenerateSummary:
    """Test suite for OutputAnalyzer.generate_summary method."""

    def test_generate_summary_creates_json(self, monkeypatch):
        """Test generate_summary writes statistics.json."""
        import polars as pl

        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            # generate_summary writes to CWD
            monkeypatch.chdir(tmpdir)

            analyzer = OutputAnalyzer(path=tmpdir, surface_tension=0.0072)

            # Build minimal DataFrames with required structure
            data = {
                'VDWAALS': [-10.0, -11.0, -10.5, -10.0, -11.0, -10.5],
                'EEL': [-200.0, -210.0, -205.0, -200.0, -210.0, -205.0],
                'EGB': [-50.0, -55.0, -52.0, -50.0, -55.0, -52.0],
                'ESURF': [7.0, 7.5, 7.2, 7.0, 7.5, 7.2],
                'system': [
                    'receptor',
                    'receptor',
                    'ligand',
                    'ligand',
                    'complex',
                    'complex',
                ],
            }

            analyzer.gb = pl.DataFrame(data)
            analyzer.pb = pl.DataFrame(
                {
                    'VDWAALS': [-10.0, -11.0, -10.5, -10.0, -11.0, -10.5],
                    'EEL': [-200.0, -210.0, -205.0, -200.0, -210.0, -205.0],
                    'EPB': [-50.0, -55.0, -52.0, -50.0, -55.0, -52.0],
                    'ECAVITY': [7.0, 7.5, 7.2, 7.0, 7.5, 7.2],
                    'system': [
                        'receptor',
                        'receptor',
                        'ligand',
                        'ligand',
                        'complex',
                        'complex',
                    ],
                }
            )

            analyzer.n_frames = 2
            analyzer.square_root_N = np.sqrt(2)
            analyzer.contributions = {
                'G gas': ['VDWAALS', 'EEL'],
                'G solv': ['EGB', 'ESURF', 'EPB', 'ECAVITY'],
            }

            analyzer.generate_summary()

            stats_file = Path('statistics.json')
            assert stats_file.exists()

            with open(stats_file) as f:
                stats = json.load(f)

            assert 'receptor' in stats
            assert 'ligand' in stats
            assert 'complex' in stats
            assert 'gb' in stats['receptor']
            assert 'pb' in stats['receptor']


class TestMMPBSACalculateSasa:
    """Test suite for MMPBSA.calculate_sasa method."""

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_calculate_sasa_runs_cpptraj(self, mock_subprocess, mock_amberhome):
        """Test calculate_sasa writes script and runs cpptraj."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')
        mock_subprocess.DEVNULL = -1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                amberhome=str(mock_amberhome),
            )

            # Create the sasa.in path so unlink works
            mmpbsa.fh.path / 'sasa.in'

            mmpbsa.calculate_sasa('complex', 'system.prmtop', 'traj.crd')

            # Verify cpptraj was called
            calls = [
                c for c in mock_subprocess.run.call_args_list if 'cpptraj' in str(c)
            ]
            assert len(calls) > 0


class TestVerifyCombinedOutputs:
    """Test suite for MMPBSA._verify_combined_outputs."""

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_verify_missing_files_raises(self, mock_subprocess, mock_amberhome):
        """Test that missing output files raise RuntimeError."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                amberhome=str(mock_amberhome),
            )

            with pytest.raises(RuntimeError, match='Output verification failed'):
                mmpbsa._verify_combined_outputs()

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_verify_empty_files_raises(self, mock_subprocess, mock_amberhome):
        """Test that empty output files raise RuntimeError."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                amberhome=str(mock_amberhome),
            )

            # Create empty files for one system
            for system in ['complex', 'receptor', 'ligand']:
                (mmpbsa.path / f'{system}_surf.dat').write_text('')
                (mmpbsa.path / f'{system}_gb.mdout').write_text('')
                (mmpbsa.path / f'{system}_pb.mdout').write_text('')

            with pytest.raises(RuntimeError, match='Empty files'):
                mmpbsa._verify_combined_outputs()

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_verify_valid_files_passes(self, mock_subprocess, mock_amberhome):
        """Test that valid output files pass verification."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text('mock topology')
            traj_file = path / 'traj.dcd'
            traj_file.write_text('mock trajectory')

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                amberhome=str(mock_amberhome),
            )

            # Create valid files
            for system in ['complex', 'receptor', 'ligand']:
                (mmpbsa.path / f'{system}_surf.dat').write_text(
                    '#Frame SASA\n1 1000.0\n'
                )
                (mmpbsa.path / f'{system}_gb.mdout').write_text(
                    ' BOND = 100.0  ANGLE = 50.0\n'
                )
                (mmpbsa.path / f'{system}_pb.mdout').write_text(
                    ' BOND = 100.0  ANGLE = 50.0\n'
                )

            # Should not raise
            mmpbsa._verify_combined_outputs()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
