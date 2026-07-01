"""Unit tests for simulate/mmpbsa.py module.

These tests run REAL, no-mock coverage of input-file generation, output parsing,
chunk combination, retry/backoff logic and output verification -- everything that
does not require an AMBER binary. The few paths that genuinely need ``cpptraj`` /
``mmpbsa_py_energy`` are skip-gated on ``shutil.which`` and execute only where the
binaries are installed.

The worker functions (``_run_energy_calculation`` / ``_run_sasa_calculation``)
shell out via ``subprocess``; rather than mocking ``subprocess`` we drive their
real logic with the ``true``/``false`` shell builtins (to control return codes)
and with a small on-disk "flaky" helper script (to exercise retry/backoff). No
``subprocess`` is mocked anywhere in this file.
"""

import importlib.util
import json
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

# ============================================================================
# Skip gates for real external-binary execution
# ============================================================================

_have_cpptraj = shutil.which('cpptraj') is not None
_have_mmpbsa_energy = shutil.which('mmpbsa_py_energy') is not None

requires_cpptraj = pytest.mark.skipif(
    not _have_cpptraj, reason='cpptraj binary not available'
)
requires_mmpbsa_energy = pytest.mark.skipif(
    not _have_mmpbsa_energy, reason='mmpbsa_py_energy binary not available'
)


@pytest.fixture
def fake_amberhome(tmp_path):
    """Create a real AMBERHOME-shaped directory tree.

    The binaries inside are placeholders; no-binary tests never execute them
    (the cpptraj/mmpbsa_py_energy paths simply fail harmlessly when run). This
    only satisfies ``MMPBSA``'s requirement that ``amberhome/bin`` exists so the
    object can be constructed.
    """
    amber_bin = tmp_path / 'amber' / 'bin'
    amber_bin.mkdir(parents=True)
    (amber_bin / 'cpptraj').write_text('#!/bin/bash\necho "placeholder cpptraj"')
    (amber_bin / 'mmpbsa_py_energy').write_text('#!/bin/bash\necho "placeholder"')
    return tmp_path / 'amber'


# Realistic two-frame mmpbsa_py_energy output (GB / PB) and a cpptraj molsurf
# SASA table. Parsed for real by OutputAnalyzer below.
SAMPLE_GB_MDOUT = """ BOND  =       10.0  ANGLE   =       20.0  DIHED      =       30.0
 VDWAALS  =      -40.0  EEL     =     -100.0  EGB        =      -25.0
 1-4 VDW  =        5.0  1-4 EEL =       15.0  RESTRAINT  =        0.0
 ESURF =        3.0
 BOND  =       11.0  ANGLE   =       21.0  DIHED      =       31.0
 VDWAALS  =      -41.0  EEL     =     -101.0  EGB        =      -26.0
 1-4 VDW  =        6.0  1-4 EEL =       16.0  RESTRAINT  =        0.0
 ESURF =        3.5
"""

SAMPLE_PB_MDOUT = """ BOND  =       10.0  ANGLE   =       20.0  DIHED      =       30.0
 VDWAALS  =      -40.0  EEL     =     -100.0  EPB        =      -22.0
 1-4 VDW  =        5.0  1-4 EEL =       15.0  RESTRAINT  =        0.0
 ECAVITY =        2.0  EDISPER =       -1.0
 BOND  =       11.0  ANGLE   =       21.0  DIHED      =       31.0
 VDWAALS  =      -41.0  EEL     =     -101.0  EPB        =      -23.0
 1-4 VDW  =        6.0  1-4 EEL =       16.0  RESTRAINT  =        0.0
 ECAVITY =        2.5  EDISPER =       -1.0
"""

SAMPLE_SASA = '#Frame SASA\n1 1000.0\n2 1100.0\n'


def _build_mmpbsa(tmp_path, fake_amberhome, **kwargs):
    """Construct a real MMPBSA object (no mocks).

    With ``n_cpus == 1`` FileHandler does not count or split frames, and the
    cpptraj calls it issues simply fail (binary absent) without raising, so the
    object constructs cleanly in CI without any external binary.
    """
    from molecular_simulations.simulate.mmpbsa import MMPBSA

    top_file = tmp_path / 'system.prmtop'
    top_file.write_text('placeholder topology')
    traj_file = tmp_path / 'traj.dcd'
    traj_file.write_text('placeholder trajectory')

    return MMPBSA(
        top=str(top_file),
        dcd=str(traj_file),
        selections=[':1-100', ':101-150'],
        amberhome=str(fake_amberhome),
        **kwargs,
    )


# ============================================================================
# Pure dataclass / settings logic - no binaries
# ============================================================================


class TestMMPBSASettings:
    """Test suite for MMPBSASettings dataclass."""

    def test_mmpbsa_settings_defaults(self):
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


# ============================================================================
# MMPBSA construction - real, no subprocess mocks
# ============================================================================


class TestMMPBSAInit:
    """Test suite for MMPBSA initialization (real construction)."""

    def test_mmpbsa_init(self, tmp_path, fake_amberhome):
        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome)

        assert mmpbsa.cpptraj == fake_amberhome / 'bin' / 'cpptraj'
        assert mmpbsa.mmpbsa_py_energy == fake_amberhome / 'bin' / 'mmpbsa_py_energy'

    def test_mmpbsa_init_no_amberhome(self, tmp_path, monkeypatch):
        """Without AMBERHOME and no explicit amberhome, init must raise."""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        monkeypatch.delenv('AMBERHOME', raising=False)

        top_file = tmp_path / 'system.prmtop'
        top_file.write_text('placeholder topology')
        traj_file = tmp_path / 'traj.dcd'
        traj_file.write_text('placeholder trajectory')

        with pytest.raises(ValueError, match='AMBERHOME not set'):
            MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
            )

    def test_mmpbsa_init_custom_output(self, tmp_path, fake_amberhome):
        out_dir = tmp_path / 'custom_output'
        out_dir.mkdir()

        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome, out=str(out_dir))

        assert mmpbsa.path.resolve() == out_dir.resolve()

    def test_mmpbsa_run_creates_output_dir(self, tmp_path, fake_amberhome):
        out_dir = tmp_path / 'output' / 'mmpbsa'
        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome, out=str(out_dir))

        assert out_dir.exists()
        assert mmpbsa.path.resolve() == out_dir.resolve()

    def test_mmpbsa_with_kwargs(self, tmp_path, fake_amberhome):
        """Extra kwargs are stored as attributes on the real object."""
        mmpbsa = _build_mmpbsa(
            tmp_path, fake_amberhome, custom_attr='custom_value', another_attr=42
        )

        assert mmpbsa.custom_attr == 'custom_value'
        assert mmpbsa.another_attr == 42


# ============================================================================
# Input-file generation - real content read back from disk
# ============================================================================


class TestMMPBSAWriteMdins:
    """write_mdins writes real GB/PB config files."""

    def test_write_mdins(self, tmp_path, fake_amberhome):
        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome)

        gb_mdin, pb_mdin = mmpbsa.write_mdins()

        assert gb_mdin.exists()
        assert pb_mdin.exists()

        gb_content = gb_mdin.read_text()
        assert gb_content.splitlines()[0] == 'GB'
        assert 'igb = 2' in gb_content
        assert 'extdiel = 78.3' in gb_content
        assert 'saltcon = 0.10' in gb_content
        assert f'surften = {mmpbsa.gb_surften}' in gb_content
        assert 'rgbmax = 25.0' in gb_content

        pb_content = pb_mdin.read_text()
        assert pb_content.splitlines()[0] == 'PB'
        assert 'inp = 2' in pb_content
        assert 'epsin = 1.0' in pb_content
        assert 'epsout = 80.0' in pb_content
        assert 'cavity_surften = 0.0378' in pb_content


class TestWriteSasaScript:
    """_write_sasa_script writes a real cpptraj molsurf script."""

    def test_write_sasa_script(self, tmp_path, fake_amberhome):
        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome)

        prefix = mmpbsa.path / 'complex'
        prm = mmpbsa.path / 'complex.prmtop'
        trj = mmpbsa.path / 'complex_chunk0.crd'

        result = mmpbsa._write_sasa_script(prefix, prm, trj, chunk_idx=0)

        assert result.exists()
        assert result.name == 'sasa_complex_chunk0.in'
        content = result.read_text()
        assert f'parm {prm}' in content
        assert f'trajin {trj}' in content
        assert 'molsurf :*' in content
        assert 'complex_chunk0_surf.dat' in content
        assert f'probe {mmpbsa.solvent_probe}' in content
        assert f'offset {mmpbsa.offset}' in content


# ============================================================================
# Chunk combination - real file I/O
# ============================================================================


class TestCombineChunks:
    """Real chunk-combination file logic."""

    def test_combine_sasa_chunks(self, tmp_path, fake_amberhome):
        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome)
        path = mmpbsa.path

        for system in ['complex', 'receptor', 'ligand']:
            (path / f'{system}_chunk0_surf.dat').write_text(
                '#Frame SASA\n1 1000.0\n2 1050.0\n'
            )
            (path / f'{system}_chunk1_surf.dat').write_text(
                '#Frame SASA\n3 1100.0\n4 1150.0\n'
            )

        mmpbsa._combine_sasa_chunks()

        for system in ['complex', 'receptor', 'ligand']:
            combined = path / f'{system}_surf.dat'
            assert combined.exists()
            content = combined.read_text()
            # Header kept once, second chunk's header dropped -> 4 data rows.
            assert content.count('#Frame SASA') == 1
            data_rows = [
                ln for ln in content.splitlines() if ln and not ln.startswith('#')
            ]
            assert len(data_rows) == 4
            assert '1000.0' in content and '1150.0' in content
            # Chunk files removed after combination.
            assert not (path / f'{system}_chunk0_surf.dat').exists()
            assert not (path / f'{system}_chunk1_surf.dat').exists()

    def test_combine_energy_chunks(self, tmp_path, fake_amberhome):
        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome)
        path = mmpbsa.path

        for system in ['complex', 'receptor', 'ligand']:
            for level in ['gb', 'pb']:
                (path / f'{system}_chunk0_{level}.mdout').write_text(' BOND = 100.0\n')
                (path / f'{system}_chunk1_{level}.mdout').write_text(' BOND = 110.0\n')

        mmpbsa._combine_energy_chunks()

        for system in ['complex', 'receptor', 'ligand']:
            for level in ['gb', 'pb']:
                combined = path / f'{system}_{level}.mdout'
                assert combined.exists()
                content = combined.read_text()
                assert 'BOND = 100.0' in content
                assert 'BOND = 110.0' in content
                assert not (path / f'{system}_chunk0_{level}.mdout').exists()


# ============================================================================
# Output verification - real file inspection
# ============================================================================


class TestVerifyCombinedOutputs:
    """_verify_combined_outputs checks real files on disk."""

    def test_verify_missing_files_raises(self, tmp_path, fake_amberhome):
        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome)

        with pytest.raises(RuntimeError, match='Output verification failed'):
            mmpbsa._verify_combined_outputs()

    def test_verify_empty_files_raises(self, tmp_path, fake_amberhome):
        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome)

        for system in ['complex', 'receptor', 'ligand']:
            (mmpbsa.path / f'{system}_surf.dat').write_text('')
            (mmpbsa.path / f'{system}_gb.mdout').write_text('')
            (mmpbsa.path / f'{system}_pb.mdout').write_text('')

        with pytest.raises(RuntimeError, match='Empty files'):
            mmpbsa._verify_combined_outputs()

    def test_verify_valid_files_passes(self, tmp_path, fake_amberhome):
        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome)

        for system in ['complex', 'receptor', 'ligand']:
            (mmpbsa.path / f'{system}_surf.dat').write_text('#Frame SASA\n1 1000.0\n')
            (mmpbsa.path / f'{system}_gb.mdout').write_text(
                ' BOND = 100.0  ANGLE = 50.0\n'
            )
            (mmpbsa.path / f'{system}_pb.mdout').write_text(
                ' BOND = 100.0  ANGLE = 50.0\n'
            )

        # Should not raise.
        mmpbsa._verify_combined_outputs()

    def test_verify_sasa_no_data_lines_raises(self, tmp_path, fake_amberhome):
        """A non-empty SASA file that holds only comment lines is invalid."""
        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome)

        for system in ['complex', 'receptor', 'ligand']:
            # Non-empty but comment-only -> zero data lines (branch at line ~669).
            (mmpbsa.path / f'{system}_surf.dat').write_text('#Frame SASA\n#comment\n')
            # Keep the energy files valid so only the SASA branch fires.
            (mmpbsa.path / f'{system}_gb.mdout').write_text(' BOND = 100.0\n')
            (mmpbsa.path / f'{system}_pb.mdout').write_text(' BOND = 100.0\n')

        with pytest.raises(RuntimeError) as excinfo:
            mmpbsa._verify_combined_outputs()

        msg = str(excinfo.value)
        assert 'Output verification failed' in msg
        assert 'Invalid files' in msg
        assert 'no data lines' in msg

    def test_verify_energy_no_bond_raises(self, tmp_path, fake_amberhome):
        """A non-empty energy file without a ' BOND' record is invalid."""
        mmpbsa = _build_mmpbsa(tmp_path, fake_amberhome)

        for system in ['complex', 'receptor', 'ligand']:
            (mmpbsa.path / f'{system}_surf.dat').write_text('#Frame SASA\n1 1000.0\n')
            # No ' BOND' token anywhere -> invalid (branch at line ~681).
            (mmpbsa.path / f'{system}_gb.mdout').write_text('ANGLE = 50.0\n')
            (mmpbsa.path / f'{system}_pb.mdout').write_text('ANGLE = 50.0\n')

        with pytest.raises(RuntimeError) as excinfo:
            mmpbsa._verify_combined_outputs()

        msg = str(excinfo.value)
        assert 'Invalid files' in msg
        assert 'no energy data' in msg


class TestRunFrameParallelFailures:
    """Real failure-aggregation logic in _run_frame_parallel.

    With no real binaries the SASA workers cannot produce output, so run()
    must raise RuntimeError reporting the SASA failures. Built with
    ``max_retries=1`` so the (real) retry loop makes a single real attempt and
    takes no backoff -- nothing is patched.
    """

    def test_sasa_failure_raises(self, tmp_path, fake_amberhome):
        mmpbsa = _build_mmpbsa(
            tmp_path, fake_amberhome, parallel_mode='frame', max_retries=1
        )

        with pytest.raises(RuntimeError, match='SASA calculations failed'):
            mmpbsa.run()


# ============================================================================
# OutputAnalyzer - real parsing
# ============================================================================


class TestOutputAnalyzer:
    """OutputAnalyzer construction and primitive parsers."""

    def test_output_analyzer_init(self, tmp_path):
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        analyzer = OutputAnalyzer(
            path=str(tmp_path), surface_tension=0.01, sasa_offset=0.5
        )

        assert analyzer.surften == 0.01
        assert analyzer.offset == 0.5
        assert analyzer.free_energy is None
        assert analyzer.systems == ['receptor', 'ligand', 'complex']
        assert analyzer.levels == ['gb', 'pb']

    def test_parse_line_single_value(self):
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        result = list(OutputAnalyzer.parse_line(' BOND = 123.456'))

        assert len(result) == 1
        assert result[0][0] == 'BOND'
        assert np.isclose(result[0][1], 123.456)

    def test_parse_line_multiple_values(self):
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        result = list(OutputAnalyzer.parse_line(' BOND =  123.456  ANGLE =  78.901'))

        result_dict = dict(result)
        assert np.isclose(result_dict['BOND'], 123.456)
        assert np.isclose(result_dict['ANGLE'], 78.901)

    def test_parse_line_energy_format(self):
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        line = ' VDWAALS = -10.5  EEL = -200.3  EGB = -50.1'
        result_dict = dict(OutputAnalyzer.parse_line(line))

        assert np.isclose(result_dict['VDWAALS'], -10.5)
        assert np.isclose(result_dict['EEL'], -200.3)
        assert np.isclose(result_dict['EGB'], -50.1)

    def test_read_sasa(self, tmp_path):
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        sasa_file = tmp_path / 'test_surf.dat'
        sasa_file.write_text('#Frame SASA\n1 1000.0\n2 1100.0\n3 1050.0\n')

        analyzer = OutputAnalyzer(path=str(tmp_path), surface_tension=0.0072)
        sasa = analyzer.read_sasa(sasa_file)

        assert len(sasa) == 3
        # Rescaled by surface tension (0.0072) plus zero offset.
        assert np.isclose(sasa[0], 7.2)
        assert np.isclose(sasa[1], 7.92)
        assert np.isclose(sasa[2], 7.56)

    def test_read_sasa_with_offset(self, tmp_path):
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        sasa_file = tmp_path / 'test_surf.dat'
        sasa_file.write_text('#Frame SASA\n1 1000.0\n')

        analyzer = OutputAnalyzer(
            path=str(tmp_path), surface_tension=0.0072, sasa_offset=2.0
        )
        sasa = analyzer.read_sasa(sasa_file)

        # 1000.0 * 0.0072 + 2.0 = 9.2
        assert np.isclose(sasa[0], 9.2)


class TestOutputAnalyzerParseOutputs:
    """Full, real parse_outputs over realistic sample mdout/SASA files."""

    def test_parse_outputs_extracts_real_energies(self, tmp_path, monkeypatch):
        import polars as pl

        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        # generate_summary writes statistics.json into CWD.
        monkeypatch.chdir(tmp_path)

        for system in ['complex', 'receptor', 'ligand']:
            (tmp_path / f'{system}_gb.mdout').write_text(SAMPLE_GB_MDOUT)
            (tmp_path / f'{system}_pb.mdout').write_text(SAMPLE_PB_MDOUT)
            (tmp_path / f'{system}_surf.dat').write_text(SAMPLE_SASA)

        analyzer = OutputAnalyzer(path=str(tmp_path), surface_tension=0.0072)
        analyzer.parse_outputs()

        # Two frames per system, three systems -> 6 rows in each table.
        assert analyzer.gb.shape[0] == 6
        assert analyzer.pb.shape[0] == 6

        # Real values parsed from the GB sample.
        receptor_gb = analyzer.gb.filter(pl.col('system') == 'receptor')
        assert receptor_gb['BOND'].to_list() == [10.0, 11.0]
        assert receptor_gb['EGB'].to_list() == [-25.0, -26.0]
        # SASA rescaled by surface tension: 1000*0.0072, 1100*0.0072.
        assert np.allclose(receptor_gb['ESURF'].to_list(), [7.2, 7.92])

        # PB-specific term parsed.
        receptor_pb = analyzer.pb.filter(pl.col('system') == 'receptor')
        assert receptor_pb['EPB'].to_list() == [-22.0, -23.0]

        # Binding free energy populated and summary/report files written for real.
        assert analyzer.free_energy is not None
        assert np.allclose(analyzer.free_energy, [78.75, 1.25])
        assert (tmp_path / 'statistics.json').exists()
        assert (tmp_path / 'deltaG.txt').exists()


class TestOutputAnalyzerGenerateSummary:
    """generate_summary writes real statistics.json."""

    def test_generate_summary_creates_json(self, monkeypatch, tmp_path):
        import polars as pl

        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        monkeypatch.chdir(tmp_path)

        analyzer = OutputAnalyzer(path=str(tmp_path), surface_tension=0.0072)

        analyzer.gb = pl.DataFrame(
            {
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
        )
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

        for system in ('receptor', 'ligand', 'complex'):
            assert system in stats
            assert 'gb' in stats[system]
            assert 'pb' in stats[system]

    def test_generate_summary_computes_real_statistics(self, monkeypatch, tmp_path):
        """Assert the real numeric means/errors written into statistics.json."""
        import polars as pl

        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        monkeypatch.chdir(tmp_path)

        analyzer = OutputAnalyzer(path=str(tmp_path), surface_tension=0.0072)
        # Two frames per system, constant within a system -> std == 0.
        analyzer.gb = pl.DataFrame(
            {
                'VDWAALS': [-10.0, -10.0, -5.0, -5.0, -20.0, -20.0],
                'EEL': [-100.0, -100.0, -50.0, -50.0, -160.0, -160.0],
                'EGB': [-20.0, -20.0, -10.0, -10.0, -35.0, -35.0],
                'ESURF': [5.0, 5.0, 3.0, 3.0, 9.0, 9.0],
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
        analyzer.pb = analyzer.gb.rename({'EGB': 'EPB', 'ESURF': 'ECAVITY'})
        analyzer.n_frames = 2
        analyzer.square_root_N = np.sqrt(2)
        analyzer.contributions = {
            'G gas': ['VDWAALS', 'EEL'],
            'G solv': ['EGB', 'ESURF', 'EPB', 'ECAVITY'],
        }

        analyzer.generate_summary()

        with open('statistics.json') as f:
            stats = json.load(f)

        rec_gb = stats['receptor']['gb']
        # Column means/stdevs are the real per-frame statistics.
        assert np.isclose(rec_gb['VDWAALS']['mean'], -10.0)
        assert np.isclose(rec_gb['VDWAALS']['std'], 0.0)
        assert np.isclose(rec_gb['VDWAALS']['err'], 0.0)
        assert np.isclose(rec_gb['EEL']['mean'], -100.0)
        # G gas pools VDWAALS + EEL over both frames -> mean of [-10,-10,-100,-100].
        assert np.isclose(rec_gb['G gas']['mean'], -55.0)
        # G solv pools EGB + ESURF -> mean of [-20,-20,5,5].
        assert np.isclose(rec_gb['G solv']['mean'], -7.5)
        # 'total' pools every energy column for the system.
        assert np.isclose(
            rec_gb['total']['mean'], np.mean([-10, -10, -100, -100, -20, -20, 5, 5])
        )


class TestOutputAnalyzerCheckBondedTerms:
    """check_bonded_terms enforces receptor + ligand == complex on bonded terms."""

    def _analyzer(self, tmp_path):
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        return OutputAnalyzer(path=str(tmp_path), surface_tension=0.0072, log=False)

    @staticmethod
    def _consistent_df():
        import polars as pl

        # complex bonded == receptor + ligand for every bonded column.
        return pl.DataFrame(
            {
                'BOND': [10.0, 5.0, 15.0],
                'ANGLE': [4.0, 2.0, 6.0],
                'DIHED': [3.0, 1.0, 4.0],
                '1-4 VDW': [2.0, 1.0, 3.0],
                '1-4 EEL': [8.0, 2.0, 10.0],
                'VDWAALS': [-10.0, -5.0, -16.0],
                'EEL': [-100.0, -50.0, -160.0],
                'EGB': [-20.0, -10.0, -35.0],
                'ESURF': [5.0, 3.0, 9.0],
                'RESTRAINT': [0.0, 0.0, 0.0],
                'system': ['receptor', 'ligand', 'complex'],
            }
        )

    def test_consistent_bonded_terms_pass(self, tmp_path):
        analyzer = self._analyzer(tmp_path)
        analyzer.gb = self._consistent_df()
        analyzer.pb = self._consistent_df()

        # Must not raise; also computes derived frame counts.
        analyzer.check_bonded_terms()

        assert analyzer.n_frames == analyzer.gb.height
        assert np.isclose(analyzer.square_root_N, np.sqrt(analyzer.n_frames))
        # RESTRAINT is stripped by check_bonded_terms.
        assert 'RESTRAINT' not in analyzer.gb.columns
        assert 'RESTRAINT' not in analyzer.pb.columns

    def test_inconsistent_bonded_terms_raise(self, tmp_path):
        import polars as pl

        analyzer = self._analyzer(tmp_path)
        bad = self._consistent_df().with_columns(
            pl.Series('BOND', [10.0, 5.0, 100.0])  # complex 100 != 10 + 5
        )
        analyzer.gb = bad
        analyzer.pb = self._consistent_df()

        with pytest.raises(
            ValueError, match=r'Bonded terms for receptor \+ ligand != complex!'
        ):
            analyzer.check_bonded_terms()


class TestOutputAnalyzerComputeDG:
    """compute_dG forms ΔG = complex - receptor - ligand and writes deltaG.txt."""

    @staticmethod
    def _make_analyzer(tmp_path):
        import polars as pl

        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        analyzer = OutputAnalyzer(path=str(tmp_path), surface_tension=0.0072, log=False)
        # Constant-per-system, two frames each; systems ordered
        # receptor, ligand, complex to match analyzer.systems.
        analyzer.gb = pl.DataFrame(
            {
                'VDWAALS': [-10.0, -10.0, -5.0, -5.0, -20.0, -20.0],
                'EEL': [-100.0, -100.0, -50.0, -50.0, -160.0, -160.0],
                'EGB': [-20.0, -20.0, -10.0, -10.0, -35.0, -35.0],
                'ESURF': [5.0, 5.0, 3.0, 3.0, 9.0, 9.0],
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
        analyzer.pb = pl.DataFrame(
            {
                'VDWAALS': [-10.0, -10.0, -5.0, -5.0, -20.0, -20.0],
                'EEL': [-100.0, -100.0, -50.0, -50.0, -160.0, -160.0],
                'EPB': [-18.0, -18.0, -9.0, -9.0, -30.0, -30.0],
                'ECAVITY': [4.0, 4.0, 2.0, 2.0, 7.0, 7.0],
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
        analyzer.square_root_N = np.sqrt(2)
        analyzer.contributions = {
            'G gas': ['VDWAALS', 'EEL'],
            'G solv': ['EGB', 'ESURF', 'EPB', 'ECAVITY'],
        }
        return analyzer

    def test_compute_dG_binding_free_energy(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path)

        analyzer.compute_dG()

        # PB ∆G Binding = sum over columns of (complex - receptor - ligand):
        #   VDWAALS: -20-(-10)-(-5) = -5
        #   EEL:     -160-(-100)-(-50) = -10
        #   EPB:     -30-(-18)-(-9)  = -3
        #   ECAVITY:  7-4-2 = 1
        #   total = -17; std = 0 across identical frames.
        assert analyzer.free_energy is not None
        assert np.allclose(analyzer.free_energy, [-17.0, 0.0])

        # deltaG.txt written for real into the analyzer's path.
        report = (tmp_path / 'deltaG.txt').read_text()
        assert 'Generalized Born' in report
        assert 'Poisson Boltzmann' in report
        # GB ΔG Binding = -5 + -10 + -5 + 1 = -19.
        assert '∆G Binding          -19.000' in report
        assert '∆G Binding          -17.000' in report
        # Per-component GB solvation ΔG (EGB -5, ESURF +1) => G solv -4.
        assert 'G solv              -4.000' in report


class TestOutputAnalyzerPrettyPrint:
    """pretty_print formats the energy table and prints it when logging is off."""

    def test_pretty_print_table_to_stdout(self, tmp_path, capsys):
        import polars as pl

        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        analyzer = OutputAnalyzer(path=str(tmp_path), surface_tension=0.0072, log=False)

        # Each column holds exactly [mean, std, err] as compute_dG produces.
        gb = pl.DataFrame(
            {
                'VDWAALS': [-5.0, 0.5, 0.1],
                'EEL': [-10.0, 1.0, 0.2],
                'G gas': [-15.0, 1.1, 0.22],
                '∆G Binding': [-19.0, 1.2, 0.24],
            }
        )
        pb = pl.DataFrame(
            {
                'VDWAALS': [-5.0, 0.5, 0.1],
                'EEL': [-10.0, 1.0, 0.2],
                'G gas': [-15.0, 1.1, 0.22],
                '∆G Binding': [-17.0, 1.3, 0.26],
            }
        )

        analyzer.pretty_print([gb, pb])

        out = capsys.readouterr().out
        assert '=== Generalized Born  ===' in out
        assert '=== Poisson Boltzmann ===' in out
        assert (
            'Energy Component    Average         Std. Dev.       Std. Err. of Mean'
            in out
        )
        # Formatted numeric rows: value<16.3f columns.
        assert 'VDWAALS             -5.000          0.500           0.100' in out
        assert '∆G Binding          -19.000         1.200           0.240' in out
        assert '∆G Binding          -17.000         1.300           0.260' in out

        # PB ∆G Binding captured into free_energy; file also written.
        assert np.allclose(analyzer.free_energy, [-17.0, 1.3])
        assert (tmp_path / 'deltaG.txt').exists()


# ============================================================================
# FileHandler - real file writing and construction
# ============================================================================


class TestFileHandlerWriteFile:
    """FileHandler.write_file static method."""

    def test_write_file_list(self, tmp_path):
        from molecular_simulations.simulate.mmpbsa import FileHandler

        outfile = tmp_path / 'test.txt'
        FileHandler.write_file(['line1', 'line2', 'line3'], outfile)

        assert outfile.read_text() == 'line1\nline2\nline3'

    def test_write_file_string(self, tmp_path):
        from molecular_simulations.simulate.mmpbsa import FileHandler

        outfile = tmp_path / 'test.txt'
        FileHandler.write_file('single string content', outfile)

        assert outfile.read_text() == 'single string content'

    def test_write_file_with_path_object(self, tmp_path):
        from molecular_simulations.simulate.mmpbsa import FileHandler

        outfile = tmp_path / 'path_test.txt'
        FileHandler.write_file(['path', 'object', 'test'], outfile)

        assert outfile.exists()
        assert outfile.read_text() == 'path\nobject\ntest'


class TestFileHandlerInit:
    """FileHandler construction (n_chunks == 1 path needs no binary)."""

    def test_filehandler_init_single_chunk(self, tmp_path):
        from molecular_simulations.simulate.mmpbsa import FileHandler

        top_file = tmp_path / 'system.prmtop'
        top_file.write_text('placeholder')
        traj_file = tmp_path / 'traj.dcd'
        traj_file.write_text('placeholder')

        fh = FileHandler(
            top=top_file,
            traj=traj_file,
            path=tmp_path,
            sels=[':1-100', ':101-150'],
            first=0,
            last=-1,
            stride=1,
            cpptraj_binary='/nonexistent/cpptraj',
            n_chunks=1,
        )

        assert fh.path == tmp_path
        # Sub-topology / trajectory / pdb names are derived deterministically.
        assert [p.name for p in fh.topologies] == [
            'complex.prmtop',
            'receptor.prmtop',
            'ligand.prmtop',
        ]
        assert [p.name for p in fh.trajectories] == [
            'complex.crd',
            'receptor.crd',
            'ligand.crd',
        ]
        # Single-chunk mapping: one trajectory per system.
        for system in ['complex', 'receptor', 'ligand']:
            assert len(fh.trajectory_chunks[system]) == 1

    def test_filehandler_files_property(self, tmp_path):
        from molecular_simulations.simulate.mmpbsa import FileHandler

        top_file = tmp_path / 'system.prmtop'
        top_file.write_text('placeholder')
        traj_file = tmp_path / 'traj.dcd'
        traj_file.write_text('placeholder')

        fh = FileHandler(
            top=top_file,
            traj=traj_file,
            path=tmp_path,
            sels=[':1-100', ':101-150'],
            first=0,
            last=-1,
            stride=1,
            cpptraj_binary='/nonexistent/cpptraj',
            n_chunks=1,
        )

        files = list(fh.files)
        assert len(files) == 3
        prefixes = [f[0].name for f in files]
        assert prefixes == ['complex', 'receptor', 'ligand']


# ============================================================================
# Worker functions - REAL subprocess via true/false builtins and a flaky script
# ============================================================================


class TestRunEnergyCalculation:
    """_run_energy_calculation worker, driven by real shell commands."""

    def test_successful_energy_calculation(self, tmp_path):
        """Output exists with energy data -> success (binary 'true', rc 0)."""
        from molecular_simulations.simulate.mmpbsa import _run_energy_calculation

        out_file = tmp_path / 'test_gb.mdout'
        out_file.write_text(' BOND = 100.0  ANGLE = 50.0\n')

        result = _run_energy_calculation(
            (
                'true',
                'gb.mdin',
                'sys.prmtop',
                'sys.pdb',
                'traj.crd',
                'test_gb.mdout',
                str(tmp_path),
            ),
            max_retries=1,
        )

        assert result[1] is True
        assert result[2] == ''

    def test_failed_energy_calculation_missing_output(self, tmp_path):
        """Successful return code but no output file -> failure."""
        from molecular_simulations.simulate.mmpbsa import _run_energy_calculation

        result = _run_energy_calculation(
            (
                'true',
                'gb.mdin',
                'sys.prmtop',
                'sys.pdb',
                'traj.crd',
                'missing_output.mdout',
                str(tmp_path),
            ),
            max_retries=1,
        )

        assert result[1] is False
        assert 'missing or empty' in result[2]

    def test_failed_energy_nonzero_return_code(self, tmp_path):
        """Nonzero return code and no energy data -> failure reports rc."""
        from molecular_simulations.simulate.mmpbsa import _run_energy_calculation

        out_file = tmp_path / 'test_gb.mdout'
        out_file.write_text('no energy here\n')

        result = _run_energy_calculation(
            (
                'false',
                'gb.mdin',
                'sys.prmtop',
                'sys.pdb',
                'traj.crd',
                'test_gb.mdout',
                str(tmp_path),
            ),
            max_retries=1,
        )

        assert result[1] is False
        assert 'Return code' in result[2]

    def test_retry_then_success(self, tmp_path):
        """Real retry/backoff: a flaky helper produces output on its 2nd run.

        The helper increments an on-disk counter and only writes the energy
        output on the second invocation, so the worker must retry once before
        succeeding. The real 2**attempt backoff (a single 1 s sleep here) runs
        for real -- no patching of time.sleep.
        """
        from molecular_simulations.simulate.mmpbsa import _run_energy_calculation

        helper = tmp_path / 'flaky_energy.py'
        helper.write_text(
            textwrap.dedent(
                """
                import sys
                from pathlib import Path
                args = sys.argv[1:]
                out = args[args.index("-o") + 1]
                counter = Path("attempts.txt")
                n = (int(counter.read_text()) if counter.exists() else 0) + 1
                counter.write_text(str(n))
                if n >= 2:
                    Path(out).write_text(" BOND = 9.0  ANGLE = 1.0\\n")
                sys.exit(0)
                """
            )
        )

        binary = f'{sys.executable} {helper}'
        result = _run_energy_calculation(
            (
                binary,
                'gb.mdin',
                'sys.prmtop',
                'sys.pdb',
                'traj.crd',
                'out.mdout',
                str(tmp_path),
            ),
            max_retries=3,
        )

        assert result[1] is True
        # Helper was invoked twice -> exactly one real retry occurred.
        assert (tmp_path / 'attempts.txt').read_text() == '2'
        assert (tmp_path / 'out.mdout').read_text().startswith(' BOND = 9.0')


class TestRunSasaCalculation:
    """_run_sasa_calculation worker, driven by real shell commands."""

    def test_successful_sasa_calculation(self, tmp_path):
        from molecular_simulations.simulate.mmpbsa import _run_sasa_calculation

        sasa_script = tmp_path / 'sasa.in'
        sasa_script.write_text(
            'parm sys.prmtop\ntrajin traj.crd\n'
            'molsurf :* out test_surf.dat probe 1.4 offset 0.0\nrun\nquit\n'
        )
        (tmp_path / 'test_surf.dat').write_text('#Frame SASA\n1 1000.0\n2 1100.0\n')

        result = _run_sasa_calculation(
            ('true', str(sasa_script), str(tmp_path)), max_retries=1
        )

        assert result[1] is True
        assert result[2] == ''

    def test_sasa_calculation_empty_output(self, tmp_path):
        from molecular_simulations.simulate.mmpbsa import _run_sasa_calculation

        sasa_script = tmp_path / 'sasa.in'
        sasa_script.write_text('molsurf :* out test_surf.dat probe 1.4 offset 0.0\n')
        (tmp_path / 'test_surf.dat').write_text('')

        result = _run_sasa_calculation(
            ('true', str(sasa_script), str(tmp_path)), max_retries=1
        )

        assert result[1] is False
        assert 'empty' in result[2]


# ============================================================================
# Skip-gated REAL external-binary execution
# ============================================================================


class TestRealBinaryExecution:
    """Real cpptraj / mmpbsa_py_energy runs, skipped where binaries are absent."""

    @requires_cpptraj
    def test_real_cpptraj_molsurf(self, real_amber_system_files):
        """Run real cpptraj molsurf SASA on a real AMBER prmtop + trajectory."""
        from molecular_simulations.simulate.mmpbsa import _run_sasa_calculation

        path = real_amber_system_files['path']
        prmtop = real_amber_system_files['prmtop']
        dcd = real_amber_system_files['dcd']

        script = path / 'sasa.in'
        script.write_text(
            f'parm {prmtop}\n'
            f'trajin {dcd}\n'
            'molsurf :* out surf.dat probe 1.4 offset 0.0\n'
            'run\nquit\n'
        )

        result = _run_sasa_calculation(
            ('cpptraj', str(script), str(path)), max_retries=1
        )

        assert result[1] is True, result[2]
        surf = path / 'surf.dat'
        assert surf.exists()
        data = [
            ln
            for ln in surf.read_text().splitlines()
            if ln.strip() and not ln.startswith('#')
        ]
        assert len(data) > 0

    @requires_mmpbsa_energy
    def test_real_mmpbsa_py_energy_gb(self, real_amber_system_files):
        """Run real mmpbsa_py_energy (GB) on a real vacuum AMBER system."""
        from molecular_simulations.simulate.mmpbsa import _run_energy_calculation

        path = real_amber_system_files['path']
        prmtop = real_amber_system_files['prmtop']
        pdb = real_amber_system_files['pdb']
        dcd = real_amber_system_files['dcd']

        gb_mdin = path / 'gb_mdin'
        gb_mdin.write_text(
            'GB\nigb = 2\nextdiel = 78.3\nsaltcon = 0.10\n'
            'surften = 0.0072\nrgbmax = 25.0\n'
        )

        result = _run_energy_calculation(
            (
                'mmpbsa_py_energy',
                str(gb_mdin),
                str(prmtop),
                str(pdb),
                str(dcd),
                'complex_gb.mdout',
                str(path),
            ),
            max_retries=1,
        )

        assert result[1] is True, result[2]
        assert ' BOND' in (path / 'complex_gb.mdout').read_text()


# ============================================================================
# Sanity: calvados has nothing to do with this module; ensure no import gate.
# ============================================================================


def test_module_imports_without_amber_binaries():
    """The module imports cleanly in CI (no AMBER binary required to import)."""
    import molecular_simulations.simulate.mmpbsa as m

    assert hasattr(m, 'MMPBSA')
    assert hasattr(m, 'OutputAnalyzer')
    # calvados is unrelated to this module.
    _ = importlib.util.find_spec  # keep import referenced


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
