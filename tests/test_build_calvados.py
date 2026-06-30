"""
Unit tests for build/build_calvados.py module

Note: These tests mock the calvados dependency at the sys.modules level
since calvados is an optional/difficult dependency.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Skip all tests if calvados is not available, OR mock it before import
# We'll use the mocking approach to test the module logic without calvados
@pytest.fixture(autouse=True)
def mock_calvados():
    """Mock calvados module before any imports"""
    # Create mock calvados module structure
    mock_calvados_cfg = MagicMock()
    mock_calvados_cfg.Config = MagicMock()
    mock_calvados_cfg.Job = MagicMock()
    mock_calvados_cfg.Components = MagicMock()

    mock_calvados = MagicMock()
    mock_calvados.cfg = mock_calvados_cfg

    # Insert into sys.modules before import
    with patch.dict(
        sys.modules,
        {
            'calvados': mock_calvados,
            'calvados.cfg': mock_calvados_cfg,
        },
    ):
        yield mock_calvados_cfg


class TestCGBuilder:
    """Test suite for CGBuilder class"""

    def test_cgbuilder_init(self, mock_calvados):
        """Test CGBuilder initialization with all parameters"""
        # Import after mocking
        from molecular_simulations.build.build_calvados import CGBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'test.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            residues_file = tmpdir / 'residues.csv'
            residues_file.write_text('resname,sigma\nALA,0.5\n')

            domains_file = tmpdir / 'domains.yaml'
            domains_file.write_text('domains: []\n')

            builder = CGBuilder(
                path=tmpdir / 'output',
                input_pdb=pdb_path,
                residues_file=residues_file,
                domains_file=domains_file,
                box_dim=[100.0, 100.0, 100.0],
                temp=310.0,
                ion_conc=0.15,
                pH=7.4,
                topol='center',
                dcd_freq=2000,
                n_steps=1000000,
                platform='CUDA',
                restart='checkpoint',
                frestart='restart.chk',
                verbose=True,
                molecule_type='protein',
                nmol=1,
                restraint=True,
                charge_termini='end-capped',
                restraint_type='harmonic',
                use_com=True,
                colabfold=0,
                k_harmonic=700.0,
            )

            assert builder.path == tmpdir / 'output'
            assert builder.input_pdb == pdb_path
            assert builder.temp == 310.0
            assert builder.ion_conc == 0.15
            assert builder.pH == 7.4
            assert builder.box_dim == [100.0, 100.0, 100.0]
            assert builder.platform == 'CUDA'
            assert builder.nmol == 1
            assert builder.restraint is True
            assert builder.k_harmonic == 700.0

    def test_cgbuilder_from_dict(self, mock_calvados):
        """Test CGBuilder.from_dict classmethod"""
        from molecular_simulations.build.build_calvados import CGBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'test.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            residues_file = tmpdir / 'residues.csv'
            residues_file.write_text('resname,sigma\nALA,0.5\n')

            domains_file = tmpdir / 'domains.yaml'
            domains_file.write_text('domains: []\n')

            cg_params = {
                'config': {
                    'path': str(tmpdir / 'output'),
                    'input_pdb': str(pdb_path),
                    'box_dim': [80.0, 80.0, 80.0],
                    'temp': 300.0,
                    'ion_conc': 0.1,
                    'pH': 7.0,
                    'topol': 'center',
                    'dcd_freq': 1000,
                    'n_steps': 500000,
                    'platform': 'CPU',
                    'restart': 'checkpoint',
                    'frestart': 'restart.chk',
                    'verbose': False,
                },
                'components': {
                    'residues_file': str(residues_file),
                    'domains_file': str(domains_file),
                    'molecule_type': 'protein',
                    'nmol': 2,
                    'restraint': False,
                    'charge_termini': 'both',
                    'restraint_type': 'go',
                    'use_com': False,
                    'colabfold': 1,
                    'k_harmonic': 500.0,
                },
            }

            builder = CGBuilder.from_dict(cg_params)

            assert builder.temp == 300.0
            assert builder.pH == 7.0
            assert builder.nmol == 2
            assert builder.restraint is False
            assert builder.platform == 'CPU'

    def test_write_config(self, mock_calvados):
        """Test write_config method"""
        # Setup the mock to return proper config dict
        mock_config_instance = MagicMock()
        mock_config_instance.config = {
            'sysname': 'test',
            'box': [100.0, 100.0, 100.0],
            'temp': 310.0,
        }
        mock_calvados.Config.return_value = mock_config_instance

        from molecular_simulations.build.build_calvados import CGBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / 'output'
            output_dir.mkdir()

            pdb_path = tmpdir / 'test.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            residues_file = tmpdir / 'residues.csv'
            residues_file.write_text('resname,sigma\nALA,0.5\n')

            domains_file = tmpdir / 'domains.yaml'
            domains_file.write_text('domains: []\n')

            builder = CGBuilder(
                path=output_dir,
                input_pdb=pdb_path,
                residues_file=residues_file,
                domains_file=domains_file,
                box_dim=[100.0, 100.0, 100.0],
            )

            builder.write_config()

            # Check config file was created
            config_file = output_dir / 'config.yaml'
            assert config_file.exists()

    def test_write_components(self, mock_calvados):
        """Test write_components method"""
        mock_components_instance = MagicMock()
        mock_components_instance.components = {'molecules': [{'name': 'test'}]}
        mock_calvados.Components.return_value = mock_components_instance

        from molecular_simulations.build.build_calvados import CGBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / 'output'
            output_dir.mkdir()

            pdb_path = tmpdir / 'test.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            residues_file = tmpdir / 'residues.csv'
            residues_file.write_text('resname,sigma\nALA,0.5\n')

            domains_file = tmpdir / 'domains.yaml'
            domains_file.write_text('domains: []\n')

            builder = CGBuilder(
                path=output_dir,
                input_pdb=pdb_path,
                residues_file=residues_file,
                domains_file=domains_file,
                box_dim=[100.0, 100.0, 100.0],
            )

            builder.write_components()

            # Check components file was created
            components_file = output_dir / 'components.yaml'
            assert components_file.exists()

    def test_build(self, mock_calvados):
        """Test build method calls write_config and write_components"""
        mock_config_instance = MagicMock()
        mock_config_instance.config = {}
        mock_calvados.Config.return_value = mock_config_instance

        mock_components_instance = MagicMock()
        mock_components_instance.components = {}
        mock_calvados.Components.return_value = mock_components_instance

        from molecular_simulations.build.build_calvados import CGBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / 'output'
            output_dir.mkdir()

            pdb_path = tmpdir / 'test.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            residues_file = tmpdir / 'residues.csv'
            residues_file.write_text('resname,sigma\nALA,0.5\n')

            domains_file = tmpdir / 'domains.yaml'
            domains_file.write_text('domains: []\n')

            builder = CGBuilder(
                path=output_dir,
                input_pdb=pdb_path,
                residues_file=residues_file,
                domains_file=domains_file,
                box_dim=[100.0, 100.0, 100.0],
            )

            builder.build()

            # Both files should be created
            assert (output_dir / 'config.yaml').exists()
            assert (output_dir / 'components.yaml').exists()


class TestCGBuilderParameters:
    """Test CGBuilder parameter handling"""

    def test_default_parameters(self, mock_calvados):
        """Test default parameter values"""
        from molecular_simulations.build.build_calvados import CGBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'test.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            residues_file = tmpdir / 'residues.csv'
            residues_file.write_text('resname,sigma\nALA,0.5\n')

            domains_file = tmpdir / 'domains.yaml'
            domains_file.write_text('domains: []\n')

            builder = CGBuilder(
                path=tmpdir,
                input_pdb=pdb_path,
                residues_file=residues_file,
                domains_file=domains_file,
                box_dim=[50.0, 50.0, 50.0],
            )

            # Check defaults
            assert builder.temp == 310.0
            assert builder.ion_conc == 0.15
            assert builder.pH == 7.4
            assert builder.topol == 'center'
            assert builder.dcd_freq == 2000
            assert builder.n_steps == 10_000_000
            assert builder.platform == 'CUDA'
            assert builder.restart == 'checkpoint'
            assert builder.verbose is True
            assert builder.molecule_type == 'protein'
            assert builder.nmol == 1
            assert builder.restraint is True
            assert builder.charge_termini == 'end-capped'
            assert builder.restraint_type == 'harmonic'
            assert builder.use_com is True
            assert builder.colabfold == 0
            assert builder.k_harmonic == 700.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
