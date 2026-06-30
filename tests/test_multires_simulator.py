"""
Unit tests for simulate/multires_simulator.py module

Note: These tests mock calvados and cg2all dependencies at the sys.modules level
since these are optional/difficult dependencies.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Create comprehensive mocks for all difficult dependencies
@pytest.fixture(autouse=True)
def mock_difficult_dependencies():
    """Mock calvados, cg2all, and related dependencies before any imports"""

    # Mock calvados
    mock_calvados = MagicMock()
    mock_calvados.sim = MagicMock()
    mock_calvados_cfg = MagicMock()
    mock_calvados_cfg.Config = MagicMock()
    mock_calvados_cfg.Job = MagicMock()
    mock_calvados_cfg.Components = MagicMock()
    mock_calvados.cfg = mock_calvados_cfg

    # Mock cg2all
    mock_cg2all = MagicMock()
    mock_cg2all_script = MagicMock()
    mock_cg2all_convert = MagicMock()
    mock_cg2all_script.convert_cg2all = mock_cg2all_convert
    mock_cg2all_convert.main = MagicMock()

    # Mock parmed
    mock_parmed = MagicMock()
    mock_parmed.openmm = MagicMock()
    mock_parmed.openmm.load_topology = MagicMock()

    # Insert mocks into sys.modules
    with patch.dict(
        sys.modules,
        {
            'calvados': mock_calvados,
            'calvados.cfg': mock_calvados_cfg,
            'calvados.sim': mock_calvados.sim,
            'cg2all': mock_cg2all,
            'cg2all.script': mock_cg2all_script,
            'cg2all.script.convert_cg2all': mock_cg2all_convert,
            'parmed': mock_parmed,
        },
    ):
        yield {
            'calvados': mock_calvados,
            'cg2all': mock_cg2all,
            'parmed': mock_parmed,
        }


class TestSanderMinDefaults:
    """Test suite for sander_min_defaults dataclass"""

    def test_sander_min_defaults_values(self, mock_difficult_dependencies):
        """Test default values for sander minimization"""
        from molecular_simulations.simulate.multires_simulator import (
            SanderMinDefaults,
        )

        defaults = SanderMinDefaults()

        assert defaults.imin == 1
        assert defaults.maxcyc == 5000
        assert defaults.ncyc == 2500
        assert defaults.ntb == 0
        assert defaults.ntr == 0
        assert defaults.cut == 10.0
        assert defaults.ntpr == 10000
        assert defaults.ntwr == 5000
        assert defaults.ntxo == 1

    def test_sander_min_defaults_mdin_contents(self, mock_difficult_dependencies):
        """Test that mdin_contents is properly formatted"""
        from molecular_simulations.simulate.multires_simulator import (
            SanderMinDefaults,
        )

        defaults = SanderMinDefaults()

        assert 'Minimization input' in defaults.mdin_contents
        assert 'imin=1' in defaults.mdin_contents
        assert 'maxcyc=5000' in defaults.mdin_contents
        assert 'ncyc=2500' in defaults.mdin_contents
        assert 'ntb=0' in defaults.mdin_contents
        assert 'cut=10.0' in defaults.mdin_contents
        assert '&cntrl' in defaults.mdin_contents


class TestSanderMinimize:
    """Test suite for sander_minimize function"""

    def test_sander_minimize_success(self, mock_difficult_dependencies):
        """Test sander_minimize runs successfully"""
        from molecular_simulations.simulate.multires_simulator import sander_minimize

        with patch(
            'molecular_simulations.simulate.multires_simulator.subprocess'
        ) as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(returncode=0)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Create dummy files
                (tmpdir / 'system.inpcrd').write_text('coords')
                (tmpdir / 'system.prmtop').write_text('topology')

                sander_minimize(
                    path=tmpdir,
                    inpcrd_file='system.inpcrd',
                    prmtop_file='system.prmtop',
                    sander_cmd='sander',
                )

                # Should have called subprocess.run
                mock_subprocess.run.assert_called_once()

    def test_sander_minimize_failure(self, mock_difficult_dependencies):
        """Test sander_minimize raises error on failure"""
        from molecular_simulations.simulate.multires_simulator import sander_minimize

        with patch(
            'molecular_simulations.simulate.multires_simulator.subprocess'
        ) as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(
                returncode=1, stderr='Error message', stdout='Output'
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                (tmpdir / 'system.inpcrd').write_text('coords')
                (tmpdir / 'system.prmtop').write_text('topology')

                with pytest.raises(RuntimeError, match='sander error'):
                    sander_minimize(
                        path=tmpdir,
                        inpcrd_file='system.inpcrd',
                        prmtop_file='system.prmtop',
                        sander_cmd='sander',
                    )


class TestMultiResolutionSimulator:
    """Test suite for MultiResolutionSimulator class"""

    def test_multires_init(self, mock_difficult_dependencies):
        """Test MultiResolutionSimulator initialization"""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            cg_params = {'config': {}, 'components': {}}
            aa_params = {
                'solvation_scheme': 'implicit',
                'protein': True,
                'rna': False,
                'dna': False,
                'phos_protein': False,
                'use_amber': True,
                'out': 'system.pdb',
                'equilibration_steps': 1000,
                'production_steps': 10000,
                'device_ids': [0],
            }

            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=3,
                cg_params=cg_params,
                aa_params=aa_params,
                cg2all_bin='convert_cg2all',
                cg2all_ckpt=None,
                amberhome='/fake/amber',
            )

            assert sim.path == tmpdir
            assert sim.input_pdb == 'protein.pdb'
            assert sim.n_rounds == 3
            assert sim.amberhome == Path('/fake/amber')

    def test_multires_init_no_amberhome(self, mock_difficult_dependencies):
        """Test MultiResolutionSimulator initialization without amberhome"""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=1,
                cg_params={},
                aa_params={},
                amberhome=None,
            )

            assert sim.amberhome is None

    def test_multires_from_toml(self, mock_difficult_dependencies):
        """Test MultiResolutionSimulator.from_toml classmethod"""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create TOML config
            toml_content = f"""
[settings]
path = "{tmpdir}"
input_pdb = "protein.pdb"
n_rounds = 2
cg2all_bin = "convert_cg2all"
cg2all_ckpt = "/path/to/ckpt"
amberhome = "/fake/amber"

[[cg_params]]
test = "value"

[aa_params]
solvation_scheme = "explicit"
"""
            config_path = tmpdir / 'config.toml'
            config_path.write_text(toml_content)

            sim = MultiResolutionSimulator.from_toml(config_path)

            assert sim.n_rounds == 2
            assert sim.cg2all_bin == 'convert_cg2all'
            assert sim.cg2all_ckpt == '/path/to/ckpt'

    def test_multires_from_toml_minimal(self, mock_difficult_dependencies):
        """Test MultiResolutionSimulator.from_toml with minimal config"""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Minimal TOML config (no optional fields)
            toml_content = f"""
[settings]
path = "{tmpdir}"
input_pdb = "protein.pdb"
n_rounds = 1

[[cg_params]]
test = "value"

[aa_params]
solvation_scheme = "implicit"
"""
            config_path = tmpdir / 'config.toml'
            config_path.write_text(toml_content)

            sim = MultiResolutionSimulator.from_toml(config_path)

            assert sim.n_rounds == 1
            assert sim.cg2all_bin == 'convert_cg2all'  # Default
            assert sim.cg2all_ckpt is None  # Default
            assert sim.amberhome is None  # Default

    def test_strip_solvent(self, mock_difficult_dependencies):
        """Test strip_solvent static method"""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        # Get the parmed mock from our fixture
        mock_parmed = mock_difficult_dependencies['parmed']
        mock_struc = MagicMock()
        mock_parmed.openmm.load_topology.return_value = mock_struc

        mock_simulation = MagicMock()
        mock_simulation.topology = MagicMock()
        mock_simulation.system = MagicMock()
        mock_simulation.context.getState.return_value.getPositions.return_value = [
            [0, 0, 0]
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'protein.pdb'

            MultiResolutionSimulator.strip_solvent(
                mock_simulation, output_pdb=str(output_path)
            )

            # Should have called strip with solvent residues
            mock_struc.strip.assert_called_once()
            mock_struc.save.assert_called_once_with(str(output_path))


class TestMultiResolutionSimulatorRunRounds:
    """Test suite for run_rounds method."""

    def test_run_rounds_invalid_solvation_scheme(self, mock_difficult_dependencies):
        """Test run_rounds raises AttributeError for invalid solvation_scheme."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            cg_params = {'config': {}, 'components': {}}
            aa_params = {
                'solvation_scheme': 'invalid_scheme',  # Invalid scheme
                'protein': True,
                'rna': False,
                'dna': False,
                'phos_protein': False,
                'use_amber': True,
                'out': 'system.pdb',
                'equilibration_steps': 1000,
                'production_steps': 10000,
                'device_ids': [0],
            }

            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=1,
                cg_params=cg_params,
                aa_params=aa_params,
                amberhome=None,
            )

            with pytest.raises(AttributeError, match='solvation_scheme must be'):
                sim.run_rounds()

    def test_run_rounds_implicit_solvation(self, mock_difficult_dependencies):
        """Test run_rounds with implicit solvation scheme."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            cg_params = {'config': {'path': str(tmpdir)}, 'components': {}}
            aa_params = {
                'solvation_scheme': 'implicit',
                'protein': True,
                'rna': False,
                'dna': False,
                'phos_protein': False,
                'use_amber': True,
                'out': 'system.pdb',
                'equilibration_steps': 1000,
                'production_steps': 10000,
                'device_ids': [0],
            }

            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=1,
                cg_params=cg_params,
                aa_params=aa_params,
                amberhome=None,
            )

            # Mock the builders and simulators
            with (
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSolvent'
                ) as mock_builder,
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSimulator'
                ) as mock_simulator,
                patch(
                    'molecular_simulations.simulate.multires_simulator.sander_minimize'
                ),
                patch(
                    'molecular_simulations.simulate.multires_simulator.CGBuilder'
                ) as mock_cg_builder,
                patch(
                    'molecular_simulations.simulate.multires_simulator.sim'
                ) as mock_calvados_sim,
                patch(
                    'molecular_simulations.simulate.multires_simulator.subprocess'
                ) as mock_subprocess,
                patch.object(MultiResolutionSimulator, 'strip_solvent'),
            ):
                mock_builder_inst = MagicMock()
                mock_builder.return_value = mock_builder_inst
                mock_simulator_inst = MagicMock()
                mock_simulator_inst.simulation = MagicMock()
                mock_simulator.return_value = mock_simulator_inst

                mock_cg_builder_inst = MagicMock()
                mock_cg_builder.from_dict.return_value = mock_cg_builder_inst

                # Mock cg2all subprocess success
                mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')

                # Directories are created by run_rounds, mock needed file reads
                def create_cg_files(*args, **kwargs):
                    # After CG build runs, create the files it would create
                    cg_path = tmpdir / 'cg_round0'
                    if cg_path.exists():
                        (cg_path / 'last_frame.pdb').write_text('mock pdb')
                        (cg_path / 'top.pdb').write_text('mock top')

                mock_calvados_sim.run.side_effect = create_cg_files

                sim.run_rounds()

                # Verify builder was called
                mock_builder.assert_called_once()
                mock_builder_inst.build.assert_called_once()
                mock_simulator.assert_called_once()
                mock_simulator_inst.run.assert_called_once()

    def test_run_rounds_explicit_solvation(self, mock_difficult_dependencies):
        """Test run_rounds with explicit solvation scheme."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            cg_params = {'config': {'path': str(tmpdir)}, 'components': {}}
            aa_params = {
                'solvation_scheme': 'explicit',
                'protein': True,
                'rna': False,
                'dna': False,
                'phos_protein': False,
                'use_amber': True,
                'out': 'system.pdb',
                'equilibration_steps': 1000,
                'production_steps': 10000,
                'device_ids': [0],
            }

            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=1,
                cg_params=cg_params,
                aa_params=aa_params,
                amberhome=None,
            )

            with (
                patch(
                    'molecular_simulations.simulate.multires_simulator.ExplicitSolvent'
                ) as mock_builder,
                patch(
                    'molecular_simulations.simulate.multires_simulator.Simulator'
                ) as mock_simulator,
                patch(
                    'molecular_simulations.simulate.multires_simulator.sander_minimize'
                ),
                patch(
                    'molecular_simulations.simulate.multires_simulator.CGBuilder'
                ) as mock_cg_builder,
                patch(
                    'molecular_simulations.simulate.multires_simulator.sim'
                ) as mock_calvados_sim,
                patch(
                    'molecular_simulations.simulate.multires_simulator.subprocess'
                ) as mock_subprocess,
                patch.object(MultiResolutionSimulator, 'strip_solvent'),
            ):
                mock_builder_inst = MagicMock()
                mock_builder.return_value = mock_builder_inst
                mock_simulator_inst = MagicMock()
                mock_simulator_inst.simulation = MagicMock()
                mock_simulator.return_value = mock_simulator_inst

                mock_cg_builder_inst = MagicMock()
                mock_cg_builder.from_dict.return_value = mock_cg_builder_inst

                mock_subprocess.run.return_value = MagicMock(returncode=0, stdout='')

                # Directories are created by run_rounds, mock needed file reads
                def create_cg_files(*args, **kwargs):
                    cg_path = tmpdir / 'cg_round0'
                    if cg_path.exists():
                        (cg_path / 'last_frame.pdb').write_text('mock pdb')
                        (cg_path / 'top.pdb').write_text('mock top')

                mock_calvados_sim.run.side_effect = create_cg_files

                sim.run_rounds()

                # Verify explicit solvation was used
                mock_builder.assert_called_once()
                mock_simulator.assert_called_once()

    def test_run_rounds_cg2all_error(self, mock_difficult_dependencies):
        """Test run_rounds raises RuntimeError when cg2all fails."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            cg_params = {'config': {'path': str(tmpdir)}, 'components': {}}
            aa_params = {
                'solvation_scheme': 'implicit',
                'protein': True,
                'rna': False,
                'dna': False,
                'phos_protein': False,
                'use_amber': True,
                'out': 'system.pdb',
                'equilibration_steps': 1000,
                'production_steps': 10000,
                'device_ids': [0],
            }

            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=1,
                cg_params=cg_params,
                aa_params=aa_params,
                amberhome=None,
            )

            with (
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSolvent'
                ) as mock_builder,
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSimulator'
                ) as mock_simulator,
                patch(
                    'molecular_simulations.simulate.multires_simulator.sander_minimize'
                ),
                patch(
                    'molecular_simulations.simulate.multires_simulator.CGBuilder'
                ) as mock_cg_builder,
                patch('molecular_simulations.simulate.multires_simulator.sim'),
                patch(
                    'molecular_simulations.simulate.multires_simulator.subprocess'
                ) as mock_subprocess,
                patch.object(MultiResolutionSimulator, 'strip_solvent'),
            ):
                mock_builder_inst = MagicMock()
                mock_builder.return_value = mock_builder_inst
                mock_simulator_inst = MagicMock()
                mock_simulator_inst.simulation = MagicMock()
                mock_simulator.return_value = mock_simulator_inst

                mock_cg_builder_inst = MagicMock()
                mock_cg_builder.from_dict.return_value = mock_cg_builder_inst

                # Mock cg2all subprocess failure
                def create_files_then_fail(*args, **kwargs):
                    cg_path = tmpdir / 'cg_round0'
                    if cg_path.exists():
                        (cg_path / 'top.pdb').write_text('mock top')
                    return MagicMock(
                        returncode=1, stderr='cg2all error message', stdout=''
                    )

                mock_subprocess.run.side_effect = create_files_then_fail

                with pytest.raises(RuntimeError, match='cg2all error'):
                    sim.run_rounds()

    def test_run_rounds_pdb4amber_error(self, mock_difficult_dependencies):
        """Test run_rounds raises RuntimeError when pdb4amber fails."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            cg_params = {'config': {'path': str(tmpdir)}, 'components': {}}
            aa_params = {
                'solvation_scheme': 'implicit',
                'protein': True,
                'rna': False,
                'dna': False,
                'phos_protein': False,
                'use_amber': True,
                'out': 'system.pdb',
                'equilibration_steps': 1000,
                'production_steps': 10000,
                'device_ids': [0],
            }

            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=1,
                cg_params=cg_params,
                aa_params=aa_params,
                amberhome=None,
            )

            with (
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSolvent'
                ) as mock_builder,
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSimulator'
                ) as mock_simulator,
                patch(
                    'molecular_simulations.simulate.multires_simulator.sander_minimize'
                ),
                patch(
                    'molecular_simulations.simulate.multires_simulator.CGBuilder'
                ) as mock_cg_builder,
                patch('molecular_simulations.simulate.multires_simulator.sim'),
                patch(
                    'molecular_simulations.simulate.multires_simulator.subprocess'
                ) as mock_subprocess,
                patch.object(MultiResolutionSimulator, 'strip_solvent'),
            ):
                mock_builder_inst = MagicMock()
                mock_builder.return_value = mock_builder_inst
                mock_simulator_inst = MagicMock()
                mock_simulator_inst.simulation = MagicMock()
                mock_simulator.return_value = mock_simulator_inst

                mock_cg_builder_inst = MagicMock()
                mock_cg_builder.from_dict.return_value = mock_cg_builder_inst

                # First subprocess call (cg2all) succeeds, second (pdb4amber) fails
                call_count = [0]

                def subprocess_side_effect(*args, **kwargs):
                    cg_path = tmpdir / 'cg_round0'
                    if cg_path.exists():
                        (cg_path / 'top.pdb').write_text('mock top')
                        (cg_path / 'last_frame.pdb').write_text('mock pdb')
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return MagicMock(returncode=0, stdout='', stderr='')
                    else:
                        return MagicMock(
                            returncode=1, stdout='', stderr='pdb4amber error'
                        )

                mock_subprocess.run.side_effect = subprocess_side_effect

                with pytest.raises(RuntimeError, match='pdb4amber error'):
                    sim.run_rounds()

    def test_run_rounds_multiple_rounds(self, mock_difficult_dependencies):
        """Test run_rounds executes correct number of rounds."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            cg_params = {'config': {'path': str(tmpdir)}, 'components': {}}
            aa_params = {
                'solvation_scheme': 'implicit',
                'protein': True,
                'rna': False,
                'dna': False,
                'phos_protein': False,
                'use_amber': True,
                'out': 'system.pdb',
                'equilibration_steps': 1000,
                'production_steps': 10000,
                'device_ids': [0],
            }

            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=3,
                cg_params=cg_params,
                aa_params=aa_params,
                amberhome=None,
            )

            with (
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSolvent'
                ) as mock_builder,
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSimulator'
                ) as mock_simulator,
                patch(
                    'molecular_simulations.simulate.multires_simulator.sander_minimize'
                ),
                patch(
                    'molecular_simulations.simulate.multires_simulator.CGBuilder'
                ) as mock_cg_builder,
                patch('molecular_simulations.simulate.multires_simulator.sim'),
                patch(
                    'molecular_simulations.simulate.multires_simulator.subprocess'
                ) as mock_subprocess,
                patch.object(MultiResolutionSimulator, 'strip_solvent'),
            ):
                mock_builder_inst = MagicMock()
                mock_builder.return_value = mock_builder_inst
                mock_simulator_inst = MagicMock()
                mock_simulator_inst.simulation = MagicMock()
                mock_simulator.return_value = mock_simulator_inst

                mock_cg_builder_inst = MagicMock()
                mock_cg_builder.from_dict.return_value = mock_cg_builder_inst

                # Create files after CG sim runs for each round
                round_counter = [0]

                def create_cg_files_multi(*args, **kwargs):
                    r = round_counter[0]
                    cg_path = tmpdir / f'cg_round{r}'
                    if cg_path.exists():
                        (cg_path / 'top.pdb').write_text('mock top')
                        (cg_path / 'last_frame.pdb').write_text('mock pdb')
                        (cg_path / 'last_frame.amber.pdb').write_text('mock amber pdb')
                    round_counter[0] += 1
                    return MagicMock(returncode=0, stdout='mock output')

                mock_subprocess.run.side_effect = create_cg_files_multi

                sim.run_rounds()

                # Builder should be called 3 times (once per round)
                assert mock_builder.call_count == 3
                # Simulator should be called 3 times
                assert mock_simulator.call_count == 3
                # CG builder should be called 3 times
                assert mock_cg_builder.from_dict.call_count == 3

    def test_run_rounds_with_amberhome(self, mock_difficult_dependencies):
        """Test run_rounds uses amberhome path for sander."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            cg_params = {'config': {'path': str(tmpdir)}, 'components': {}}
            aa_params = {
                'solvation_scheme': 'implicit',
                'protein': True,
                'rna': False,
                'dna': False,
                'phos_protein': False,
                'use_amber': True,
                'out': 'system.pdb',
                'equilibration_steps': 1000,
                'production_steps': 10000,
                'device_ids': [0],
            }

            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=1,
                cg_params=cg_params,
                aa_params=aa_params,
                amberhome='/custom/amber',
            )

            with (
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSolvent'
                ) as mock_builder,
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSimulator'
                ) as mock_simulator,
                patch(
                    'molecular_simulations.simulate.multires_simulator.sander_minimize'
                ) as mock_sander,
                patch(
                    'molecular_simulations.simulate.multires_simulator.CGBuilder'
                ) as mock_cg_builder,
                patch('molecular_simulations.simulate.multires_simulator.sim'),
                patch(
                    'molecular_simulations.simulate.multires_simulator.subprocess'
                ) as mock_subprocess,
                patch.object(MultiResolutionSimulator, 'strip_solvent'),
            ):
                mock_builder_inst = MagicMock()
                mock_builder.return_value = mock_builder_inst
                mock_simulator_inst = MagicMock()
                mock_simulator_inst.simulation = MagicMock()
                mock_simulator.return_value = mock_simulator_inst

                mock_cg_builder_inst = MagicMock()
                mock_cg_builder.from_dict.return_value = mock_cg_builder_inst

                def create_cg_files(*args, **kwargs):
                    cg_path = tmpdir / 'cg_round0'
                    if cg_path.exists():
                        (cg_path / 'top.pdb').write_text('mock top')
                        (cg_path / 'last_frame.pdb').write_text('mock pdb')
                    return MagicMock(returncode=0, stdout='mock output')

                mock_subprocess.run.side_effect = create_cg_files

                sim.run_rounds()

                # sander should be called with amberhome path
                mock_sander.assert_called_once()
                call_args = mock_sander.call_args
                assert '/custom/amber/bin/sander' in call_args[0][3]

    def test_run_rounds_with_cg2all_checkpoint(self, mock_difficult_dependencies):
        """Test run_rounds includes cg2all checkpoint when provided."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            cg_params = {'config': {'path': str(tmpdir)}, 'components': {}}
            aa_params = {
                'solvation_scheme': 'implicit',
                'protein': True,
                'rna': False,
                'dna': False,
                'phos_protein': False,
                'use_amber': True,
                'out': 'system.pdb',
                'equilibration_steps': 1000,
                'production_steps': 10000,
                'device_ids': [0],
            }

            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=1,
                cg_params=cg_params,
                aa_params=aa_params,
                cg2all_ckpt='/path/to/checkpoint.ckpt',
                amberhome=None,
            )

            with (
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSolvent'
                ) as mock_builder,
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSimulator'
                ) as mock_simulator,
                patch(
                    'molecular_simulations.simulate.multires_simulator.sander_minimize'
                ),
                patch(
                    'molecular_simulations.simulate.multires_simulator.CGBuilder'
                ) as mock_cg_builder,
                patch('molecular_simulations.simulate.multires_simulator.sim'),
                patch(
                    'molecular_simulations.simulate.multires_simulator.subprocess'
                ) as mock_subprocess,
                patch.object(MultiResolutionSimulator, 'strip_solvent'),
            ):
                mock_builder_inst = MagicMock()
                mock_builder.return_value = mock_builder_inst
                mock_simulator_inst = MagicMock()
                mock_simulator_inst.simulation = MagicMock()
                mock_simulator.return_value = mock_simulator_inst

                mock_cg_builder_inst = MagicMock()
                mock_cg_builder.from_dict.return_value = mock_cg_builder_inst

                def create_cg_files(*args, **kwargs):
                    cg_path = tmpdir / 'cg_round0'
                    if cg_path.exists():
                        (cg_path / 'top.pdb').write_text('mock top')
                        (cg_path / 'last_frame.pdb').write_text('mock pdb')
                    return MagicMock(returncode=0, stdout='mock output')

                mock_subprocess.run.side_effect = create_cg_files

                sim.run_rounds()

                # cg2all subprocess should include --ckpt argument
                cg2all_call = mock_subprocess.run.call_args_list[0]
                command = cg2all_call[0][0]
                assert '--ckpt' in command
                assert '/path/to/checkpoint.ckpt' in command

    def test_run_rounds_uses_previous_round_pdb(self, mock_difficult_dependencies):
        """Test run_rounds uses CG output from previous round as input."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text(
                'ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n'
            )

            cg_params = {'config': {'path': str(tmpdir)}, 'components': {}}
            aa_params = {
                'solvation_scheme': 'implicit',
                'protein': True,
                'rna': False,
                'dna': False,
                'phos_protein': False,
                'use_amber': True,
                'out': 'system.pdb',
                'equilibration_steps': 1000,
                'production_steps': 10000,
                'device_ids': [0],
            }

            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=2,
                cg_params=cg_params,
                aa_params=aa_params,
                amberhome=None,
            )

            with (
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSolvent'
                ) as mock_builder,
                patch(
                    'molecular_simulations.simulate.multires_simulator.ImplicitSimulator'
                ) as mock_simulator,
                patch(
                    'molecular_simulations.simulate.multires_simulator.sander_minimize'
                ),
                patch(
                    'molecular_simulations.simulate.multires_simulator.CGBuilder'
                ) as mock_cg_builder,
                patch('molecular_simulations.simulate.multires_simulator.sim'),
                patch(
                    'molecular_simulations.simulate.multires_simulator.subprocess'
                ) as mock_subprocess,
                patch.object(MultiResolutionSimulator, 'strip_solvent'),
            ):
                mock_builder_inst = MagicMock()
                mock_builder.return_value = mock_builder_inst
                mock_simulator_inst = MagicMock()
                mock_simulator_inst.simulation = MagicMock()
                mock_simulator.return_value = mock_simulator_inst

                mock_cg_builder_inst = MagicMock()
                mock_cg_builder.from_dict.return_value = mock_cg_builder_inst

                round_counter = [0]

                def create_cg_files_multi(*args, **kwargs):
                    r = round_counter[0]
                    cg_path = tmpdir / f'cg_round{r}'
                    if cg_path.exists():
                        (cg_path / 'top.pdb').write_text('mock top')
                        (cg_path / 'last_frame.pdb').write_text('mock pdb')
                        (cg_path / 'last_frame.amber.pdb').write_text('mock amber pdb')
                    round_counter[0] += 1
                    return MagicMock(returncode=0, stdout='mock output')

                mock_subprocess.run.side_effect = create_cg_files_multi

                sim.run_rounds()

                # Check input PDB for second round came from cg_round0
                second_call = mock_builder.call_args_list[1]
                input_pdb = second_call[0][1]  # Second positional arg is input_pdb
                assert 'cg_round0' in input_pdb


class TestSanderMinDefaultsCustom:
    """Test suite for sander_min_defaults with custom values."""

    def test_sander_min_defaults_attributes_modifiable(
        self, mock_difficult_dependencies
    ):
        """Test sander_min_defaults attributes can be modified after creation."""
        from molecular_simulations.simulate.multires_simulator import (
            SanderMinDefaults,
        )

        defaults = SanderMinDefaults()

        # Modify attributes directly and regenerate mdin_contents
        defaults.maxcyc = 10000
        defaults.ncyc = 5000
        defaults.cut = 12.0
        defaults.__post_init__()

        assert defaults.maxcyc == 10000
        assert defaults.ncyc == 5000
        assert defaults.cut == 12.0
        assert 'maxcyc=10000' in defaults.mdin_contents
        assert 'ncyc=5000' in defaults.mdin_contents
        assert 'cut=12.0' in defaults.mdin_contents


class TestStripSolventMask:
    """Test suite for strip_solvent solvent mask."""

    def test_strip_solvent_mask_content(self, mock_difficult_dependencies):
        """Test strip_solvent uses correct solvent mask."""
        from molecular_simulations.simulate.multires_simulator import (
            MultiResolutionSimulator,
        )

        mock_parmed = mock_difficult_dependencies['parmed']
        mock_struc = MagicMock()
        mock_parmed.openmm.load_topology.return_value = mock_struc

        mock_simulation = MagicMock()
        mock_simulation.topology = MagicMock()
        mock_simulation.system = MagicMock()
        mock_simulation.context.getState.return_value.getPositions.return_value = [
            [0, 0, 0]
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'protein.pdb'

            MultiResolutionSimulator.strip_solvent(
                mock_simulation, output_pdb=str(output_path)
            )

            # Check that strip was called with the correct mask
            strip_call = mock_struc.strip.call_args
            mask = strip_call[0][0]
            # Mask should contain common solvent names
            assert 'WAT' in mask
            assert 'HOH' in mask
            assert 'NA' in mask or 'Na+' in mask


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
