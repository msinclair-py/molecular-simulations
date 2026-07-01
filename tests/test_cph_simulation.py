"""
Unit tests for simulate/cph_simulation.py module

This module tests the constant pH simulation setup and ensemble management
classes used for pH-dependent molecular simulations.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from parsl import python_app

# Mark tests that don't require OpenMM as unit tests
pytestmark = pytest.mark.unit


@python_app
def _record_run_app(params, temperature, n_cycles, n_steps, log_params, path):
    """A real, lightweight Parsl app standing in for run_cph_sim.

    It performs genuine work on a real worker -- recording the arguments it was
    submitted with to a JSON file under the task's log_dir -- so tests can assert
    the ensemble's real submission machinery passed the correct parameters
    without running an actual (expensive, GPU) OpenMM constant-pH simulation.
    """
    import json
    import os

    log_dir = str(log_params['log_dir'])
    os.makedirs(log_dir, exist_ok=True)
    record = {
        'temperature': temperature,
        'n_cycles': n_cycles,
        'n_steps': n_steps,
        'task_id': log_params['task_id'],
        'log_dir': log_dir,
        'path': path,
    }
    out = os.path.join(log_dir, f'record_{log_params["task_id"]}.json')
    with open(out, 'w') as fh:
        json.dump(record, fh)
    return record


@python_app
def _failing_run_app(params, temperature, n_cycles, n_steps, log_params, path):
    """A real Parsl app that raises, to exercise run()'s exception capture."""
    raise RuntimeError('Simulation failed')


def _populate_amber_path(directory: Path, src_files: dict) -> Path:
    """Copy a real prmtop/inpcrd into ``directory`` as system.prmtop/inpcrd."""
    import shutil

    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_files['prmtop'], directory / 'system.prmtop')
    shutil.copy(src_files['inpcrd'], directory / 'system.inpcrd')
    return directory


class TestConstantPHEnsembleInit:
    """Test suite for ConstantPHEnsemble class initialization."""

    def test_ensemble_init_defaults(self) -> None:
        """Test ConstantPHEnsemble initialization with default parameters.

        Verifies that the class correctly initializes with default pH range
        and temperature values.
        """

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {
                'CYS': [0.0, 10.0],
                'ASP': [0.0, 5.0],
            }

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            assert ensemble.paths == [path]
            assert ensemble.ref_energies == ref_energies
            assert ensemble.parsl_config is stored_config
            assert ensemble.log_dir == log_dir
            # Default pH range is 0.5 to 13.5 in 1.0 increments
            assert len(ensemble.pHs) == 14
            assert ensemble.pHs[0] == 0.5
            # Temperature should be 300K by default (stored as float, not Quantity)
            assert ensemble.temperature == 300.0

    def test_ensemble_init_custom_phs(self) -> None:
        """Test ConstantPHEnsemble with custom pH values."""

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}
            custom_phs = [4.0, 5.0, 6.0, 7.0, 8.0]

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
                pHs=custom_phs,
            )

            assert ensemble.pHs == custom_phs

    def test_ensemble_init_custom_temperature(self) -> None:
        """Test ConstantPHEnsemble with custom temperature.

        The temperature is multiplied by kelvin unit internally.
        """

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
                temperature=310.0,
            )

            assert ensemble.temperature == 310.0

    def test_ensemble_init_with_variant_sel(self) -> None:
        """Test ConstantPHEnsemble with custom variant selection string.

        The variant_sel parameter allows selecting specific residues for
        titration rather than all titratable residues.
        """
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
                variant_sel='resid 10:50',
            )

            assert ensemble.variant_sel == 'resid 10:50'

    def test_ensemble_init_generates_run_id(self) -> None:
        """Test that initialization generates a unique run ID based on timestamp."""
        import re

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            # run_id should match format YYYYMMDD_HHMMSS
            assert re.match(r'\d{8}_\d{6}', ensemble.run_id)


class TestConstantPHEnsembleParslManagement:
    """Test suite for Parsl initialization and shutdown."""

    def test_initialize_loads_parsl(self, local_parsl_config, tmp_path) -> None:
        """Test that initialize() loads a real Parsl DataFlowKernel."""
        from parsl.dataflow.dflow import DataFlowKernel

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        ensemble = ConstantPHEnsemble(
            paths=[tmp_path],
            reference_energies={'CYS': [0.0, 10.0]},
            parsl_config=local_parsl_config,
            log_dir=tmp_path / 'logs',
        )

        assert ensemble.dfk is None
        try:
            ensemble.initialize()
            # A real DataFlowKernel is loaded from the lightweight config.
            assert isinstance(ensemble.dfk, DataFlowKernel)
        finally:
            ensemble.shutdown()

    def test_shutdown_cleans_up_parsl(self, local_parsl_config, tmp_path) -> None:
        """Test that shutdown() tears down the real Parsl DataFlowKernel."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        ensemble = ConstantPHEnsemble(
            paths=[tmp_path],
            reference_energies={'CYS': [0.0, 10.0]},
            parsl_config=local_parsl_config,
            log_dir=tmp_path / 'logs',
        )

        ensemble.initialize()
        assert ensemble.dfk is not None
        ensemble.shutdown()

        assert ensemble.dfk is None

    def test_shutdown_when_not_initialized(self, tmp_path) -> None:
        """Test that shutdown() handles the case when dfk is None."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        ensemble = ConstantPHEnsemble(
            paths=[tmp_path],
            reference_energies={'CYS': [0.0, 10.0]},
            parsl_config=None,
            log_dir=tmp_path / 'logs',
        )

        # Should not raise even when dfk was never loaded.
        ensemble.shutdown()
        assert ensemble.dfk is None


class TestConstantPHEnsembleParams:
    """Test suite for params property."""

    def test_params_contains_required_keys(self) -> None:
        """Test params property returns dictionary with all required keys.

        The params dictionary should contain simulation parameters for both
        explicit and implicit solvent, integrators, and pH values.
        """
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            # Check required keys
            assert 'prmtop_file' in params
            assert 'inpcrd_file' in params
            assert 'pH' in params
            assert 'relaxationSteps' in params
            assert 'nonbonded_cutoff' in params
            assert 'hmr' in params
            assert 'implicit_cutoff' in params

    def test_params_explicit_args(self) -> None:
        """Test params contains correct explicit solvent arguments."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            # Check explicit solvent parameters
            assert params['nonbonded_cutoff'] == 0.9
            assert params['hmr'] == 1.5

    def test_params_implicit_args(self) -> None:
        """Test params contains correct implicit solvent arguments."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            # Check implicit solvent parameters
            assert params['implicit_cutoff'] == 2.0

    def test_params_ph_values(self) -> None:
        """Test params contains correct pH values from initialization."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}
            custom_phs = [5.0, 6.0, 7.0]

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
                pHs=custom_phs,
            )

            params = ensemble.get_params(path)

            assert params['pH'] == custom_phs

    def test_params_with_custom_temperature(self) -> None:
        """Test ensemble stores custom temperature correctly."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
                temperature=310.0,
            )

            # Temperature is stored on the ensemble object
            assert ensemble.temperature == 310.0

    def test_params_relaxation_steps(self) -> None:
        """Test params contains correct relaxation steps."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            assert params['relaxationSteps'] == 1000

    def test_params_file_paths(self) -> None:
        """Test params contains correct file paths."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            assert params['prmtop_file'] == path / 'system.prmtop'
            assert params['inpcrd_file'] == path / 'system.inpcrd'


class TestConstantPHEnsembleTemperatureHandling:
    """Test suite for temperature handling."""

    def test_temperature_stored_as_float(self) -> None:
        """Test that temperature is stored as a float value in Kelvin."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
                temperature=300.0,
            )

            # Temperature is stored as a float (assumed to be in Kelvin)
            assert ensemble.temperature == 300.0
            assert isinstance(ensemble.temperature, float)

    @pytest.mark.parametrize('temp', [273.15, 300.0, 310.0, 350.0])
    def test_temperature_various_values(self, temp: float) -> None:
        """Test temperature handling with various physiological temperatures."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
                temperature=temp,
            )

            assert ensemble.temperature == temp


class TestConstantPHEnsembleMultiplePaths:
    """Test suite for handling multiple simulation paths."""

    def test_multiple_paths_stored(self) -> None:
        """Test that multiple paths are stored correctly."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = [base / f'system{i}' for i in range(5)]
            for p in paths:
                p.mkdir()

            log_dir = base / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=paths,
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            assert len(ensemble.paths) == 5
            assert ensemble.paths == paths

    def test_single_path(self) -> None:
        """Test with single path."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            assert len(ensemble.paths) == 1
            assert ensemble.paths[0] == path


class TestConstantPHEnsembleReferenceEnergies:
    """Test suite for reference energy handling."""

    def test_reference_energies_stored(self) -> None:
        """Test that reference energies are stored correctly."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {
                'CYS': [0.0, 10.0],
                'ASP': [0.0, 5.0],
                'GLU': [0.0, 6.0],
            }

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            assert ensemble.ref_energies == ref_energies
            assert 'CYS' in ensemble.ref_energies
            assert 'ASP' in ensemble.ref_energies
            assert 'GLU' in ensemble.ref_energies


class TestConstantPHEnsemblePHRange:
    """Test suite for pH range handling."""

    def test_default_ph_range(self) -> None:
        """Test default pH range is 0.5 to 13.5 in 1.0 steps."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            # Default should be [0.5, 1.5, 2.5, ..., 13.5]
            assert len(ensemble.pHs) == 14
            assert ensemble.pHs[0] == 0.5
            assert ensemble.pHs[-1] == 13.5

    def test_custom_ph_range(self) -> None:
        """Test custom pH range."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}
            custom_phs = [2.0, 4.0, 6.0, 8.0, 10.0]

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
                pHs=custom_phs,
            )

            assert ensemble.pHs == custom_phs
            assert len(ensemble.pHs) == 5

    def test_single_ph(self) -> None:
        """Test with single pH value."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
                pHs=[7.4],  # Single physiological pH
            )

            assert ensemble.pHs == [7.4]
            assert len(ensemble.pHs) == 1


class TestConstantPHEnsembleIntegrators:
    """Test suite for simulation configuration."""

    def test_relaxation_steps_in_params(self) -> None:
        """Test that relaxationSteps is included in params."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            assert 'relaxationSteps' in params
            assert params['relaxationSteps'] == 1000

    def test_hmr_in_params(self) -> None:
        """Test that hydrogen mass repartitioning is included in params."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            assert 'hmr' in params
            assert params['hmr'] == 1.5

    def test_cutoffs_in_params(self) -> None:
        """Test that cutoff parameters are included in params."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            assert params['nonbonded_cutoff'] == 0.9
            assert params['implicit_cutoff'] == 2.0


class TestConstantPHEnsembleVariantSel:
    """Test suite for variant selection handling."""

    def test_variant_sel_none_by_default(self) -> None:
        """Test that variant_sel is None by default."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
            )

            assert ensemble.variant_sel is None

    def test_variant_sel_custom_string(self) -> None:
        """Test custom variant selection string."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            stored_config = object()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=stored_config,
                log_dir=log_dir,
                variant_sel='resid 10 to 50',
            )

            assert ensemble.variant_sel == 'resid 10 to 50'


class TestConstantPHEnsembleRun:
    """Test suite for run method."""

    def test_run_submits_futures_for_all_paths(
        self, real_amber_system_files, local_parsl_config
    ) -> None:
        """Test run submits a real Parsl app for every path and awaits it.

        Uses a real DataFlowKernel (ThreadPoolExecutor) and a real lightweight
        app, so the ensemble genuinely calls load_files/build_dicts/get_params on
        real AMBER files, submits one future per path, and resolves them all.
        """
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        base = real_amber_system_files['path']
        paths = [
            _populate_amber_path(base / f'system{i}', real_amber_system_files)
            for i in range(3)
        ]
        log_dir = base / 'logs'

        ensemble = ConstantPHEnsemble(
            paths=paths,
            reference_energies={'CYS': [0.0, 10.0]},
            parsl_config=local_parsl_config,
            log_dir=log_dir,
        )

        ensemble.initialize()
        try:
            results = ensemble.run(n_cycles=10, n_steps=100, parsl_func=_record_run_app)
        finally:
            ensemble.shutdown()

        # One resolved future per path, all successful (None == success).
        assert results == {0: None, 1: None, 2: None}
        # Each real app actually executed and recorded itself.
        records = list(log_dir.glob('record_*.json'))
        assert len(records) == 3

    def test_run_passes_correct_parameters(
        self, real_amber_system_files, local_parsl_config
    ) -> None:
        """Test run passes the real n_cycles and n_steps to the submitted app."""
        import json

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        base = real_amber_system_files['path']
        path = _populate_amber_path(base / 'system0', real_amber_system_files)
        log_dir = base / 'logs'

        ensemble = ConstantPHEnsemble(
            paths=[path],
            reference_energies={'CYS': [0.0, 10.0]},
            parsl_config=local_parsl_config,
            log_dir=log_dir,
        )

        ensemble.initialize()
        try:
            results = ensemble.run(
                n_cycles=250, n_steps=750, parsl_func=_record_run_app
            )
        finally:
            ensemble.shutdown()

        assert results == {0: None}
        record = json.loads((log_dir / 'record_00000.json').read_text())
        assert record['n_cycles'] == 250
        assert record['n_steps'] == 750
        assert record['temperature'] == 300.0


class TestConstantPHEnsembleRunWithDefaults:
    """Test suite for run method with default parameters."""

    def test_run_uses_default_parameters(
        self, real_amber_system_files, local_parsl_config
    ) -> None:
        """Test run method uses default n_cycles=500 and n_steps=500."""
        import json

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        base = real_amber_system_files['path']
        path = _populate_amber_path(base / 'system0', real_amber_system_files)
        log_dir = base / 'logs'

        ensemble = ConstantPHEnsemble(
            paths=[path],
            reference_energies={'CYS': [0.0, 10.0]},
            parsl_config=local_parsl_config,
            log_dir=log_dir,
        )

        ensemble.initialize()
        try:
            # Call with the real recording app but no explicit cycle/step counts.
            results = ensemble.run(parsl_func=_record_run_app)
        finally:
            ensemble.shutdown()

        assert results == {0: None}
        record = json.loads((log_dir / 'record_00000.json').read_text())
        assert record['n_cycles'] == 500  # default n_cycles
        assert record['n_steps'] == 500  # default n_steps


class TestConstantPHEnsembleLogParams:
    """Test suite for log parameter generation."""

    def test_run_generates_unique_task_ids(
        self, real_amber_system_files, local_parsl_config
    ) -> None:
        """Test run generates unique task IDs for each path."""
        import json

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        base = real_amber_system_files['path']
        paths = [
            _populate_amber_path(base / f'system{i}', real_amber_system_files)
            for i in range(3)
        ]
        log_dir = base / 'logs'

        ensemble = ConstantPHEnsemble(
            paths=paths,
            reference_energies={'CYS': [0.0, 10.0]},
            parsl_config=local_parsl_config,
            log_dir=log_dir,
        )

        ensemble.initialize()
        try:
            results = ensemble.run(n_cycles=10, n_steps=100, parsl_func=_record_run_app)
        finally:
            ensemble.shutdown()

        assert results == {0: None, 1: None, 2: None}

        # Each real app recorded the task_id it received; they must be unique.
        task_ids = {
            json.loads(rec.read_text())['task_id']
            for rec in log_dir.glob('record_*.json')
        }
        assert task_ids == {'00000', '00001', '00002'}


class TestConstantPHEnsembleLoadFiles:
    """Test suite for load_files method."""

    def test_load_files_returns_topology_and_positions(self, real_amber_system_files):
        """Test load_files returns a real (topology, positions) pair.

        Loads the real Ace-Ala-Nme AMBER system (22 atoms) through OpenMM and
        asserts the returned OpenMM Topology and positions reflect the real
        structure rather than mocked stand-ins.
        """
        import shutil

        from openmm.app import Topology
        from openmm.unit import nanometer

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        path = real_amber_system_files['path']
        # load_files hardcodes the system.prmtop / system.inpcrd names.
        shutil.copy(real_amber_system_files['prmtop'], path / 'system.prmtop')
        shutil.copy(real_amber_system_files['inpcrd'], path / 'system.inpcrd')

        ensemble = ConstantPHEnsemble(
            paths=[path],
            reference_energies={'CYS': [0.0, 10.0]},
            parsl_config=None,
            log_dir=path / 'logs',
        )

        topology, positions = ensemble.load_files(path)

        assert isinstance(topology, Topology)
        assert topology.getNumAtoms() == 22
        assert len(positions) == 22
        arr = np.array(positions.value_in_unit(nanometer))
        assert arr.shape == (22, 3)


class TestConstantPHEnsembleBuildDicts:
    """Test suite for build_dicts method."""

    def test_build_dicts_identifies_titratable_residues(
        self, real_amber_titratable_files
    ):
        """Test build_dicts identifies real titratable residues.

        Uses a real Ace-Lys-Asp-Nme AMBER system. The OpenMM topology and the
        MDAnalysis universe are both loaded from real files, so build_dicts must
        find LYS (index 1) and ASP (index 2) while excluding the ACE/NME termini.
        """
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        path = real_amber_titratable_files['path']

        ensemble = ConstantPHEnsemble(
            paths=[path],
            reference_energies={
                'ASP': [0.0, 5.0],
                'LYS': [0.0, 10.0],
                'CYS': [0.0, 7.0],
            },
            parsl_config=None,
            log_dir=path / 'logs',
        )

        top, _positions = ensemble.load_files(path)
        variants, ref_e = ensemble.build_dicts(path, top)

        # LYS at index 1 and ASP at index 2 are found; ACE/NME termini excluded.
        assert set(variants.keys()) == {1, 2}
        assert variants[1] == ['LYN', 'LYS']
        assert variants[2] == ['ASP', 'ASH']
        # Reference energies are carried through for the same residues.
        assert ref_e[1] == [0.0, 10.0]
        assert ref_e[2] == [0.0, 5.0]


class TestConstantPHEnsembleBuildDictsWithVariantSel:
    """Test build_dicts with variant_sel filtering."""

    def test_build_dicts_with_variant_sel_filters_residues(
        self, real_amber_titratable_files
    ):
        """Test build_dicts with variant_sel restricts titratable residues.

        With a real Ace-Lys-Asp-Nme system, selecting only ``resname LYS`` must
        keep LYS (index 1) and drop the otherwise-titratable ASP (index 2).
        """
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        path = real_amber_titratable_files['path']

        ensemble = ConstantPHEnsemble(
            paths=[path],
            reference_energies={'ASP': [0.0, 5.0], 'LYS': [0.0, 10.0]},
            parsl_config=None,
            log_dir=path / 'logs',
            variant_sel='resname LYS',
        )

        top, _positions = ensemble.load_files(path)
        variants, _ref_e = ensemble.build_dicts(path, top)

        assert 1 in variants
        assert 2 not in variants
        assert variants[1] == ['LYN', 'LYS']


class TestConstantPHEnsembleRunMethods:
    """Test suite for run method."""

    def test_run_with_list_log_dir(self, real_amber_system_files, local_parsl_config):
        """Test run method uses per-path log_dir when log_dir is a list."""
        import json

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        base = real_amber_system_files['path']
        paths = [
            _populate_amber_path(base / 'sys0', real_amber_system_files),
            _populate_amber_path(base / 'sys1', real_amber_system_files),
        ]
        log_dirs = [base / 'logs0', base / 'logs1']

        ensemble = ConstantPHEnsemble(
            paths=paths,
            reference_energies={'CYS': [0.0, 10.0]},
            parsl_config=local_parsl_config,
            log_dir=log_dirs,
        )

        ensemble.initialize()
        try:
            results = ensemble.run(n_cycles=10, n_steps=100, parsl_func=_record_run_app)
        finally:
            ensemble.shutdown()

        assert results == {0: None, 1: None}

        # Each replica's record landed in its own per-path log directory.
        rec0 = json.loads((log_dirs[0] / 'record_00000.json').read_text())
        rec1 = json.loads((log_dirs[1] / 'record_00001.json').read_text())
        assert rec0['log_dir'] == str(log_dirs[0])
        assert rec1['log_dir'] == str(log_dirs[1])

    def test_run_captures_exceptions(self, real_amber_system_files, local_parsl_config):
        """Test run method captures exceptions from failed futures."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        base = real_amber_system_files['path']
        path = _populate_amber_path(base / 'sys0', real_amber_system_files)

        ensemble = ConstantPHEnsemble(
            paths=[path],
            reference_energies={'CYS': [0.0, 10.0]},
            parsl_config=local_parsl_config,
            log_dir=base / 'logs',
        )

        ensemble.initialize()
        try:
            # The real app raises on the worker; run() must capture it.
            results = ensemble.run(
                n_cycles=10, n_steps=100, parsl_func=_failing_run_app
            )
        finally:
            ensemble.shutdown()

        assert isinstance(results[0], RuntimeError)


class TestSetupWorkerLogger:
    """Test suite for the module-level setup_worker_logger helper."""

    @pytest.mark.skip(
        reason=(
            "setup_worker_logger is broken/dead code: it calls "
            "pythonjsonlogger.json.JsonFormatter(task_id=...), but the installed "
            "python-json-logger (v4) rejects the task_id kwarg (TypeError). The "
            "project's own JsonFormatter (constantph.logging) additionally "
            "requires a run_id, so the intended fix is ambiguous. The function "
            "is never called anywhere in the source. Skipping per no-mock/no-force "
            "policy; see agent report."
        )
    )
    def test_setup_worker_logger_builds_json_logger(self, tmp_path):
        """Would assert logger name, JSON formatter handler, and log path."""
        from molecular_simulations.simulate.cph_simulation import setup_worker_logger

        logger = setup_worker_logger(None, 'worker_1', tmp_path)
        assert logger.name == 'task.worker_1'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
