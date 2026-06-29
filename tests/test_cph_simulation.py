"""
Unit tests for simulate/cph_simulation.py module

This module tests the constant pH simulation setup and ensemble management
classes used for pH-dependent molecular simulations.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mark tests that don't require OpenMM as unit tests
pytestmark = pytest.mark.unit


class TestConstantPHEnsembleInit:
    """Test suite for ConstantPHEnsemble class initialization."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_ensemble_init_defaults(self, mock_parsl: MagicMock) -> None:
        """Test ConstantPHEnsemble initialization with default parameters.

        Verifies that the class correctly initializes with default pH range
        and temperature values.
        """

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {
                "CYS": [0.0, 10.0],
                "ASP": [0.0, 5.0],
            }

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert ensemble.paths == [path]
            assert ensemble.ref_energies == ref_energies
            assert ensemble.parsl_config is mock_config
            assert ensemble.log_dir == log_dir
            # Default pH range is 0.5 to 13.5 in 1.0 increments
            assert len(ensemble.pHs) == 14
            assert ensemble.pHs[0] == 0.5
            # Temperature should be 300K by default (stored as float, not Quantity)
            assert ensemble.temperature == 300.0

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_ensemble_init_custom_phs(self, mock_parsl: MagicMock) -> None:
        """Test ConstantPHEnsemble with custom pH values."""

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}
            custom_phs = [4.0, 5.0, 6.0, 7.0, 8.0]

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                pHs=custom_phs,
            )

            assert ensemble.pHs == custom_phs

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_ensemble_init_custom_temperature(self, mock_parsl: MagicMock) -> None:
        """Test ConstantPHEnsemble with custom temperature.

        The temperature is multiplied by kelvin unit internally.
        """

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                temperature=310.0,
            )

            assert ensemble.temperature == 310.0

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_ensemble_init_with_variant_sel(self, mock_parsl: MagicMock) -> None:
        """Test ConstantPHEnsemble with custom variant selection string.

        The variant_sel parameter allows selecting specific residues for
        titration rather than all titratable residues.
        """
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                variant_sel="resid 10:50",
            )

            assert ensemble.variant_sel == "resid 10:50"

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_ensemble_init_generates_run_id(self, mock_parsl: MagicMock) -> None:
        """Test that initialization generates a unique run ID based on timestamp."""
        import re

        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            # run_id should match format YYYYMMDD_HHMMSS
            assert re.match(r"\d{8}_\d{6}", ensemble.run_id)


class TestConstantPHEnsembleParslManagement:
    """Test suite for Parsl initialization and shutdown."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_initialize_loads_parsl(self, mock_parsl: MagicMock) -> None:
        """Test that initialize() loads the Parsl configuration."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            mock_dfk = MagicMock()
            mock_parsl.load.return_value = mock_dfk

            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert ensemble.dfk is None
            ensemble.initialize()

            mock_parsl.load.assert_called_once_with(mock_config)
            assert ensemble.dfk is mock_dfk

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_shutdown_cleans_up_parsl(self, mock_parsl: MagicMock) -> None:
        """Test that shutdown() properly cleans up Parsl resources."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            mock_dfk = MagicMock()
            mock_parsl.load.return_value = mock_dfk

            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            ensemble.initialize()
            ensemble.shutdown()

            mock_dfk.cleanup.assert_called_once()
            mock_parsl.clear.assert_called()
            assert ensemble.dfk is None

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_shutdown_when_not_initialized(self, mock_parsl: MagicMock) -> None:
        """Test that shutdown() handles case when dfk is None."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            # Should not raise even when dfk is None
            ensemble.shutdown()
            mock_parsl.clear.assert_called()


class TestConstantPHEnsembleParams:
    """Test suite for params property."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_params_contains_required_keys(self, mock_parsl: MagicMock) -> None:
        """Test params property returns dictionary with all required keys.

        The params dictionary should contain simulation parameters for both
        explicit and implicit solvent, integrators, and pH values.
        """
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            # Check required keys
            assert "prmtop_file" in params
            assert "inpcrd_file" in params
            assert "pH" in params
            assert "relaxationSteps" in params
            assert "nonbonded_cutoff" in params
            assert "hmr" in params
            assert "implicit_cutoff" in params

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_params_explicit_args(self, mock_parsl: MagicMock) -> None:
        """Test params contains correct explicit solvent arguments."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            # Check explicit solvent parameters
            assert params["nonbonded_cutoff"] == 0.9
            assert params["hmr"] == 1.5

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_params_implicit_args(self, mock_parsl: MagicMock) -> None:
        """Test params contains correct implicit solvent arguments."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            # Check implicit solvent parameters
            assert params["implicit_cutoff"] == 2.0

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_params_ph_values(self, mock_parsl: MagicMock) -> None:
        """Test params contains correct pH values from initialization."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}
            custom_phs = [5.0, 6.0, 7.0]

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                pHs=custom_phs,
            )

            params = ensemble.get_params(path)

            assert params["pH"] == custom_phs

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_params_with_custom_temperature(self, mock_parsl: MagicMock) -> None:
        """Test ensemble stores custom temperature correctly."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                temperature=310.0,
            )

            # Temperature is stored on the ensemble object
            assert ensemble.temperature == 310.0

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_params_relaxation_steps(self, mock_parsl: MagicMock) -> None:
        """Test params contains correct relaxation steps."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            assert params["relaxationSteps"] == 1000

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_params_file_paths(self, mock_parsl: MagicMock) -> None:
        """Test params contains correct file paths."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            assert params["prmtop_file"] == path / "system.prmtop"
            assert params["inpcrd_file"] == path / "system.inpcrd"


class TestConstantPHEnsembleTemperatureHandling:
    """Test suite for temperature handling."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_temperature_stored_as_float(self, mock_parsl: MagicMock) -> None:
        """Test that temperature is stored as a float value in Kelvin."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                temperature=300.0,
            )

            # Temperature is stored as a float (assumed to be in Kelvin)
            assert ensemble.temperature == 300.0
            assert isinstance(ensemble.temperature, float)

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    @pytest.mark.parametrize("temp", [273.15, 300.0, 310.0, 350.0])
    def test_temperature_various_values(
        self, mock_parsl: MagicMock, temp: float
    ) -> None:
        """Test temperature handling with various physiological temperatures."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                temperature=temp,
            )

            assert ensemble.temperature == temp


class TestConstantPHEnsembleMultiplePaths:
    """Test suite for handling multiple simulation paths."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_multiple_paths_stored(self, mock_parsl: MagicMock) -> None:
        """Test that multiple paths are stored correctly."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = [base / f"system{i}" for i in range(5)]
            for p in paths:
                p.mkdir()

            log_dir = base / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=paths,
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert len(ensemble.paths) == 5
            assert ensemble.paths == paths

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_single_path(self, mock_parsl: MagicMock) -> None:
        """Test with single path."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert len(ensemble.paths) == 1
            assert ensemble.paths[0] == path


class TestConstantPHEnsembleReferenceEnergies:
    """Test suite for reference energy handling."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_reference_energies_stored(self, mock_parsl: MagicMock) -> None:
        """Test that reference energies are stored correctly."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {
                "CYS": [0.0, 10.0],
                "ASP": [0.0, 5.0],
                "GLU": [0.0, 6.0],
            }

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert ensemble.ref_energies == ref_energies
            assert "CYS" in ensemble.ref_energies
            assert "ASP" in ensemble.ref_energies
            assert "GLU" in ensemble.ref_energies


class TestConstantPHEnsemblePHRange:
    """Test suite for pH range handling."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_default_ph_range(self, mock_parsl: MagicMock) -> None:
        """Test default pH range is 0.5 to 13.5 in 1.0 steps."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            # Default should be [0.5, 1.5, 2.5, ..., 13.5]
            assert len(ensemble.pHs) == 14
            assert ensemble.pHs[0] == 0.5
            assert ensemble.pHs[-1] == 13.5

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_custom_ph_range(self, mock_parsl: MagicMock) -> None:
        """Test custom pH range."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}
            custom_phs = [2.0, 4.0, 6.0, 8.0, 10.0]

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                pHs=custom_phs,
            )

            assert ensemble.pHs == custom_phs
            assert len(ensemble.pHs) == 5

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_single_ph(self, mock_parsl: MagicMock) -> None:
        """Test with single pH value."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                pHs=[7.4],  # Single physiological pH
            )

            assert ensemble.pHs == [7.4]
            assert len(ensemble.pHs) == 1


class TestConstantPHEnsembleIntegrators:
    """Test suite for simulation configuration."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_relaxation_steps_in_params(self, mock_parsl: MagicMock) -> None:
        """Test that relaxationSteps is included in params."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            assert "relaxationSteps" in params
            assert params["relaxationSteps"] == 1000

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_hmr_in_params(self, mock_parsl: MagicMock) -> None:
        """Test that hydrogen mass repartitioning is included in params."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            assert "hmr" in params
            assert params["hmr"] == 1.5

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_cutoffs_in_params(self, mock_parsl: MagicMock) -> None:
        """Test that cutoff parameters are included in params."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            params = ensemble.get_params(path)

            assert params["nonbonded_cutoff"] == 0.9
            assert params["implicit_cutoff"] == 2.0


class TestConstantPHEnsembleVariantSel:
    """Test suite for variant selection handling."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_variant_sel_none_by_default(self, mock_parsl: MagicMock) -> None:
        """Test that variant_sel is None by default."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert ensemble.variant_sel is None

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_variant_sel_custom_string(self, mock_parsl: MagicMock) -> None:
        """Test custom variant selection string."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                variant_sel="resid 10 to 50",
            )

            assert ensemble.variant_sel == "resid 10 to 50"


class TestConstantPHEnsembleRun:
    """Test suite for run method."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    @patch("molecular_simulations.simulate.cph_simulation.run_cph_sim")
    def test_run_submits_futures_for_all_paths(
        self, mock_run_cph_sim: MagicMock, mock_parsl: MagicMock
    ) -> None:
        """Test run method submits jobs for all paths."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_run_cph_sim.return_value = mock_future

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = [base / f"system{i}" for i in range(3)]
            for p in paths:
                p.mkdir()

            log_dir = base / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=paths,
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            # Mock load_files and build_dicts
            mock_topology = MagicMock()
            mock_positions = MagicMock()

            with patch.object(
                ensemble, "load_files", return_value=(mock_topology, mock_positions)
            ), patch.object(ensemble, "build_dicts", return_value=({}, {})):
                with patch.object(ensemble, "get_params") as mock_get_params:
                    mock_get_params.return_value = {
                        "prmtop_file": base / "system.prmtop",
                        "inpcrd_file": base / "system.inpcrd",
                        "pH": [7.0],
                        "relaxationSteps": 1000,
                        "explicitArgs": {},
                        "implicitArgs": {},
                        "integrator": MagicMock(),
                        "relaxationIntegrator": MagicMock(),
                    }

                    ensemble.run(
                        n_cycles=10, n_steps=100, parsl_func=mock_run_cph_sim
                    )

            # Should call run_cph_sim for each path
            assert mock_run_cph_sim.call_count == 3
            # Should wait for all futures
            assert mock_future.result.call_count == 3

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    @patch("molecular_simulations.simulate.cph_simulation.run_cph_sim")
    def test_run_passes_correct_parameters(
        self, mock_run_cph_sim: MagicMock, mock_parsl: MagicMock
    ) -> None:
        """Test run method passes correct n_cycles and n_steps."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_run_cph_sim.return_value = mock_future

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            mock_topology = MagicMock()
            mock_positions = MagicMock()

            with patch.object(
                ensemble, "load_files", return_value=(mock_topology, mock_positions)
            ), patch.object(ensemble, "build_dicts", return_value=({}, {})):
                with patch.object(ensemble, "get_params") as mock_get_params:
                    mock_get_params.return_value = {
                        "prmtop_file": path / "system.prmtop",
                        "inpcrd_file": path / "system.inpcrd",
                        "pH": [7.0],
                        "relaxationSteps": 1000,
                        "explicitArgs": {},
                        "implicitArgs": {},
                        "integrator": MagicMock(),
                        "relaxationIntegrator": MagicMock(),
                    }

                    ensemble.run(
                        n_cycles=250, n_steps=750, parsl_func=mock_run_cph_sim
                    )

            call_args = mock_run_cph_sim.call_args
            assert call_args[0][2] == 250  # n_cycles
            assert call_args[0][3] == 750  # n_steps


class TestConstantPHEnsembleRunWithDefaults:
    """Test suite for run method with default parameters."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    @patch("molecular_simulations.simulate.cph_simulation.run_cph_sim")
    def test_run_uses_default_parameters(
        self, mock_run_cph_sim: MagicMock, mock_parsl: MagicMock
    ) -> None:
        """Test run method uses default n_cycles=500 and n_steps=500."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_run_cph_sim.return_value = mock_future

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            mock_topology = MagicMock()
            mock_positions = MagicMock()

            with patch.object(
                ensemble, "load_files", return_value=(mock_topology, mock_positions)
            ), patch.object(ensemble, "build_dicts", return_value=({}, {})):
                with patch.object(ensemble, "get_params") as mock_get_params:
                    mock_get_params.return_value = {
                        "prmtop_file": path / "system.prmtop",
                        "inpcrd_file": path / "system.inpcrd",
                        "pH": [7.0],
                        "relaxationSteps": 1000,
                        "explicitArgs": {},
                        "implicitArgs": {},
                        "integrator": MagicMock(),
                        "relaxationIntegrator": MagicMock(),
                    }

                    # Call with parsl_func to use mocked function
                    ensemble.run(parsl_func=mock_run_cph_sim)

            call_args = mock_run_cph_sim.call_args
            assert call_args[0][2] == 500  # default n_cycles
            assert call_args[0][3] == 500  # default n_steps


class TestConstantPHEnsembleLogParams:
    """Test suite for log parameter generation."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    @patch("molecular_simulations.simulate.cph_simulation.run_cph_sim")
    def test_run_generates_unique_task_ids(
        self, mock_run_cph_sim: MagicMock, mock_parsl: MagicMock
    ) -> None:
        """Test run generates unique task IDs for each path."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_run_cph_sim.return_value = mock_future

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = [base / f"system{i}" for i in range(3)]
            for p in paths:
                p.mkdir()

            log_dir = base / "logs"

            mock_config = MagicMock()
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=paths,
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            mock_topology = MagicMock()
            mock_positions = MagicMock()

            with patch.object(
                ensemble, "load_files", return_value=(mock_topology, mock_positions)
            ), patch.object(ensemble, "build_dicts", return_value=({}, {})):
                with patch.object(ensemble, "get_params") as mock_get_params:
                    mock_get_params.return_value = {
                        "prmtop_file": base / "system.prmtop",
                        "inpcrd_file": base / "system.inpcrd",
                        "pH": [7.0],
                        "relaxationSteps": 1000,
                        "explicitArgs": {},
                        "implicitArgs": {},
                        "integrator": MagicMock(),
                        "relaxationIntegrator": MagicMock(),
                    }

                    ensemble.run(
                        n_cycles=10, n_steps=100, parsl_func=mock_run_cph_sim
                    )

            # Check each call had unique task_id
            task_ids = []
            for call in mock_run_cph_sim.call_args_list:
                log_params = call[0][4]
                task_ids.append(log_params["task_id"])

            assert len(set(task_ids)) == 3
            assert "00000" in task_ids
            assert "00001" in task_ids
            assert "00002" in task_ids


class TestConstantPHEnsembleLoadFiles:
    """Test suite for load_files method."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    @patch("molecular_simulations.simulate.cph_simulation.AmberInpcrdFile")
    @patch("molecular_simulations.simulate.cph_simulation.AmberPrmtopFile")
    def test_load_files_returns_topology_and_positions(
        self, mock_prmtop_cls, mock_inpcrd_cls, mock_parsl
    ):
        """Test load_files returns (topology, positions) from AMBER files."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_top = MagicMock()
        mock_top.topology = MagicMock()
        mock_prmtop_cls.return_value = mock_top

        mock_crd = MagicMock()
        mock_crd.positions = MagicMock()
        mock_inpcrd_cls.return_value = mock_crd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / "logs"
            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=MagicMock(),
                log_dir=log_dir,
            )

            topology, positions = ensemble.load_files(path)

            mock_prmtop_cls.assert_called_once_with(str(path / "system.prmtop"))
            mock_inpcrd_cls.assert_called_once_with(str(path / "system.inpcrd"))
            assert topology is mock_top.topology
            assert positions is mock_crd.positions


class TestConstantPHEnsembleBuildDicts:
    """Test suite for build_dicts method."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    @patch("molecular_simulations.simulate.cph_simulation.mda")
    def test_build_dicts_identifies_titratable_residues(self, mock_mda, mock_parsl):
        """Test build_dicts identifies titratable residues in topology."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        # Create mock topology with ASP and LYS residues
        mock_residues = []
        for i, (name, idx) in enumerate(
            [("ALA", 0), ("ASP", 1), ("LYS", 2), ("GLY", 3)]
        ):
            res = MagicMock()
            res.name = name
            res.index = idx
            mock_residues.append(res)

        mock_top = MagicMock()
        mock_top.residues.return_value = mock_residues

        # Mock MDAnalysis Universe
        mock_u = MagicMock()
        protein_atoms = MagicMock()
        protein_atoms.__getitem__ = MagicMock(
            side_effect=lambda x: MagicMock(resindex=x)
        )
        first_atom = MagicMock()
        first_atom.resindex = 0
        last_atom = MagicMock()
        last_atom.resindex = 3
        protein_atoms.__getitem__ = MagicMock(
            side_effect=lambda idx: first_atom if idx == 0 else last_atom
        )
        mock_u.select_atoms.return_value = protein_atoms
        mock_mda.Universe.return_value = mock_u

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            # Create dummy files for MDAnalysis
            (path / "system.prmtop").write_text("mock")
            (path / "system.inpcrd").write_text("mock")
            log_dir = path / "logs"

            ref_energies = {
                "ASP": [0.0, 5.0],
                "LYS": [0.0, 10.0],
                "CYS": [0.0, 7.0],
            }

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=MagicMock(),
                log_dir=log_dir,
            )

            variants, ref_e = ensemble.build_dicts(path, mock_top)

            # ASP at index 1 should be found, LYS at index 2 should be found
            # Termini (index 0 and 3) are excluded
            assert 1 in variants
            assert variants[1] == ["ASP", "ASH"]
            assert 2 in variants
            assert variants[2] == ["LYN", "LYS"]


class TestConstantPHEnsembleBuildDictsWithVariantSel:
    """Test build_dicts with variant_sel filtering."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    @patch("molecular_simulations.simulate.cph_simulation.mda")
    def test_build_dicts_with_variant_sel_filters_residues(self, mock_mda, mock_parsl):
        """Test build_dicts with variant_sel restricts titratable residues."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        # Topology with ASP at indices 1 and 3
        mock_residues = []
        for i, (name, idx) in enumerate(
            [("ALA", 0), ("ASP", 1), ("GLY", 2), ("ASP", 3), ("ALA", 4)]
        ):
            res = MagicMock()
            res.name = name
            res.index = idx
            mock_residues.append(res)

        mock_top = MagicMock()
        mock_top.residues.return_value = mock_residues

        # Mock Universe for MDAnalysis
        mock_u = MagicMock()
        protein_sel = MagicMock()
        protein_sel.__getitem__ = MagicMock(
            side_effect=lambda idx: MagicMock(resindex={0: 0, -1: 4}[idx])
        )
        protein_res = MagicMock()
        protein_res.resindices = np.array([0, 1, 2, 3, 4])
        protein_sel.residues = protein_res

        var_sel = MagicMock()
        var_res = MagicMock()
        var_res.resindices = np.array([1])
        var_sel.residues = var_res

        mock_u.select_atoms.side_effect = [protein_sel, var_sel]
        mock_mda.Universe.return_value = mock_u

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "system.prmtop").write_text("mock")
            (path / "system.inpcrd").write_text("mock")
            log_dir = path / "logs"

            ref_energies = {"ASP": [0.0, 5.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=MagicMock(),
                log_dir=log_dir,
                variant_sel="resid 2",
            )

            variants, ref_e = ensemble.build_dicts(path, mock_top)

            assert 1 in variants
            assert 3 not in variants


class TestConstantPHEnsembleRunMethods:
    """Test suite for run method."""

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_run_with_list_log_dir(self, mock_parsl):
        """Test run method uses per-path log_dir when log_dir is a list."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_run_func = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_run_func.return_value = mock_future

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = [base / "sys0", base / "sys1"]
            for p in paths:
                p.mkdir()
            log_dirs = [base / "logs0", base / "logs1"]

            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=paths,
                reference_energies=ref_energies,
                parsl_config=MagicMock(),
                log_dir=log_dirs,
            )

            mock_topology = MagicMock()
            mock_positions = MagicMock()

            with patch.object(
                ensemble, "load_files", return_value=(mock_topology, mock_positions)
            ), patch.object(ensemble, "build_dicts", return_value=({}, {})):
                with patch.object(ensemble, "get_params") as mock_get_params:
                    mock_get_params.return_value = {
                        "prmtop_file": base / "system.prmtop",
                        "inpcrd_file": base / "system.inpcrd",
                        "pH": [7.0],
                        "relaxationSteps": 1000,
                    }

                    results = ensemble.run(
                        n_cycles=10, n_steps=100, parsl_func=mock_run_func
                    )

            # Verify per-path log dirs were used
            assert mock_run_func.call_count == 2
            call0_log_params = mock_run_func.call_args_list[0][0][4]
            call1_log_params = mock_run_func.call_args_list[1][0][4]
            assert call0_log_params["log_dir"] == log_dirs[0]
            assert call1_log_params["log_dir"] == log_dirs[1]

    @patch("molecular_simulations.simulate.cph_simulation.parsl")
    def test_run_captures_exceptions(self, mock_parsl):
        """Test run method captures exceptions from failed futures."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_run_func = MagicMock()
        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("Simulation failed")
        mock_run_func.return_value = mock_future

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = [base / "sys0"]
            for p in paths:
                p.mkdir()

            ref_energies = {"CYS": [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=paths,
                reference_energies=ref_energies,
                parsl_config=MagicMock(),
                log_dir=base / "logs",
            )

            with patch.object(
                ensemble, "load_files", return_value=(MagicMock(), MagicMock())
            ), patch.object(ensemble, "build_dicts", return_value=({}, {})):
                with patch.object(
                    ensemble,
                    "get_params",
                    return_value={
                        "prmtop_file": base / "system.prmtop",
                        "inpcrd_file": base / "system.inpcrd",
                        "pH": [7.0],
                        "relaxationSteps": 1000,
                    },
                ):
                    results = ensemble.run(
                        n_cycles=10, n_steps=100, parsl_func=mock_run_func
                    )

            assert isinstance(results[0], RuntimeError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
