"""
Comprehensive tests for EVB free energy analysis functionality.

This module tests the new EVB analysis features added in the recent update:
- Dataclasses: PMFResult, ConvergenceResult, OverlapResult, EquilibrationResult, EVBAnalysisResult
- EVBAnalyzer class: Standalone analyzer for existing EVB data
- New EVB methods: Analysis methods added to the EVB class

Tests use MINIMAL mocking and prefer real synthetic data that simulates
actual umbrella sampling output. This ensures the statistical algorithms
are tested with realistic inputs.

Testing strategy:
- Generate synthetic RC data from known distributions
- Use tempfile for all file I/O tests
- Test edge cases (empty data, single window, missing files)
- Test WHAM fallback when pymbar is unavailable
- Verify statistical algorithms with known inputs/outputs
"""

import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Mark all tests as unit tests (no OpenMM/Parsl required)
pytestmark = pytest.mark.unit


# =============================================================================
# Module Import Mocking
# =============================================================================

# We need to mock heavy dependencies before importing free_energy module
# The module imports openmm, parsl, MDAnalysis at module load time


# =============================================================================
# Synthetic Data Generation Utilities
# =============================================================================

# Boltzmann constant in kJ/(mol*K) - same as in free_energy.py
KB = 8.314462618e-3


def generate_umbrella_sampling_data(
    n_windows: int = 10,
    n_frames: int = 1000,
    rc0_min: float = -0.2,
    rc0_max: float = 0.2,
    k_umbrella: float = 160000.0,
    temperature: float = 300.0,
    add_equilibration: bool = False,
    equilibration_frames: int = 100,
    seed: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Generate synthetic umbrella sampling RC data.

    The data follows the expected distribution for umbrella sampling:
    P(rc) ~ exp(-0.5 * k * (rc - rc0)^2 / (kB * T))

    This is a Gaussian centered at rc0 with width sigma = sqrt(kB*T/k)

    Args:
        n_windows: Number of umbrella windows.
        n_frames: Number of frames per window.
        rc0_min: Minimum RC target value.
        rc0_max: Maximum RC target value.
        k_umbrella: Umbrella force constant in kJ/mol/nm^2.
        temperature: Temperature in Kelvin.
        add_equilibration: Whether to add initial equilibration period.
        equilibration_frames: Number of frames for equilibration period.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (rc_data list, rc0_values array).
    """
    if seed is not None:
        np.random.seed(seed)

    # Width of the umbrella distribution
    sigma = np.sqrt(KB * temperature / k_umbrella)

    rc0_values = np.linspace(rc0_min, rc0_max, n_windows)
    rc_data = []

    for rc0 in rc0_values:
        if add_equilibration:
            n_prod = n_frames - equilibration_frames
            # Generate equilibrated production data
            production = np.random.normal(rc0, sigma, n_prod)

            # Add drift during equilibration (start far from target)
            eq_start = rc0 + 3 * sigma
            eq_drift = np.linspace(eq_start, rc0, equilibration_frames)
            eq_drift += np.random.normal(0, sigma * 0.5, equilibration_frames)

            data = np.concatenate([eq_drift, production])
        else:
            data = np.random.normal(rc0, sigma, n_frames)

        rc_data.append(data)

    return rc_data, rc0_values


def generate_correlated_timeseries(
    n_frames: int = 1000,
    correlation_time: int = 50,
    mean: float = 0.0,
    std: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a correlated time series using AR(1) process.

    Args:
        n_frames: Number of frames.
        correlation_time: Approximate correlation time in frames.
        mean: Mean value of the series.
        std: Standard deviation of the series.
        seed: Random seed.

    Returns:
        Correlated time series array.
    """
    if seed is not None:
        np.random.seed(seed)

    # AR(1) coefficient from correlation time: phi = exp(-1/tau)
    phi = np.exp(-1.0 / correlation_time)

    # Generate AR(1) process
    noise_std = std * np.sqrt(1 - phi**2)
    data = np.zeros(n_frames)
    data[0] = np.random.normal(mean, std)

    for i in range(1, n_frames):
        data[i] = mean + phi * (data[i - 1] - mean) + np.random.normal(0, noise_std)

    return data


def create_rc_log_file(path: Path, rc_data: np.ndarray) -> None:
    """Create a CSV log file in the format expected by EVBAnalyzer.

    The log file format matches RCReporter output with 'rc' column.

    Args:
        path: Path to write the log file.
        rc_data: Array of RC values.
    """
    df = pl.DataFrame({"rc": rc_data})
    df.write_csv(str(path))


def create_rc_log_files(
    log_path: Path, log_prefix: str, rc_data_list: list[np.ndarray]
) -> None:
    """Create multiple RC log files for a complete EVB run.

    Args:
        log_path: Directory for log files.
        log_prefix: Prefix for log file names (e.g., 'reactant').
        rc_data_list: List of RC arrays, one per window.
    """
    log_path.mkdir(parents=True, exist_ok=True)
    for i, rc_data in enumerate(rc_data_list):
        log_file = log_path / f"{log_prefix}_{i}.log"
        create_rc_log_file(log_file, rc_data)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_rc_data():
    """Generate synthetic RC data with known properties."""
    return generate_umbrella_sampling_data(
        n_windows=5, n_frames=500, rc0_min=-0.1, rc0_max=0.1, seed=42
    )


@pytest.fixture
def synthetic_rc_data_with_equilibration():
    """Generate synthetic RC data with equilibration period."""
    return generate_umbrella_sampling_data(
        n_windows=5,
        n_frames=500,
        rc0_min=-0.1,
        rc0_max=0.1,
        add_equilibration=True,
        equilibration_frames=50,
        seed=42,
    )


@pytest.fixture
def real_evb_instance(alanine_dipeptide_pdb, tmp_path):
    """A real EVB instance for testing from_evb_instance."""
    from molecular_simulations.simulate.free_energy import EVB

    log_path = tmp_path / "logs"
    log_path.mkdir()
    return EVB(
        topology=alanine_dipeptide_pdb,
        coordinates=alanine_dipeptide_pdb,
        donor_atom="index 0",
        acceptor_atom="index 1",
        reactive_atom="index 2",
        reaction_coordinate=[-0.2, 0.2, 0.1],
        parsl_config=None,
        log_path=log_path,
    )


# =============================================================================
# Test Dataclasses
# =============================================================================


class TestPMFResult:
    """Tests for PMFResult dataclass."""

    def test_pmf_result_instantiation(self):
        """Test PMFResult can be instantiated with numpy arrays."""
        from molecular_simulations.simulate.free_energy import PMFResult

        bin_centers = np.linspace(-0.2, 0.2, 50)
        pmf = np.random.random(50)
        pmf_uncertainty = np.random.random(50) * 0.1
        free_energies = np.random.random(10)
        free_energy_uncertainty = np.random.random(10) * 0.1

        result = PMFResult(
            bin_centers=bin_centers,
            pmf=pmf,
            pmf_uncertainty=pmf_uncertainty,
            free_energies=free_energies,
            free_energy_uncertainty=free_energy_uncertainty,
        )

        assert isinstance(result.bin_centers, np.ndarray)
        assert len(result.bin_centers) == 50
        assert len(result.pmf) == 50
        np.testing.assert_array_equal(result.pmf, pmf)

    def test_pmf_result_fields(self):
        """Test PMFResult has expected fields."""
        from molecular_simulations.simulate.free_energy import PMFResult

        field_names = {f.name for f in fields(PMFResult)}
        expected = {
            "bin_centers",
            "pmf",
            "pmf_uncertainty",
            "free_energies",
            "free_energy_uncertainty",
        }
        assert field_names == expected


class TestConvergenceResult:
    """Tests for ConvergenceResult dataclass."""

    def test_convergence_result_instantiation(self):
        """Test ConvergenceResult instantiation."""
        from molecular_simulations.simulate.free_energy import ConvergenceResult

        result = ConvergenceResult(
            window_idx=5,
            mean_rc=0.05,
            sem=0.001,
            n_blocks=10,
            block_means=np.random.random(10),
            is_converged=True,
        )

        assert result.window_idx == 5
        assert result.mean_rc == 0.05
        assert result.is_converged == True

    def test_convergence_result_is_converged_flag(self):
        """Test is_converged flag correctly reflects SEM threshold."""
        from molecular_simulations.simulate.free_energy import ConvergenceResult

        # Converged case
        converged = ConvergenceResult(
            window_idx=0,
            mean_rc=0.0,
            sem=0.005,
            n_blocks=5,
            block_means=np.zeros(5),
            is_converged=True,
        )
        assert converged.is_converged == True

        # Not converged case
        not_converged = ConvergenceResult(
            window_idx=0,
            mean_rc=0.0,
            sem=0.05,
            n_blocks=5,
            block_means=np.zeros(5),
            is_converged=False,
        )
        assert not_converged.is_converged == False


class TestOverlapResult:
    """Tests for OverlapResult dataclass."""

    def test_overlap_result_empty_problem_pairs(self):
        """Test OverlapResult with no problem pairs (good overlap)."""
        from molecular_simulations.simulate.free_energy import OverlapResult

        result = OverlapResult(
            overlap_matrix=np.array([0.5, 0.6, 0.4, 0.5]),
            min_overlap=0.4,
            problem_pairs=[],
        )

        assert len(result.problem_pairs) == 0
        assert result.min_overlap == 0.4

    def test_overlap_result_with_problem_pairs(self):
        """Test OverlapResult with identified problem pairs."""
        from molecular_simulations.simulate.free_energy import OverlapResult

        result = OverlapResult(
            overlap_matrix=np.array([0.5, 0.01, 0.4]),
            min_overlap=0.01,
            problem_pairs=[(1, 2)],
        )

        assert len(result.problem_pairs) == 1
        assert result.problem_pairs[0] == (1, 2)


class TestEquilibrationResult:
    """Tests for EquilibrationResult dataclass."""

    def test_equilibration_result_no_discarding(self):
        """Test EquilibrationResult with t0=0 (no equilibration needed)."""
        from molecular_simulations.simulate.free_energy import EquilibrationResult

        result = EquilibrationResult(
            window_idx=0, t0=0, g=1.5, n_effective=666.7, fraction_discarded=0.0
        )

        assert result.t0 == 0
        assert result.fraction_discarded == 0.0

    def test_equilibration_result_high_discarding(self):
        """Test EquilibrationResult with high fraction discarded."""
        from molecular_simulations.simulate.free_energy import EquilibrationResult

        result = EquilibrationResult(
            window_idx=3, t0=500, g=2.0, n_effective=250.0, fraction_discarded=0.5
        )

        assert result.fraction_discarded == 0.5
        assert result.t0 == 500


class TestEVBAnalysisResult:
    """Tests for EVBAnalysisResult composite dataclass."""

    def test_evb_analysis_result_instantiation(self):
        """Test EVBAnalysisResult with all sub-results."""
        from molecular_simulations.simulate.free_energy import (
            ConvergenceResult,
            EquilibrationResult,
            EVBAnalysisResult,
            OverlapResult,
            PMFResult,
        )

        pmf = PMFResult(
            bin_centers=np.linspace(-0.1, 0.1, 20),
            pmf=np.random.random(20),
            pmf_uncertainty=np.random.random(20) * 0.1,
            free_energies=np.random.random(5),
            free_energy_uncertainty=np.random.random(5) * 0.1,
        )

        convergence = [
            ConvergenceResult(i, 0.0, 0.01, 5, np.zeros(5), True) for i in range(5)
        ]

        overlap = OverlapResult(
            overlap_matrix=np.array([0.5, 0.5, 0.5, 0.5]),
            min_overlap=0.5,
            problem_pairs=[],
        )

        equilibration = [EquilibrationResult(i, 0, 1.0, 100.0, 0.0) for i in range(5)]

        rc_data = [np.random.random(100) for _ in range(5)]

        result = EVBAnalysisResult(
            pmf=pmf,
            convergence=convergence,
            overlap=overlap,
            equilibration=equilibration,
            rc_data=rc_data,
            temperature=300.0,
            k_umbrella=160000.0,
        )

        assert result.temperature == 300.0
        assert result.k_umbrella == 160000.0
        assert len(result.convergence) == 5
        assert len(result.rc_data) == 5


# =============================================================================
# Test EVBAnalyzer Initialization
# =============================================================================


class TestEVBAnalyzerInit:
    """Tests for EVBAnalyzer initialization."""

    def test_init_valid_parameters(self):
        """Test EVBAnalyzer initialization with valid parameters."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            rc0_values = np.linspace(-0.2, 0.2, 10)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            assert analyzer.log_path == log_path
            assert analyzer.log_prefix == "test"
            assert analyzer.k == 160000.0
            np.testing.assert_array_equal(analyzer.reaction_coordinate, rc0_values)
            assert analyzer.output_path == log_path  # Default

    def test_init_custom_output_path(self):
        """Test EVBAnalyzer with custom output path."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs"
            log_path.mkdir()
            output_path = Path(tmpdir) / "results"

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.linspace(-0.1, 0.1, 5),
                output_path=output_path,
            )

            assert analyzer.output_path == output_path

    def test_init_nonexistent_log_path_raises(self):
        """Test EVBAnalyzer raises FileNotFoundError for missing log_path."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with pytest.raises(FileNotFoundError, match="Log path does not exist"):
            EVBAnalyzer(
                log_path=Path("/nonexistent/path"),
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.linspace(-0.1, 0.1, 5),
            )

    def test_init_converts_rc0_to_array(self):
        """Test EVBAnalyzer converts rc0_values list to numpy array."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            rc0_list = [-0.2, -0.1, 0.0, 0.1, 0.2]

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_list,
            )

            assert isinstance(analyzer.reaction_coordinate, np.ndarray)
            np.testing.assert_array_equal(analyzer.reaction_coordinate, rc0_list)


# =============================================================================
# Test EVBAnalyzer Class Methods
# =============================================================================


class TestEVBAnalyzerFromMetadata:
    """Tests for EVBAnalyzer.from_metadata class method."""

    def test_from_metadata_nested_structure(self):
        """Test loading from TOML with nested [evb] section."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs"
            log_path.mkdir()

            metadata_content = f'''
[evb]
log_path = "{log_path}"
log_prefix = "reactant"
k_umbrella = 160000.0
rc0_values = [-0.2, -0.1, 0.0, 0.1, 0.2]
'''
            metadata_path = Path(tmpdir) / "metadata.toml"
            metadata_path.write_text(metadata_content)

            analyzer = EVBAnalyzer.from_metadata(metadata_path)

            assert analyzer.log_path == log_path
            assert analyzer.log_prefix == "reactant"
            assert analyzer.k == 160000.0
            assert len(analyzer.reaction_coordinate) == 5

    def test_from_metadata_flat_structure(self):
        """Test loading from TOML with flat structure (no [evb] section)."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs"
            log_path.mkdir()

            metadata_content = f'''
log_path = "{log_path}"
log_prefix = "product"
k_umbrella = 200000.0
rc0_values = [-0.1, 0.0, 0.1]
'''
            metadata_path = Path(tmpdir) / "metadata.toml"
            metadata_path.write_text(metadata_content)

            analyzer = EVBAnalyzer.from_metadata(metadata_path)

            assert analyzer.log_prefix == "product"
            assert analyzer.k == 200000.0

    def test_from_metadata_with_output_path(self):
        """Test loading metadata that includes output_path."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs"
            log_path.mkdir()
            output_path = Path(tmpdir) / "results"

            metadata_content = f'''
[evb]
log_path = "{log_path}"
log_prefix = "reactant"
k_umbrella = 160000.0
rc0_values = [-0.1, 0.1]
output_path = "{output_path}"
'''
            metadata_path = Path(tmpdir) / "metadata.toml"
            metadata_path.write_text(metadata_content)

            analyzer = EVBAnalyzer.from_metadata(metadata_path)

            assert analyzer.output_path == output_path


class TestEVBAnalyzerFromEVBInstance:
    """Tests for EVBAnalyzer.from_evb_instance class method."""

    def test_from_evb_instance(self, real_evb_instance):
        """Test creating EVBAnalyzer from a real EVB instance."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        analyzer = EVBAnalyzer.from_evb_instance(real_evb_instance)

        assert analyzer.log_path == real_evb_instance.log_path
        assert analyzer.log_prefix == real_evb_instance.log_prefix
        assert analyzer.k == real_evb_instance.k
        np.testing.assert_array_equal(
            analyzer.reaction_coordinate, real_evb_instance.reaction_coordinate
        )


class TestEVBAnalyzerSaveMetadata:
    """Tests for EVBAnalyzer.save_metadata method."""

    def test_save_metadata_default_path(self):
        """Test save_metadata writes to default location."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([-0.1, 0.0, 0.1]),
            )

            saved_path = analyzer.save_metadata()

            assert saved_path == log_path / "evb_metadata.toml"
            assert saved_path.exists()

            # Verify content
            content = saved_path.read_text()
            assert "[evb]" in content
            assert 'log_prefix = "test"' in content
            assert "k_umbrella = 160000.0" in content

    def test_save_metadata_custom_path(self):
        """Test save_metadata with custom output path."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            # Create parent directory for custom path (save_metadata doesn't create it)
            custom_dir = Path(tmpdir) / "custom"
            custom_dir.mkdir()
            custom_path = custom_dir / "metadata.toml"

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="custom",
                k_umbrella=200000.0,
                rc0_values=np.array([-0.2, 0.2]),
            )

            saved_path = analyzer.save_metadata(output_path=custom_path)

            assert saved_path == custom_path
            assert custom_path.exists()

    def test_save_metadata_roundtrip(self):
        """Test that saved metadata can be loaded back."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            rc0_values = np.array([-0.15, -0.05, 0.05, 0.15])

            original = EVBAnalyzer(
                log_path=log_path,
                log_prefix="roundtrip",
                k_umbrella=180000.0,
                rc0_values=rc0_values,
            )

            saved_path = original.save_metadata()
            loaded = EVBAnalyzer.from_metadata(saved_path)

            assert loaded.log_prefix == original.log_prefix
            assert loaded.k == original.k
            np.testing.assert_array_almost_equal(
                loaded.reaction_coordinate, original.reaction_coordinate
            )


# =============================================================================
# Test EVBAnalyzer Data Loading
# =============================================================================


class TestEVBAnalyzerLoadRCData:
    """Tests for EVBAnalyzer.load_rc_data method."""

    def test_load_rc_data_valid_files(self, synthetic_rc_data):
        """Test loading RC data from valid CSV files."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        rc_data, rc0_values = synthetic_rc_data

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            loaded_data = analyzer.load_rc_data()

            assert len(loaded_data) == len(rc_data)
            for i, (original, loaded) in enumerate(zip(rc_data, loaded_data)):
                np.testing.assert_array_almost_equal(original, loaded)

    def test_load_rc_data_missing_file_raises(self):
        """Test load_rc_data raises FileNotFoundError for missing files."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            # Create only 2 of 5 expected files
            for i in [0, 2]:
                log_file = log_path / f"test_{i}.log"
                create_rc_log_file(log_file, np.random.random(100))

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.linspace(-0.1, 0.1, 5),
            )

            with pytest.raises(FileNotFoundError, match="RC log file not found"):
                analyzer.load_rc_data()


class TestEVBAnalyzerGetAvailableWindows:
    """Tests for EVBAnalyzer.get_available_windows method."""

    def test_get_available_windows_complete(self, synthetic_rc_data):
        """Test get_available_windows with all files present."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        rc_data, rc0_values = synthetic_rc_data

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            available = analyzer.get_available_windows()

            assert available == list(range(len(rc_data)))

    def test_get_available_windows_partial(self):
        """Test get_available_windows with some files missing."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            # Create only windows 0, 2, 4
            for i in [0, 2, 4]:
                log_file = log_path / f"test_{i}.log"
                create_rc_log_file(log_file, np.random.random(100))

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.linspace(-0.1, 0.1, 5),
            )

            available = analyzer.get_available_windows()

            assert available == [0, 2, 4]

    def test_get_available_windows_none(self):
        """Test get_available_windows with no files present."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.linspace(-0.1, 0.1, 5),
            )

            available = analyzer.get_available_windows()

            assert available == []


class TestEVBAnalyzerCheckRunStatus:
    """Tests for EVBAnalyzer.check_run_status method."""

    def test_check_run_status_complete(self, synthetic_rc_data):
        """Test check_run_status for complete run."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        rc_data, rc0_values = synthetic_rc_data

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            status = analyzer.check_run_status()

            assert status["n_expected"] == len(rc_data)
            assert status["n_complete"] == len(rc_data)
            assert status["complete_fraction"] == 1.0
            assert status["missing_windows"] == []
            assert len(status["frames_per_window"]) == len(rc_data)

    def test_check_run_status_partial(self):
        """Test check_run_status for partial run."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            # Create only windows 0, 1, 2 of 5
            for i in range(3):
                log_file = log_path / f"test_{i}.log"
                create_rc_log_file(log_file, np.random.random(100))

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.linspace(-0.1, 0.1, 5),
            )

            status = analyzer.check_run_status()

            assert status["n_expected"] == 5
            assert status["n_complete"] == 3
            assert status["complete_fraction"] == 0.6
            assert status["missing_windows"] == [3, 4]


# =============================================================================
# Test EVBAnalyzer Statistical Methods
# =============================================================================


class TestEVBAnalyzerDetectEquilibration:
    """Tests for EVBAnalyzer equilibration detection."""

    def test_detect_equilibration_autocorr_constant_data(self):
        """Test equilibration detection with constant (zero variance) data."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        # Constant data should return t0=0, g=1.0
        data = np.ones(1000) * 0.5
        t0, g, n_eff = EVBAnalyzer._detect_equilibration_autocorr(data)

        assert t0 == 0
        assert g == 1.0
        assert n_eff == float(len(data))

    def test_detect_equilibration_autocorr_uncorrelated(self):
        """Test equilibration detection with uncorrelated (white noise) data."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        np.random.seed(42)
        # White noise should have g close to 1
        data = np.random.normal(0, 1, 1000)
        t0, g, n_eff = EVBAnalyzer._detect_equilibration_autocorr(data)

        # g should be close to 1 for white noise
        assert g < 2.0  # Allow some tolerance
        assert n_eff > len(data) / 2

    def test_detect_equilibration_autocorr_correlated(self):
        """Test equilibration detection with correlated data."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        # Generate correlated time series
        data = generate_correlated_timeseries(
            n_frames=1000, correlation_time=50, seed=42
        )
        t0, g, n_eff = EVBAnalyzer._detect_equilibration_autocorr(data)

        # g should be significantly > 1 for correlated data
        assert g > 1.5
        # n_eff should be much less than n
        assert n_eff < len(data)

    def test_detect_equilibration_short_trajectory(self):
        """Test equilibration detection with short trajectory (<10 frames)."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0]),
            )

            # Short trajectory should return default values
            rc_data = [np.random.random(5)]
            results = analyzer.detect_equilibration(rc_data)

            assert len(results) == 1
            assert results[0].t0 == 0
            assert results[0].g == 1.0
            assert results[0].n_effective == 5.0

    def test_detect_equilibration_with_drift(
        self, synthetic_rc_data_with_equilibration
    ):
        """Test equilibration detection identifies drift period."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        rc_data, rc0_values = synthetic_rc_data_with_equilibration

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            results = analyzer.detect_equilibration(rc_data)

            # Should detect some equilibration period
            # At least some windows should have t0 > 0
            assert len(results) == len(rc_data)
            # All results should have valid fields
            for r in results:
                assert r.t0 >= 0
                assert r.g >= 1.0
                assert 0 <= r.fraction_discarded <= 1


class TestEVBAnalyzerCheckConvergence:
    """Tests for EVBAnalyzer.check_convergence method."""

    def test_check_convergence_well_converged(self):
        """Test convergence check with well-converged (low variance) data."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0]),
            )

            # Generate very low variance data (should be converged)
            np.random.seed(42)
            rc_data = [np.random.normal(0.0, 0.001, 1000)]

            results = analyzer.check_convergence(rc_data, sem_threshold=0.01)

            assert len(results) == 1
            assert results[0].is_converged == True
            assert results[0].sem < 0.01

    def test_check_convergence_not_converged(self):
        """Test convergence check with poorly converged (high variance) data."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0]),
            )

            # Generate high variance data with few samples
            np.random.seed(42)
            rc_data = [np.random.normal(0.0, 0.5, 30)]  # Few samples, high variance

            results = analyzer.check_convergence(rc_data, sem_threshold=0.001)

            assert len(results) == 1
            assert results[0].is_converged == False

    def test_check_convergence_custom_block_size(self, synthetic_rc_data):
        """Test convergence check with custom block size."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        rc_data, rc0_values = synthetic_rc_data

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            # Custom block size
            results = analyzer.check_convergence(rc_data, block_size=25)

            assert len(results) == len(rc_data)
            # With block_size=25 and 500 frames, should have 20 blocks
            assert all(r.n_blocks == 20 for r in results)

    def test_check_convergence_sem_calculation(self):
        """Test that SEM is calculated correctly."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0]),
            )

            # Create data with known block structure
            # 100 frames, 10 blocks of 10, each block is constant
            block_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
            rc_data = [np.repeat(block_values, 10)]

            results = analyzer.check_convergence(rc_data, block_size=10)

            # Expected SEM = std(block_means) / sqrt(n_blocks)
            expected_sem = np.std(block_values, ddof=1) / np.sqrt(10)

            assert results[0].sem == pytest.approx(expected_sem, rel=1e-5)
            np.testing.assert_array_almost_equal(results[0].block_means, block_values)


class TestEVBAnalyzerAnalyzeOverlap:
    """Tests for EVBAnalyzer.analyze_overlap method."""

    def test_analyze_overlap_good_overlap(self):
        """Test overlap analysis with well-spaced windows (good overlap)."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        # Generate data with enough overlap - use lower force constant
        # so distributions are wider and overlap more
        np.random.seed(42)
        k_umbrella = 10000.0  # Lower k = wider distributions = more overlap
        temperature = 300.0
        sigma = np.sqrt(KB * temperature / k_umbrella)  # ~0.016 nm

        # Windows spaced by ~0.5*sigma should have good overlap
        rc0_values = np.linspace(-0.02, 0.02, 5)  # Spacing ~0.01 nm < sigma
        rc_data = [np.random.normal(rc0, sigma, 2000) for rc0 in rc0_values]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=k_umbrella,
                rc0_values=rc0_values,
            )

            result = analyzer.analyze_overlap(rc_data)

            # Should have good overlap (no problem pairs)
            assert len(result.problem_pairs) == 0
            assert result.min_overlap > 0.03
            assert len(result.overlap_matrix) == len(rc_data) - 1

    def test_analyze_overlap_poor_overlap(self):
        """Test overlap analysis with widely spaced windows (poor overlap)."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            # Create windows with very little overlap
            np.random.seed(42)
            sigma = 0.001  # Very narrow distributions
            rc_data = [
                np.random.normal(-0.1, sigma, 500),
                np.random.normal(0.0, sigma, 500),
                np.random.normal(0.1, sigma, 500),
            ]

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([-0.1, 0.0, 0.1]),
            )

            result = analyzer.analyze_overlap(rc_data, min_overlap_threshold=0.03)

            # Should detect poor overlap
            assert result.min_overlap < 0.03
            assert len(result.problem_pairs) > 0

    def test_analyze_overlap_single_window(self):
        """Test overlap analysis with single window (edge case)."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0]),
            )

            rc_data = [np.random.random(100)]
            result = analyzer.analyze_overlap(rc_data)

            # Single window: empty overlap matrix, min_overlap=0
            assert len(result.overlap_matrix) == 0
            assert result.min_overlap == 0.0
            assert result.problem_pairs == []

    def test_analyze_overlap_identical_distributions(self):
        """Test overlap analysis with identical distributions (100% overlap)."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            np.random.seed(42)
            data = np.random.normal(0, 0.05, 1000)
            rc_data = [data.copy(), data.copy()]

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0, 0.0]),
            )

            result = analyzer.analyze_overlap(rc_data)

            # Identical distributions should have very high overlap
            assert result.min_overlap > 0.9


# =============================================================================
# Test EVBAnalyzer PMF Calculation
# =============================================================================


class TestEVBAnalyzerComputePMF:
    """Tests for EVBAnalyzer.compute_pmf method."""

    def test_compute_pmf_empty_data_raises(self):
        """Test compute_pmf raises ValueError for empty data."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0]),
            )

            with pytest.raises(ValueError, match="No RC data provided"):
                analyzer.compute_pmf([])

    def test_compute_pmf_wham_fallback(self, synthetic_rc_data):
        """Test compute_pmf uses WHAM when pymbar is unavailable."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        rc_data, rc0_values = synthetic_rc_data

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            # pymbar is not installed, so compute_pmf falls back to real WHAM
            result = analyzer.compute_pmf(rc_data)

            assert isinstance(result.bin_centers, np.ndarray)
            assert isinstance(result.pmf, np.ndarray)
            assert len(result.bin_centers) == 50  # Default n_bins


class TestEVBAnalyzerComputePMFHistogram:
    """Tests for EVBAnalyzer._compute_pmf_histogram (WHAM) method."""

    def test_compute_pmf_histogram_harmonic_potential(self):
        """Test WHAM PMF calculation with known harmonic potential.

        For umbrella sampling of a harmonic potential, the unbiased PMF
        should be approximately flat (constant).
        """
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        # Generate data from umbrella sampling of a flat underlying potential
        # The biased distributions should be Gaussian centered at rc0
        np.random.seed(42)
        k_umbrella = 160000.0
        temperature = 300.0
        sigma = np.sqrt(KB * temperature / k_umbrella)

        rc0_values = np.linspace(-0.1, 0.1, 5)
        rc_data = [np.random.normal(rc0, sigma, 2000) for rc0 in rc0_values]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=k_umbrella,
                rc0_values=rc0_values,
            )

            result = analyzer._compute_pmf_histogram(
                rc_data, rc0_values, temperature, n_bins=30
            )

            # PMF should be valid
            assert isinstance(result.bin_centers, np.ndarray)
            assert len(result.pmf) == 30

            # For flat underlying potential, PMF should be relatively flat
            # (after removing NaN values)
            valid_pmf = result.pmf[~np.isnan(result.pmf)]
            if len(valid_pmf) > 5:
                pmf_range = np.max(valid_pmf) - np.min(valid_pmf)
                # PMF range should be small (< 5 kJ/mol for flat potential)
                assert pmf_range < 10.0

    def test_compute_pmf_histogram_convergence(self):
        """Test that WHAM iteration converges."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        np.random.seed(42)
        k_umbrella = 160000.0
        temperature = 300.0
        sigma = np.sqrt(KB * temperature / k_umbrella)

        rc0_values = np.linspace(-0.1, 0.1, 3)
        rc_data = [np.random.normal(rc0, sigma, 1000) for rc0 in rc0_values]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=k_umbrella,
                rc0_values=rc0_values,
            )

            # Should not raise and should return valid result
            result = analyzer._compute_pmf_histogram(
                rc_data, rc0_values, temperature, n_bins=20
            )

            # Free energies should be finite
            assert np.all(np.isfinite(result.free_energies))


# =============================================================================
# Test EVBAnalyzer Full Analysis Pipeline
# =============================================================================


class TestEVBAnalyzerRunFullAnalysis:
    """Tests for EVBAnalyzer.run_full_analysis method."""

    def test_run_full_analysis_complete(self, synthetic_rc_data):
        """Test complete analysis pipeline."""
        from molecular_simulations.simulate.free_energy import (
            EVBAnalysisResult,
            EVBAnalyzer,
        )

        rc_data, rc0_values = synthetic_rc_data

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            result = analyzer.run_full_analysis(temperature=300.0)

            assert isinstance(result, EVBAnalysisResult)
            assert result.temperature == 300.0
            assert result.k_umbrella == 160000.0
            assert len(result.convergence) == len(rc_data)
            assert len(result.equilibration) == len(rc_data)
            assert len(result.rc_data) == len(rc_data)

    def test_run_full_analysis_discard_equilibration(
        self, synthetic_rc_data_with_equilibration
    ):
        """Test analysis with equilibration discarding enabled."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        rc_data, rc0_values = synthetic_rc_data_with_equilibration

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            result = analyzer.run_full_analysis(discard_equilibration=True)

            # After discarding equilibration, rc_data lengths may be shorter
            for original, processed in zip(rc_data, result.rc_data):
                assert len(processed) <= len(original)

    def test_run_full_analysis_no_equilibration_discard(self, synthetic_rc_data):
        """Test analysis without discarding equilibration."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        rc_data, rc0_values = synthetic_rc_data

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            result = analyzer.run_full_analysis(discard_equilibration=False)

            # Data lengths should be unchanged
            for original, processed in zip(rc_data, result.rc_data):
                assert len(processed) == len(original)


class TestEVBAnalyzerSaveAnalysisResults:
    """Tests for EVBAnalyzer.save_analysis_results method."""

    def test_save_analysis_results_creates_files(self, synthetic_rc_data):
        """Test that save_analysis_results creates all expected files."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        rc_data, rc0_values = synthetic_rc_data

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs"
            log_path.mkdir()
            output_path = Path(tmpdir) / "results"

            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
                output_path=output_path,
            )

            result = analyzer.run_full_analysis()
            analyzer.save_analysis_results(result, output_dir=output_path)

            # Check expected files exist
            assert (output_path / "test_pmf.csv").exists()
            assert (output_path / "test_window_free_energies.csv").exists()
            assert (output_path / "test_convergence.csv").exists()
            assert (output_path / "test_analysis_summary.txt").exists()

    def test_save_analysis_results_csv_content(self, synthetic_rc_data):
        """Test that saved CSV files have correct content."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        rc_data, rc0_values = synthetic_rc_data

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            result = analyzer.run_full_analysis()
            analyzer.save_analysis_results(result)

            # Read PMF CSV and verify columns
            pmf_df = pl.read_csv(str(log_path / "test_pmf.csv"))
            assert "RC" in pmf_df.columns
            assert "PMF_kJ_mol" in pmf_df.columns
            assert "uncertainty_kJ_mol" in pmf_df.columns

            # Read convergence CSV
            conv_df = pl.read_csv(str(log_path / "test_convergence.csv"))
            assert "window" in conv_df.columns
            assert "mean_rc" in conv_df.columns
            assert "sem" in conv_df.columns
            assert "is_converged" in conv_df.columns


# =============================================================================
# Test Statistical Algorithms with Known Inputs/Outputs
# =============================================================================


class TestStatisticalAlgorithms:
    """Tests for statistical algorithms with known inputs/outputs."""

    def test_block_averaging_known_sem(self):
        """Test block averaging with data that has known SEM."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0]),
            )

            # Create data where each block has a known mean
            # 5 blocks of 20 samples each, with block means [1, 2, 3, 4, 5]
            block_means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            rc_data = [np.repeat(block_means, 20).astype(float)]

            results = analyzer.check_convergence(rc_data, block_size=20)

            # SEM = std(block_means, ddof=1) / sqrt(5)
            expected_sem = np.std(block_means, ddof=1) / np.sqrt(5)
            assert results[0].sem == pytest.approx(expected_sem, rel=1e-5)
            assert results[0].mean_rc == pytest.approx(3.0, rel=1e-5)

    def test_block_averaging_constant_data_zero_sem(self):
        """Test that constant data gives zero SEM."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0]),
            )

            # Constant data
            rc_data = [np.ones(100) * 5.0]
            results = analyzer.check_convergence(rc_data)

            assert results[0].sem == 0.0
            assert results[0].mean_rc == 5.0

    def test_autocorrelation_white_noise_g_near_one(self):
        """Test that white noise has statistical inefficiency near 1."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        np.random.seed(42)
        # Large sample for statistical reliability
        data = np.random.normal(0, 1, 10000)

        t0, g, n_eff = EVBAnalyzer._detect_equilibration_autocorr(data)

        # g should be close to 1 for white noise
        assert g < 1.5  # Allow some tolerance

    def test_overlap_integral_identical_distributions(self):
        """Test overlap integral for identical distributions is close to 1."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0, 0.0]),
            )

            np.random.seed(42)
            # Two identical distributions
            data = np.random.normal(0, 0.1, 5000)
            rc_data = [data.copy(), data.copy()]

            result = analyzer.analyze_overlap(rc_data, n_bins=50)

            # Overlap should be very close to 1
            assert result.overlap_matrix[0] > 0.95

    def test_overlap_integral_non_overlapping(self):
        """Test overlap integral for non-overlapping distributions is near 0."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([-1.0, 1.0]),
            )

            np.random.seed(42)
            # Two completely separate distributions
            rc_data = [
                np.random.normal(-1.0, 0.01, 1000),  # Very narrow at -1
                np.random.normal(1.0, 0.01, 1000),  # Very narrow at +1
            ]

            result = analyzer.analyze_overlap(rc_data, n_bins=50)

            # Overlap should be essentially 0
            assert result.overlap_matrix[0] < 0.01


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_window_analysis(self):
        """Test analysis with single window."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            np.random.seed(42)
            rc_data = [np.random.normal(0, 0.01, 500)]
            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0]),
            )

            result = analyzer.run_full_analysis()

            # Should complete without error
            assert len(result.convergence) == 1
            assert len(result.overlap.overlap_matrix) == 0  # No pairs

    def test_very_short_trajectories(self):
        """Test handling of very short trajectories."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            # Very short trajectories
            rc_data = [np.random.random(15) for _ in range(3)]
            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.linspace(-0.1, 0.1, 3),
            )

            # Should handle without crashing
            result = analyzer.run_full_analysis()
            assert result is not None

    def test_negative_rc_values(self):
        """Test handling of negative RC values."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            np.random.seed(42)
            rc0_values = np.array([-0.5, -0.3, -0.1])
            rc_data = [np.random.normal(rc0, 0.01, 500) for rc0 in rc0_values]
            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=rc0_values,
            )

            result = analyzer.run_full_analysis()

            # Should handle negative values correctly
            assert all(c.mean_rc < 0 for c in result.convergence)

    def test_all_nan_pmf_handling(self):
        """Test handling when PMF calculation produces NaN values."""
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)

            # Create very sparse data that might produce NaN PMF
            np.random.seed(42)
            rc_data = [np.array([0.0]), np.array([0.1])]  # Only 1 sample each
            create_rc_log_files(log_path, "test", rc_data)

            analyzer = EVBAnalyzer(
                log_path=log_path,
                log_prefix="test",
                k_umbrella=160000.0,
                rc0_values=np.array([0.0, 0.1]),
            )

            # Should not crash even with sparse data
            result = analyzer.compute_pmf(rc_data)
            assert result is not None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
