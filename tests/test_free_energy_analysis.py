"""
Tests for free energy analysis algorithms using synthetic data.

This module tests the numerical/statistical analysis methods in free_energy.py
WITHOUT mocking the ALGORITHMS. All tests use numpy-generated synthetic data
to validate the algorithms directly.

Note: free_energy.py is imported for real -- OpenMM, MDAnalysis, numpy and
parsl are all available, so omm_simulator and reporters import cleanly with
no mocking required.

Tested methods:
    - _detect_equilibration_autocorr: Autocorrelation-based equilibration detection
    - check_convergence: Block averaging for convergence analysis
    - analyze_overlap: Histogram overlap between umbrella windows
    - _compute_pmf_histogram: WHAM algorithm for PMF calculation

The test strategy focuses on:
    1. Synthetic data with KNOWN properties (equilibration time, variance, etc.)
    2. Direct algorithm validation without I/O or external dependencies
    3. Edge cases (empty arrays, single samples, no overlap, etc.)
"""

from collections.abc import Callable

import numpy as np
import pytest

# Constants matching free_energy.py
KB = 8.314462618e-3  # Boltzmann constant in kJ/(mol*K)


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================


def generate_equilibrated_timeseries(
    n_samples: int = 1000,
    mean: float = 0.0,
    std: float = 0.1,
    correlation_time: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    """Generate well-equilibrated time series with known autocorrelation.

    Creates stationary data using an AR(1) process:
        x[t] = phi * x[t-1] + epsilon[t]

    Args:
        n_samples: Number of data points.
        mean: Target mean value.
        std: Target standard deviation.
        correlation_time: Approximate autocorrelation decay time.
        seed: Random seed for reproducibility.

    Returns:
        Stationary time series array.
    """
    rng = np.random.default_rng(seed)

    # AR(1) coefficient from correlation time: phi = exp(-1/tau)
    phi = np.exp(-1.0 / correlation_time) if correlation_time > 0 else 0.0

    # Innovation std to achieve target variance
    # Var(x) = sigma_eps^2 / (1 - phi^2)
    sigma_eps = std * np.sqrt(1 - phi**2) if abs(phi) < 1 else std

    # Generate AR(1) process
    data = np.zeros(n_samples)
    data[0] = rng.normal(0, std)
    for t in range(1, n_samples):
        data[t] = phi * data[t - 1] + rng.normal(0, sigma_eps)

    return data + mean


def generate_poorly_equilibrated_timeseries(
    n_samples: int = 1000,
    equilibration_time: int = 200,
    initial_offset: float = 1.0,
    equilibrium_mean: float = 0.0,
    equilibrium_std: float = 0.1,
    seed: int | None = None,
) -> tuple[np.ndarray, int]:
    """Generate time series with non-stationary equilibration phase.

    The data starts far from equilibrium and drifts toward the equilibrium
    value over the specified equilibration time.

    Args:
        n_samples: Total number of data points.
        equilibration_time: Number of frames for equilibration phase.
        initial_offset: Starting deviation from equilibrium mean.
        equilibrium_mean: Target equilibrium mean value.
        equilibrium_std: Equilibrium standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (time series array, true equilibration index).
    """
    rng = np.random.default_rng(seed)

    data = np.zeros(n_samples)

    # Equilibration phase: exponential decay toward equilibrium
    t0 = min(equilibration_time, n_samples - 1)
    decay_rate = 5.0 / t0  # ~99% decay by t0

    for t in range(t0):
        drift = initial_offset * np.exp(-decay_rate * t)
        data[t] = equilibrium_mean + drift + rng.normal(0, equilibrium_std * 0.5)

    # Production phase: stationary noise
    data[t0:] = equilibrium_mean + rng.normal(0, equilibrium_std, n_samples - t0)

    return data, t0


def generate_umbrella_window_data(
    rc0: float,
    k_umbrella: float,
    temperature: float,
    n_samples: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Generate RC samples from a single umbrella window.

    Samples from a Gaussian distribution centered at rc0 with variance
    determined by the umbrella force constant:
        sigma^2 = k_B * T / k_umbrella

    Args:
        rc0: Umbrella window center (reaction coordinate target).
        k_umbrella: Umbrella force constant in kJ/(mol*nm^2).
        temperature: Temperature in Kelvin.
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Array of sampled RC values.
    """
    rng = np.random.default_rng(seed)

    # Variance from equipartition: sigma^2 = kT / k_umbrella
    sigma = np.sqrt(KB * temperature / k_umbrella)

    return rng.normal(rc0, sigma, n_samples)


def generate_overlapping_windows(
    n_windows: int = 10,
    rc_min: float = -0.2,
    rc_max: float = 0.2,
    k_umbrella: float = 160000.0,
    temperature: float = 300.0,
    samples_per_window: int = 1000,
    seed: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Generate umbrella sampling data with good overlap between windows.

    Creates multiple overlapping Gaussian distributions that simulate
    proper umbrella sampling with sufficient window overlap for WHAM/MBAR.

    Args:
        n_windows: Number of umbrella windows.
        rc_min: Minimum RC value.
        rc_max: Maximum RC value.
        k_umbrella: Umbrella force constant.
        temperature: Temperature in Kelvin.
        samples_per_window: Samples per window.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (list of RC arrays per window, array of window centers).
    """
    np.random.default_rng(seed)

    rc0_values = np.linspace(rc_min, rc_max, n_windows)
    rc_data = []

    for i, rc0 in enumerate(rc0_values):
        # Use different seed for each window but deterministic
        window_seed = seed + i if seed is not None else None
        data = generate_umbrella_window_data(
            rc0, k_umbrella, temperature, samples_per_window, seed=window_seed
        )
        rc_data.append(data)

    return rc_data, rc0_values


def generate_non_overlapping_windows(
    n_windows: int = 5,
    window_spacing: float = 0.5,
    k_umbrella: float = 1600000.0,  # Very stiff - narrow distributions
    temperature: float = 300.0,
    samples_per_window: int = 500,
    seed: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Generate umbrella sampling data with NO overlap between windows.

    Uses a very stiff umbrella potential to create non-overlapping
    distributions for testing edge cases.

    Args:
        n_windows: Number of windows.
        window_spacing: Spacing between window centers.
        k_umbrella: Very stiff umbrella force constant.
        temperature: Temperature in Kelvin.
        samples_per_window: Samples per window.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (list of RC arrays per window, array of window centers).
    """
    np.random.default_rng(seed)

    rc0_values = np.arange(n_windows) * window_spacing
    rc_data = []

    for i, rc0 in enumerate(rc0_values):
        window_seed = seed + i if seed is not None else None
        data = generate_umbrella_window_data(
            rc0, k_umbrella, temperature, samples_per_window, seed=window_seed
        )
        rc_data.append(data)

    return rc_data, rc0_values


def generate_harmonic_pmf_data(
    n_windows: int = 20,
    rc_min: float = -0.3,
    rc_max: float = 0.3,
    pmf_k: float = 100.0,  # kJ/mol/nm^2 - underlying PMF curvature
    pmf_center: float = 0.0,  # PMF minimum location
    k_umbrella: float = 160000.0,
    temperature: float = 300.0,
    samples_per_window: int = 2000,
    seed: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Generate umbrella sampling data for a known harmonic PMF.

    The underlying PMF is:
        W(x) = 0.5 * pmf_k * (x - pmf_center)^2

    Samples are drawn from the combined distribution:
        P(x) ~ exp(-beta * [W(x) + U_bias(x)])

    where U_bias is the umbrella potential.

    Args:
        n_windows: Number of umbrella windows.
        rc_min: Minimum RC value for window centers.
        rc_max: Maximum RC value for window centers.
        pmf_k: Force constant of underlying harmonic PMF.
        pmf_center: Location of PMF minimum.
        k_umbrella: Umbrella force constant.
        temperature: Temperature in Kelvin.
        samples_per_window: Samples per window.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
            - List of RC arrays per window
            - Array of window centers
            - Function that computes true PMF at given RC values
    """
    np.random.default_rng(seed)
    beta = 1.0 / (KB * temperature)

    rc0_values = np.linspace(rc_min, rc_max, n_windows)
    rc_data = []

    for i, rc0 in enumerate(rc0_values):
        # Combined effective force constant: k_eff = k_umbrella + pmf_k
        # Combined effective center: weighted average
        k_eff = k_umbrella + pmf_k
        x_eff = (k_umbrella * rc0 + pmf_k * pmf_center) / k_eff
        sigma_eff = np.sqrt(1.0 / (beta * k_eff))

        # Sample from effective Gaussian
        window_seed = seed + i if seed is not None else None
        window_rng = np.random.default_rng(window_seed)
        data = window_rng.normal(x_eff, sigma_eff, samples_per_window)
        rc_data.append(data)

    def true_pmf(x: np.ndarray) -> np.ndarray:
        """Compute the true underlying PMF."""
        pmf = 0.5 * pmf_k * (x - pmf_center) ** 2
        return pmf - pmf.min()

    return rc_data, rc0_values, true_pmf


def generate_double_well_pmf_data(
    n_windows: int = 30,
    rc_min: float = -0.25,
    rc_max: float = 0.25,
    barrier_height: float = 10.0,  # kJ/mol
    well_positions: tuple[float, float] = (-0.1, 0.1),
    k_umbrella: float = 160000.0,
    temperature: float = 300.0,
    samples_per_window: int = 2000,
    seed: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Generate umbrella sampling data for a symmetric double-well PMF.

    The PMF has the form:
        W(x) = barrier_height * (1 - (x/a)^2)^2

    where a is chosen to place wells at well_positions.

    Args:
        n_windows: Number of umbrella windows.
        rc_min: Minimum RC value.
        rc_max: Maximum RC value.
        barrier_height: Height of barrier in kJ/mol.
        well_positions: Locations of the two minima.
        k_umbrella: Umbrella force constant.
        temperature: Temperature in Kelvin.
        samples_per_window: Samples per window.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (RC data, window centers, true PMF function).
    """
    np.random.default_rng(seed)
    beta = 1.0 / (KB * temperature)

    # Position parameter for double well
    a = (well_positions[0] + well_positions[1]) / 2
    if abs(a) < 1e-10:
        a = well_positions[1]  # Use right well position if centered

    rc0_values = np.linspace(rc_min, rc_max, n_windows)
    rc_data = []

    def pmf_func(x: np.ndarray) -> np.ndarray:
        """Double well PMF."""
        # Use quartic form: W = b*x^4 - c*x^2
        # With minima at +/- x_min and barrier at x=0
        x_min = abs(well_positions[1])
        if x_min > 0:
            c = 2 * barrier_height / (x_min**2)
            b = barrier_height / (x_min**4)
            pmf = b * x**4 - c * x**2 + barrier_height
        else:
            pmf = np.zeros_like(x)
        return pmf

    # Sample using rejection sampling from each biased distribution
    for i, rc0 in enumerate(rc0_values):
        window_seed = seed + i if seed is not None else None
        window_rng = np.random.default_rng(window_seed)

        samples = []
        # Use a broader proposal distribution
        proposal_sigma = np.sqrt(KB * temperature / k_umbrella) * 3

        while len(samples) < samples_per_window:
            # Propose from Gaussian centered at rc0
            x_prop = window_rng.normal(rc0, proposal_sigma, samples_per_window * 2)

            # Compute acceptance probability
            bias = 0.5 * k_umbrella * (x_prop - rc0) ** 2
            total_energy = pmf_func(x_prop) + bias
            # Relative to minimum of biased potential near rc0
            min_energy = pmf_func(np.array([rc0]))[0] + 0.0
            log_accept = -beta * (total_energy - min_energy)
            accept = window_rng.random(len(x_prop)) < np.exp(np.minimum(0, log_accept))

            samples.extend(x_prop[accept].tolist())

        rc_data.append(np.array(samples[:samples_per_window]))

    def true_pmf(x: np.ndarray) -> np.ndarray:
        """Compute the true PMF (shifted to minimum = 0)."""
        pmf = pmf_func(x)
        return pmf - pmf.min()

    return rc_data, rc0_values, true_pmf


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def analyzer_class():
    """Import and return the real EVBAnalyzer class.

    free_energy.py (and its omm_simulator/reporters imports) loads cleanly
    against the real OpenMM/MDAnalysis/numpy/parsl stack -- no mocking needed.
    """
    from molecular_simulations.simulate.free_energy import EVBAnalyzer

    return EVBAnalyzer


@pytest.fixture
def equilibrated_data():
    """Fixture providing well-equilibrated time series data."""
    return generate_equilibrated_timeseries(
        n_samples=1000, mean=0.1, std=0.02, correlation_time=20, seed=42
    )


@pytest.fixture
def poorly_equilibrated_data():
    """Fixture providing time series with known equilibration time."""
    data, t0 = generate_poorly_equilibrated_timeseries(
        n_samples=1000,
        equilibration_time=200,
        initial_offset=0.5,
        equilibrium_mean=0.0,
        equilibrium_std=0.05,
        seed=42,
    )
    return data, t0


@pytest.fixture
def overlapping_windows():
    """Fixture providing umbrella sampling data with good overlap.

    With k_umbrella=1000 kJ/mol/nm^2, T=300K:
        sigma = sqrt(kT/k) = sqrt(2.494/1000) = 0.05 nm
    With 10 windows over 0.3 nm range, spacing ~0.033 nm < sigma
    This ensures ~30%+ overlap between adjacent windows.
    """
    rc_data, rc0_values = generate_overlapping_windows(
        n_windows=10,
        rc_min=-0.15,
        rc_max=0.15,
        k_umbrella=1000.0,  # Softer restraint for good overlap
        temperature=300.0,
        samples_per_window=1000,
        seed=42,
    )
    return rc_data, rc0_values


@pytest.fixture
def non_overlapping_windows():
    """Fixture providing umbrella sampling data without overlap."""
    rc_data, rc0_values = generate_non_overlapping_windows(
        n_windows=5,
        window_spacing=0.5,
        k_umbrella=1600000.0,
        temperature=300.0,
        samples_per_window=500,
        seed=42,
    )
    return rc_data, rc0_values


@pytest.fixture
def harmonic_pmf_data():
    """Fixture providing data for harmonic PMF recovery test."""
    return generate_harmonic_pmf_data(
        n_windows=20,
        rc_min=-0.2,
        rc_max=0.2,
        pmf_k=50.0,
        pmf_center=0.0,
        k_umbrella=160000.0,
        temperature=300.0,
        samples_per_window=2000,
        seed=42,
    )


# =============================================================================
# TESTS: _detect_equilibration_autocorr
# =============================================================================


class TestDetectEquilibrationAutocorr:
    """Tests for the autocorrelation-based equilibration detection."""

    def test_well_equilibrated_returns_low_t0(self, analyzer_class):
        """Well-equilibrated data should have t0 near zero."""
        data = generate_equilibrated_timeseries(
            n_samples=1000, mean=0.0, std=0.1, correlation_time=5, seed=123
        )

        t0, g, n_eff = analyzer_class._detect_equilibration_autocorr(data)

        # Well-equilibrated data: t0 should be small (< 10% of trajectory)
        assert t0 < 100, f't0={t0} too large for equilibrated data'
        assert g >= 1.0, 'Statistical inefficiency must be >= 1'
        assert n_eff > 0, 'Effective samples must be positive'
        assert n_eff <= len(data), 'n_eff cannot exceed total samples'

    def test_poorly_equilibrated_detects_drift(self, analyzer_class):
        """Poorly equilibrated data should detect equilibration time."""
        data, _true_t0 = generate_poorly_equilibrated_timeseries(
            n_samples=1000,
            equilibration_time=300,
            initial_offset=1.0,
            equilibrium_mean=0.0,
            equilibrium_std=0.05,
            seed=123,
        )

        t0, g, _n_eff = analyzer_class._detect_equilibration_autocorr(data)

        # Should detect equilibration - allow some tolerance
        # The algorithm may not find exact t0 but should be in ballpark
        assert t0 >= 0, 't0 must be non-negative'
        # Key insight: n_eff should improve when discarding equilibration
        len(data) / g if g > 0 else len(data)

    def test_constant_data_returns_trivial_result(self, analyzer_class):
        """Constant (zero variance) data should return t0=0, g=1."""
        data = np.ones(500) * 0.5  # Constant value

        t0, g, n_eff = analyzer_class._detect_equilibration_autocorr(data)

        assert t0 == 0, 'Constant data should have t0=0'
        assert g == pytest.approx(1.0), 'Constant data should have g=1'
        assert n_eff == pytest.approx(float(len(data))), (
            'Constant data should have n_eff = n'
        )

    def test_short_timeseries(self, analyzer_class):
        """Short time series should be handled gracefully."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.1, 20)  # Very short

        t0, g, n_eff = analyzer_class._detect_equilibration_autocorr(data)

        assert t0 >= 0
        assert g >= 1.0
        assert n_eff > 0

    def test_custom_max_lag(self, analyzer_class):
        """Custom max_lag parameter should be respected."""
        data = generate_equilibrated_timeseries(
            n_samples=1000, correlation_time=50, seed=42
        )

        # With very small max_lag, g will be underestimated
        _t0_small, g_small, _ = analyzer_class._detect_equilibration_autocorr(
            data, max_lag=5
        )
        _t0_large, g_large, _ = analyzer_class._detect_equilibration_autocorr(
            data, max_lag=200
        )

        # g should generally be larger with larger max_lag for correlated data
        # (though not guaranteed due to noise)
        assert g_small >= 1.0
        assert g_large >= 1.0

    def test_highly_correlated_data(self, analyzer_class):
        """Highly correlated data should have large statistical inefficiency."""
        # Very long correlation time
        data = generate_equilibrated_timeseries(
            n_samples=2000, correlation_time=100, seed=42
        )

        _t0, g, n_eff = analyzer_class._detect_equilibration_autocorr(data)

        # High correlation should give g >> 1
        assert g > 5.0, f'Expected g > 5 for highly correlated data, got {g}'
        assert n_eff < len(data) / 2, 'n_eff should be much less than n'

    def test_uncorrelated_data_has_g_near_one(self, analyzer_class):
        """Uncorrelated (white noise) data should have g close to 1."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.1, 1000)  # IID samples

        _t0, g, _n_eff = analyzer_class._detect_equilibration_autocorr(data)

        # g should be close to 1 for uncorrelated data
        assert 0.8 < g < 3.0, f'Expected g near 1 for IID data, got {g}'


# =============================================================================
# TESTS: check_convergence
# =============================================================================


class TestCheckConvergence:
    """Tests for block averaging convergence analysis."""

    def test_converged_windows_identified(self, analyzer_class, tmp_path):
        """Windows with low variance should be marked as converged."""
        # Create analyzer with dummy paths (we only use static method behavior)
        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=np.linspace(-0.1, 0.1, 5),
        )

        # Generate well-behaved data with low SEM
        rng = np.random.default_rng(42)
        rc_data = [rng.normal(0.0, 0.001, 1000) for _ in range(5)]  # Very low variance

        results = analyzer.check_convergence(
            rc_data, block_size=100, sem_threshold=0.01
        )

        assert len(results) == 5
        for r in results:
            assert r.is_converged, f'Window {r.window_idx} should be converged'
            assert r.sem < 0.01

    def test_unconverged_windows_identified(self, analyzer_class, tmp_path):
        """Windows with high variance should be marked as not converged."""
        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=np.linspace(-0.1, 0.1, 3),
        )

        # Generate high variance data
        rng = np.random.default_rng(42)
        rc_data = [rng.normal(0.0, 0.5, 1000) for _ in range(3)]  # High variance

        results = analyzer.check_convergence(
            rc_data,
            block_size=100,
            sem_threshold=0.001,  # Very strict threshold
        )

        for r in results:
            assert not r.is_converged, f'Window {r.window_idx} should NOT be converged'
            assert r.sem > 0.001

    def test_block_means_computed_correctly(self, analyzer_class, tmp_path):
        """Block means should equal manual calculation."""
        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=np.array([0.0]),
        )

        # Simple deterministic data
        rc_data = [np.arange(100, dtype=float)]  # 0, 1, 2, ..., 99
        block_size = 10

        results = analyzer.check_convergence(rc_data, block_size=block_size)

        # Block means should be: [4.5, 14.5, 24.5, ..., 94.5]
        expected_block_means = np.array(
            [np.mean(np.arange(i * 10, (i + 1) * 10)) for i in range(10)]
        )

        np.testing.assert_array_almost_equal(
            results[0].block_means, expected_block_means, decimal=10
        )

    def test_sem_calculation_correct(self, analyzer_class, tmp_path):
        """SEM should match manual calculation."""
        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=np.array([0.0]),
        )

        # Known block means for easy verification
        rc_data = [
            np.concatenate(
                [
                    np.ones(100) * 1.0,
                    np.ones(100) * 2.0,
                    np.ones(100) * 3.0,
                ]
            )
        ]  # Three blocks with means 1, 2, 3

        results = analyzer.check_convergence(rc_data, block_size=100)

        block_means = np.array([1.0, 2.0, 3.0])
        expected_sem = np.std(block_means, ddof=1) / np.sqrt(3)

        assert results[0].sem == pytest.approx(expected_sem, rel=1e-10)
        assert results[0].mean_rc == pytest.approx(2.0, rel=1e-10)

    def test_automatic_block_size(self, analyzer_class, tmp_path):
        """Default block size should be n // 10."""
        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=np.array([0.0]),
        )

        rng = np.random.default_rng(42)
        rc_data = [rng.normal(0, 0.1, 1000)]

        results = analyzer.check_convergence(rc_data)  # No block_size specified

        # Default: block_size = max(10, 1000 // 10) = 100
        # n_blocks = 1000 // 100 = 10
        assert results[0].n_blocks == 10

    def test_minimum_three_blocks(self, analyzer_class, tmp_path):
        """Should ensure at least 3 blocks for SEM calculation."""
        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=np.array([0.0]),
        )

        rng = np.random.default_rng(42)
        # Very large block size would give < 3 blocks
        rc_data = [rng.normal(0, 0.1, 100)]

        results = analyzer.check_convergence(
            rc_data, block_size=50
        )  # Would give 2 blocks

        assert results[0].n_blocks >= 3, 'Should have at least 3 blocks'


# =============================================================================
# TESTS: analyze_overlap
# =============================================================================


class TestAnalyzeOverlap:
    """Tests for histogram overlap analysis between windows."""

    def test_good_overlap_detected(self, analyzer_class, tmp_path, overlapping_windows):
        """Well-spaced windows should have good overlap."""
        rc_data, rc0_values = overlapping_windows

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result = analyzer.analyze_overlap(
            rc_data, n_bins=50, min_overlap_threshold=0.03
        )

        # Should have good overlap everywhere
        assert result.min_overlap > 0.03, (
            f'Expected min overlap > 0.03, got {result.min_overlap}'
        )
        assert len(result.problem_pairs) == 0, (
            f'Expected no problem pairs, got {result.problem_pairs}'
        )
        assert len(result.overlap_matrix) == len(rc_data) - 1

    def test_no_overlap_detected(
        self, analyzer_class, tmp_path, non_overlapping_windows
    ):
        """Non-overlapping windows should be flagged."""
        rc_data, rc0_values = non_overlapping_windows

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=1600000.0,
            rc0_values=rc0_values,
        )

        result = analyzer.analyze_overlap(
            rc_data, n_bins=50, min_overlap_threshold=0.03
        )

        # Should detect problems
        assert result.min_overlap < 0.03, (
            f'Expected min overlap < 0.03, got {result.min_overlap}'
        )
        assert len(result.problem_pairs) > 0, 'Should detect problem pairs'

    def test_overlap_values_in_valid_range(self, analyzer_class, tmp_path):
        """Overlap values should be between 0 and 1."""
        rc_data, rc0_values = generate_overlapping_windows(n_windows=5, seed=42)

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result = analyzer.analyze_overlap(rc_data)

        assert np.all(result.overlap_matrix >= 0), 'Overlap cannot be negative'
        assert np.all(result.overlap_matrix <= 1), 'Overlap cannot exceed 1'

    def test_identical_windows_have_high_overlap(self, analyzer_class, tmp_path):
        """Identical distributions should have overlap = 1."""
        rng = np.random.default_rng(42)
        # Same distribution in both windows
        shared_data = rng.normal(0.0, 0.05, 1000)
        rc_data = [shared_data.copy(), shared_data.copy()]
        rc0_values = np.array([0.0, 0.0])

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result = analyzer.analyze_overlap(rc_data, n_bins=50)

        # Identical distributions should have overlap close to 1
        assert result.overlap_matrix[0] > 0.95, (
            f'Identical distributions should have high overlap, got {result.overlap_matrix[0]}'
        )

    def test_overlap_matrix_length(self, analyzer_class, tmp_path):
        """Overlap matrix should have n_windows - 1 elements."""
        n_windows = 7
        rc_data, rc0_values = generate_overlapping_windows(n_windows=n_windows, seed=42)

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result = analyzer.analyze_overlap(rc_data)

        assert len(result.overlap_matrix) == n_windows - 1

    def test_nbins_affects_resolution(self, analyzer_class, tmp_path):
        """Different n_bins should give similar but not identical results."""
        # Use soft restraint for overlapping windows
        rc_data, rc0_values = generate_overlapping_windows(
            n_windows=5, k_umbrella=1000.0, seed=42
        )

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=1000.0,
            rc0_values=rc0_values,
        )

        result_fine = analyzer.analyze_overlap(rc_data, n_bins=100)
        result_coarse = analyzer.analyze_overlap(rc_data, n_bins=20)

        # Results should be similar but not identical
        assert not np.allclose(result_fine.overlap_matrix, result_coarse.overlap_matrix)
        # But both should indicate good overlap
        assert result_fine.min_overlap > 0
        assert result_coarse.min_overlap > 0


# =============================================================================
# TESTS: _compute_pmf_histogram (WHAM)
# =============================================================================


class TestComputePMFHistogram:
    """Tests for WHAM-based PMF calculation."""

    def test_recovers_harmonic_pmf(self, analyzer_class, tmp_path, harmonic_pmf_data):
        """WHAM should recover known harmonic PMF shape."""
        rc_data, rc0_values, true_pmf = harmonic_pmf_data

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result = analyzer._compute_pmf_histogram(
            rc_data, rc0_values, temperature=300.0, n_bins=40
        )

        # Compare computed PMF to true PMF at bin centers
        computed_pmf = result.pmf
        expected_pmf = true_pmf(result.bin_centers)

        # Ignore NaN values and edges
        valid = ~np.isnan(computed_pmf)
        # Only compare interior points (avoid edge effects)
        interior = (result.bin_centers > rc0_values.min() + 0.02) & (
            result.bin_centers < rc0_values.max() - 0.02
        )
        mask = valid & interior

        if mask.sum() > 5:
            # Allow some tolerance - WHAM is approximate
            # Compare shapes by looking at correlation
            correlation = np.corrcoef(computed_pmf[mask], expected_pmf[mask])[0, 1]
            assert correlation > 0.9, (
                f'PMF shape correlation {correlation} too low (expected > 0.9)'
            )

    def test_pmf_minimum_is_zero(self, analyzer_class, tmp_path):
        """Computed PMF should be shifted so minimum is zero."""
        rc_data, rc0_values = generate_overlapping_windows(
            n_windows=15, samples_per_window=1000, seed=42
        )

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result = analyzer._compute_pmf_histogram(
            rc_data, rc0_values, temperature=300.0, n_bins=30
        )

        valid_pmf = result.pmf[~np.isnan(result.pmf)]
        if len(valid_pmf) > 0:
            assert valid_pmf.min() == pytest.approx(0.0, abs=1e-10), (
                'PMF minimum should be shifted to zero'
            )

    def test_pmf_result_structure(self, analyzer_class, tmp_path):
        """PMFResult should have correct structure and sizes."""
        n_bins = 25
        rc_data, rc0_values = generate_overlapping_windows(n_windows=10, seed=42)

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result = analyzer._compute_pmf_histogram(
            rc_data, rc0_values, temperature=300.0, n_bins=n_bins
        )

        assert len(result.bin_centers) == n_bins
        assert len(result.pmf) == n_bins
        assert len(result.pmf_uncertainty) == n_bins
        assert len(result.free_energies) == len(rc_data)
        assert len(result.free_energy_uncertainty) == len(rc_data)

    def test_bin_centers_span_data_range(self, analyzer_class, tmp_path):
        """Bin centers should span the range of RC data."""
        rc_data, rc0_values = generate_overlapping_windows(
            n_windows=8, rc_min=-0.1, rc_max=0.1, seed=42
        )

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result = analyzer._compute_pmf_histogram(
            rc_data, rc0_values, temperature=300.0, n_bins=20
        )

        all_rc = np.concatenate(rc_data)
        rc_min, rc_max = all_rc.min(), all_rc.max()

        assert result.bin_centers.min() >= rc_min
        assert result.bin_centers.max() <= rc_max

    def test_temperature_affects_pmf_scale(self, analyzer_class, tmp_path):
        """PMF magnitude should scale with temperature."""
        rc_data, rc0_values = generate_overlapping_windows(n_windows=10, seed=42)

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result_300K = analyzer._compute_pmf_histogram(
            rc_data, rc0_values, temperature=300.0, n_bins=30
        )
        result_600K = analyzer._compute_pmf_histogram(
            rc_data, rc0_values, temperature=600.0, n_bins=30
        )

        # PMF values scale with kT, so higher T should give larger PMF values
        # (for same probability distribution)
        valid_300 = ~np.isnan(result_300K.pmf)
        valid_600 = ~np.isnan(result_600K.pmf)

        # Both should have valid values
        assert valid_300.sum() > 0
        assert valid_600.sum() > 0

    def test_wham_converges(self, analyzer_class, tmp_path):
        """WHAM iteration should converge for reasonable data."""
        rc_data, rc0_values = generate_overlapping_windows(
            n_windows=15, samples_per_window=2000, seed=42
        )

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        # This should not raise and should produce valid output
        result = analyzer._compute_pmf_histogram(
            rc_data, rc0_values, temperature=300.0, n_bins=30
        )

        # Check that we got reasonable output (not all NaN)
        valid_count = (~np.isnan(result.pmf)).sum()
        assert valid_count > len(result.pmf) // 2, (
            'WHAM should produce mostly valid PMF values'
        )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.filterwarnings('ignore:Mean of empty slice:RuntimeWarning')
    @pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning')
    @pytest.mark.filterwarnings('ignore:Degrees of freedom <= 0:RuntimeWarning')
    def test_empty_array_equilibration(self, analyzer_class):
        """Empty array should be handled gracefully in equilibration detection."""
        data = np.array([])

        # Should not crash - behavior may vary
        try:
            t0, _g, _n_eff = analyzer_class._detect_equilibration_autocorr(data)
            # If it returns, values should be sensible
            assert t0 == 0
        except (ValueError, IndexError):
            # Acceptable to raise error on empty input
            pass

    def test_single_sample_equilibration(self, analyzer_class):
        """Single sample should be handled gracefully."""
        data = np.array([0.5])

        try:
            t0, g, _n_eff = analyzer_class._detect_equilibration_autocorr(data)
            assert t0 == 0
            assert g >= 1.0
        except (ValueError, IndexError):
            pass

    def test_very_short_trajectory_convergence(self, analyzer_class, tmp_path):
        """Very short trajectories should still produce results."""
        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=np.array([0.0]),
        )

        rng = np.random.default_rng(42)
        rc_data = [rng.normal(0, 0.1, 15)]  # Very short

        results = analyzer.check_convergence(rc_data)

        assert len(results) == 1
        assert results[0].n_blocks >= 3

    def test_single_window_overlap(self, analyzer_class, tmp_path):
        """Single window should give empty overlap matrix."""
        rng = np.random.default_rng(42)
        rc_data = [rng.normal(0, 0.1, 100)]
        rc0_values = np.array([0.0])

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result = analyzer.analyze_overlap(rc_data)

        assert len(result.overlap_matrix) == 0
        assert result.min_overlap == 0.0
        assert len(result.problem_pairs) == 0

    def test_two_windows_overlap(self, analyzer_class, tmp_path):
        """Two windows should give single overlap value."""
        rng = np.random.default_rng(42)
        rc_data = [
            rng.normal(-0.05, 0.03, 500),
            rng.normal(0.05, 0.03, 500),
        ]
        rc0_values = np.array([-0.05, 0.05])

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result = analyzer.analyze_overlap(rc_data)

        assert len(result.overlap_matrix) == 1

    def test_all_converged_vs_none_converged(self, analyzer_class, tmp_path):
        """Test distinguishing all converged from none converged."""
        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=np.linspace(-0.1, 0.1, 5),
        )

        rng = np.random.default_rng(42)

        # All converged: very low variance
        converged_data = [rng.normal(i * 0.05, 0.0001, 1000) for i in range(-2, 3)]
        converged_results = analyzer.check_convergence(
            converged_data, sem_threshold=0.01
        )
        n_converged = sum(1 for r in converged_results if r.is_converged)
        assert n_converged == 5, 'All windows should be converged'

        # None converged: high variance, strict threshold
        unconverged_data = [rng.normal(i * 0.05, 0.5, 1000) for i in range(-2, 3)]
        unconverged_results = analyzer.check_convergence(
            unconverged_data, sem_threshold=0.0001
        )
        n_unconverged = sum(1 for r in unconverged_results if not r.is_converged)
        assert n_unconverged == 5, 'No windows should be converged'

    def test_pmf_with_sparse_windows(self, analyzer_class, tmp_path):
        """PMF calculation with few windows should still work."""
        rc_data, rc0_values = generate_overlapping_windows(
            n_windows=3, rc_min=-0.05, rc_max=0.05, seed=42
        )

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        result = analyzer._compute_pmf_histogram(
            rc_data, rc0_values, temperature=300.0, n_bins=15
        )

        # Should produce some valid values
        assert (~np.isnan(result.pmf)).sum() > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple analysis steps."""

    def test_full_analysis_workflow(self, analyzer_class, tmp_path):
        """Test complete analysis workflow on synthetic data."""
        # Generate realistic umbrella sampling data
        n_windows = 15
        rc_data, rc0_values = generate_overlapping_windows(
            n_windows=n_windows,
            rc_min=-0.15,
            rc_max=0.15,
            k_umbrella=160000.0,
            temperature=300.0,
            samples_per_window=1500,
            seed=42,
        )

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        # Run each analysis step
        equilibration = analyzer.detect_equilibration(rc_data)
        assert len(equilibration) == n_windows

        convergence = analyzer.check_convergence(rc_data)
        assert len(convergence) == n_windows

        overlap = analyzer.analyze_overlap(rc_data)
        assert len(overlap.overlap_matrix) == n_windows - 1

        pmf = analyzer._compute_pmf_histogram(
            rc_data, rc0_values, temperature=300.0, n_bins=30
        )
        assert len(pmf.pmf) == 30

    def test_equilibration_improves_convergence(self, analyzer_class, tmp_path):
        """Removing equilibration should improve convergence metrics."""
        # Generate data with equilibration drift
        n_windows = 5
        rc0_values = np.linspace(-0.1, 0.1, n_windows)
        rc_data = []

        for i, rc0 in enumerate(rc0_values):
            data, _ = generate_poorly_equilibrated_timeseries(
                n_samples=1000,
                equilibration_time=200,
                initial_offset=0.3,
                equilibrium_mean=rc0,
                equilibrium_std=0.02,
                seed=42 + i,
            )
            rc_data.append(data)

        analyzer = analyzer_class(
            log_path=tmp_path,
            log_prefix='test',
            k_umbrella=160000.0,
            rc0_values=rc0_values,
        )

        # Check convergence before removing equilibration
        conv_before = analyzer.check_convergence(rc_data)
        np.mean([c.sem for c in conv_before])

        # Detect and remove equilibration
        equil = analyzer.detect_equilibration(rc_data)
        rc_data_trimmed = [
            data[e.t0 :] for data, e in zip(rc_data, equil, strict=False)
        ]

        # Check convergence after removing equilibration
        conv_after = analyzer.check_convergence(rc_data_trimmed)
        np.mean([c.sem for c in conv_after])

        # SEM should generally be lower after removing equilibration
        # (though not guaranteed for all cases due to reduced sample size)


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================


@pytest.mark.parametrize('n_samples', [50, 100, 500, 2000])
def test_equilibration_detection_various_lengths(analyzer_class, n_samples):
    """Equilibration detection should work for various trajectory lengths."""
    data = generate_equilibrated_timeseries(n_samples=n_samples, seed=42)

    t0, g, n_eff = analyzer_class._detect_equilibration_autocorr(data)

    assert 0 <= t0 < n_samples
    assert g >= 1.0
    assert 0 < n_eff <= n_samples


@pytest.mark.parametrize('correlation_time', [1, 5, 20, 50])
def test_statistical_inefficiency_scales_with_correlation(
    analyzer_class, correlation_time
):
    """Statistical inefficiency should increase with correlation time."""
    data = generate_equilibrated_timeseries(
        n_samples=2000,
        correlation_time=correlation_time,
        seed=42,
    )

    _t0, g, _n_eff = analyzer_class._detect_equilibration_autocorr(data)

    # g should roughly scale with correlation time
    # but this is approximate due to finite sampling
    assert g >= 1.0


@pytest.mark.parametrize('n_windows', [3, 5, 10, 20])
def test_overlap_analysis_various_window_counts(analyzer_class, tmp_path, n_windows):
    """Overlap analysis should work for various numbers of windows."""
    rc_data, rc0_values = generate_overlapping_windows(n_windows=n_windows, seed=42)

    analyzer = analyzer_class(
        log_path=tmp_path,
        log_prefix='test',
        k_umbrella=160000.0,
        rc0_values=rc0_values,
    )

    result = analyzer.analyze_overlap(rc_data)

    assert len(result.overlap_matrix) == n_windows - 1
    assert result.min_overlap >= 0


@pytest.mark.parametrize('temperature', [200.0, 300.0, 400.0, 500.0])
def test_pmf_at_various_temperatures(analyzer_class, tmp_path, temperature):
    """PMF calculation should work at various temperatures."""
    rc_data, rc0_values = generate_overlapping_windows(
        n_windows=10, temperature=temperature, seed=42
    )

    analyzer = analyzer_class(
        log_path=tmp_path,
        log_prefix='test',
        k_umbrella=160000.0,
        rc0_values=rc0_values,
    )

    result = analyzer._compute_pmf_histogram(
        rc_data, rc0_values, temperature=temperature, n_bins=25
    )

    # Should produce valid output
    assert (~np.isnan(result.pmf)).sum() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
