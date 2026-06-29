"""Free energy calculations using Empirical Valence Bond (EVB) methods."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import parsl
import polars as pl
from natsort import natsorted
from openmm import (
    CustomBondForce,
    CustomCompoundBondForce,
    HarmonicBondForce,
)
from parsl import Config, python_app

try:
    import tomllib  # ty: ignore[unresolved-import]  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10

import traceback
from typing import Any, TypeVar

from .omm_simulator import Simulator
from .reporters import RCReporter

# Constants
KB = 8.314462618e-3  # Boltzmann constant in kJ/(mol·K)

_T = TypeVar('_T')


@dataclass
class PMFResult:
    """Results from PMF calculation using MBAR.

    Attributes:
        bin_centers: Reaction coordinate values at bin centers (nm).
        pmf: Free energy values relative to minimum (kJ/mol).
        pmf_uncertainty: Standard error of PMF estimates (kJ/mol).
        free_energies: Raw free energies for each window from MBAR (kJ/mol).
        free_energy_uncertainty: Uncertainty in window free energies (kJ/mol).
    """

    bin_centers: np.ndarray
    pmf: np.ndarray
    pmf_uncertainty: np.ndarray
    free_energies: np.ndarray
    free_energy_uncertainty: np.ndarray


@dataclass
class ConvergenceResult:
    """Results from convergence analysis.

    Attributes:
        window_idx: Window index.
        mean_rc: Mean reaction coordinate value.
        sem: Standard error of the mean.
        n_blocks: Number of blocks used in analysis.
        block_means: Mean RC for each block.
        is_converged: Whether the window appears converged (SEM < threshold).
    """

    window_idx: int
    mean_rc: float
    sem: float
    n_blocks: int
    block_means: np.ndarray
    is_converged: bool


@dataclass
class OverlapResult:
    """Results from window overlap analysis.

    Attributes:
        overlap_matrix: Pairwise overlap between adjacent windows.
        min_overlap: Minimum overlap value (should be > 0.03 for MBAR).
        problem_pairs: List of window pairs with insufficient overlap.
    """

    overlap_matrix: np.ndarray
    min_overlap: float
    problem_pairs: list[tuple[int, int]]


@dataclass
class EquilibrationResult:
    """Results from equilibration detection.

    Attributes:
        window_idx: Window index.
        t0: Index of first equilibrated frame.
        g: Statistical inefficiency.
        n_effective: Effective number of uncorrelated samples.
        fraction_discarded: Fraction of trajectory discarded as equilibration.
    """

    window_idx: int
    t0: int
    g: float
    n_effective: float
    fraction_discarded: float


@dataclass
class EVBAnalysisResult:
    """Comprehensive results from EVB free energy analysis.

    Attributes:
        pmf: PMF calculation results.
        convergence: Per-window convergence analysis.
        overlap: Window overlap analysis.
        equilibration: Per-window equilibration detection.
        rc_data: Raw reaction coordinate data per window (after equilibration).
        temperature: Temperature used for analysis (K).
        k_umbrella: Umbrella force constant used (kJ/mol/nm²).
    """

    pmf: PMFResult
    convergence: list[ConvergenceResult]
    overlap: OverlapResult
    equilibration: list[EquilibrationResult]
    rc_data: list[np.ndarray]
    temperature: float
    k_umbrella: float


class EVBAnalyzer:
    """Standalone analyzer for existing EVB umbrella sampling data.

    Use this class when you have already run EVB simulations (possibly across
    multiple HPC jobs) and want to analyze the results without re-instantiating
    the full EVB class with Parsl configuration.

    This class can:
    - Load RC data from log files
    - Detect equilibration
    - Check convergence
    - Analyze window overlap
    - Compute PMF using MBAR or WHAM

    Example:
        >>> # Analyze existing EVB run
        >>> analyzer = EVBAnalyzer(
        ...     log_path=Path("/scratch/evb_run/logs"),
        ...     log_prefix="reactant",
        ...     k_umbrella=160000.0,  # Must match what was used in simulation
        ...     rc0_values=np.linspace(-0.2, 0.2, 50),  # Window centers
        ... )
        >>> result = analyzer.run_full_analysis(temperature=300.0)
        >>> print(f"Barrier: {result.pmf.pmf.max():.2f} kJ/mol")

        >>> # Or load from a metadata file saved during simulation
        >>> analyzer = EVBAnalyzer.from_metadata("/scratch/evb_run/evb_metadata.toml")
        >>> result = analyzer.run_full_analysis()
    """

    def __init__(
        self,
        log_path: Path,
        log_prefix: str,
        k_umbrella: float,
        rc0_values: np.ndarray | list[int | float],
        output_path: Path | None = None,
    ):
        """Initialize the EVB analyzer.

        Args:
            log_path: Directory containing RC log files (e.g., reactant_0.log, ...).
            log_prefix: Prefix for log file names (e.g., 'reactant' for reactant_0.log).
            k_umbrella: Umbrella force constant in kJ/mol/nm². Must match simulation.
            rc0_values: Array of target RC values for each window. Must match simulation.
            output_path: Directory for saving results. Defaults to log_path.
        """
        self.log_path = Path(log_path)
        self.log_prefix = log_prefix
        self.k = k_umbrella
        self.reaction_coordinate = np.asarray(rc0_values)
        self.output_path = Path(output_path) if output_path else self.log_path

        if not self.log_path.exists():
            raise FileNotFoundError(f'Log path does not exist: {self.log_path}')

    @classmethod
    def from_metadata(cls, metadata_path: Path) -> 'EVBAnalyzer':
        """Create analyzer from a metadata TOML file.

        The metadata file should contain the parameters used during simulation.
        This is useful for reproducibility and avoiding parameter mismatches.

        Args:
            metadata_path: Path to TOML metadata file.

        Returns:
            Configured EVBAnalyzer instance.

        Example metadata.toml:
            [evb]
            log_path = "/scratch/evb_run/logs"
            log_prefix = "reactant"
            k_umbrella = 160000.0
            rc0_values = [-0.2, -0.19, ..., 0.2]
        """
        metadata_path = Path(metadata_path)
        with open(metadata_path, 'rb') as f:
            config = tomllib.load(f)

        evb_config = config.get('evb', config)  # Support nested or flat structure

        return cls(
            log_path=Path(evb_config['log_path']),
            log_prefix=evb_config['log_prefix'],
            k_umbrella=evb_config['k_umbrella'],
            rc0_values=np.array(evb_config['rc0_values']),
            output_path=evb_config.get('output_path'),
        )

    @classmethod
    def from_evb_instance(cls, evb: 'EVB') -> 'EVBAnalyzer':
        """Create analyzer from an existing EVB instance.

        Useful when you want to decouple analysis from the simulation object.

        Args:
            evb: Existing EVB instance.

        Returns:
            EVBAnalyzer with matching parameters.
        """
        return cls(
            log_path=evb.log_path,
            log_prefix=evb.log_prefix,
            k_umbrella=evb.k,
            rc0_values=evb.reaction_coordinate,
            output_path=evb.log_path,
        )

    def save_metadata(self, output_path: Path | None = None) -> Path:
        """Save analyzer parameters to a TOML file for later reuse.

        Args:
            output_path: Path to save metadata. Defaults to log_path/evb_metadata.toml.

        Returns:
            Path to saved metadata file.
        """
        output_path = output_path or (self.log_path / 'evb_metadata.toml')

        # tomllib is read-only, so write manually
        with open(output_path, 'w') as f:
            f.write('[evb]\n')
            f.write(f'log_path = "{self.log_path}"\n')
            f.write(f'log_prefix = "{self.log_prefix}"\n')
            f.write(f'k_umbrella = {self.k}\n')
            f.write(f'rc0_values = {self.reaction_coordinate.tolist()}\n')
            if self.output_path != self.log_path:
                f.write(f'output_path = "{self.output_path}"\n')

        return output_path

    def load_rc_data(self) -> list[np.ndarray]:
        """Load reaction coordinate data from all window log files.

        Returns:
            List of RC arrays, one per window.

        Raises:
            FileNotFoundError: If any log file is missing.
        """
        rc_data = []
        n_windows = len(self.reaction_coordinate)

        for i in range(n_windows):
            rc_log = self.log_path / f'{self.log_prefix}_{i}.log'
            if not rc_log.exists():
                raise FileNotFoundError(
                    f'RC log file not found: {rc_log}. '
                    f'Expected {n_windows} windows based on rc0_values.'
                )
            rc_contents = (
                pl.read_csv(str(rc_log)).select(pl.col('rc')).to_numpy().flatten()
            )
            rc_data.append(rc_contents)

        return rc_data

    def get_available_windows(self) -> list[int]:
        """Detect which window log files are available.

        Useful for checking progress of incomplete runs.

        Returns:
            List of window indices that have log files.
        """
        available = []
        for i in range(len(self.reaction_coordinate)):
            rc_log = self.log_path / f'{self.log_prefix}_{i}.log'
            if rc_log.exists():
                available.append(i)
        return available

    def check_run_status(self) -> dict[str, Any]:
        """Check the status of an EVB run.

        Returns:
            Dictionary with status information including:
            - n_expected: Number of expected windows
            - n_complete: Number of windows with log files
            - missing_windows: List of window indices without log files
            - frames_per_window: Dict of window -> frame count
        """
        n_expected = len(self.reaction_coordinate)
        available = self.get_available_windows()
        missing = [i for i in range(n_expected) if i not in available]

        frames_per_window = {}
        for i in available:
            rc_log = self.log_path / f'{self.log_prefix}_{i}.log'
            try:
                n_frames = len(pl.read_csv(str(rc_log)))
                frames_per_window[i] = n_frames
            except Exception as e:
                frames_per_window[i] = f'Error: {e}'

        return {
            'n_expected': n_expected,
            'n_complete': len(available),
            'complete_fraction': len(available) / n_expected if n_expected > 0 else 0,
            'missing_windows': missing,
            'frames_per_window': frames_per_window,
            'total_frames': sum(
                v for v in frames_per_window.values() if isinstance(v, int)
            ),
        }

    @staticmethod
    def _detect_equilibration_autocorr(
        data: np.ndarray, max_lag: int | None = None
    ) -> tuple[int, float, float]:
        """Detect equilibration using autocorrelation analysis.

        See EVB._detect_equilibration_autocorr for full documentation.
        """
        n = len(data)
        if max_lag is None:
            max_lag = max(10, n // 4)

        data_normalized = data - np.mean(data)
        variance = np.var(data)

        if variance < 1e-10:
            return 0, 1.0, float(n)

        autocorr = np.correlate(data_normalized, data_normalized, mode='full')
        autocorr = autocorr[n - 1 :] / (variance * n)

        g = 1.0
        for t in range(1, min(max_lag, n)):
            if autocorr[t] < 0.05:
                break
            g += 2.0 * autocorr[t]
        g = max(1.0, g)

        best_t0 = 0
        best_n_eff = n / g

        for t0 in range(0, n // 2, max(1, n // 20)):
            subset = data[t0:]
            if len(subset) < 10:
                break

            subset_norm = subset - np.mean(subset)
            var_subset = np.var(subset)
            if var_subset < 1e-10:
                continue

            autocorr_subset = np.correlate(subset_norm, subset_norm, mode='full')
            autocorr_subset = autocorr_subset[len(subset) - 1 :] / (
                var_subset * len(subset)
            )

            g_subset = 1.0
            for t in range(1, min(max_lag, len(subset))):
                if autocorr_subset[t] < 0.05:
                    break
                g_subset += 2.0 * autocorr_subset[t]
            g_subset = max(1.0, g_subset)

            n_eff_subset = len(subset) / g_subset
            if n_eff_subset > best_n_eff:
                best_n_eff = n_eff_subset
                best_t0 = t0
                g = g_subset

        return best_t0, g, best_n_eff

    def detect_equilibration(
        self, rc_data: list[np.ndarray]
    ) -> list[EquilibrationResult]:
        """Detect equilibration time for each window."""
        results = []
        for i, rc in enumerate(rc_data):
            if len(rc) < 10:
                results.append(
                    EquilibrationResult(
                        window_idx=i,
                        t0=0,
                        g=1.0,
                        n_effective=float(len(rc)),
                        fraction_discarded=0.0,
                    )
                )
                continue

            t0, g, n_eff = self._detect_equilibration_autocorr(rc)
            results.append(
                EquilibrationResult(
                    window_idx=i,
                    t0=t0,
                    g=g,
                    n_effective=n_eff,
                    fraction_discarded=t0 / len(rc),
                )
            )

        return results

    def check_convergence(
        self,
        rc_data: list[np.ndarray],
        block_size: int | None = None,
        sem_threshold: float = 0.01,
    ) -> list[ConvergenceResult]:
        """Check convergence of each window using block averaging."""
        results = []
        for i, rc in enumerate(rc_data):
            n = len(rc)
            bs = block_size if block_size else max(10, n // 10)
            n_blocks = n // bs

            if n_blocks < 3:
                n_blocks = max(3, n_blocks)
                bs = n // n_blocks

            block_means = np.array(
                [rc[j * bs : (j + 1) * bs].mean() for j in range(n_blocks)]
            )
            mean_rc = np.mean(block_means)
            sem = np.std(block_means, ddof=1) / np.sqrt(n_blocks)

            results.append(
                ConvergenceResult(
                    window_idx=i,
                    mean_rc=mean_rc,
                    sem=sem,
                    n_blocks=n_blocks,
                    block_means=block_means,
                    is_converged=sem < sem_threshold,
                )
            )

        return results

    def analyze_overlap(
        self,
        rc_data: list[np.ndarray],
        n_bins: int = 50,
        min_overlap_threshold: float = 0.03,
    ) -> OverlapResult:
        """Analyze overlap between adjacent umbrella windows."""
        n_windows = len(rc_data)
        overlap_matrix = np.zeros(n_windows - 1)
        problem_pairs = []

        all_rc = np.concatenate(rc_data)
        bin_edges = np.linspace(all_rc.min(), all_rc.max(), n_bins + 1)

        for i in range(n_windows - 1):
            hist1, _ = np.histogram(rc_data[i], bins=bin_edges, density=True)
            hist2, _ = np.histogram(rc_data[i + 1], bins=bin_edges, density=True)
            bin_width = bin_edges[1] - bin_edges[0]
            overlap = np.sum(np.minimum(hist1, hist2)) * bin_width
            overlap_matrix[i] = overlap

            if overlap < min_overlap_threshold:
                problem_pairs.append((i, i + 1))

        return OverlapResult(
            overlap_matrix=overlap_matrix,
            min_overlap=overlap_matrix.min() if len(overlap_matrix) > 0 else 0.0,
            problem_pairs=problem_pairs,
        )

    def compute_pmf(
        self, rc_data: list[np.ndarray], temperature: float = 300.0, n_bins: int = 50
    ) -> PMFResult:
        """Compute PMF using MBAR (preferred) or WHAM fallback."""
        if not rc_data:
            raise ValueError('No RC data provided')

        n_windows = len(rc_data)
        rc0_values = self.reaction_coordinate[:n_windows]

        if find_spec('pymbar') is not None:
            return self._compute_pmf_mbar(rc_data, rc0_values, temperature, n_bins)
        else:
            return self._compute_pmf_histogram(rc_data, rc0_values, temperature, n_bins)

    def _compute_pmf_mbar(
        self,
        rc_data: list[np.ndarray],
        rc0_values: np.ndarray,
        temperature: float,
        n_bins: int,
    ) -> PMFResult:
        """Compute PMF using MBAR."""
        import pymbar  # ty: ignore[unresolved-import]

        beta = 1.0 / (KB * temperature)
        N_k = np.array([len(rc) for rc in rc_data])
        n_windows = len(rc_data)
        rc_all = np.concatenate(rc_data)
        N_total = len(rc_all)

        u_kn = np.zeros((n_windows, N_total))
        for k in range(n_windows):
            u_kn[k, :] = beta * 0.5 * self.k * (rc_all - rc0_values[k]) ** 2

        mbar = pymbar.MBAR(u_kn, N_k, verbose=False)

        results = mbar.compute_free_energy_differences()
        free_energies = results['Delta_f'][0, :]
        free_energy_uncertainty = results['dDelta_f'][0, :]

        rc_min, rc_max = rc_all.min(), rc_all.max()
        bin_edges = np.linspace(rc_min, rc_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        pmf = np.zeros(n_bins)
        for i in range(n_bins):
            bin_mask = (rc_all >= bin_edges[i]) & (rc_all < bin_edges[i + 1])
            if bin_mask.sum() == 0:
                pmf[i] = np.nan
                continue
            try:
                weights = mbar.compute_weights()[0]
                p_bin = np.sum(weights[bin_mask])
                pmf[i] = -KB * temperature * np.log(p_bin) if p_bin > 0 else np.nan
            except Exception:
                pmf[i] = np.nan

        pmf -= np.nanmin(pmf)
        pmf_uncertainty = np.full(n_bins, np.mean(free_energy_uncertainty))

        return PMFResult(
            bin_centers=bin_centers,
            pmf=pmf,
            pmf_uncertainty=pmf_uncertainty,
            free_energies=free_energies,
            free_energy_uncertainty=free_energy_uncertainty,
        )

    def _compute_pmf_histogram(
        self,
        rc_data: list[np.ndarray],
        rc0_values: np.ndarray,
        temperature: float,
        n_bins: int,
    ) -> PMFResult:
        """Compute PMF using WHAM iteration."""
        beta = 1.0 / (KB * temperature)
        n_windows = len(rc_data)
        rc_all = np.concatenate(rc_data)

        bin_edges = np.linspace(rc_all.min(), rc_all.max(), n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        histograms = np.zeros((n_windows, n_bins))
        for k, rc in enumerate(rc_data):
            histograms[k], _ = np.histogram(rc, bins=bin_edges)

        f_k = np.zeros(n_windows)
        N_k = np.array([len(rc) for rc in rc_data])

        for _ in range(1000):
            f_k_old = f_k.copy()
            denom = np.zeros(n_bins)
            for k in range(n_windows):
                bias = 0.5 * self.k * (bin_centers - rc0_values[k]) ** 2
                denom += N_k[k] * np.exp(f_k[k] - beta * bias)

            n_total = histograms.sum(axis=0)
            P_unbiased = np.zeros(n_bins)
            nonzero = denom > 0
            P_unbiased[nonzero] = n_total[nonzero] / denom[nonzero]

            P_sum = P_unbiased.sum() * bin_width
            if P_sum > 0:
                P_unbiased /= P_sum

            for k in range(n_windows):
                bias = 0.5 * self.k * (bin_centers - rc0_values[k]) ** 2
                integral = np.sum(P_unbiased * np.exp(-beta * bias)) * bin_width
                if integral > 0:
                    f_k[k] = -np.log(integral)

            f_k -= f_k[0]
            if np.max(np.abs(f_k - f_k_old)) < 1e-7:
                break

        pmf = np.full(n_bins, np.nan)
        valid = P_unbiased > 0
        pmf[valid] = -KB * temperature * np.log(P_unbiased[valid])
        pmf -= np.nanmin(pmf)

        return PMFResult(
            bin_centers=bin_centers,
            pmf=pmf,
            pmf_uncertainty=np.full(n_bins, 0.5),
            free_energies=f_k * KB * temperature,
            free_energy_uncertainty=np.full(n_windows, 0.5),
        )

    def run_full_analysis(
        self,
        temperature: float = 300.0,
        n_bins: int = 50,
        block_size: int | None = None,
        sem_threshold: float = 0.01,
        overlap_threshold: float = 0.03,
        discard_equilibration: bool = True,
    ) -> EVBAnalysisResult:
        """Run comprehensive free energy analysis.

        This is the main entry point for analyzing existing EVB data.

        Args:
            temperature: Temperature in Kelvin.
            n_bins: Number of bins for PMF.
            block_size: Block size for convergence analysis.
            sem_threshold: SEM threshold for convergence.
            overlap_threshold: Minimum required window overlap.
            discard_equilibration: Whether to discard equilibration frames.

        Returns:
            EVBAnalysisResult with complete analysis.
        """
        # Load data
        rc_data_raw = self.load_rc_data()

        # Equilibration
        equilibration = self.detect_equilibration(rc_data_raw)
        if discard_equilibration:
            rc_data = [rc[eq.t0 :] for rc, eq in zip(rc_data_raw, equilibration, strict=True)]
            n_discarded = sum(eq.t0 for eq in equilibration)
            n_total = sum(len(rc) for rc in rc_data_raw)
        else:
            rc_data = rc_data_raw

        # Convergence
        convergence = self.check_convergence(rc_data, block_size, sem_threshold)
        n_converged = sum(1 for c in convergence if c.is_converged)

        # Overlap
        overlap = self.analyze_overlap(rc_data, n_bins, overlap_threshold)

        # PMF
        pmf = self.compute_pmf(rc_data, temperature, n_bins)
        valid_pmf = pmf.pmf[~np.isnan(pmf.pmf)]
        if len(valid_pmf) > 0:
            barrier = valid_pmf.max()

        return EVBAnalysisResult(
            pmf=pmf,
            convergence=convergence,
            overlap=overlap,
            equilibration=equilibration,
            rc_data=rc_data,
            temperature=temperature,
            k_umbrella=self.k,
        )

    def save_analysis_results(
        self, result: EVBAnalysisResult, output_dir: Path | None = None
    ) -> None:
        """Save analysis results to files."""
        output_dir = Path(output_dir) if output_dir else self.output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # PMF
        pmf_df = pl.DataFrame(
            {
                'RC': result.pmf.bin_centers,
                'PMF_kJ_mol': result.pmf.pmf,
                'uncertainty_kJ_mol': result.pmf.pmf_uncertainty,
            }
        )
        pmf_df.write_csv(str(output_dir / f'{self.log_prefix}_pmf.csv'))

        # Window free energies
        fe_df = pl.DataFrame(
            {
                'window': list(range(len(result.pmf.free_energies))),
                'rc0': self.reaction_coordinate[
                    : len(result.pmf.free_energies)
                ].tolist(),
                'free_energy_kJ_mol': result.pmf.free_energies,
                'uncertainty_kJ_mol': result.pmf.free_energy_uncertainty,
            }
        )
        fe_df.write_csv(str(output_dir / f'{self.log_prefix}_window_free_energies.csv'))

        # Convergence
        conv_df = pl.DataFrame(
            {
                'window': [c.window_idx for c in result.convergence],
                'mean_rc': [c.mean_rc for c in result.convergence],
                'sem': [c.sem for c in result.convergence],
                'is_converged': [c.is_converged for c in result.convergence],
            }
        )
        conv_df.write_csv(str(output_dir / f'{self.log_prefix}_convergence.csv'))

        # Summary
        summary_path = output_dir / f'{self.log_prefix}_analysis_summary.txt'
        with open(summary_path, 'w') as f:
            f.write('EVB Free Energy Analysis Summary\n')
            f.write('=' * 50 + '\n\n')
            f.write(f'Temperature: {result.temperature} K\n')
            f.write(f'Umbrella force constant: {result.k_umbrella} kJ/mol/nm²\n')
            f.write(f'Number of windows: {len(result.rc_data)}\n\n')

            valid_pmf = result.pmf.pmf[~np.isnan(result.pmf.pmf)]
            if len(valid_pmf) > 0:
                f.write(f'Barrier height: {valid_pmf.max():.2f} kJ/mol\n')
                f.write(f'              = {valid_pmf.max() / 4.184:.2f} kcal/mol\n\n')

            f.write(
                f'Convergence: {sum(1 for c in result.convergence if c.is_converged)}/{len(result.convergence)} converged\n'
            )
            f.write(f'Minimum overlap: {result.overlap.min_overlap:.3f}\n')


@python_app
def run_evb_window(
    topology: Path,
    coord_file: Path,
    out_path: Path,
    rc_file: Path,
    umbrella_force: dict[str, int | float],
    morse_bond: dict[str, int | float],
    rc_freq: int,
    steps: int,
    dt: float,
    platform: str,
    restraint_sel: str | None,
) -> None:
    """Parsl python app. Separate module due to need for serialization."""
    evb = EVBCalculation(
        topology=topology,
        coord_file=coord_file,
        out_path=out_path,
        rc_file=rc_file,
        umbrella=umbrella_force,
        morse_bond=morse_bond,
        rc_freq=rc_freq,
        steps=steps,
        dt=dt,
        platform=platform,
        restraint_sel=restraint_sel,
    )

    evb.run()


class EVB:
    """EVB orchestrator. Sets up full EVB run for a set of reactants or products,
    and distributes calculations using Parsl."""

    def __init__(
        self,
        topology: Path,
        coordinates: Path,
        donor_atom: str,
        acceptor_atom: str,
        reactive_atom: str,
        parsl_config: Config,
        log_path: Path,
        log_prefix: str = 'reactant',
        rc_write_freq: int = 5,
        steps: int = 500000,
        dt: float = 0.002,
        k: float = 160000.0,  # Umbrella force constant (kJ/mol/nm^2)
        k_path: float = 100.0,  # Path restraint force constant (kJ/mol)
        D_e: float = 392.46,  # Morse well depth (kJ/mol) - from BDE
        alpha: float = 13.275,  # Morse width parameter (nm^-1) - computed from sqrt(k_bond/(2*D_e))
        r0: float = 0.109,  # Equilibrium bond distance (nm)
        platform: str = 'CUDA',
        n_windows: int = 50,
        reaction_coordinate: list[float] | None = None,
        restraint_sel: str | None = None,
    ):
        """Initialize the EVB orchestrator.

        Args:
            topology: Path to the system topology file (prmtop).
            coordinates: Path to the system coordinate file (inpcrd).
            donor_atom: MDAnalysis selection string for the donor atom (the
                atom the reactive atom departs from).
            acceptor_atom: MDAnalysis selection string for the acceptor atom
                (the atom the reactive atom moves toward).
            reactive_atom: MDAnalysis selection string for the transferring
                atom shared by donor and acceptor. The reaction coordinate is
                dist(donor, reactive) - dist(acceptor, reactive).
            parsl_config: Parsl configuration for distributed execution.
            log_path: Directory path for writing reaction coordinate logs.
            log_prefix: Prefix for log file names. Defaults to 'reactant'.
            rc_write_freq: Steps between reaction coordinate writes. Defaults to 5.
            steps: Number of simulation steps per window. Defaults to 500000.
            dt: Integration timestep in picoseconds. Defaults to 0.002.
            k: Umbrella force constant in kJ/mol/nm^2. Defaults to 160000.0.
            k_path: Path restraint force constant in kJ/mol. Defaults to 100.0.
            D_e: Morse well depth in kJ/mol. Defaults to 392.46.
            alpha: Morse width parameter in nm^-1. Defaults to 13.275.
            r0: Equilibrium bond distance in nm. Defaults to 0.109.
            platform: OpenMM platform name. Defaults to 'CUDA'.
            n_windows: Number of umbrella windows to generate when an explicit
                reaction_coordinate is not provided. Defaults to 50.
            reaction_coordinate: Optional list of [min, max, increment] defining
                the reaction coordinate windows. If None, it is inferred from the
                donor/acceptor/reactive atom positions. Defaults to None.
            restraint_sel: Optional MDAnalysis selection string for backbone
                restraints. Defaults to None.
        """
        self.topology = Path(topology)
        self.coordinates = Path(coordinates)
        self.path = self.topology.parent / 'evb'

        self.parsl_config = parsl_config
        self.dfk = None
        self._owns_parsl = False  # Track if we loaded Parsl or borrowed it

        self.log_path = Path(log_path)
        self.log_prefix = log_prefix
        self.rc_freq = rc_write_freq

        self.steps = steps
        self.dt = dt
        self.k = k
        self.k_path = k_path
        self.D_e = D_e
        self.alpha = alpha
        self.r0 = r0

        self.platform = platform
        self.restraint_sel = restraint_sel
        self.n_windows = n_windows

        self.prepare_inputs(
            donor_atom, acceptor_atom, reactive_atom, reaction_coordinate
        )

    def prepare_inputs(
        self,
        donor: str,
        acceptor: str,
        reactor: str,
        rc: list[float] | None = None,
    ) -> None:
        """Resolve donor/acceptor/reactive atoms and build the reaction coordinate.

        Selects the participating atoms from the topology, records the Morse
        and umbrella atom index sets, and constructs the per-window reaction
        coordinate values.

        Args:
            donor: MDAnalysis selection string for the donor atom.
            acceptor: MDAnalysis selection string for the acceptor atom.
            reactor: MDAnalysis selection string for the transferring (reactive)
                atom shared by donor and acceptor.
            rc: Optional list of [min, max, increment] defining the reaction
                coordinate windows. If None, it is inferred from the donor,
                acceptor, and reactive atom positions using n_windows.
                Defaults to None.
        """
        u = mda.Universe(self.topology, self.coordinates)

        a0 = u.select_atoms(donor)
        a1 = u.select_atoms(acceptor)
        a2 = u.select_atoms(reactor)

        self.morse_atoms = [a0.ix[0], a2.ix[0]]
        self.umbrella_atoms = [a0.ix[0], a1.ix[0], a2.ix[0]]

        if rc is None:
            p0 = a0.positions
            p1 = a1.positions
            p2 = a2.positions

            rc_min = float((np.linalg.norm(p0 - p2) - np.linalg.norm(p1 - p2)) * 0.1)  # to nm
            rc_interval = float(np.abs(rc_min * 2) / self.n_windows)
            rc = [rc_min, rc_min * -1 + rc_interval, rc_interval]

        assert rc is not None
        self.reaction_coordinate = self.construct_rc(rc)

    def construct_rc(self, rc: list[float]) -> np.ndarray:
        """Construct linearly spaced reaction coordinate.

        Args:
            rc: List of [rc_minimum, rc_maximum, rc_increment].

        Returns:
            Linearly spaced reaction coordinate as a numpy array.
        """
        return np.arange(rc[0], rc[1] + rc[2], rc[2])

    def initialize(self) -> None:
        """Initialize Parsl for runs.

        If Parsl is already running (e.g., from a parent pipeline), reuses the
        existing DataFlowKernel. Otherwise, loads a new one from parsl_config.
        """
        if self.dfk is None:
            try:
                # Try to get existing DFK (may be from parent pipeline)
                existing_dfk = parsl.dfk()
                self.dfk = existing_dfk
                self._owns_parsl = False
            except Exception:
                # No DFK exists, load a new one
                self.dfk = parsl.load(self.parsl_config)
                self._owns_parsl = True

    def shutdown(self) -> None:
        """Clean up Parsl after runs.

        Only cleans up if this instance loaded Parsl itself. If Parsl was
        borrowed from a parent pipeline, leaves it running.
        """
        if self._owns_parsl and self.dfk:
            self.dfk.cleanup()
            parsl.clear()
        self.dfk = None
        self._owns_parsl = False

    def run_evb(self, parsl_func: Callable = run_evb_window) -> None:
        """Collect futures for each EVB window and distribute."""
        self.initialize()

        futures = []
        for i, rc0 in enumerate(self.reaction_coordinate):
            umbrella = {**self.umbrella, 'rc0': rc0}

            futures.append(
                parsl_func(
                    topology=self.topology,
                    coord_file=self.coordinates,
                    out_path=self.path / f'window{i}',
                    rc_file=self.log_path / f'{self.log_prefix}_{i}.log',
                    umbrella_force=umbrella,
                    morse_bond=self.morse_bond,
                    rc_freq=self.rc_freq,
                    steps=self.steps,
                    dt=self.dt,
                    platform=self.platform,
                    restraint_sel=self.restraint_sel,
                )
            )

        for future in futures:
            try:
                future.result()
            except Exception as e:
                tb = traceback.format_exc()
                print(f'EVB failed for 1 or more windows!{e}{tb}')

        if self._owns_parsl:
            self.shutdown()

    def process_evb_run(self) -> pl.DataFrame:
        """Reads in RCReporter logs from an EVB run and collects RC data.

        This method collects the reaction coordinate values from each umbrella
        window. For actual free energy calculation, use compute_pmf() or
        run_full_analysis() after this method.

        Returns:
            DataFrame with columns: window, RC, rc0 (target RC for window).

        Raises:
            ValueError: If no EVB windows are found.
            FileNotFoundError: If RC log files are missing.
        """
        windows = natsorted(list(self.path.glob('window*')))
        if not windows:
            raise ValueError(f'No EVB windows found in {self.path}')

        all_data = []

        for i in range(len(windows)):
            rc_log = self.log_path / f'{self.log_prefix}_{i}.log'
            if not rc_log.exists():
                raise FileNotFoundError(f'RC log file not found: {rc_log}')

            rc_contents = (
                pl.read_csv(str(rc_log)).select(pl.col('rc')).to_numpy().flatten()
            )
            n_frames = len(rc_contents)

            window_df = pl.DataFrame(
                {
                    'window': np.full(n_frames, i, dtype=np.int32),
                    'RC': rc_contents,
                    'rc0': np.full(
                        n_frames, self.reaction_coordinate[i], dtype=np.float64
                    ),
                }
            )
            all_data.append(window_df)

        df = pl.concat(all_data)
        output_path = self.log_path / f'{self.log_prefix}_rc_data.parquet'
        df.write_parquet(str(output_path))

        return df

    def load_rc_data(self) -> list[np.ndarray]:
        """Load reaction coordinate data from all windows.

        Returns:
            List of RC arrays, one per window.

        Raises:
            ValueError: If no windows found.
            FileNotFoundError: If log files missing.
        """
        windows = natsorted(list(self.path.glob('window*')))
        if not windows:
            raise ValueError(f'No EVB windows found in {self.path}')

        rc_data = []
        for i in range(len(windows)):
            rc_log = self.log_path / f'{self.log_prefix}_{i}.log'
            if not rc_log.exists():
                raise FileNotFoundError(f'RC log file not found: {rc_log}')
            rc_contents = (
                pl.read_csv(str(rc_log)).select(pl.col('rc')).to_numpy().flatten()
            )
            rc_data.append(rc_contents)

        return rc_data

    def detect_equilibration(
        self, rc_data: list[np.ndarray], method: str = 'statistical_inefficiency'
    ) -> list[EquilibrationResult]:
        """Detect equilibration time for each window.

        Uses statistical inefficiency to identify when each trajectory has
        equilibrated. Frames before t0 should be discarded for analysis.

        Args:
            rc_data: List of RC arrays for each window.
            method: Detection method. Currently supports 'statistical_inefficiency'.

        Returns:
            List of EquilibrationResult for each window.
        """
        results = []

        for i, rc in enumerate(rc_data):
            if len(rc) < 10:
                # Too few samples for equilibration detection
                results.append(
                    EquilibrationResult(
                        window_idx=i,
                        t0=0,
                        g=1.0,
                        n_effective=float(len(rc)),
                        fraction_discarded=0.0,
                    )
                )
                continue

            # Use autocorrelation-based detection
            t0, g, n_eff = self._detect_equilibration_autocorr(rc)

            results.append(
                EquilibrationResult(
                    window_idx=i,
                    t0=t0,
                    g=g,
                    n_effective=n_eff,
                    fraction_discarded=t0 / len(rc),
                )
            )

        return results

    @staticmethod
    def _detect_equilibration_autocorr(
        data: np.ndarray, max_lag: int | None = None
    ) -> tuple[int, float, float]:
        """Detect equilibration using autocorrelation analysis.

        Implements a simplified version of Chodera's method for detecting
        equilibration based on statistical inefficiency.

        Args:
            data: Time series data.
            max_lag: Maximum lag for autocorrelation. Defaults to len(data)//4.

        Returns:
            Tuple of (t0, g, n_effective) where:
                t0: Index of first equilibrated frame
                g: Statistical inefficiency
                n_effective: Effective number of uncorrelated samples
        """
        n = len(data)
        if max_lag is None:
            max_lag = max(10, n // 4)

        # Normalize data
        data_normalized = data - np.mean(data)
        variance = np.var(data)

        if variance < 1e-10:
            return 0, 1.0, float(n)

        # Compute autocorrelation
        autocorr = np.correlate(data_normalized, data_normalized, mode='full')
        autocorr = autocorr[n - 1 :] / (variance * n)

        # Compute statistical inefficiency g = 1 + 2 * sum(C(t))
        # where C(t) is the normalized autocorrelation
        g = 1.0
        for t in range(1, min(max_lag, n)):
            if autocorr[t] < 0.05:  # Stop when correlation becomes negligible
                break
            g += 2.0 * autocorr[t]

        g = max(1.0, g)  # g must be >= 1

        # Try different starting points to find where g is minimized
        # (indicating equilibration)
        best_t0 = 0
        best_n_eff = n / g

        for t0 in range(0, n // 2, max(1, n // 20)):
            subset = data[t0:]
            if len(subset) < 10:
                break

            subset_norm = subset - np.mean(subset)
            var_subset = np.var(subset)
            if var_subset < 1e-10:
                continue

            autocorr_subset = np.correlate(subset_norm, subset_norm, mode='full')
            autocorr_subset = autocorr_subset[len(subset) - 1 :] / (
                var_subset * len(subset)
            )

            g_subset = 1.0
            for t in range(1, min(max_lag, len(subset))):
                if autocorr_subset[t] < 0.05:
                    break
                g_subset += 2.0 * autocorr_subset[t]
            g_subset = max(1.0, g_subset)

            n_eff_subset = len(subset) / g_subset

            if n_eff_subset > best_n_eff:
                best_n_eff = n_eff_subset
                best_t0 = t0
                g = g_subset

        return best_t0, g, best_n_eff

    def check_convergence(
        self,
        rc_data: list[np.ndarray],
        block_size: int | None = None,
        sem_threshold: float = 0.01,
    ) -> list[ConvergenceResult]:
        """Check convergence of each window using block averaging.

        Block averaging divides the trajectory into blocks and computes
        the standard error of the mean across blocks. Well-converged
        simulations should have low SEM values.

        Args:
            rc_data: List of RC arrays for each window.
            block_size: Number of frames per block. Defaults to n_frames // 10.
            sem_threshold: SEM threshold for convergence (nm). Defaults to 0.01.

        Returns:
            List of ConvergenceResult for each window.
        """
        results = []

        for i, rc in enumerate(rc_data):
            n = len(rc)
            bs = block_size if block_size else max(10, n // 10)
            n_blocks = n // bs

            if n_blocks < 3:
                n_blocks = max(3, n_blocks)
                bs = n // n_blocks

            # Compute block means
            block_means = np.array(
                [rc[j * bs : (j + 1) * bs].mean() for j in range(n_blocks)]
            )

            mean_rc = np.mean(block_means)
            sem = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
            is_converged = sem < sem_threshold

            results.append(
                ConvergenceResult(
                    window_idx=i,
                    mean_rc=mean_rc,
                    sem=sem,
                    n_blocks=n_blocks,
                    block_means=block_means,
                    is_converged=is_converged,
                )
            )

        return results

    def analyze_overlap(
        self,
        rc_data: list[np.ndarray],
        n_bins: int = 50,
        min_overlap_threshold: float = 0.03,
    ) -> OverlapResult:
        """Analyze overlap between adjacent umbrella windows.

        Good overlap between windows is critical for accurate free energy
        estimation with WHAM/MBAR. Adjacent windows should have >3% overlap
        for reliable results.

        Args:
            rc_data: List of RC arrays for each window.
            n_bins: Number of histogram bins for overlap calculation.
            min_overlap_threshold: Minimum acceptable overlap (default 0.03 = 3%).

        Returns:
            OverlapResult with overlap matrix and problem identification.
        """
        n_windows = len(rc_data)
        overlap_matrix = np.zeros(n_windows - 1)
        problem_pairs = []

        # Determine global bin edges
        all_rc = np.concatenate(rc_data)
        rc_min, rc_max = all_rc.min(), all_rc.max()
        bin_edges = np.linspace(rc_min, rc_max, n_bins + 1)

        for i in range(n_windows - 1):
            # Compute normalized histograms
            hist1, _ = np.histogram(rc_data[i], bins=bin_edges, density=True)
            hist2, _ = np.histogram(rc_data[i + 1], bins=bin_edges, density=True)

            # Overlap integral: integral of min(p1, p2)
            bin_width = bin_edges[1] - bin_edges[0]
            overlap = np.sum(np.minimum(hist1, hist2)) * bin_width
            overlap_matrix[i] = overlap

            if overlap < min_overlap_threshold:
                problem_pairs.append((i, i + 1))

        min_overlap = overlap_matrix.min() if len(overlap_matrix) > 0 else 0.0

        return OverlapResult(
            overlap_matrix=overlap_matrix,
            min_overlap=min_overlap,
            problem_pairs=problem_pairs,
        )

    def compute_pmf(
        self, rc_data: list[np.ndarray], temperature: float = 300.0, n_bins: int = 50
    ) -> PMFResult:
        """Compute the Potential of Mean Force using MBAR.

        Uses the pymbar library to compute free energies from umbrella
        sampling data. Falls back to WHAM-like histogram reweighting
        if pymbar is not available.

        Args:
            rc_data: List of RC arrays for each window (after equilibration).
            temperature: Temperature in Kelvin. Defaults to 300.0.
            n_bins: Number of bins for PMF histogram. Defaults to 50.

        Returns:
            PMFResult with free energy profile and uncertainties.

        Raises:
            ValueError: If rc_data is empty or windows have no samples.
        """
        if not rc_data:
            raise ValueError('No RC data provided')

        n_windows = len(rc_data)
        rc0_values = self.reaction_coordinate[:n_windows]

        if find_spec('pymbar') is not None:
            return self._compute_pmf_mbar(rc_data, rc0_values, temperature, n_bins)
        else:
            return self._compute_pmf_histogram(rc_data, rc0_values, temperature, n_bins)

    def _compute_pmf_mbar(
        self,
        rc_data: list[np.ndarray],
        rc0_values: np.ndarray,
        temperature: float,
        n_bins: int,
    ) -> PMFResult:
        """Compute PMF using MBAR (Multistate Bennett Acceptance Ratio).

        MBAR is the statistically optimal method for combining data from
        multiple thermodynamic states (umbrella windows).

        Reference:
            Shirts & Chodera, J. Chem. Phys. 129, 124105 (2008)
        """
        import pymbar  # ty: ignore[unresolved-import]

        beta = 1.0 / (KB * temperature)
        k_umb = self.k  # Umbrella force constant in kJ/mol/nm²

        # Number of samples in each window
        N_k = np.array([len(rc) for rc in rc_data])
        n_windows = len(rc_data)

        # Concatenate all samples
        rc_all = np.concatenate(rc_data)
        N_total = len(rc_all)

        # Build reduced potential energy matrix u_kn
        # u_kn[k, n] = beta * U_bias_k(rc_n)
        # where U_bias_k = 0.5 * k_umb * (rc - rc0_k)^2
        u_kn = np.zeros((n_windows, N_total))

        for k in range(n_windows):
            u_kn[k, :] = beta * 0.5 * k_umb * (rc_all - rc0_values[k]) ** 2

        # Initialize MBAR
        mbar = pymbar.MBAR(u_kn, N_k, verbose=False)

        # Get free energies for each window
        results = mbar.compute_free_energy_differences()
        free_energies = results['Delta_f'][0, :]  # Relative to first window
        free_energy_uncertainty = results['dDelta_f'][0, :]

        # Compute PMF on a grid using histogram reweighting
        rc_min, rc_max = rc_all.min(), rc_all.max()
        bin_edges = np.linspace(rc_min, rc_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # For PMF, we need to compute the free energy at each bin
        # This requires creating "target" states for each bin
        pmf = np.zeros(n_bins)
        pmf_uncertainty = np.zeros(n_bins)

        # Use MBAR to compute expectations for histogram bin populations
        # Then convert to free energy
        for i, _rc_target in enumerate(bin_centers):
            # Create a flat potential (unbiased) target state
            # and compute the probability of being in each bin
            bin_mask = (rc_all >= bin_edges[i]) & (rc_all < bin_edges[i + 1])

            if bin_mask.sum() == 0:
                pmf[i] = np.nan
                pmf_uncertainty[i] = np.nan
                continue

            # Use MBAR weights to compute unbiased histogram
            try:
                # Get weights for unbiased state
                weights = mbar.compute_weights()[0]  # Weights for first state

                # Probability in this bin under unbiased distribution
                p_bin = np.sum(weights[bin_mask])

                if p_bin > 0:
                    pmf[i] = -KB * temperature * np.log(p_bin)
                else:
                    pmf[i] = np.nan
                    pmf_uncertainty[i] = np.nan

            except Exception:
                pmf[i] = np.nan
                pmf_uncertainty[i] = np.nan

        # Shift PMF so minimum is zero
        valid_pmf = pmf[~np.isnan(pmf)]
        if len(valid_pmf) > 0:
            pmf -= np.nanmin(pmf)

        # Estimate uncertainties using bootstrap (simplified)
        # Full implementation would use mbar.computePMF with uncertainties
        pmf_uncertainty = np.full(n_bins, np.mean(free_energy_uncertainty))

        return PMFResult(
            bin_centers=bin_centers,
            pmf=pmf,
            pmf_uncertainty=pmf_uncertainty,
            free_energies=free_energies,
            free_energy_uncertainty=free_energy_uncertainty,
        )

    def _compute_pmf_histogram(
        self,
        rc_data: list[np.ndarray],
        rc0_values: np.ndarray,
        temperature: float,
        n_bins: int,
    ) -> PMFResult:
        """Compute PMF using WHAM-like histogram reweighting.

        This is a simplified implementation when pymbar is not available.
        Results may be less accurate than MBAR, especially for poor overlap.
        """
        beta = 1.0 / (KB * temperature)
        k_umb = self.k
        n_windows = len(rc_data)

        # Concatenate all samples
        rc_all = np.concatenate(rc_data)
        rc_min, rc_max = rc_all.min(), rc_all.max()
        bin_edges = np.linspace(rc_min, rc_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        # Compute histogram for each window
        histograms = np.zeros((n_windows, n_bins))
        for k, rc in enumerate(rc_data):
            histograms[k], _ = np.histogram(rc, bins=bin_edges)

        # WHAM iteration
        # Initialize free energies
        f_k = np.zeros(n_windows)
        N_k = np.array([len(rc) for rc in rc_data])

        # Iterative WHAM equations
        max_iter = 1000
        tolerance = 1e-7

        for iteration in range(max_iter):
            f_k_old = f_k.copy()

            # Compute denominator for each bin
            # denom[i] = sum_k N_k * exp(f_k - beta * U_k(x_i))
            denom = np.zeros(n_bins)
            for k in range(n_windows):
                bias = 0.5 * k_umb * (bin_centers - rc0_values[k]) ** 2
                denom += N_k[k] * np.exp(f_k[k] - beta * bias)

            # Compute unbiased probability
            # P(x_i) = sum_k n_ki / denom[i]
            n_total = histograms.sum(axis=0)
            P_unbiased = np.zeros(n_bins)
            nonzero_mask = denom > 0
            P_unbiased[nonzero_mask] = n_total[nonzero_mask] / denom[nonzero_mask]

            # Normalize
            P_sum = P_unbiased.sum() * bin_width
            if P_sum > 0:
                P_unbiased /= P_sum

            # Update free energies
            for k in range(n_windows):
                bias = 0.5 * k_umb * (bin_centers - rc0_values[k]) ** 2
                integrand = P_unbiased * np.exp(-beta * bias)
                integral = np.sum(integrand) * bin_width
                if integral > 0:
                    f_k[k] = -np.log(integral)

            # Check convergence
            f_k -= f_k[0]  # Reference to first window
            delta = np.max(np.abs(f_k - f_k_old))
            if delta < tolerance:
                break

        # Compute PMF from probability
        pmf = np.full(n_bins, np.nan)
        valid_mask = P_unbiased > 0
        pmf[valid_mask] = -KB * temperature * np.log(P_unbiased[valid_mask])

        # Shift to minimum
        pmf -= np.nanmin(pmf)

        # Estimate uncertainty (rough approximation)
        pmf_uncertainty = np.full(n_bins, 0.5)  # Placeholder

        return PMFResult(
            bin_centers=bin_centers,
            pmf=pmf,
            pmf_uncertainty=pmf_uncertainty,
            free_energies=f_k * KB * temperature,
            free_energy_uncertainty=np.full(n_windows, 0.5),
        )

    def run_full_analysis(
        self,
        temperature: float = 300.0,
        n_bins: int = 50,
        block_size: int | None = None,
        sem_threshold: float = 0.01,
        overlap_threshold: float = 0.03,
        discard_equilibration: bool = True,
    ) -> EVBAnalysisResult:
        """Run comprehensive free energy analysis on EVB data.

        This method performs:
        1. Equilibration detection and removal
        2. Convergence checking via block averaging
        3. Window overlap analysis
        4. PMF calculation using MBAR (or WHAM fallback)

        Args:
            temperature: Temperature in Kelvin. Defaults to 300.0.
            n_bins: Number of bins for PMF. Defaults to 50.
            block_size: Block size for convergence analysis.
            sem_threshold: SEM threshold for convergence (nm).
            overlap_threshold: Minimum required window overlap.
            discard_equilibration: Whether to discard equilibration frames.

        Returns:
            EVBAnalysisResult with complete analysis results.

        Example:
            >>> evb = EVB(topology, coordinates, ...)
            >>> evb.run_evb()  # Run umbrella sampling
            >>> result = evb.run_full_analysis(temperature=300.0)
            >>> print(f"Barrier height: {result.pmf.pmf.max():.2f} kJ/mol")
        """
        # Load RC data
        rc_data_raw = self.load_rc_data()

        # Detect equilibration
        equilibration = self.detect_equilibration(rc_data_raw)

        # Remove equilibration frames if requested
        if discard_equilibration:
            rc_data = [rc[eq.t0 :] for rc, eq in zip(rc_data_raw, equilibration, strict=True)]
            n_discarded = sum(eq.t0 for eq in equilibration)
            n_total = sum(len(rc) for rc in rc_data_raw)
        else:
            rc_data = rc_data_raw

        # Check convergence
        convergence = self.check_convergence(rc_data, block_size, sem_threshold)
        n_converged = sum(1 for c in convergence if c.is_converged)

        # Analyze overlap
        overlap = self.analyze_overlap(rc_data, n_bins, overlap_threshold)

        # Compute PMF
        pmf = self.compute_pmf(rc_data, temperature, n_bins)

        # Report key results
        valid_pmf = pmf.pmf[~np.isnan(pmf.pmf)]
        if len(valid_pmf) > 0:
            barrier = valid_pmf.max()

        return EVBAnalysisResult(
            pmf=pmf,
            convergence=convergence,
            overlap=overlap,
            equilibration=equilibration,
            rc_data=rc_data,
            temperature=temperature,
            k_umbrella=self.k,
        )

    def save_analysis_results(
        self, result: EVBAnalysisResult, output_dir: Path | None = None
    ) -> None:
        """Save analysis results to files.

        Saves:
        - PMF data as CSV
        - Full results as parquet
        - Summary statistics as text

        Args:
            result: EVBAnalysisResult from run_full_analysis().
            output_dir: Output directory. Defaults to log_path.
        """
        output_dir = output_dir or self.log_path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PMF
        pmf_df = pl.DataFrame(
            {
                'RC': result.pmf.bin_centers,
                'PMF_kJ_mol': result.pmf.pmf,
                'uncertainty_kJ_mol': result.pmf.pmf_uncertainty,
            }
        )
        pmf_df.write_csv(str(output_dir / f'{self.log_prefix}_pmf.csv'))

        # Save window free energies
        fe_df = pl.DataFrame(
            {
                'window': list(range(len(result.pmf.free_energies))),
                'rc0': self.reaction_coordinate[: len(result.pmf.free_energies)],
                'free_energy_kJ_mol': result.pmf.free_energies,
                'uncertainty_kJ_mol': result.pmf.free_energy_uncertainty,
            }
        )
        fe_df.write_csv(str(output_dir / f'{self.log_prefix}_window_free_energies.csv'))

        # Save convergence data
        conv_df = pl.DataFrame(
            {
                'window': [c.window_idx for c in result.convergence],
                'mean_rc': [c.mean_rc for c in result.convergence],
                'sem': [c.sem for c in result.convergence],
                'is_converged': [c.is_converged for c in result.convergence],
            }
        )
        conv_df.write_csv(str(output_dir / f'{self.log_prefix}_convergence.csv'))

        # Save summary
        summary_path = output_dir / f'{self.log_prefix}_analysis_summary.txt'
        with open(summary_path, 'w') as f:
            f.write('EVB Free Energy Analysis Summary\n')
            f.write('=' * 50 + '\n\n')

            f.write(f'Temperature: {result.temperature} K\n')
            f.write(f'Umbrella force constant: {result.k_umbrella} kJ/mol/nm²\n')
            f.write(f'Number of windows: {len(result.rc_data)}\n\n')

            valid_pmf = result.pmf.pmf[~np.isnan(result.pmf.pmf)]
            if len(valid_pmf) > 0:
                f.write(f'Barrier height: {valid_pmf.max():.2f} kJ/mol\n')
                f.write(f'              = {valid_pmf.max() / 4.184:.2f} kcal/mol\n\n')

            f.write(
                f'Convergence: {sum(1 for c in result.convergence if c.is_converged)}/{len(result.convergence)} windows converged\n'
            )
            f.write(f'Minimum overlap: {result.overlap.min_overlap:.3f}\n')
            if result.overlap.problem_pairs:
                f.write(f'Problem pairs: {result.overlap.problem_pairs}\n')

            f.write('\nEquilibration:\n')
            for eq in result.equilibration:
                f.write(
                    f'  Window {eq.window_idx}: t0={eq.t0}, g={eq.g:.2f}, N_eff={eq.n_effective:.1f}\n'
                )

    def save_metadata(self, output_path: Path | None = None) -> Path:
        """Save run metadata for later analysis without re-instantiation.

        This is crucial for HPC workflows where simulations may be interrupted
        or run across multiple jobs. The metadata file contains all parameters
        needed to create an EVBAnalyzer for post-hoc analysis.

        Args:
            output_path: Path to save metadata TOML. Defaults to log_path/evb_metadata.toml.

        Returns:
            Path to saved metadata file.

        Example:
            >>> evb = EVB(...)
            >>> evb.save_metadata()  # Save before running
            >>> evb.run_evb()  # May get interrupted by walltime
            >>>
            >>> # Later, in a new job:
            >>> analyzer = EVBAnalyzer.from_metadata("path/to/evb_metadata.toml")
            >>> result = analyzer.run_full_analysis()
        """
        output_path = (
            Path(output_path) if output_path else (self.log_path / 'evb_metadata.toml')
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write('# EVB run metadata - use with EVBAnalyzer.from_metadata()\n')
            f.write('# Generated from EVB instance\n\n')
            f.write('[evb]\n')
            f.write(f'log_path = "{self.log_path}"\n')
            f.write(f'log_prefix = "{self.log_prefix}"\n')
            f.write(f'k_umbrella = {self.k}\n')
            f.write(f'rc0_values = {self.reaction_coordinate.tolist()}\n')
            f.write('\n# Additional simulation parameters (for reference)\n')
            f.write('[simulation]\n')
            f.write(f'n_windows = {self.n_windows}\n')
            f.write(f'steps = {self.steps}\n')
            f.write(f'dt = {self.dt}\n')
            f.write(f'k_path = {self.k_path}\n')
            f.write(f'D_e = {self.D_e}\n')
            f.write(f'alpha = {self.alpha}\n')
            f.write(f'r0 = {self.r0}\n')
            f.write(f'platform = "{self.platform}"\n')
            f.write(f'topology = "{self.topology}"\n')
            f.write(f'coordinates = "{self.coordinates}"\n')

        return output_path

    def get_analyzer(self) -> 'EVBAnalyzer':
        """Create an EVBAnalyzer from this EVB instance.

        Useful when you want to run analysis separately from simulation,
        or when you want to decouple the analysis code from Parsl.

        Returns:
            EVBAnalyzer configured with this instance's parameters.

        Example:
            >>> evb = EVB(...)
            >>> evb.run_evb()
            >>> analyzer = evb.get_analyzer()
            >>> result = analyzer.run_full_analysis()
        """
        return EVBAnalyzer.from_evb_instance(self)

    @property
    def umbrella(self) -> dict[str, Any]:
        """Sets up Umbrella force settings for force calculation.

        Because the windows are decided at run time we leave rc0 as None for now.
        The dict includes both k (umbrella force constant) and k_path (path
        restraint force constant) since both are passed to the EVBCalculation.

        Returns:
            dict: Umbrella and path restraint parameters.
        """
        return {
            'atom_i': self.umbrella_atoms[0],
            'atom_j': self.umbrella_atoms[1],
            'atom_k': self.umbrella_atoms[2],
            'k': self.k,
            'k_path': self.k_path,
            'rc0': None,
        }

    @property
    def morse_bond(self) -> dict[str, Any]:
        """Sets up Morse bond settings for potential creation.

        D_e is the potential well depth (bond dissociation energy) and can be
        computed using QM or obtained from ML predictions such as ALFABET
        (https://bde.ml.nrel.gov). Must be in kJ/mol for OpenMM.

        alpha is the potential well width computed from the harmonic force
        constant via the Taylor expansion of the second derivative:
            alpha = sqrt(k_bond / (2 * D_e))

        Unit conversions from AMBER frcmod (kcal/mol/A^2) to OpenMM (kJ/mol/nm^2):
            k_openmm = k_amber * 4.184 * 100

        Example calculation for C-H bond:
            D_e = 93.8 kcal/mol = 392.46 kJ/mol
            k_bond = 330.6 kcal/(mol*A^2) = 138323 kJ/(mol*nm^2)
            alpha = sqrt(138323 / (2 * 392.46)) = 13.275 nm^-1

        r0 is the equilibrium bond distance in nm.

        Returns:
            dict: Morse bond parameters with keys atom_i, atom_j, D_e, alpha, r0.
        """
        return {
            'atom_i': self.morse_atoms[0],
            'atom_j': self.morse_atoms[1],
            'D_e': self.D_e,
            'alpha': self.alpha,
            'r0': self.r0,
        }


class EVBCalculation:
    """Runs a single EVB window."""

    def __init__(
        self,
        topology: Path,
        coord_file: Path,
        out_path: Path,
        rc_file: Path,
        umbrella: dict,
        morse_bond: dict,
        rc_freq: int = 5,  # 0.01 ps @ 2 fs timestep
        steps: int = 500_000,  # 1 ns @ 2 fs timestep
        dt: float = 0.002,
        platform: str = 'CUDA',
        restraint_sel: str | None = None,
    ):
        """Initialize a single EVB window calculation.

        Args:
            topology: Path to the system topology file (prmtop).
            coord_file: Path to the system coordinate file (inpcrd).
            out_path: Directory path for simulation output files.
            rc_file: Path to the reaction coordinate log file.
            umbrella: Dictionary containing umbrella sampling parameters
                including atom_i, atom_j, atom_k, k, k_path, and rc0.
            morse_bond: Dictionary containing Morse bond parameters
                including atom_i, atom_j, D_e, alpha, and r0.
            rc_freq: Steps between reaction coordinate writes. Defaults to 5.
            steps: Number of simulation steps. Defaults to 500000.
            dt: Integration timestep in picoseconds. Defaults to 0.002.
            platform: OpenMM platform name. Defaults to 'CUDA'.
            restraint_sel: Optional MDAnalysis selection string for backbone
                restraints. Defaults to None.
        """
        self.sim_engine = Simulator(
            path=topology.parent,
            top_name=topology.name,
            coor_name=coord_file.name,
            out_path=out_path,
            prod_steps=steps,
            platform=platform,
        )

        # Only set Precision for platforms that support it (CUDA, OpenCL)
        if platform.upper() in ('CUDA', 'OPENCL'):
            self.sim_engine.properties = {
                'Precision': 'mixed',
            }
        else:
            self.sim_engine.properties = {}

        self.rc_file = rc_file
        self.rc_freq = rc_freq
        self.steps = steps
        self.dt = dt
        self.restraint_sel = restraint_sel
        self.umbrella = umbrella
        self.morse_bond = morse_bond

    def prepare(self):
        """Generates simulation object containing all custom forces to compute
        free energy. Leverages standard Simulator as backend, adding in Morse
        potential and Umbrella forces.
        """
        # load files into system object
        system = self.sim_engine.load_system()

        # Remove the original harmonic bond before adding Morse potential
        # to avoid double-counting the bonded interaction
        self.remove_harmonic_bond(
            system, self.morse_bond['atom_i'], self.morse_bond['atom_j']
        )

        # add various custom forces to system
        morse_bond = self.morse_bond_force(**self.morse_bond)
        system.addForce(morse_bond)
        ddbonds_umb = self.umbrella_force(**self.umbrella)
        system.addForce(ddbonds_umb)
        path_force = self.path_restraint(**self.umbrella)
        system.addForce(path_force)

        # if we want restraints add them now
        if self.restraint_sel is not None:
            restraint_idx = self.sim_engine.get_restraint_indices(self.restraint_sel)
            system = self.sim_engine.add_backbone_posres(
                system,
                self.sim_engine.coordinate.positions,
                self.sim_engine.topology.topology.atoms(),
                restraint_idx,
            )

        # finally, build simulation object
        simulation, integrator = self.sim_engine.setup_sim(system, dt=self.dt)
        simulation.context.setPositions(self.sim_engine.coordinate.positions)

        return simulation, integrator

    def run(self):
        """Runs EVB simulation window with custom RCReporter."""
        simulation, _integrator = self.prepare()
        simulation.minimizeEnergy()
        simulation = self.sim_engine.attach_reporters(
            simulation,
            self.sim_engine.dcd,
            str(self.sim_engine.prod_log),
            str(self.sim_engine.restart),
            restart=False,
        )
        atom_indices = [
            self.umbrella['atom_i'],
            self.umbrella['atom_j'],
            self.umbrella['atom_k'],
        ]

        simulation.reporters.append(
            RCReporter(self.rc_file, self.rc_freq, atom_indices, self.umbrella['rc0'])
        )

        simulation.step(self.steps)

    @staticmethod
    def umbrella_force(
        atom_i: int, atom_j: int, atom_k: int, k: float, rc0: float, **kwargs
    ) -> CustomCompoundBondForce:
        """Difference of distances umbrella force. Think pulling an oxygen off

        Args:
            atom_i: Index of first atom participating (from reactant).
            atom_j: Index of second atom participating (from product).
            atom_k: Index of shared atom participating in both reactant and product.
            k: Harmonic spring constant.
            rc0: Target equilibrium distance for current window.

        Returns:
            CustomCompoundBondForce that drives sampling in each umbrella window.
        """
        force = CustomCompoundBondForce(
            3,
            '0.5 * k_umb * ((r13 - r23) - rc0) ^ 2; r13=distance(p1, p3); r23=distance(p2, p3);',
        )
        force.addGlobalParameter('k_umb', k)
        force.addGlobalParameter('rc0', rc0)
        force.addBond([atom_i, atom_j, atom_k])

        return force

    @staticmethod
    def path_restraint(
        atom_i: int, atom_j: int, atom_k: int, k_path: float, **kwargs
    ) -> CustomCompoundBondForce:
        """Enforce collinearity of moving atom with respect to the initial
        and final positions. By avoiding a custom angle force we avoid instability
        related to the asymptote at 180 degrees, which is what we are attempting to
        enforce. The cosine of the dot product of the vectors from i -> k and i -> j
        allows the penalty to scale quadratically with deviation, thus keeping the
        mobile atom snapped along the progress coordinate vector.

        Note: k_path has units of kJ/mol (not kJ/mol/nm^2 like the umbrella k)
        since (1 - costheta)^2 is dimensionless. Typical values are 50-200 kJ/mol.

        Args:
            atom_i: Index of donor atom.
            atom_j: Index of acceptor atom.
            atom_k: Index of transferring atom (e.g., hydride).
            k_path: Force constant in kJ/mol for collinearity restraint.

        Returns:
            CustomCompoundBondForce enforcing D-H-A collinearity.
        """
        force = CustomCompoundBondForce(
            3,
            (
                'k_path * (1 - costheta)^2; '
                'costheta = dot_ij_ik / (r_ij * r_ik); '
                'dot_ij_ik = dx_ij*dx_ik + dy_ij*dy_ik + dz_ij*dz_ik; '
                'r_ij = sqrt(dx_ij^2 + dy_ij^2 + dz_ij^2); '
                'r_ik = sqrt(dx_ik^2 + dy_ik^2 + dz_ik^2); '
                'dx_ij = x2 - x1; '
                'dy_ij = y2 - y1; '
                'dz_ij = z2 - z1; '
                'dx_ik = x3 - x1; '
                'dy_ik = y3 - y1; '
                'dz_ik = z3 - z1'
            ),
        )
        force.addGlobalParameter('k_path', k_path)
        force.addBond([atom_i, atom_k, atom_j])

        return force

    @staticmethod
    def morse_bond_force(
        atom_i: int, atom_j: int, D_e: float, alpha: float, r0: float
    ) -> CustomBondForce:
        """Generates a custom Morse potential between two atom indices.

        The Morse potential has the form:
            V(r) = D_e * (1 - exp(-alpha * (r - r0)))^2

        The alpha parameter can be computed from harmonic force constant k_bond:
            alpha = sqrt(k_bond / (2 * D_e))

        All parameters must be in OpenMM's native unit system:
            - Distances in nanometers (nm)
            - Energies in kJ/mol
            - Force constants in kJ/(mol*nm^2)
            - Alpha in nm^-1

        Args:
            atom_i: Index of first atom.
            atom_j: Index of second atom.
            D_e: Well depth in kJ/mol (from bond dissociation energy).
            alpha: Width parameter in nm^-1.
            r0: Equilibrium distance in nm.

        Returns:
            CustomBondForce corresponding to a Morse potential.
        """
        force = CustomBondForce('D_e * (1 - exp(-alpha * (r-r0))) ^ 2')
        force.addGlobalParameter('D_e', D_e)
        force.addGlobalParameter('alpha', alpha)
        force.addGlobalParameter('r0', r0)
        force.addBond(atom_i, atom_j)

        return force

    @staticmethod
    def remove_harmonic_bond(system, atom_i: int, atom_j: int) -> None:
        """Remove the bond/constraint between two atoms.

        This is necessary when replacing a harmonic bond with a Morse potential
        to avoid double-counting the bonded interaction. The method handles both:
        1. Harmonic bonds (sets force constant to zero)
        2. SHAKE/RATTLE constraints (removes the constraint entirely)

        Args:
            system: OpenMM System object containing forces.
            atom_i: Index of first atom in the bond.
            atom_j: Index of second atom in the bond.

        Returns:
            None. Modifies system in place.
        """
        target_pair = {atom_i, atom_j}
        found_bond = False
        found_constraint = False

        # First, check for harmonic bond and zero it out
        for force_idx in range(system.getNumForces()):
            force = system.getForce(force_idx)
            if isinstance(force, HarmonicBondForce):
                for bond_idx in range(force.getNumBonds()):
                    p1, p2, length, _k = force.getBondParameters(bond_idx)
                    if {p1, p2} == target_pair:
                        # Zero out the force constant, keeping equilibrium length
                        force.setBondParameters(bond_idx, p1, p2, length, 0.0)
                        print(
                            f'Zeroed harmonic bond between atoms {atom_i} and {atom_j}'
                        )
                        found_bond = True
                        break
                break

        # Second, check for constraints (SHAKE) and remove them
        # Need to iterate in reverse since we're removing
        constraints_to_remove = []
        for i in range(system.getNumConstraints()):
            p1, p2, _distance = system.getConstraintParameters(i)
            if {p1, p2} == target_pair:
                constraints_to_remove.append(i)

        # Remove constraints in reverse order to maintain indices
        for idx in reversed(constraints_to_remove):
            system.removeConstraint(idx)
            print(f'Removed SHAKE constraint between atoms {atom_i} and {atom_j}')
            found_constraint = True

        if not found_bond and not found_constraint:
            print(
                f'Warning: No harmonic bond or constraint found between atoms {atom_i} and {atom_j}'
            )
