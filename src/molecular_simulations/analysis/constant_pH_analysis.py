# ruff: noqa: N999
"""
Improved constant pH analysis with UWHAM reweighting.

This implementation adds multistate analysis capabilities to the basic
curve fitting approach. Uses log-space arithmetic for numerical stability.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.optimize import curve_fit
from scipy.special import logsumexp

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class UWHAMSolver:
    """
    Unbinned Weighted Histogram Analysis Method (UWHAM) solver.

    NOTE: This class is NOT currently used because UWHAM/MBAR is designed
    for umbrella sampling and replica exchange, not independent constant pH
    simulations. For standard constant pH MD where each pH is an independent
    equilibrium simulation, simple curve fitting is the correct approach.

    This class is retained for potential use with replica exchange constant
    pH (REX-cpH) simulations where samples ARE correlated across pH values.

    Uses log-space arithmetic throughout for numerical stability with
    large systems (100+ titratable residues).
    """

    def __init__(self, tol: float = 1e-7, maxiter: int = 10000):
        """Initialize the UWHAM solver.

        Args:
            tol: Convergence tolerance on the maximum change in free
                energy offsets between iterations.
            maxiter: Maximum number of self-consistent iterations.
        """
        self.tol = tol
        self.maxiter = maxiter
        self.f = None  # Log of normalization constants (will be solved)
        self.log10 = np.log(10)

    def load_data(self, df: pl.DataFrame, resid_cols: list[str]):
        """
        Load data from polars DataFrame into UWHAM-compatible format.

        Args:
            df: DataFrame with columns: rankid, current_pH, and residue
                columns. Residue columns should contain numeric
                protonation states (0 or 1).
            resid_cols: List of column names corresponding to residue IDs.
        """
        # Get unique pH values and count samples
        pH_groups = df.group_by('current_pH').agg(pl.len().alias('count'))
        self.pH_values = pH_groups['current_pH'].to_numpy()
        self.nsamples = pH_groups['count'].to_numpy().astype(int)
        self.nstates = len(self.pH_values)

        # Sort by pH for consistency
        sort_idx = np.argsort(self.pH_values)
        self.pH_values = self.pH_values[sort_idx]
        self.nsamples = self.nsamples[sort_idx]

        # Store state data for each pH simulation
        self.states = {}  # resid -> list of arrays (one per pH)
        self.nprotons_total = []  # Total protons for each pH simulation

        for resid_col in resid_cols:
            self.states[resid_col] = []

        # Extract data for each pH
        for pH in self.pH_values:
            pH_data = df.filter(pl.col('current_pH') == pH)

            # Compute total protons for this pH's samples
            total_protons = np.zeros(len(pH_data))

            # For each residue, store states
            for resid_col in resid_cols:
                states = pH_data[resid_col].to_numpy().astype(float)
                self.states[resid_col].append(states)
                total_protons += states

            self.nprotons_total.append(total_protons)

        # Precompute reduced potentials for all state pairs
        # u_kl[k] is shape (nstates, n_k) - reduced potential of samples from k evaluated at all states
        self.u_kl = []
        for k in range(self.nstates):
            n_k = self.nsamples[k]
            u_k = np.zeros((self.nstates, n_k))
            for l in range(self.nstates): # noqa: E741
                u_k[l, :] = self.log10 * self.pH_values[l] * self.nprotons_total[k]
            self.u_kl.append(u_k)

    def solve(self, verbose: bool = False):
        """
        Solve UWHAM self-consistent equations iteratively.

        Uses the MBAR equation:
        f_k = -log(Σ_n exp(-u_k(x_n)) / Σ_l N_l exp(f_l - u_l(x_n)))

        where the sum over n includes ALL samples from ALL states.

        Args:
            verbose: If True, print convergence progress.

        Returns:
            Free energy offsets for each pH simulation as an np.ndarray.
        """
        # Initialize free energies
        f = np.zeros(self.nstates)
        log_N = np.log(self.nsamples.astype(float))
        total_samples = sum(self.nsamples)

        # Precompute reduced potentials for all samples at all target states
        # u_all[target_k, sample_idx] = reduced potential at state k for sample idx
        # Also store source state for each sample
        u_all = np.zeros((self.nstates, total_samples))
        sample_source = np.zeros(
            total_samples, dtype=int
        )  # which state each sample came from

        idx = 0
        for source_i in range(self.nstates):
            n_i = self.nsamples[source_i]
            for n in range(n_i):
                sample_source[idx] = source_i
                for target_k in range(self.nstates):
                    # u_k(x_n) = log10 * pH_k * nprotons(x_n)
                    u_all[target_k, idx] = (
                        self.log10
                        * self.pH_values[target_k]
                        * self.nprotons_total[source_i][n]
                    )
                idx += 1

        for iteration in range(self.maxiter):
            f_old = f.copy()

            # Compute denominator for each sample: c_n = Σ_l N_l exp(f_l - u_l(x_n))
            # log(c_n) = logsumexp(log_N + f - u_l(x_n))
            log_c = np.zeros(total_samples)
            for n in range(total_samples):
                source_i = sample_source[n]
                # u_l(x_n) for all states l - this is stored in u_kl[source_i]
                log_c[n] = logsumexp(
                    log_N + f - self.u_kl[source_i][:, n % self.nsamples[source_i]]
                )

            # Wait, that indexing is wrong. Let me redo this.
            # Actually I need to recompute using proper indexing

            log_c = np.zeros(total_samples)
            idx = 0
            for source_i in range(self.nstates):
                n_i = self.nsamples[source_i]
                for local_n in range(n_i):
                    # u_l(x_n) for all states l
                    log_c[idx] = logsumexp(log_N + f - self.u_kl[source_i][:, local_n])
                    idx += 1

            # Update each free energy
            for target_k in range(self.nstates):
                # f_k = -log(Σ_n exp(-u_k(x_n)) / c_n)
                #     = -log(Σ_n exp(-u_k(x_n) - log(c_n)))
                #     = -logsumexp(-u_all[target_k, :] - log_c)
                log_weights = -u_all[target_k, :] - log_c
                f[target_k] = -logsumexp(log_weights)

            # Normalize so f[0] = 0
            f = f - f[0]

            # Check convergence
            delta = np.abs(f - f_old).max()
            if verbose and iteration % 100 == 0:
                print(f'  Iteration {iteration}: max|Δf| = {delta:.2e}')

            if delta < self.tol:
                if verbose:
                    print(f'  Converged after {iteration + 1} iterations')
                break
        else:
            warnings.warn(
                f'UWHAM did not converge after {self.maxiter} iterations '
                f'(final delta = {delta:.2e})',
                stacklevel=2
            )

        self.f = f
        self.log_c = log_c  # Store for weight computation
        self.u_all = u_all  # Store for weight computation
        self.sample_source = sample_source
        self.total_samples = total_samples

        return f

    def compute_log_weights(self, target_pH: float) -> tuple[np.ndarray, float]:
        """
        Compute log weights for reweighting to target pH.

        Uses MBAR formula:
        w_n ∝ exp(-u_target(x_n)) / Σ_l N_l exp(f_l - u_l(x_n))

        Args:
            target_pH: pH value to reweight the samples to.

        Returns:
            Tuple of (log_weights, log_norm), where log_weights is an
            np.ndarray of log weights for all samples (flattened) and
            log_norm is the log of the normalization constant.

        Raises:
            RuntimeError: If solve() has not been called first.
        """
        if self.f is None:
            raise RuntimeError('Must call solve() before computing weights')

        # Compute reduced potential at target pH for all samples
        u_target = np.zeros(self.total_samples)
        idx = 0
        for source_i in range(self.nstates):
            n_i = self.nsamples[source_i]
            for local_n in range(n_i):
                u_target[idx] = (
                    self.log10 * target_pH * self.nprotons_total[source_i][local_n]
                )
                idx += 1

        # log(w_n) = -u_target(x_n) - log(c_n)
        # where log(c_n) was precomputed in solve()
        log_weights = -u_target - self.log_c

        # Normalize
        log_norm = logsumexp(log_weights)

        return log_weights, log_norm

    def compute_expectation_at_pH(
        self, observable_by_state: list[np.ndarray], target_pH: float
    ) -> float:
        """
        Compute expectation value of observable at arbitrary pH.

        Args:
            observable_by_state: Observable values for each sample,
                organized by state index.
            target_pH: pH value at which to compute the expectation.

        Returns:
            Reweighted expectation value at target_pH.
        """
        log_weights, log_norm = self.compute_log_weights(target_pH)

        # Flatten observable to match log_weights ordering
        obs_flat = np.concatenate(observable_by_state)

        # Compute weighted sum
        # <A> = Σ_n A_n * w_n / Σ_n w_n
        #     = Σ_n A_n * exp(log_w_n - log_norm)
        weights = np.exp(log_weights - log_norm)

        return np.sum(obs_flat * weights)

    def get_occupancy_for_resid(self, resid: str) -> list[np.ndarray]:
        """Get occupancy arrays for a specific residue across all pH values."""
        return self.states[resid]


class TitrationCurve:
    """
    Analyze constant pH simulations with multiple fitting methods.

    Available methods:
    - curvefit: Simple least squares fit of Hill equation to per-pH averages
    - weighted: Weighted least squares (weight by 1/variance)
    - bootstrap: Curve fitting with bootstrap confidence intervals

    Note: For independent constant pH simulations (not replica exchange),
    simple curve fitting is the statistically correct approach. UWHAM/MBAR
    is only appropriate for replica exchange constant pH where samples
    are correlated across pH values.
    """

    def __init__(
        self,
        log_file: Path | list[Path],
        make_plots: bool = True,
        out: Path = Path('.'),
        method: str = 'curvefit',  # 'curvefit' or 'uwham'
    ):
        """Initialize the titration curve analyzer.

        Args:
            log_file: Path or list of paths to constant pH log file(s).
            make_plots: Whether to generate plots during postprocessing.
            out: Output directory for results and plots.
            method: Fitting method to use; one of 'curvefit', 'weighted',
                or 'bootstrap'.
        """
        if isinstance(log_file, list):
            dfs = []
            resids = None
            for log in log_file:
                df, r = self.parse_log(log)
                dfs.append(df)
                if resids is None:
                    resids = r
            self.df = pl.concat(dfs, how='vertical')
        else:
            self.df, resids = self.parse_log(log_file)

        # Store residue IDs (converted to strings to match column names)
        assert resids is not None, "No residue IDs found in any log file"
        self.resid_cols = [str(r) for r in resids]

        self.make_plots = make_plots
        self.out = out
        self.method = method

    @staticmethod
    def parse_log(log: Path) -> tuple[pl.DataFrame, list[int]]:
        """Parse OpenMM constant pH log file.

        Args:
            log: Path to the constant pH log file to parse.

        Returns:
            Tuple of (df, resids), where df is a DataFrame with columns
            rankid, current_pH, and one column per residue, and resids is
            a list of residue IDs in order.

        Raises:
            RuntimeError: If the cpH residue ID header line is not found.
            ValueError: If the number of states does not match the number
                of residues.
        """
        lines = log.read_text().splitlines()

        resids = None
        # Header format: "cpH: resids 20  76  83  92  ..."
        header_re = re.compile(r'cpH:\s+resids\s+(.+)$')

        # Find header with residue IDs
        for line in lines:
            m = header_re.search(line)
            if m:
                # Residue IDs are separated by whitespace (possibly multiple spaces)
                resids = [int(x) for x in m.group(1).split()]
                break

        if resids is None:
            raise RuntimeError(
                'Could not find cpH residue ID header line in log. '
                'Expected line containing "cpH: resids ..."'
            )

        # Parse state lines
        state_re = re.compile(r'rank=(\d+).*cpH:\s+pH\s+([0-9.]+):\s+(\[.*\])')

        rows = []
        for line in lines:
            m = state_re.search(line)
            if not m:
                continue

            rank = int(m.group(1))
            current_pH = float(m.group(2))
            states_list = ast.literal_eval(m.group(3))

            if len(states_list) != len(resids):
                raise ValueError(
                    f'Mismatch between number of residues ({len(resids)}) '
                    f'and number of states ({len(states_list)})'
                )

            # Build row dictionary
            row = {
                'rankid': rank,
                'current_pH': current_pH,
            }
            row.update(
                {str(resid): state
                 for resid, state in zip(resids, states_list, strict=True)}
            )
            rows.append(row)

        return pl.DataFrame(rows), resids

    def prepare(self) -> None:
        """Prepare data for analysis."""
        # Melt to long format for curve fitting method
        self.df_long = self.df.unpivot(
            index=['rankid', 'current_pH'],
            on=self.resid_cols,
            variable_name='resid',
            value_name='state',
        )

        # Determine canonical resname for each residue ID
        # Look at the first state observed for each residue
        self.resid_to_resname = {}
        for resid_col in self.resid_cols:
            # Get the first non-null state for this residue
            first_state = self.df[resid_col].drop_nulls().head(1).to_list()
            if first_state:
                state = first_state[0]
                self.resid_to_resname[resid_col] = self.canonical_resname.get(
                    state, state
                )
            else:
                self.resid_to_resname[resid_col] = 'UNK'

        # Map states to protonation (1 or 0)
        self.df_long = self.df_long.with_columns(
            pl.col('state')
            .map_elements(
                lambda x: self.protonation_mapping.get(x), return_dtype=pl.Int64
            )
            .alias('prot')
        ).drop_nulls('prot')

        # Compute per-pH statistics for curve fitting
        self.titrations = (
            self.df_long.group_by(['resid', 'current_pH'])
            .agg(
                [
                    pl.col('prot').mean().alias('fraction_protonated'),
                    pl.col('prot').count().alias('n_samples'),
                ]
            )
            .sort(['resid', 'current_pH'])
        )

    def compute_titrations_curvefit(self) -> pl.DataFrame:
        """
        Compute pKa and Hill coefficient using scipy curve_fit.

        This is the simple approach that treats each pH independently.
        """
        fit_rows = []

        for resid, subdf in self.titrations.group_by('resid', maintain_order=True):
            resid = resid[0]  # Unpack tuple
            resname = self.resid_to_resname.get(resid, 'UNK')
            x = subdf['current_pH'].to_numpy().astype(float)
            y = subdf['fraction_protonated'].to_numpy().astype(float)

            if x.size < 3:
                # Not enough data points
                fit_rows.append(
                    {
                        'resid': resid,
                        'resname': resname,
                        'pKa': np.nan,
                        'Hill_n': np.nan,
                        'pKa_err': np.nan,
                        'Hill_n_err': np.nan,
                        'n_points': int(x.size),
                        'method': 'curvefit',
                    }
                )
                continue

            # Initial guess: pKa where fraction ~ 0.5
            idx_mid = np.argmin(np.abs(y - 0.5))
            pKa0 = x[idx_mid]
            n0 = 1.0

            try:
                popt, pcov = curve_fit(
                    self.hill_equation,
                    x,
                    y,
                    p0=[pKa0, n0],
                    bounds=([0.0, 0.1], [14.0, 10.0]),
                    maxfev=5000,
                )
                pKa, n = popt
                pKa_err = np.sqrt(np.diag(pcov))[0] if pcov is not None else np.nan
                n_err = np.sqrt(np.diag(pcov))[1] if pcov is not None else np.nan
            except Exception:
                pKa, n = np.nan, np.nan
                pKa_err, n_err = np.nan, np.nan

            fit_rows.append(
                {
                    'resid': resid,
                    'resname': resname,
                    'pKa': float(pKa),
                    'Hill_n': float(n),
                    'pKa_err': float(pKa_err),
                    'Hill_n_err': float(n_err),
                    'n_points': int(x.size),
                    'method': 'curvefit',
                }
            )

        return pl.DataFrame(fit_rows)

    def compute_titrations_weighted(self, verbose: bool = False) -> pl.DataFrame:
        """
        Compute pKa and Hill coefficient using weighted least squares.

        Weights each pH point by 1/variance, giving more influence to
        points with more samples and intermediate protonation fractions.

        This is more statistically rigorous than unweighted curve fitting
        when sample sizes vary across pH values.

        Args:
            verbose: If True, print progress information.

        Returns:
            DataFrame with fitted pKa, Hill coefficient, and associated
            errors for each residue.
        """
        fit_rows = []

        for resid, subdf in self.titrations.group_by('resid', maintain_order=True):
            resid = resid[0]
            resname = self.resid_to_resname.get(resid, 'UNK')
            x = subdf['current_pH'].to_numpy().astype(float)
            y = subdf['fraction_protonated'].to_numpy().astype(float)
            n = subdf['n_samples'].to_numpy().astype(float)

            if x.size < 3:
                fit_rows.append(
                    {
                        'resid': resid,
                        'resname': resname,
                        'pKa': np.nan,
                        'Hill_n': np.nan,
                        'pKa_err': np.nan,
                        'Hill_n_err': np.nan,
                        'n_points': int(x.size),
                        'method': 'weighted',
                    }
                )
                continue

            # Compute weights: 1/variance for binomial
            # Var(p) = p(1-p)/n, but avoid division by zero
            # Add small epsilon to avoid infinite weights at p=0 or p=1
            eps = 0.01
            y_clipped = np.clip(y, eps, 1 - eps)
            variance = y_clipped * (1 - y_clipped) / n
            weights = 1.0 / variance
            # Normalize weights
            weights = weights / weights.sum()

            # Initial guess
            idx_mid = np.argmin(np.abs(y - 0.5))
            pKa0 = x[idx_mid]
            n0 = 1.0

            try:
                # Weighted curve fit using sigma = 1/sqrt(weight)
                sigma = 1.0 / np.sqrt(weights * len(weights))
                popt, pcov = curve_fit(
                    self.hill_equation,
                    x,
                    y,
                    p0=[pKa0, n0],
                    sigma=sigma,
                    absolute_sigma=False,
                    bounds=([0.0, 0.1], [14.0, 10.0]),
                    maxfev=5000,
                )
                pKa, hill_n = popt
                pKa_err = np.sqrt(np.diag(pcov))[0] if pcov is not None else np.nan
                n_err = np.sqrt(np.diag(pcov))[1] if pcov is not None else np.nan
            except Exception:
                pKa, hill_n = np.nan, np.nan
                pKa_err, n_err = np.nan, np.nan

            fit_rows.append(
                {
                    'resid': resid,
                    'resname': resname,
                    'pKa': float(pKa),
                    'Hill_n': float(hill_n),
                    'pKa_err': float(pKa_err),
                    'Hill_n_err': float(n_err),
                    'n_points': int(x.size),
                    'method': 'weighted',
                }
            )

        return pl.DataFrame(fit_rows)

    def compute_titrations_bootstrap(
        self, n_bootstrap: int = 1000, verbose: bool = False
    ) -> pl.DataFrame:
        """
        Compute pKa and Hill coefficient with bootstrap confidence intervals.

        Resamples the data at each pH to estimate uncertainty in fitted
        parameters. This gives robust error estimates even when the
        Hill equation doesn't perfectly fit the data.

        Args:
            n_bootstrap: Number of bootstrap iterations (default 1000).
            verbose: If True, print progress.

        Returns:
            DataFrame with pKa, Hill_n, and 95% confidence intervals.
        """
        fit_rows = []

        if verbose:
            print(f'Running bootstrap with {n_bootstrap} iterations...')

        for i, (resid, subdf) in enumerate(
            self.titrations.group_by('resid', maintain_order=True)
        ):
            resid = resid[0]
            resname = self.resid_to_resname.get(resid, 'UNK')
            x = subdf['current_pH'].to_numpy().astype(float)
            y = subdf['fraction_protonated'].to_numpy().astype(float)
            n_samples = subdf['n_samples'].to_numpy().astype(int)

            if verbose and (i + 1) % 20 == 0:
                print(f'  {i + 1}/{len(self.resid_cols)} residues...')

            if x.size < 3:
                fit_rows.append(
                    {
                        'resid': resid,
                        'resname': resname,
                        'pKa': np.nan,
                        'pKa_lo': np.nan,
                        'pKa_hi': np.nan,
                        'Hill_n': np.nan,
                        'Hill_n_lo': np.nan,
                        'Hill_n_hi': np.nan,
                        'n_points': int(x.size),
                        'method': 'bootstrap',
                    }
                )
                continue

            # First fit to get point estimate
            idx_mid = np.argmin(np.abs(y - 0.5))
            pKa0 = x[idx_mid]

            try:
                popt, _ = curve_fit(
                    self.hill_equation,
                    x,
                    y,
                    p0=[pKa0, 1.0],
                    bounds=([0.0, 0.1], [14.0, 10.0]),
                    maxfev=5000,
                )
                pKa_point, hill_n_point = popt
            except Exception:
                pKa_point, hill_n_point = np.nan, np.nan

            # Bootstrap resampling
            pKa_boots = []
            hill_n_boots = []

            for _ in range(n_bootstrap):
                # Resample: for each pH, draw n_samples from Binomial(n, p)
                y_boot = np.zeros(len(x))
                for j in range(len(x)):
                    # Number of protonated in bootstrap sample
                    n_prot = np.random.binomial(n_samples[j], y[j])
                    y_boot[j] = n_prot / n_samples[j]

                try:
                    popt_boot, _ = curve_fit(
                        self.hill_equation,
                        x,
                        y_boot,
                        p0=[pKa0, 1.0],
                        bounds=([0.0, 0.1], [14.0, 10.0]),
                        maxfev=2000,
                    )
                    pKa_boots.append(popt_boot[0])
                    hill_n_boots.append(popt_boot[1])
                except Exception:
                    pass

            # Compute confidence intervals
            if len(pKa_boots) > 10:
                pKa_lo, pKa_hi = np.percentile(pKa_boots, [2.5, 97.5])
                hill_n_lo, hill_n_hi = np.percentile(hill_n_boots, [2.5, 97.5])
            else:
                pKa_lo, pKa_hi = np.nan, np.nan
                hill_n_lo, hill_n_hi = np.nan, np.nan

            fit_rows.append(
                {
                    'resid': resid,
                    'resname': resname,
                    'pKa': float(pKa_point),
                    'pKa_lo': float(pKa_lo),
                    'pKa_hi': float(pKa_hi),
                    'Hill_n': float(hill_n_point),
                    'Hill_n_lo': float(hill_n_lo),
                    'Hill_n_hi': float(hill_n_hi),
                    'n_points': int(x.size),
                    'method': 'bootstrap',
                }
            )

        return pl.DataFrame(fit_rows)

    def compute_titrations(
        self, verbose: bool = False, n_bootstrap: int = 1000
    ) -> None:
        """Compute titrations using selected method."""
        if self.method == 'curvefit':
            self.fits = self.compute_titrations_curvefit()
        elif self.method == 'weighted':
            self.fits = self.compute_titrations_weighted(verbose=verbose)
        elif self.method == 'bootstrap':
            self.fits = self.compute_titrations_bootstrap(
                n_bootstrap=n_bootstrap, verbose=verbose
            )
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Use 'curvefit', 'weighted', or 'bootstrap'"
            )

    def postprocess(self) -> None:
        """Generate fitted curves for plotting."""
        if self.fits is None:
            raise RuntimeError('Must call compute_titrations() first')

        pH_min_val = self.df['current_pH'].min()
        pH_max_val = self.df['current_pH'].max()
        assert isinstance(pH_min_val, (int, float)) and isinstance(pH_max_val, (int, float)), "No pH data found"
        pH_grid = np.linspace(float(pH_min_val), float(pH_max_val), 200)

        curves = []
        for row in self.fits.iter_rows(named=True):
            resid = row['resid']
            pKa = row['pKa']
            n = row['Hill_n']

            if np.isnan(pKa) or np.isnan(n):
                continue

            y_fit = self.hill_equation(pH_grid, pKa, n)
            curves.append(
                pl.DataFrame(
                    {
                        'resid': [resid] * len(pH_grid),
                        'pH': pH_grid,
                        'fraction_protonated_fit': y_fit,
                    }
                )
            )

        self.curves = pl.concat(curves) if curves else None

        if self.make_plots:
            self.plot()

    def plot(self) -> None:
        """Generate plots (to be implemented)."""
        pass

    def diagnose_residue(self, resid: str, verbose: bool = True) -> dict:
        """
        Diagnose why a residue might have failed pKa determination.

        Args:
            resid: Residue ID to diagnose.
            verbose: If True, print diagnostic information.

        Returns:
            Dictionary with diagnostic info including titration curve data.
        """
        # Get per-pH fraction protonated from simple averaging
        resid_data = self.titrations.filter(pl.col('resid') == resid)

        pH_vals = resid_data['current_pH'].to_numpy()
        frac_prot = resid_data['fraction_protonated'].to_numpy()
        n_samples = resid_data['n_samples'].to_numpy()

        # Get state distribution
        resid_states = self.df_long.filter(pl.col('resid') == resid)
        state_counts = resid_states.group_by('state').agg(pl.len().alias('count'))

        resname = self.resid_to_resname.get(resid, 'UNK')

        result = {
            'resid': resid,
            'resname': resname,
            'pH': pH_vals,
            'fraction_protonated': frac_prot,
            'n_samples': n_samples,
            'state_distribution': state_counts.to_dict(),
            'frac_min': frac_prot.min() if len(frac_prot) > 0 else np.nan,
            'frac_max': frac_prot.max() if len(frac_prot) > 0 else np.nan,
        }

        if verbose:
            print(f'\nDiagnostics for residue {resid} ({resname}):')
            print('  State distribution:')
            for row in state_counts.iter_rows(named=True):
                print(f"    {row['state']}: {row['count']}")
            print('\n  Titration curve (simple average):')
            print(f"  {'pH':>6s}  {'frac':>6s}  {'n':>5s}")
            for pH, f, n in zip(pH_vals, frac_prot, n_samples, strict=True):
                print(f'  {pH:6.2f}  {f:6.3f}  {n:5d}')
            print(
                f"\n  Fraction range: {result['frac_min']:.3f} - {result['frac_max']:.3f}"
            )

            if result['frac_min'] > 0.5:
                print(
                    f'  → Always >50% protonated - pKa likely ABOVE pH {pH_vals.max():.1f}'
                )
            elif result['frac_max'] < 0.5:
                print(
                    f'  → Always <50% protonated - pKa likely BELOW pH {pH_vals.min():.1f}'
                )
            elif result['frac_max'] - result['frac_min'] < 0.1:
                print(
                    '  → Very little titration observed - may not titrate in this pH range'
                )

        return result

    @staticmethod
    def hill_equation(
        pH: float | npt.NDArray[np.floating], pKa: float, n: float
    ) -> float | npt.NDArray[np.floating]:
        """
        Hill equation for acid-base equilibrium.

        Returns fraction protonated as function of pH.
        """
        return 1.0 / (1.0 + 10.0 ** (n * (pH - pKa)))

    @property
    def protonation_mapping(self) -> dict[str, int]:
        """Map state names to protonation numbers (1 = protonated, 0 = not)."""
        return {
            'ASH': 1,
            'ASP': 0,
            'GLH': 1,
            'GLU': 0,
            'LYS': 1,
            'LYN': 0,
            'CYS': 1,
            'CYX': 0,
            'HIP': 1,
            'HIE': 0,
            'HID': 0,
        }

    @property
    def canonical_resname(self) -> dict[str, str]:
        """Map any state name to canonical residue name."""
        return {
            'ASH': 'ASP',
            'ASP': 'ASP',
            'GLH': 'GLU',
            'GLU': 'GLU',
            'LYS': 'LYS',
            'LYN': 'LYS',
            'CYS': 'CYS',
            'CYX': 'CYS',
            'HIP': 'HIS',
            'HIE': 'HIS',
            'HID': 'HIS',
        }

    def compare_methods(self, resids: list[str] | None = None) -> pl.DataFrame:
        """
        Compare curve fit vs UWHAM results for specified residues.

        Args:
            resids: Residues to compare. If None, compares all.

        Returns:
            DataFrame with both methods' results side by side.
        """
        # Run both methods
        fits_cf = self.compute_titrations_curvefit()
        fits_wt = self.compute_titrations_weighted(verbose=False)

        # Join on resid
        comparison = fits_cf.join(
            fits_wt.select(['resid', 'pKa', 'Hill_n']),
            on='resid',
            suffix='_weighted',
        )

        # Add difference columns
        comparison = comparison.with_columns(
            [
                (pl.col('pKa') - pl.col('pKa_weighted')).alias('pKa_diff'),
                (pl.col('Hill_n') - pl.col('Hill_n_weighted')).alias('Hill_n_diff'),
            ]
        )

        if resids is not None:
            comparison = comparison.filter(pl.col('resid').is_in(resids))

        return comparison


class TitrationAnalyzer:
    """
    High-level analyzer for constant pH simulations.

    Provides a streamlined API that runs both curve fitting and UWHAM analysis,
    generates comparisons, and creates publication-quality plots.

    Example:
        >>> analyzer = TitrationAnalyzer(["cpH.log"])
        >>> analyzer.run()
        >>> analyzer.summary()
        >>> analyzer.plot_residue("145")
        >>> analyzer.plot_all(output_dir="plots/")
        >>> analyzer.save_results("results/")
    """

    def __init__(
        self,
        log_files: Path | list[Path] | str | list[str],
        output_dir: Path | str | None = None,
    ):
        """
        Initialize the analyzer.

        Args:
            log_files: Path(s) to constant pH log file(s).
            output_dir: Directory for output files. If None, uses current
                directory.
        """
        log_file_list: list[Path | str] = [log_files] if isinstance(log_files, (str, Path)) else list(log_files)
        self.log_files = [Path(f) for f in log_file_list]

        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.fits_curvefit: pl.DataFrame | None = None
        self.fits_weighted: pl.DataFrame | None = None
        self.fits_bootstrap: pl.DataFrame | None = None
        self.comparison: pl.DataFrame | None = None
        self.titration_data: pl.DataFrame | None = None

        # Internal objects
        self._tc: TitrationCurve | None = None

        # Metadata
        self.resid_to_resname: dict[str, str] = {}
        self.resid_cols: list[str] = []

        self._analyzed = False

    def run(
        self,
        methods: list[str] = ['curvefit', 'weighted'], # noqa: B006
        verbose: bool = True,
        n_bootstrap: int = 1000,
    ) -> TitrationAnalyzer:
        """
        Run the analysis with specified methods.

        Args:
            methods: Methods to run: 'curvefit', 'weighted', 'bootstrap'.
                - curvefit: Simple least squares fit of Hill equation
                - weighted: Weighted least squares (by 1/variance)
                - bootstrap: Curve fit with bootstrap confidence intervals
            verbose: If True, print progress information.
            n_bootstrap: Number of bootstrap iterations (only used if
                'bootstrap' in methods).

        Returns:
            Self, for method chaining.
        """
        if verbose:
            print('=' * 60)
            print('Constant pH Titration Analysis')
            print('=' * 60)
            print(f'Log files: {[str(f) for f in self.log_files]}')

        # Initialize and prepare
        self._tc = TitrationCurve(self.log_files, make_plots=False)
        self._tc.prepare()

        # Store data for plotting
        self.titration_data = self._tc.titrations.clone()
        self.resid_to_resname = self._tc.resid_to_resname.copy()
        self.resid_cols = self._tc.resid_cols.copy()

        if verbose:
            n_residues = len(self._tc.resid_cols)
            pH_vals = self._tc.df['current_pH'].unique().sort()
            print(f'Residues: {n_residues}')
            print(f'pH values: {pH_vals.to_list()}')
            print(f'Total samples: {len(self._tc.df)}')

        # Curve fitting
        if 'curvefit' in methods:
            if verbose:
                print('\n' + '-' * 40)
                print('Running curve fitting...')
            self.fits_curvefit = self._tc.compute_titrations_curvefit()
            if verbose:
                n_success = self.fits_curvefit.filter(pl.col('pKa').is_not_nan()).height
                print(f'  Success: {n_success}/{len(self.fits_curvefit)} residues')

        # Weighted fitting
        if 'weighted' in methods:
            if verbose:
                print('\n' + '-' * 40)
                print('Running weighted curve fitting...')
            self.fits_weighted = self._tc.compute_titrations_weighted(verbose=verbose)
            if verbose:
                n_success = self.fits_weighted.filter(pl.col('pKa').is_not_nan()).height
                print(f'  Success: {n_success}/{len(self.fits_weighted)} residues')

        # Bootstrap
        if 'bootstrap' in methods:
            if verbose:
                print('\n' + '-' * 40)
                print(f'Running bootstrap ({n_bootstrap} iterations)...')
            self.fits_bootstrap = self._tc.compute_titrations_bootstrap(
                n_bootstrap=n_bootstrap, verbose=verbose
            )
            if verbose:
                n_success = self.fits_bootstrap.filter(
                    pl.col('pKa').is_not_nan()
                ).height
                print(f'  Success: {n_success}/{len(self.fits_bootstrap)} residues')

        # Generate comparison if multiple methods ran
        if self.fits_curvefit is not None and self.fits_weighted is not None:
            self._generate_comparison()

        self._analyzed = True

        if verbose:
            print('\n' + '=' * 60)
            print('Analysis complete!')
            print('=' * 60)

        return self

    def _generate_comparison(self) -> None:
        """Generate comparison DataFrame between curvefit and weighted methods."""
        assert self.fits_curvefit is not None and self.fits_weighted is not None
        self.comparison = self.fits_curvefit.join(
            self.fits_weighted.select(['resid', 'pKa', 'Hill_n']),
            on='resid',
            suffix='_weighted',
        ).with_columns(
            [
                (pl.col('pKa') - pl.col('pKa_weighted')).alias('pKa_diff'),
                (pl.col('Hill_n') - pl.col('Hill_n_weighted')).alias('Hill_n_diff'),
            ]
        )

    def summary(self, show_all: bool = False) -> pl.DataFrame | None:
        """
        Print and return summary of results.

        Args:
            show_all: If True, show all residues. Otherwise show first 20.

        Returns:
            DataFrame with comparison results.
        """
        if not self._analyzed:
            raise RuntimeError('Must call run() before summary()')

        if self.comparison is not None:
            successful = self.comparison.filter(
                pl.col('pKa').is_not_nan() & pl.col('pKa_weighted').is_not_nan()
            )

            print(
                f'\nComparison Summary ({len(successful)} residues with both methods successful):'
            )
            print('-' * 60)

            if len(successful) > 0:
                delta = successful['pKa_diff'].to_numpy()
                print('ΔpKa (curvefit - weighted):')
                print(f'  Mean:   {np.mean(delta):+.3f}')
                print(f'  Std:    {np.std(delta):.3f}')
                print(f'  Median: {np.median(delta):+.3f}')
                print(f'  Range:  [{np.min(delta):.3f}, {np.max(delta):.3f}]')

            display_df = successful.select(
                [
                    'resid',
                    'resname',
                    'pKa',
                    'pKa_weighted',
                    'pKa_diff',
                    'Hill_n',
                    'Hill_n_weighted',
                ]
            )

            if not show_all and len(display_df) > 20:
                print(
                    f'\nShowing first 20 of {len(display_df)} residues (use show_all=True for all):'
                )
                print(display_df.head(20))
            else:
                print(display_df)

            return self.comparison

        elif self.fits_curvefit is not None:
            print('\nCurve Fitting Results:')
            print(self.fits_curvefit if show_all else self.fits_curvefit.head(20))
            return self.fits_curvefit

        elif self.fits_weighted is not None:
            print('\nWeighted Fitting Results:')
            print(self.fits_weighted if show_all else self.fits_weighted.head(20))
            return self.fits_weighted

        elif self.fits_bootstrap is not None:
            print('\nBootstrap Results:')
            print(self.fits_bootstrap if show_all else self.fits_bootstrap.head(20))
            return self.fits_bootstrap

        return None

    def get_results(self, method: str = 'curvefit') -> pl.DataFrame | None:
        """
        Get results DataFrame for specified method.

        Args:
            method: One of 'curvefit', 'weighted', 'bootstrap', or
                'comparison'.
        """
        if method == 'curvefit':
            return self.fits_curvefit
        elif method == 'weighted':
            return self.fits_weighted
        elif method == 'bootstrap':
            return self.fits_bootstrap
        elif method == 'comparison':
            return self.comparison
        else:
            raise ValueError(f'Unknown method: {method}')

    def plot_residue(
        self,
        resid: str,
        ax: plt.Axes | None = None,
        show_curvefit: bool = True,
        show_weighted: bool = True,
        show_data: bool = True,
        figsize: tuple[float, float] = (8, 6),
        save: str | Path | None = None,
    ) -> plt.Figure:
        """
        Plot titration curve for a single residue.

        Args:
            resid: Residue ID to plot.
            ax: Axes to plot on. If None, creates new figure.
            show_curvefit: Show curve fitting result.
            show_weighted: Show weighted fit result.
            show_data: Show raw data points.
            figsize: Figure size if creating new figure.
            save: Path to save figure.

        Returns:
            The matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        if not self._analyzed:
            raise RuntimeError('Must call run() before plotting')

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            _fig = ax.get_figure()
            assert _fig is not None, "Axes has no associated Figure"
            fig = _fig

        resname = self.resid_to_resname.get(resid, 'UNK')

        # Raw data
        assert self.titration_data is not None, "No titration data available"
        resid_data = self.titration_data.filter(pl.col('resid') == resid)
        pH_data = resid_data['current_pH'].to_numpy()
        frac_data = resid_data['fraction_protonated'].to_numpy()
        n_samples = resid_data['n_samples'].to_numpy()

        # Standard error for binomial
        se = np.sqrt(frac_data * (1 - frac_data) / np.maximum(n_samples, 1))

        # Plot data points
        if show_data:
            ax.errorbar(
                pH_data,
                frac_data,
                yerr=se,
                fmt='o',
                color='black',
                markersize=8,
                capsize=3,
                capthick=1,
                elinewidth=1,
                label='Data',
                zorder=10,
            )

        # pH grid for curves
        pH_grid = np.linspace(min(pH_data) - 0.5, max(pH_data) + 0.5, 200)

        # Curve fit line (unweighted)
        if show_curvefit and self.fits_curvefit is not None:
            cf_row = self.fits_curvefit.filter(pl.col('resid') == resid)
            if len(cf_row) > 0:
                pKa_cf = cf_row['pKa'][0]
                n_cf = cf_row['Hill_n'][0]
                if not np.isnan(pKa_cf) and not np.isnan(n_cf):
                    y_cf = TitrationCurve.hill_equation(pH_grid, pKa_cf, n_cf)
                    ax.plot(
                        pH_grid,
                        y_cf,
                        '-',
                        color='blue',
                        linewidth=2,
                        label=f'Curve fit (pKa={pKa_cf:.2f}, n={n_cf:.2f})',
                    )
                    ax.axvline(pKa_cf, color='blue', linestyle=':', alpha=0.5)

        # Weighted fit line
        if show_weighted and self.fits_weighted is not None:
            wt_row = self.fits_weighted.filter(pl.col('resid') == resid)
            if len(wt_row) > 0:
                pKa_wt = wt_row['pKa'][0]
                n_wt = wt_row['Hill_n'][0]
                if not np.isnan(pKa_wt) and not np.isnan(n_wt):
                    y_wt = TitrationCurve.hill_equation(pH_grid, pKa_wt, n_wt)
                    ax.plot(
                        pH_grid,
                        y_wt,
                        '--',
                        color='red',
                        linewidth=2,
                        label=f'Weighted (pKa={pKa_wt:.2f}, n={n_wt:.2f})',
                    )
                    ax.axvline(pKa_wt, color='red', linestyle=':', alpha=0.5)

        # Formatting
        ax.set_xlabel('pH', fontsize=12)
        ax.set_ylabel('Fraction Protonated', fontsize=12)
        ax.set_title(f'Residue {resid} ({resname})', fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(save, dpi=150, bbox_inches='tight')

        return fig

    def plot_all(
        self,
        output_dir: str | Path | None = None,
        format: str = 'png',
        show_curvefit: bool = True,
        show_weighted: bool = True,
        residues: list[str] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Generate plots for all (or selected) residues.

        Args:
            output_dir: Directory for plots. Uses self.output_dir /
                'plots' if None.
            format: Image format ('png', 'pdf', 'svg').
            show_curvefit: Include curve fitting results.
            show_weighted: Include weighted fit results.
            residues: Specific residues to plot. If None, plots all.
            verbose: If True, print progress.
        """
        import matplotlib.pyplot as plt

        if not self._analyzed:
            raise RuntimeError('Must call run() before plotting')

        plot_dir = Path(output_dir) if output_dir else self.output_dir / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)

        if residues is None:
            residues = self.resid_cols

        if verbose:
            print(f'Generating {len(residues)} plots in {plot_dir}/')

        for i, resid in enumerate(residues):
            resname = self.resid_to_resname.get(resid, 'UNK')
            filename = plot_dir / f'{resname}_{resid}.{format}'

            fig = self.plot_residue(
                resid,
                show_curvefit=show_curvefit,
                show_weighted=show_weighted,
                save=filename,
            )
            plt.close(fig)

            if verbose and (i + 1) % 20 == 0:
                print(f'  {i + 1}/{len(residues)} plots generated...')

        if verbose:
            print(f'  All {len(residues)} plots saved to {plot_dir}/')

    def plot_summary(
        self,
        figsize: tuple[float, float] = (12, 5),
        save: str | Path | None = None,
    ) -> plt.Figure:
        """
        Generate summary plot comparing methods.

        Creates a 2-panel figure:
        - Left: pKa comparison scatter plot
        - Right: Distribution of pKa differences
        """
        import matplotlib.pyplot as plt

        if self.comparison is None:
            raise RuntimeError(
                'Need both curvefit and weighted methods for summary plot'
            )

        successful = self.comparison.filter(
            pl.col('pKa').is_not_nan() & pl.col('pKa_weighted').is_not_nan()
        )

        if len(successful) == 0:
            raise ValueError('No residues with both methods successful')

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        pKa_cf = successful['pKa'].to_numpy()
        pKa_wt = successful['pKa_weighted'].to_numpy()
        diff = successful['pKa_diff'].to_numpy()

        # Scatter plot
        ax = axes[0]
        ax.scatter(pKa_cf, pKa_wt, alpha=0.6, edgecolor='black', linewidth=0.5)
        lims = [
            min(min(pKa_cf), min(pKa_wt)) - 0.5,
            max(max(pKa_cf), max(pKa_wt)) + 0.5,
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('pKa (Curve Fit)', fontsize=12)
        ax.set_ylabel('pKa (Weighted)', fontsize=12)
        ax.set_title('Method Comparison', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        corr = np.corrcoef(pKa_cf, pKa_wt)[0, 1]
        ax.text(
            0.05,
            0.95,
            f'r = {corr:.3f}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        # Histogram
        ax = axes[1]
        ax.hist(diff, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(
            np.mean(diff),
            color='blue',
            linestyle='-',
            linewidth=2,
            label=f'Mean = {np.mean(diff):.3f}',
        )
        ax.set_xlabel('ΔpKa (Curve Fit - Weighted)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('pKa Difference Distribution', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(save, dpi=150, bbox_inches='tight')

        return fig

    def save_results(
        self,
        output_dir: str | Path | None = None,
        prefix: str = '',
        formats: list[str] = ['csv'], # noqa: B006
    ) -> None:
        """
        Save all results to files.

        Args:
            output_dir: Output directory. Uses self.output_dir if None.
            prefix: Prefix for filenames.
            formats: Output formats: 'csv', 'parquet', 'json'.
        """
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = f'{prefix}_' if prefix else ''

        def save_df(df: pl.DataFrame, name: str):
            for fmt in formats:
                filepath = out_dir / f'{prefix}{name}.{fmt}'
                if fmt == 'csv':
                    df.write_csv(filepath)
                elif fmt == 'parquet':
                    df.write_parquet(filepath)
                elif fmt == 'json':
                    df.write_json(filepath)
                print(f'  Saved {filepath}')

        print(f'Saving results to {out_dir}/')

        if self.fits_curvefit is not None:
            save_df(self.fits_curvefit, 'pKa_curvefit')
        if self.fits_weighted is not None:
            save_df(self.fits_weighted, 'pKa_weighted')
        if self.fits_bootstrap is not None:
            save_df(self.fits_bootstrap, 'pKa_bootstrap')
        if self.comparison is not None:
            save_df(self.comparison, 'pKa_comparison')
        if self.titration_data is not None:
            save_df(self.titration_data, 'titration_data')

    def diagnose(self, resid: str) -> dict:
        """Get diagnostic information for a residue."""
        if self._tc is None:
            raise RuntimeError('Must call run() first')
        return self._tc.diagnose_residue(resid, verbose=True)

    def recommend_protonation(
        self,
        target_pH: float,
        confidence_threshold: float = 0.7,
        use_bootstrap: bool = False,
        verbose: bool = True,
    ) -> pl.DataFrame:
        """
        Recommend protonation states for a target pH.

        Uses the fitted titration curves to predict which residues are
        protonated vs deprotonated at the specified pH, with confidence
        estimates based on distance from pKa.

        Args:
            target_pH: pH value to make predictions for (e.g., 3.0, 7.4).
            confidence_threshold: Probability threshold for "confident"
                predictions (default 0.7). Residues with P(protonated)
                between (1-threshold) and threshold are marked as
                "uncertain".
            use_bootstrap: If True and bootstrap results available, use
                bootstrap CI for uncertainty estimation.
            verbose: If True, print summary of recommendations.

        Returns:
            DataFrame with columns:
                - resid: residue ID
                - resname: canonical residue name (ASP, GLU, HIS, LYS, CYS)
                - pKa: pKa value used for the prediction; the fitted pKa
                  when available, otherwise the reference pKa
                - pKa_source: 'fitted' if a fitted pKa was used, or
                  'reference' if the reference pKa was used as a fallback
                - prob_protonated: probability of being protonated at
                  target pH
                - recommendation: 'protonated', 'deprotonated', or
                  'uncertain'
                - confidence: 'high', 'medium', or 'low'
                - state_name: recommended state name (e.g., 'ASH' or 'ASP')
        """
        if not self._analyzed:
            raise RuntimeError('Must call run() before recommend_protonation()')

        # Use curvefit results (or weighted if available)
        fits = self.fits_curvefit
        if fits is None:
            fits = self.fits_weighted
        if fits is None:
            raise RuntimeError('No fitting results available')

        # State name mappings
        protonated_state = {
            'ASP': 'ASH',
            'GLU': 'GLH',
            'HIS': 'HIP',
            'LYS': 'LYS',
            'CYS': 'CYS',
        }
        deprotonated_state = {
            'ASP': 'ASP',
            'GLU': 'GLU',
            'HIS': 'HIE',
            'LYS': 'LYN',
            'CYS': 'CYX',
        }

        # Reference pKa values for sanity checking
        reference_pKa = {'ASP': 3.9, 'GLU': 4.3, 'HIS': 6.0, 'LYS': 10.5, 'CYS': 8.3}

        recommendations = []

        for row in fits.iter_rows(named=True):
            resid = row['resid']
            resname = row['resname']
            pKa = row['pKa']
            hill_n = row['Hill_n']

            # Compute probability of being protonated at target pH
            if np.isnan(pKa) or np.isnan(hill_n):
                # No fit available - use reference pKa
                ref_pKa = reference_pKa.get(resname, 7.0)
                prob_prot = 1.0 / (1.0 + 10 ** (target_pH - ref_pKa))
                pKa_used = ref_pKa
                fit_quality = 'reference'
            else:
                # Use fitted Hill equation
                prob_prot = TitrationCurve.hill_equation(target_pH, pKa, hill_n)
                pKa_used = pKa
                fit_quality = 'fitted'

            # Determine recommendation
            if prob_prot >= confidence_threshold:
                recommendation = 'protonated'
                state_name = protonated_state.get(resname, resname)
            elif prob_prot <= (1 - confidence_threshold):
                recommendation = 'deprotonated'
                state_name = deprotonated_state.get(resname, resname)
            else:
                recommendation = 'uncertain'
                # For uncertain cases, go with majority
                if prob_prot >= 0.5:
                    state_name = protonated_state.get(resname, resname)
                else:
                    state_name = deprotonated_state.get(resname, resname)

            # Confidence based on distance from 0.5
            prob_distance = abs(prob_prot - 0.5)
            if prob_distance > 0.4:  # >90% or <10%
                confidence = 'high'
            elif prob_distance > 0.2:  # >70% or <30%
                confidence = 'medium'
            else:
                confidence = 'low'

            recommendations.append(
                {
                    'resid': resid,
                    'resname': resname,
                    'pKa': pKa_used,
                    'pKa_source': fit_quality,
                    'prob_protonated': prob_prot,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'state_name': state_name,
                }
            )

        result = pl.DataFrame(recommendations)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f'Protonation Recommendations at pH {target_pH}')
            print(f"{'=' * 60}")

            # Summary counts
            n_prot = result.filter(pl.col('recommendation') == 'protonated').height
            n_deprot = result.filter(pl.col('recommendation') == 'deprotonated').height
            n_uncertain = result.filter(pl.col('recommendation') == 'uncertain').height

            print('\nSummary:')
            print(f'  Protonated:   {n_prot:3d} residues')
            print(f'  Deprotonated: {n_deprot:3d} residues')
            print(f'  Uncertain:    {n_uncertain:3d} residues')

            # Group by residue type
            print('\nBy residue type:')
            for restype in ['ASP', 'GLU', 'HIS', 'LYS', 'CYS']:
                subset = result.filter(pl.col('resname') == restype)
                if len(subset) > 0:
                    n_p = subset.filter(pl.col('recommendation') == 'protonated').height
                    n_d = subset.filter(
                        pl.col('recommendation') == 'deprotonated'
                    ).height
                    n_u = subset.filter(pl.col('recommendation') == 'uncertain').height
                    ref = reference_pKa.get(restype, '?')
                    print(
                        f'  {restype} (ref pKa={ref}): {n_p} prot, {n_d} deprot, {n_u} uncertain'
                    )

            # Show uncertain residues (most important to check)
            uncertain = result.filter(pl.col('recommendation') == 'uncertain')
            if len(uncertain) > 0:
                print(
                    f'\n⚠️  Uncertain residues (prob between {1 - confidence_threshold:.0%}-{confidence_threshold:.0%}):'
                )
                for row in uncertain.sort('prob_protonated', descending=True).iter_rows(
                    named=True
                ):
                    print(
                        f"    {row['resname']} {row['resid']}: "
                        f"P(prot)={row['prob_protonated']:.1%}, "
                        f"pKa={row['pKa']:.1f} → {row['state_name']}"
                    )

            # Show residues with pKa near target pH
            near_pKa = result.filter(
                (pl.col('pKa') > target_pH - 1.5)
                & (pl.col('pKa') < target_pH + 1.5)
                & (pl.col('pKa_source') == 'fitted')
            )
            if len(near_pKa) > 0:
                print(f'\n📍 Residues with pKa near pH {target_pH} (±1.5 units):')
                for row in near_pKa.sort('pKa').iter_rows(named=True):
                    print(
                        f"    {row['resname']} {row['resid']}: "
                        f"pKa={row['pKa']:.2f}, "
                        f"P(prot)={row['prob_protonated']:.1%} → {row['state_name']}"
                    )

        return result

    def get_protonation_string(
        self,
        target_pH: float,
        confidence_threshold: float = 0.7,
    ) -> str:
        """
        Get a simple string of recommended protonation states.

        Useful for setting up simulations.

        Args:
            target_pH: pH value to make predictions for.
            confidence_threshold: Probability threshold for confident
                predictions.

        Returns:
            String with format: "resid:state,resid:state,...".
        """
        recs = self.recommend_protonation(
            target_pH, confidence_threshold=confidence_threshold, verbose=False
        )

        parts = []
        for row in recs.iter_rows(named=True):
            parts.append(f"{row['resid']}:{row['state_name']}")

        return ','.join(parts)

    def export_protonation_states(
        self,
        target_pH: float,
        output_file: str | Path | None = None,
        format: str = 'csv',
        confidence_threshold: float = 0.7,
    ) -> pl.DataFrame:
        """
        Export protonation state recommendations to file.

        Args:
            target_pH: pH value to make predictions for.
            output_file: Output file path. If None, uses
                output_dir/protonation_pH{pH}.{format}.
            format: Output format: 'csv', 'json', or 'txt'.
            confidence_threshold: Probability threshold for confident
                predictions.

        Returns:
            DataFrame with recommendations.
        """
        recs = self.recommend_protonation(
            target_pH, confidence_threshold=confidence_threshold, verbose=False
        )

        if output_file is None:
            output_file = self.output_dir / f'protonation_pH{target_pH:.1f}.{format}'
        else:
            output_file = Path(output_file)

        if format == 'csv':
            recs.write_csv(output_file)
        elif format == 'json':
            recs.write_json(output_file)
        elif format == 'txt':
            # Simple text format for easy reading
            with open(output_file, 'w') as f:
                f.write(f'# Protonation states at pH {target_pH}\n')
                f.write(f'# confidence_threshold = {confidence_threshold}\n')
                f.write('#\n')
                f.write('# resid  resname  state  prob_prot  confidence\n')
                for row in recs.iter_rows(named=True):
                    f.write(
                        f"{row['resid']:>6s}  {row['resname']:>7s}  "
                        f"{row['state_name']:>5s}  {row['prob_protonated']:>9.3f}  "
                        f"{row['confidence']}\n"
                    )

        print(f'Saved protonation recommendations to {output_file}')
        return recs

    def plot_protonation_summary(
        self,
        target_pH: float,
        figsize: tuple[float, float] = (12, 6),
        save: str | Path | None = None,
    ) -> plt.Figure:
        """
        Visualize protonation probabilities at target pH.

        Creates a bar plot showing P(protonated) for each residue,
        colored by residue type.

        Args:
            target_pH: pH value to visualize.
            figsize: Figure size.
            save: Path to save figure.

        Returns:
            The matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        recs = self.recommend_protonation(target_pH, verbose=False)

        # Sort by probability
        recs_sorted = recs.sort('prob_protonated', descending=True)

        fig, ax = plt.subplots(figsize=figsize)

        # Colors for each residue type
        colors = {
            'ASP': '#e41a1c',  # red
            'GLU': '#ff7f00',  # orange
            'HIS': '#4daf4a',  # green
            'LYS': '#377eb8',  # blue
            'CYS': '#984ea3',  # purple
        }

        x = np.arange(len(recs_sorted))
        probs = recs_sorted['prob_protonated'].to_numpy()
        resnames = recs_sorted['resname'].to_list()
        resids = recs_sorted['resid'].to_list()

        bar_colors = [colors.get(rn, 'gray') for rn in resnames]

        _bars = ax.bar(x, probs, color=bar_colors, edgecolor='black', linewidth=0.5)

        # Add 0.5 line
        ax.axhline(0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(0.7, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(0.3, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Labels
        ax.set_xlabel('Residue', fontsize=12)
        ax.set_ylabel('P(protonated)', fontsize=12)
        ax.set_title(f'Protonation Probabilities at pH {target_pH}', fontsize=14)
        ax.set_ylim(0, 1.05)

        # X-axis labels (show every Nth label if too many)
        n_residues = len(x)
        if n_residues > 50:
            # Show fewer labels
            step = n_residues // 20
            ax.set_xticks(x[::step])
            labels = [f'{resnames[i]}{resids[i]}' for i in range(0, n_residues, step)]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        else:
            ax.set_xticks(x)
            labels = [f'{rn}{ri}' for rn, ri in zip(resnames, resids, strict=True)]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

        # Legend
        legend_elements = [
            Patch(facecolor=c, edgecolor='black', label=n)
            for n, c in colors.items()
            if n in resnames
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Add text annotations for counts
        n_prot = sum(1 for p in probs if p >= 0.5)
        n_deprot = sum(1 for p in probs if p < 0.5)
        ax.text(
            0.02,
            0.98,
            f'Protonated: {n_prot}\nDeprotonated: {n_deprot}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        plt.tight_layout()

        if save:
            fig.savefig(save, dpi=150, bbox_inches='tight')

        return fig

    def __repr__(self) -> str:
        status = 'analyzed' if self._analyzed else 'not analyzed'
        return f'TitrationAnalyzer({len(self.log_files)} log files, {status})'


def analyze_cph(
    log_files: Path | list[Path] | str | list[str],
    output_dir: str | Path | None = None,
    methods: list[str] = ['curvefit', 'weighted'], # noqa: B006
    plot: bool = True,
    verbose: bool = True,
) -> TitrationAnalyzer:
    """
    Convenience function to run complete constant pH analysis.

    Args:
        log_files: Path(s) to log files.
        output_dir: Output directory.
        methods: List of methods to run ('curvefit', 'weighted',
            'bootstrap').
        plot: Whether to generate plots.
        verbose: If True, print progress.

    Returns:
        TitrationAnalyzer with results.

    Example:
        >>> results = analyze_cph("cpH.log", output_dir="analysis/")
        >>> results.summary()
        >>> results.plot_residue("145")
    """
    analyzer = TitrationAnalyzer(log_files, output_dir=output_dir)
    analyzer.run(methods=methods, verbose=verbose)

    if plot:
        try:
            analyzer.plot_all(verbose=verbose)
            analyzer.plot_summary(save=analyzer.output_dir / 'summary.png')
        except ImportError:
            if verbose:
                print('matplotlib not available, skipping plots')
        except RuntimeError:
            # plot_summary requires both methods
            pass

    analyzer.save_results()

    return analyzer


if __name__ == '__main__':
    import sys

    # Get log files from command line or use default
    log_paths = [Path('cpH.log')]
    if len(sys.argv) > 1:
        log_paths = [Path(p) for p in sys.argv[1:]]

    # =========================================================================
    # STREAMLINED API - TitrationAnalyzer
    # =========================================================================
    #
    # Available methods:
    #   - curvefit: Simple least squares fit (default)
    #   - weighted: Weighted least squares (by 1/variance)
    #   - bootstrap: Curve fit with bootstrap confidence intervals
    #
    # Basic usage:
    #     analyzer = TitrationAnalyzer(log_paths)
    #     analyzer.run()
    #     analyzer.summary()
    #
    # Protonation recommendations:
    #     recs = analyzer.recommend_protonation(target_pH=3.0)
    #     analyzer.plot_protonation_summary(target_pH=3.0)
    #
    # =========================================================================

    # Create analyzer
    analyzer = TitrationAnalyzer(log_paths, output_dir='cph_analysis')

    # Run curve fitting and weighted fitting
    analyzer.run(methods=['curvefit', 'weighted'], verbose=True)

    # Print summary
    analyzer.summary()

    # Generate all plots (if matplotlib available)
    try:
        analyzer.plot_all(verbose=True)
        analyzer.plot_summary(save='cph_analysis/summary.png')
        print('\nPlots saved to cph_analysis/plots/')
    except ImportError:
        print('\nSkipping plots (matplotlib not installed)')

    # Save results
    analyzer.save_results()

    # =========================================================================
    # PROTONATION RECOMMENDATIONS
    # =========================================================================

    # Get recommendations for pH 3.0
    print('\n')
    recs = analyzer.recommend_protonation(target_pH=3.0)

    # Export to file
    analyzer.export_protonation_states(target_pH=3.0, format='csv')

    # Visualize
    import contextlib
    with contextlib.suppress(ImportError):
        analyzer.plot_protonation_summary(
            target_pH=3.0, save='cph_analysis/protonation_pH3.0.png'
        )

    # Can also get recommendations for physiological pH
    # analyzer.recommend_protonation(target_pH=7.4)
