"""Unit tests for the pure logic in scripts/benchmark_evb.py.

Mirrors tests/test_benchmark_hewl.py: the script is loaded by path (it is not an
installed module) and only its dependency-light, deterministic pieces are
exercised -- the analytic surfaces, the umbrella sampler, the PMF observable
extractors, and the reference loader. One fast end-to-end check drives the real
EVBAnalyzer over synthetic double-well data and confirms the analytic barrier is
recovered, which is the whole point of the benchmark.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_evb.py'


def _load():
    spec = importlib.util.spec_from_file_location('benchmark_evb', _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


# ---------------------------------------------------------------------------
# Reference table
# ---------------------------------------------------------------------------


class TestLoadReference:
    def test_default_reference_parses(self):
        table = mod.load_reference(mod.DEFAULT_REFERENCE)
        # Keyed by (case, observable) with float ref/tol.
        assert ('double_well', 'barrier') in table
        row = table[('double_well', 'barrier')]
        assert row['ref'] == pytest.approx(mod.A_BARRIER)
        assert row['tol'] > 0
        assert isinstance(row['note'], str) and row['note']

    def test_reported_only_rows_are_nan(self):
        table = mod.load_reference(mod.DEFAULT_REFERENCE)
        row = table[('proton_transfer', 'barrier')]
        assert np.isnan(row['ref'])
        assert np.isnan(row['tol'])

    def test_barrier_ref_matches_module_constant(self):
        # The CSV is the source of truth the selftest asserts against; it must
        # stay in sync with the surface the generator draws from.
        table = mod.load_reference(mod.DEFAULT_REFERENCE)
        assert table[('double_well', 'barrier')]['ref'] == mod.A_BARRIER


# ---------------------------------------------------------------------------
# Analytic surfaces
# ---------------------------------------------------------------------------


class TestAnalyticU0:
    def test_flat_is_zero(self):
        rc = np.linspace(-0.1, 0.1, 11)
        np.testing.assert_array_equal(mod.analytic_u0('flat', rc), np.zeros_like(rc))

    def test_harmonic_is_parabola(self):
        rc = np.array([-0.05, 0.0, 0.05])
        expected = 0.5 * mod.K0_HARMONIC * rc**2
        np.testing.assert_allclose(mod.analytic_u0('harmonic', rc), expected)

    def test_double_well_minima_and_barrier(self):
        # Zero at the minima (+/- b), barrier height A at the top (rc = 0).
        assert mod.analytic_u0('double_well', mod.B_WELL) == pytest.approx(0.0)
        assert mod.analytic_u0('double_well', -mod.B_WELL) == pytest.approx(0.0)
        assert mod.analytic_u0('double_well', 0.0) == pytest.approx(mod.A_BARRIER)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match='unknown surface'):
            mod.analytic_u0('bogus', 0.0)


class TestDefaultWindows:
    def test_spans_and_count(self):
        w = mod.default_windows(0.06, 25)
        assert len(w) == 25
        assert w[0] == pytest.approx(-0.06)
        assert w[-1] == pytest.approx(0.06)
        # Symmetric about zero.
        np.testing.assert_allclose(w, -w[::-1], atol=1e-12)


# ---------------------------------------------------------------------------
# Umbrella sampler
# ---------------------------------------------------------------------------


class TestSampleWindow:
    def test_flat_surface_centered_with_expected_width(self):
        # On a flat surface a window samples a Gaussian centered at rc0 with
        # sigma = sqrt(kT/k).
        T, k = 300.0, 100000.0
        sigma = np.sqrt(mod.KB * T / k)
        rng = np.random.default_rng(0)
        s = mod.sample_window(
            lambda x: 0.0,
            0.03,
            k,
            T,
            n_frames=20000,
            burn=2000,
            step=1.5 * sigma,
            rng=rng,
        )
        assert s.mean() == pytest.approx(0.03, abs=3 * sigma / np.sqrt(1000))
        assert s.std() == pytest.approx(sigma, rel=0.15)

    def test_deterministic_with_seed(self):
        def make():
            rng = np.random.default_rng(7)
            return mod.sample_window(
                lambda x: 0.0, 0.0, 100000.0, 300.0, 500, 100, 0.005, rng
            )

        np.testing.assert_array_equal(make(), make())

    def test_returns_requested_length(self):
        rng = np.random.default_rng(1)
        s = mod.sample_window(lambda x: 0.0, 0.0, 100000.0, 300.0, 333, 50, 0.005, rng)
        assert len(s) == 333


# ---------------------------------------------------------------------------
# PMF observable extractors
# ---------------------------------------------------------------------------


class TestPmfRange:
    def test_ignores_nan(self):
        pmf = np.array([0.0, np.nan, 3.0, 1.0])
        assert mod.pmf_range(pmf) == pytest.approx(3.0)


class TestExtractBarrierDg:
    def test_symmetric_double_well(self):
        bc = np.linspace(-0.06, 0.06, 61)
        pmf = mod.analytic_u0('double_well', bc)  # exact surface as a "PMF"
        obs = mod.extract_barrier_dg(bc, pmf)
        assert obs['barrier'] == pytest.approx(mod.A_BARRIER, abs=0.5)
        assert obs['dG_rxn'] == pytest.approx(0.0, abs=1e-6)
        assert obs['rc_reactant'] < 0 < obs['rc_product']
        assert obs['rc_ts'] == pytest.approx(0.0, abs=0.002)

    def test_asymmetric_double_well_nonzero_dg(self):
        bc = np.linspace(-0.06, 0.06, 61)
        # Tilt the well so the product basin sits 5 kJ/mol below the reactant.
        pmf = mod.analytic_u0('double_well', bc) - 5.0 * (bc > 0)
        obs = mod.extract_barrier_dg(bc, pmf)
        assert obs['dG_rxn'] == pytest.approx(-5.0, abs=0.5)

    def test_single_sided_returns_nan_barrier(self):
        bc = np.linspace(0.01, 0.06, 20)  # entirely rc > 0
        pmf = 0.5 * 8000.0 * bc**2
        obs = mod.extract_barrier_dg(bc, pmf)
        assert np.isnan(obs['barrier'])


class TestPmfAsymmetry:
    def test_symmetric_is_zero(self):
        bc = np.linspace(-0.06, 0.06, 61)
        pmf = mod.analytic_u0('double_well', bc)
        assert mod.pmf_asymmetry(bc, pmf) == pytest.approx(0.0, abs=1e-6)

    def test_asymmetric_is_positive(self):
        bc = np.linspace(-0.06, 0.06, 61)
        pmf = mod.analytic_u0('harmonic', bc) + 50.0 * bc  # linear tilt
        assert mod.pmf_asymmetry(bc, pmf) > 1.0


class TestPmfRmseVsAnalytic:
    def test_zero_when_pmf_equals_surface(self):
        bc = np.linspace(-0.06, 0.06, 61)
        pmf = mod.analytic_u0('harmonic', bc)
        assert mod.pmf_rmse_vs_analytic(bc, pmf, 'harmonic') == pytest.approx(
            0.0, abs=1e-6
        )

    def test_shift_invariant(self):
        bc = np.linspace(-0.06, 0.06, 61)
        pmf = mod.analytic_u0('harmonic', bc) + 123.0  # additive constant
        assert mod.pmf_rmse_vs_analytic(bc, pmf, 'harmonic') == pytest.approx(
            0.0, abs=1e-6
        )

    def test_grows_with_mismatch(self):
        bc = np.linspace(-0.06, 0.06, 61)
        pmf = mod.analytic_u0('flat', bc)  # compare flat against harmonic
        assert mod.pmf_rmse_vs_analytic(bc, pmf, 'harmonic') > 1.0


# ---------------------------------------------------------------------------
# Pass/fail comparison helper
# ---------------------------------------------------------------------------


class TestCheck:
    def test_pass(self):
        passed, err = mod._check(20.1, {'ref': 20.0, 'tol': 3.0})
        assert passed is True
        assert err == pytest.approx(0.1)

    def test_fail(self):
        passed, _ = mod._check(30.0, {'ref': 20.0, 'tol': 3.0})
        assert passed is False

    def test_reported_only(self):
        passed, err = mod._check(5.0, {'ref': float('nan'), 'tol': float('nan')})
        assert passed is None
        assert np.isnan(err)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


class TestPlots:
    def _rows(self):
        return [
            {
                'case': 'c',
                'observable': 'barrier',
                'value': 20.2,
                'ref': {'ref': 20.0, 'tol': 3.0},
            },
            {
                'case': 'c',
                'observable': 'dG_rxn',
                'value': 0.3,
                'ref': {'ref': 0.0, 'tol': 1.5},
            },
            {
                'case': 'c',
                'observable': 'reported',
                'value': 5.0,
                'ref': {'ref': float('nan'), 'tol': float('nan')},
            },
        ]

    def test_parity_plot_writes_file(self, tmp_path):
        out = tmp_path / 'parity.png'
        mod._parity_plot(self._rows(), out, 'title')
        assert out.exists() and out.stat().st_size > 0

    def test_parity_plot_no_finite_refs_is_noop(self, tmp_path):
        rows = [
            {
                'case': 'c',
                'observable': 'x',
                'value': 5.0,
                'ref': {'ref': float('nan'), 'tol': float('nan')},
            }
        ]
        out = tmp_path / 'empty.png'
        mod._parity_plot(rows, out, 'title')
        assert not out.exists()  # nothing to plot -> no file

    def test_plot_pmf_writes_file(self, tmp_path):
        bc = np.linspace(-0.06, 0.06, 61)
        pmf = mod.analytic_u0('double_well', bc)
        out = tmp_path / 'pmf.png'
        mod._plot_pmf(bc, pmf, out)
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# End-to-end: recover the analytic barrier through the real EVBAnalyzer
# ---------------------------------------------------------------------------


class TestEndToEndRecovery:
    def test_double_well_barrier_recovered(self, tmp_path):
        """Synthetic double-well data -> EVBAnalyzer -> recovered barrier ~ A.

        This is the benchmark in miniature: it validates that the sampler +
        WHAM/MBAR reweighting recover a known free-energy barrier.
        """
        from molecular_simulations.simulate.free_energy import EVBAnalyzer

        T, k = 300.0, 100000.0
        rc0_values = mod.default_windows(0.06, 15)
        mod.generate_umbrella_logs(
            'double_well',
            rc0_values,
            k,
            T,
            n_frames=1500,
            out_dir=tmp_path,
            prefix='e2e',
            seed=99,
            burn=500,
        )
        analyzer = EVBAnalyzer(
            log_path=tmp_path, log_prefix='e2e', k_umbrella=k, rc0_values=rc0_values
        )
        result = analyzer.run_full_analysis(temperature=T, n_bins=50)
        obs = mod.extract_barrier_dg(result.pmf.bin_centers, result.pmf.pmf)

        # Generous tolerances for a short, few-window run.
        assert obs['barrier'] == pytest.approx(mod.A_BARRIER, abs=5.0)
        assert obs['dG_rxn'] == pytest.approx(0.0, abs=3.0)
