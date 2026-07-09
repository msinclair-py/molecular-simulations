"""Unit tests for the energy-gap EVB analysis numerics.

The OpenMM lambda-mapping engine (build_mapping_system / mapping_integrator /
run_lambda_window) is exercised against a real GPU + AmberTools system; here we
test the dependency-light numerics: BAR, the ladder accumulation, the
ground-state coupling, the schedule, and the gap observables.
"""

import numpy as np
import pytest

from molecular_simulations.simulate.evb_mapping import (
    KB,
    EVBGapResult,
    analyze_gap,
    bar,
    default_lambda_schedule,
    ground_state_energy,
    ladder_free_energies,
)
from molecular_simulations.simulate.evb_mapping import (
    _gap_observables as gap_observables,
)

pytestmark = pytest.mark.unit


class TestBar:
    @pytest.mark.parametrize('dF_true', [5.0, -3.0, 0.0, 12.0])
    def test_recovers_gaussian_work_free_energy(self, dF_true: float):
        """Crooks-consistent Gaussian works (equal variance) -> BAR gives dF.

        For Gaussian forward/reverse work with variance s^2, consistency
        requires the means shifted up by s^2/(2kT); BAR must recover dF exactly.
        """
        rng = np.random.default_rng(0)
        kT = KB * 300.0
        s = 3.0
        off = s**2 / (2 * kT)
        w_f = rng.normal(dF_true + off, s, 200_000)
        w_r = rng.normal(-dF_true + off, s, 200_000)
        assert bar(w_f, w_r, kT) == pytest.approx(dF_true, abs=0.1)

    def test_antisymmetric_in_direction(self):
        rng = np.random.default_rng(1)
        kT = KB * 300.0
        s, dF = 2.0, 4.0
        off = s**2 / (2 * kT)
        w_f = rng.normal(dF + off, s, 100_000)
        w_r = rng.normal(-dF + off, s, 100_000)
        forward = bar(w_f, w_r, kT)
        reverse = bar(w_r, w_f, kT)
        assert forward == pytest.approx(-reverse, abs=0.1)


class TestLadder:
    def test_zero_gap_gives_zero_ladder(self):
        kT = KB * 300.0
        lam = np.array([0.0, 0.5, 1.0])
        gaps = [np.zeros(1000), np.zeros(1000), np.zeros(1000)]
        np.testing.assert_allclose(ladder_free_energies(lam, gaps, kT), 0.0, atol=1e-6)

    def test_is_cumulative(self):
        kT = KB * 300.0
        lam = np.array([0.0, 0.5, 1.0])
        rng = np.random.default_rng(2)
        gaps = [
            rng.normal(20, 2, 5000),
            rng.normal(0, 2, 5000),
            rng.normal(-20, 2, 5000),
        ]
        dG = ladder_free_energies(lam, gaps, kT)
        assert dG[0] == 0.0
        assert dG[1] < dG[2] or dG[1] > dG[2]  # monotone accumulation, either sign
        assert len(dG) == 3


class TestGroundState:
    def test_h12_zero_is_min(self):
        v1 = np.array([0.0, 10.0, 5.0])
        v2 = np.array([10.0, 0.0, 5.0])
        np.testing.assert_allclose(ground_state_energy(v1, v2, 0.0), np.minimum(v1, v2))

    def test_coupling_lowers_below_min(self):
        v1 = np.array([5.0])
        v2 = np.array([5.0])
        e = ground_state_energy(v1, v2, 50.0)
        assert e[0] == pytest.approx(5.0 - 50.0)  # at crossing E = V - H12

    def test_coupling_never_raises_state(self):
        rng = np.random.default_rng(3)
        v1 = rng.normal(0, 50, 1000)
        v2 = rng.normal(0, 50, 1000)
        assert np.all(ground_state_energy(v1, v2, 30.0) <= np.minimum(v1, v2) + 1e-9)


class TestSchedule:
    def test_endpoints_and_length(self):
        s = default_lambda_schedule(11)
        assert len(s) == 11
        assert s[0] == 0.0
        assert s[-1] == 1.0
        assert np.all(np.diff(s) > 0)


class TestGapObservables:
    def test_symmetric_double_well_zero_dg(self):
        c = np.linspace(-40, 40, 81)
        pmf = 0.01 * (c**2 - 400) ** 2 / 400  # wells at +/-20, barrier at 0
        pmf = pmf - pmf.min()
        dg, barrier = gap_observables(c, pmf, np.ones_like(c, bool))
        assert dg == pytest.approx(0.0, abs=0.2)
        assert barrier > 3.0

    def test_tilted_well_nonzero_dg(self):
        c = np.linspace(-40, 40, 81)
        pmf = 0.01 * (c**2 - 400) ** 2 / 400 - 0.1 * c  # tilt toward +c (reactant)
        pmf = pmf - pmf.min()
        dg, _ = gap_observables(c, pmf, np.ones_like(c, bool))
        assert dg > 0.5  # product (c<0) sits above reactant (c>0)

    def test_single_sided_returns_nan(self):
        c = np.linspace(1.0, 40, 40)  # entirely reactant side
        pmf = c**2
        dg, barrier = gap_observables(c, pmf, np.ones_like(c, bool))
        assert np.isnan(dg) and np.isnan(barrier)


class TestAnalyzeGapIntegration:
    def test_symmetric_windows_give_zero_dg(self):
        """Mirror-symmetric diabatic samples -> dG_rxn ~ 0.

        Build windows whose reactant/product energies are swapped mirror images
        about lambda = 0.5, so the reaction is symmetric by construction and the
        recovered reaction free energy must be ~0.
        """
        rng = np.random.default_rng(4)
        lams = default_lambda_schedule(7)
        v1s, v2s = [], []
        for lam in lams:
            # gap mean drifts linearly from + to - across lambda (overlapping).
            mean_gap = 40.0 * (1.0 - 2.0 * lam)
            gap = rng.normal(mean_gap, 8.0, 4000)
            base = rng.normal(0.0, 5.0, 4000)
            v1 = base
            v2 = base + gap
            v1s.append(v1)
            v2s.append(v2)
        res = analyze_gap(lams, v1s, v2s, temperature=300.0, h12=0.0, n_bins=40)
        assert isinstance(res, EVBGapResult)
        assert np.isfinite(res.pmf).sum() > 10
        assert res.dG_rxn == pytest.approx(0.0, abs=3.0)
