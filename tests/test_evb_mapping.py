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
    BARConvergenceWarning,
    EVBCalibration,
    EVBGapResult,
    EVBMapping,
    GapSamplingWarning,
    aggregate_replicas,
    analyze_gap,
    bar,
    calibrate_evb,
    crossing_dense_schedule,
    ddg_barrier,
    default_lambda_schedule,
    endpoint_dense_schedule,
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

    def test_nonoverlapping_returns_nan_and_warns(self):
        """Work distributions with no overlap -> nan + BARConvergenceWarning,
        not a brentq 'different signs' crash."""
        kT = KB * 300.0
        # forward/reverse works both huge-positive: their distributions are
        # disjoint, so the BAR root lies far outside +/-500 kT.
        w_f = np.full(1000, 3.0e5)
        w_r = np.full(1000, 3.0e5)
        with pytest.warns(BARConvergenceWarning):
            result = bar(w_f, w_r, kT)
        assert np.isnan(result)


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

    def test_broken_segment_nans_downstream_and_warns(self):
        """A non-overlapping window breaks the ladder: free energies up to the
        break stay finite, everything past it is nan (with a warning)."""
        kT = KB * 300.0
        lam = default_lambda_schedule(5)
        rng = np.random.default_rng(5)
        gaps = [
            rng.normal(20, 2, 4000),
            rng.normal(10, 2, 4000),
            rng.normal(0, 2, 4000),
            rng.normal(3.0e5, 2, 4000),  # disjoint from window 2 -> breaks 2->3
            rng.normal(3.0e5, 2, 4000),
        ]
        with pytest.warns(BARConvergenceWarning):
            dG = ladder_free_energies(lam, gaps, kT)
        assert np.all(np.isfinite(dG[:3]))  # resolved up to the break
        assert np.all(np.isnan(dG[3:]))  # undefined past it


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


class TestEndpointDenseSchedule:
    def test_endpoints_length_monotone(self):
        s = endpoint_dense_schedule(11, 0.0, 0.55)
        assert len(s) == 11
        assert s[0] == pytest.approx(0.0)
        assert s[-1] == pytest.approx(0.55)
        assert np.all(np.diff(s) > 0)

    def test_denser_at_the_ends(self):
        s = endpoint_dense_schedule(11)
        d = np.diff(s)
        assert d[0] < d[len(d) // 2]  # first step tighter than the middle
        assert d[-1] < d[len(d) // 2]  # last step tighter than the middle

    def test_single_window(self):
        assert endpoint_dense_schedule(1, 0.2, 0.8).tolist() == [0.2]


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

    def test_broken_ladder_does_not_crash(self):
        """A non-overlapping tail window must not crash analyze_gap: BAR warns,
        the ladder resolves up to the break, and a result is still returned."""
        rng = np.random.default_rng(6)
        lams = default_lambda_schedule(6)
        v1s, v2s = [], []
        for i, lam in enumerate(lams):
            base = rng.normal(0.0, 5.0, 4000)
            if i < 4:
                gap = rng.normal(40.0 * (1.0 - 2.0 * lam), 8.0, 4000)
            else:
                gap = rng.normal(3.0e5, 8.0, 4000)  # disjoint tail -> broken ladder
            v1s.append(base)
            v2s.append(base + gap)
        with pytest.warns(BARConvergenceWarning):
            res = analyze_gap(lams, v1s, v2s, temperature=300.0, h12=0.0, n_bins=40)
        assert isinstance(res, EVBGapResult)
        assert np.isfinite(res.ladder[:4]).all()  # resolved up to the break
        assert np.isnan(res.ladder[4:]).any()  # undefined past it

    def test_bootstrap_sets_finite_error(self):
        rng = np.random.default_rng(7)
        lams = default_lambda_schedule(7)
        v1s, v2s = [], []
        for lam in lams:
            base = rng.normal(0.0, 5.0, 3000)
            gap = rng.normal(40.0 * (1.0 - 2.0 * lam), 8.0, 3000)
            v1s.append(base)
            v2s.append(base + gap)
        # no bootstrap -> nan errors; bootstrap -> finite, positive errors
        r0 = analyze_gap(lams, v1s, v2s, n_bins=40, n_boot=0)
        assert np.isnan(r0.dG_barrier_err)
        rb = analyze_gap(lams, v1s, v2s, n_bins=40, n_boot=40)
        assert np.isfinite(rb.dG_barrier_err) and rb.dG_barrier_err > 0
        assert np.isfinite(rb.dG_rxn_err) and rb.dG_rxn_err > 0

    def test_wide_gap_range_resolves_barrier(self):
        """A deep-basin window with a huge gap must not wash out the crossing:
        equal-population (quantile) binning still resolves both basins."""
        rng = np.random.default_rng(8)
        lams = np.array([0.0, 0.1, 0.3, 0.5, 0.6, 0.7])
        means = [25000.0, 3000.0, 800.0, 100.0, -100.0, -800.0]
        v1s, v2s = [], []
        for mu in means:
            base = rng.normal(0.0, 5.0, 3000)
            gap = rng.normal(mu, max(50.0, 0.05 * abs(mu)), 3000)
            v1s.append(base)
            v2s.append(base + gap)
        res = analyze_gap(lams, v1s, v2s, n_bins=40)
        assert np.isfinite(res.dG_barrier)  # crossing resolved despite the tail
        assert np.isfinite(res.dG_rxn)

    def test_single_sided_warns(self):
        """Windows that never cross the gap=0 line -> nan + GapSamplingWarning."""
        rng = np.random.default_rng(9)
        lams = np.array([0.0, 0.2, 0.4])
        v1s, v2s = [], []
        for mu in [3000.0, 2000.0, 1000.0]:  # gap stays positive: no product side
            base = rng.normal(0.0, 5.0, 2000)
            v1s.append(base)
            v2s.append(base + rng.normal(mu, 50.0, 2000))
        with pytest.warns(GapSamplingWarning):
            res = analyze_gap(lams, v1s, v2s, n_bins=30)
        assert np.isnan(res.dG_barrier)


class TestAggregateReplicas:
    def _result(self, barrier, rxn):
        return EVBGapResult(
            gap_centers=np.zeros(2),
            pmf=np.zeros(2),
            dG_rxn=rxn,
            dG_barrier=barrier,
            ladder=np.zeros(2),
        )

    def test_mean_and_sem(self):
        reps = [
            self._result(10.0, -5.0),
            self._result(12.0, -7.0),
            self._result(14.0, -6.0),
        ]
        agg = aggregate_replicas(reps)
        assert agg['dG_barrier'] == pytest.approx(12.0)
        assert agg['dG_rxn'] == pytest.approx(-6.0)
        assert agg['n'] == 3
        expect_sem = float(np.std([10, 12, 14], ddof=1) / np.sqrt(3))
        assert agg['dG_barrier_sem'] == pytest.approx(expect_sem)

    def test_ignores_nan_replicas(self):
        reps = [
            self._result(10.0, -5.0),
            self._result(float('nan'), float('nan')),
            self._result(12.0, -5.0),
        ]
        agg = aggregate_replicas(reps)
        assert agg['n'] == 2
        assert agg['dG_barrier'] == pytest.approx(11.0)


def _marcus_windows(seed, k=1.0, d=14.0, dG0=-10.0, n_windows=11, n=3000):
    """Marcus two-state model: V1=0.5k x^2, V2=0.5k(x-d)^2+dG0. Window m samples
    x ~ N(lam*d, sqrt(kT/k)) so the gap sweeps from + (reactant) to - (product)."""
    rng = np.random.default_rng(seed)
    kT = KB * 300.0
    sigma = np.sqrt(kT / k)
    lams = default_lambda_schedule(n_windows)
    v1s, v2s = [], []
    for lam in lams:
        x = rng.normal(lam * d, sigma, n)
        v1s.append(0.5 * k * x**2)
        v2s.append(0.5 * k * (x - d) ** 2 + dG0)
    return lams, v1s, v2s


class TestAlphaShift:
    def test_alpha_shifts_dG_rxn(self):
        """Raising V2 by alpha raises the product basin: dG_rxn increases ~alpha."""
        lams, v1s, v2s = _marcus_windows(10)
        r0 = analyze_gap(lams, v1s, v2s, n_bins=60, alpha=0.0)
        r1 = analyze_gap(lams, v1s, v2s, n_bins=60, alpha=20.0)
        assert (r1.dG_rxn - r0.dG_rxn) == pytest.approx(20.0, abs=4.0)


class TestCalibrate:
    def test_hits_targets(self):
        """calibrate_evb finds (alpha, H12) reproducing a reference dG_rxn+dG dagger."""
        lams, v1s, v2s = _marcus_windows(11)
        cal = calibrate_evb(
            lams,
            v1s,
            v2s,
            dG_rxn_ref=-15.0,
            dG_barrier_ref=8.0,
            n_bins=60,
            tol=1.0,
        )
        assert isinstance(cal, EVBCalibration)
        assert cal.converged
        assert cal.dG_rxn == pytest.approx(-15.0, abs=1.0)
        assert cal.dG_barrier == pytest.approx(8.0, abs=1.0)
        assert cal.h12 > 0.0  # coupling lowered the diabatic barrier to the target

    def test_reapplying_calibration_reproduces_targets(self):
        """analyze_gap with the fitted (alpha, h12) reproduces the calibrated values."""
        lams, v1s, v2s = _marcus_windows(12)
        cal = calibrate_evb(
            lams,
            v1s,
            v2s,
            dG_rxn_ref=-20.0,
            dG_barrier_ref=6.0,
            n_bins=60,
            tol=1.0,
        )
        res = analyze_gap(lams, v1s, v2s, n_bins=60, h12=cal.h12, alpha=cal.alpha)
        assert res.dG_rxn == pytest.approx(cal.dG_rxn, abs=0.5)
        assert res.dG_barrier == pytest.approx(cal.dG_barrier, abs=0.5)

    def test_alpha_only_defers_h12(self):
        """dG_barrier_ref=None calibrates alpha alone and keeps H12=0 (the diabatic
        upper bound), converging on the dG_rxn match -- for when the barrier
        reference (kcat) is not yet in hand."""
        lams, v1s, v2s = _marcus_windows(13)
        cal = calibrate_evb(lams, v1s, v2s, dG_rxn_ref=-18.0, n_bins=60, tol=1.0)
        assert cal.converged
        assert cal.h12 == 0.0  # H12 untouched
        assert cal.dG_rxn == pytest.approx(-18.0, abs=1.0)

    def test_bracket_finds_root_past_nan_window_edges(self):
        """A dG_rxn target reachable only at a large alpha shift -- where the raw
        valid-window edges are nan (a basin empty) -- still converges via the
        finite-straddle bracket, instead of failing to start brentq."""
        lams, v1s, v2s = _marcus_windows(14)
        r0 = analyze_gap(lams, v1s, v2s, n_bins=60).dG_rxn
        target = r0 - 40.0  # far from 0 -> alpha shift lands near the window edge
        cal = calibrate_evb(lams, v1s, v2s, dG_rxn_ref=target, n_bins=60, tol=1.0)
        assert cal.converged
        assert cal.dG_rxn == pytest.approx(target, abs=1.0)

    def test_finite_straddle_bracket_skips_nan_gap(self):
        """The bracket scanner returns a finite adjacent pair straddling the target
        and never brackets across a nan region."""
        from molecular_simulations.simulate.evb_mapping import _finite_straddle_bracket

        def f(x):
            return (
                float('nan') if x < 0.2 or x > 0.8 else 10.0 * x
            )  # finite on [0.2,0.8]

        br = _finite_straddle_bracket(f, 0.0, 1.0, target=5.0, n=41)  # root at x=0.5
        assert br is not None
        lo, hi = br
        assert 0.2 <= lo < hi <= 0.8
        assert (f(lo) - 5.0) * (f(hi) - 5.0) <= 0
        # unreachable target -> None
        assert _finite_straddle_bracket(f, 0.0, 1.0, target=99.0, n=41) is None


class TestDDGBarrier:
    def test_difference_and_error(self):
        def r(barrier, err):
            return EVBGapResult(
                gap_centers=np.zeros(2),
                pmf=np.zeros(2),
                dG_rxn=0.0,
                dG_barrier=barrier,
                ladder=np.zeros(2),
                dG_barrier_err=err,
            )

        ddg, err = ddg_barrier(r(58.0, 3.0), r(50.0, 4.0))
        assert ddg == pytest.approx(8.0)
        assert err == pytest.approx(5.0)  # hypot(3, 4)


class TestReconcileExceptions:
    def test_makes_exception_pairsets_equal(self):
        """After reconciliation both forces share one exception-pair set."""
        import openmm as mm

        from molecular_simulations.simulate.evb_mapping import _reconcile_exceptions

        a, b = mm.NonbondedForce(), mm.NonbondedForce()
        for f in (a, b):
            for _ in range(4):
                f.addParticle(0.1, 0.3, 0.5)
        a.addException(0, 1, 0.0, 0.3, 0.0)  # excluded in a only
        a.addException(2, 3, 0.0, 0.3, 0.0)  # common
        b.addException(2, 3, 0.0, 0.3, 0.0)  # common
        b.addException(0, 2, 0.0, 0.3, 0.0)  # excluded in b only

        _reconcile_exceptions(a, b)

        def pairs(f):
            out = set()
            for k in range(f.getNumExceptions()):
                i, j, *_ = f.getExceptionParameters(k)
                out.add(frozenset((i, j)))
            return out

        assert pairs(a) == pairs(b)
        # the pair added to 'a' carries a's real interaction (nonzero charge).
        added = next(
            a.getExceptionParameters(k)
            for k in range(a.getNumExceptions())
            if {a.getExceptionParameters(k)[0], a.getExceptionParameters(k)[1]}
            == {0, 2}
        )
        assert added[2].value_in_unit(added[2].unit) != 0.0  # chargeProd nonzero

    def test_reports_added_pairs(self):
        """_reconcile_exceptions returns the (i, j) pairs it adds to each force."""
        import openmm as mm

        from molecular_simulations.simulate.evb_mapping import _reconcile_exceptions

        a, b = mm.NonbondedForce(), mm.NonbondedForce()
        for f in (a, b):
            for _ in range(4):
                f.addParticle(0.1, 0.3, 0.5)
        a.addException(0, 1, 0.0, 0.3, 0.0)  # excluded in a only
        b.addException(0, 2, 0.0, 0.3, 0.0)  # excluded in b only
        added_a, added_b = _reconcile_exceptions(a, b)
        assert {frozenset(p) for p in added_a} == {frozenset((0, 2))}  # b's -> a
        assert {frozenset(p) for p in added_b} == {frozenset((0, 1))}  # a's -> b


class TestSoftCore:
    def test_bounds_reactive_clash(self):
        """Soft-core moves a reactive pair's hard LJ into a bounded CustomBondForce:
        the exception LJ is zeroed and the energy at contact stays finite."""
        import openmm as mm
        from openmm import unit

        from molecular_simulations.simulate.evb_mapping import _soft_core_reactive_pairs

        system = mm.System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        system.setDefaultPeriodicBoxVectors(
            mm.Vec3(5, 0, 0) * unit.nanometer,
            mm.Vec3(0, 5, 0) * unit.nanometer,
            mm.Vec3(0, 0, 5) * unit.nanometer,
        )
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(1.0 * unit.nanometer)
        nb.addParticle(0.0, 0.3, 0.5)
        nb.addParticle(0.0, 0.3, 0.5)
        nb.addException(
            0, 1, 0.0, 0.3, 0.5
        )  # real-interaction pair (as reconcile adds)
        system.addForce(nb)

        _soft_core_reactive_pairs(system, nb, reactive=0, pairs=[(0, 1)], sc_alpha=0.5)

        # the exception's hard LJ is zeroed...
        _, _, _, _, eps = nb.getExceptionParameters(0)
        assert eps.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(0.0)
        # ...and a CustomBondForce was added.
        cbfs = [f for f in system.getForces() if isinstance(f, mm.CustomBondForce)]
        assert len(cbfs) == 1
        cbfs[0].setForceGroup(5)

        ctx = mm.Context(
            system, mm.VerletIntegrator(1e-3), mm.Platform.getPlatform('Reference')
        )
        # 0.05 nm apart: a bare LJ (sigma=0.3) would be ~1e9 kJ/mol; soft-core
        # plateaus at 4*eps*(1/a^2 - 1/a) = 4 kJ/mol for eps=0.5, a=0.5.
        ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.05, 0, 0)] * unit.nanometer)
        e_sc = (
            ctx.getState(getEnergy=True, groups={5})
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        assert e_sc == pytest.approx(4.0, abs=0.5)


class TestEVBMappingAnalyze:
    def test_reads_windows_and_reduces(self, tmp_path):
        """EVBMapping.analyze loads per-window npz and reduces to a result."""
        from molecular_simulations.simulate.evb_mapping import EVBMapping

        lams = default_lambda_schedule(7)
        evb = EVBMapping(
            reactant_prmtop=tmp_path / 'r.prmtop',
            product_prmtop=tmp_path / 'p.prmtop',
            coordinates=tmp_path / 'c.inpcrd',
            out_path=tmp_path / 'windows',
            lambdas=lams,
            temperature=300.0,
        )
        for i, lam in enumerate(lams):
            rng = np.random.default_rng(i)
            gap = rng.normal(40.0 * (1.0 - 2.0 * lam), 8.0, 3000)
            base = rng.normal(0.0, 5.0, 3000)
            np.savez(evb.out_path / f'window{i}.npz', v1=base, v2=base + gap, lam=lam)
        res = evb.analyze(h12=0.0, n_bins=40)
        assert isinstance(res, EVBGapResult)
        assert res.dG_rxn == pytest.approx(0.0, abs=3.0)


class TestPerWindowSampling:
    """n_equil/n_prod may be a scalar (broadcast) or one value per window."""

    def _mapping(self, tmp_path, **kw):
        return EVBMapping(
            reactant_prmtop=tmp_path / 'r.prmtop',
            product_prmtop=tmp_path / 'p.prmtop',
            coordinates=tmp_path / 'c.inpcrd',
            out_path=tmp_path / 'windows',
            n_windows=4,
            **kw,
        )

    def test_scalar_broadcasts_to_every_window(self, tmp_path):
        evb = self._mapping(tmp_path, n_equil=500, n_prod=1000)
        assert evb.n_equil == [500, 500, 500, 500]
        assert evb.n_prod == [1000, 1000, 1000, 1000]

    def test_per_window_sequence_preserved(self, tmp_path):
        evb = self._mapping(tmp_path, n_prod=[10, 20, 30, 40])
        assert evb.n_prod == [10, 20, 30, 40]

    def test_wrong_length_sequence_raises(self, tmp_path):
        with pytest.raises(ValueError, match='lambda windows'):
            self._mapping(tmp_path, n_prod=[10, 20])


class TestCrossingDenseSchedule:
    """Dense at both endpoints AND the crossing, unlike endpoint_dense."""

    def test_count_endpoints_and_monotonic(self):
        s = crossing_dense_schedule(12, 0.0, 1.0, 0.5)
        assert len(s) == 12
        assert s[0] == pytest.approx(0.0)
        assert s[-1] == pytest.approx(1.0)
        assert np.all(np.diff(s) > 0)  # strictly increasing (no dup crossing)
        assert np.isclose(s, 0.5).any()  # crossing point present

    def test_denser_at_crossing_than_endpoint_only(self):
        band = lambda s: int(np.sum((s >= 0.35) & (s <= 0.65)))  # noqa: E731
        cross = crossing_dense_schedule(20, 0.0, 1.0, 0.5)
        endp = endpoint_dense_schedule(20, 0.0, 1.0)
        assert band(cross) > band(endp)

    def test_invalid_crossing_raises(self):
        with pytest.raises(ValueError, match='strictly inside'):
            crossing_dense_schedule(10, 0.0, 1.0, 1.5)


class TestCheckpointAppend:
    """_write_or_append_window accumulates samples across resumes."""

    def _fn(self):
        from molecular_simulations.simulate.evb_mapping import (
            _write_or_append_window,
        )

        return _write_or_append_window

    def test_fresh_write(self, tmp_path):
        p = tmp_path / 'window0.npz'
        self._fn()(p, np.arange(5.0), np.arange(5.0) + 1, 0.3, append=False)
        d = np.load(p)
        assert len(d['v1']) == 5
        assert float(d['lam']) == pytest.approx(0.3)

    def test_append_concatenates_onto_existing(self, tmp_path):
        p = tmp_path / 'window0.npz'
        fn = self._fn()
        fn(p, np.zeros(3), np.ones(3), 0.5, append=False)
        _, v2 = fn(p, np.full(2, 9.0), np.full(2, 8.0), 0.5, append=True)
        d = np.load(p)
        assert len(d['v1']) == 5
        assert list(d['v1']) == [0, 0, 0, 9, 9]
        assert list(v2) == [1, 1, 1, 8, 8]

    def test_append_without_existing_writes_fresh(self, tmp_path):
        p = tmp_path / 'window0.npz'
        self._fn()(p, np.zeros(3), np.ones(3), 0.5, append=True)
        assert len(np.load(p)['v1']) == 3
