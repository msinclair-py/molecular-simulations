"""Tests for simulate/constantph/remd.py (pH replica exchange)."""

import csv
import math
from types import SimpleNamespace

import numpy as np
import pytest

from molecular_simulations.simulate.constantph.remd import (
    PHREMDDriver,
    accept_exchange,
    exchange_delta,
    total_protons,
    uwham_titration_curves,
)


def _fake_titration(current_hydrogens):
    state = SimpleNamespace(numHydrogens=current_hydrogens)
    return SimpleNamespace(currentIndex=0, explicitStates=[state])


def _fake_replica(pH, hydrogens_by_resid):
    return SimpleNamespace(
        pH=[pH],
        titrations={
            resid: _fake_titration(h) for resid, h in hydrogens_by_resid.items()
        },
    )


class TestExchangeDelta:
    @pytest.mark.unit
    def test_zero_when_symmetric(self) -> None:
        assert exchange_delta(4.0, 10, 6.0, 10) == 0.0

    @pytest.mark.unit
    def test_matches_hand_derivation(self) -> None:
        delta = exchange_delta(4.0, 10, 6.0, 8)
        assert delta == pytest.approx(math.log(10.0) * (6.0 - 4.0) * (10 - 8))

    @pytest.mark.unit
    def test_favorable_direction_is_negative(self) -> None:
        # A (pH 4, 8 protons) trading with B (pH 6, 10 protons): A already has
        # fewer protons than B, so sending A to the higher pH (where fewer
        # protons is favored) is thermodynamically favorable -- Delta < 0.
        assert exchange_delta(pH_a=4.0, n_a=8, pH_b=6.0, n_b=10) < 0

    @pytest.mark.unit
    def test_unfavorable_direction_is_positive(self) -> None:
        assert exchange_delta(pH_a=4.0, n_a=10, pH_b=6.0, n_b=8) > 0

    @pytest.mark.unit
    def test_symmetric_under_relabeling(self) -> None:
        # Relabeling which replica is "a" and which is "b" describes the same
        # physical exchange move, so both the (pH_b-pH_a) and (n_a-n_b) factors
        # flip sign together and the product -- Delta -- is unchanged.
        d1 = exchange_delta(4.0, 10, 6.0, 8)
        d2 = exchange_delta(6.0, 8, 4.0, 10)
        assert d1 == pytest.approx(d2)


class TestAcceptExchange:
    @pytest.mark.unit
    def test_nonpositive_delta_always_accepts(self) -> None:
        rng = np.random.default_rng(0)
        assert accept_exchange(0.0, rng) is True
        assert accept_exchange(-5.0, rng) is True

    @pytest.mark.unit
    def test_acceptance_rate_matches_boltzmann_factor(self) -> None:
        rng = np.random.default_rng(42)
        delta = 1.5
        n = 20000
        accepted = sum(accept_exchange(delta, rng) for _ in range(n))
        assert accepted / n == pytest.approx(math.exp(-delta), abs=0.02)

    @pytest.mark.unit
    def test_works_without_explicit_rng(self) -> None:
        assert accept_exchange(0.0) is True


class TestTotalProtons:
    @pytest.mark.unit
    def test_sums_current_state_hydrogens_across_residues(self) -> None:
        replica = SimpleNamespace(
            titrations={1: _fake_titration(13), 2: _fake_titration(4)}
        )
        assert total_protons(replica) == 17


class TestAttemptExchangeRound:
    @pytest.mark.unit
    def test_accept_swaps_ph_labels_and_replica_of_rung(self) -> None:
        driver = object.__new__(PHREMDDriver)
        driver.pH_ladder = [4.0, 6.0]
        driver.replicas = [_fake_replica(4.0, {1: 8}), _fake_replica(6.0, {1: 10})]
        driver.replica_of_rung = [0, 1]
        driver._rng = np.random.default_rng(0)

        # Guaranteed favorable (Delta < 0, see TestExchangeDelta) -> always accepts.
        records = driver.attempt_exchange_round(cycle=0, parity=0)

        assert len(records) == 1
        rec = records[0]
        assert rec.accepted is True
        assert (rec.rung_low, rec.rung_high) == (0, 1)
        assert driver.replicas[0].pH[0] == 6.0
        assert driver.replicas[1].pH[0] == 4.0
        assert driver.replica_of_rung == [1, 0]

    @pytest.mark.unit
    def test_reject_leaves_state_untouched(self) -> None:
        driver = object.__new__(PHREMDDriver)
        driver.pH_ladder = [4.0, 6.0]
        # Unfavorable direction (Delta > 0, see TestExchangeDelta).
        driver.replicas = [_fake_replica(4.0, {1: 10}), _fake_replica(6.0, {1: 8})]
        driver.replica_of_rung = [0, 1]
        # A "never accept" RNG: random() always returns 1.0, which is never
        # less than exp(-delta) for delta > 0.
        driver._rng = SimpleNamespace(random=lambda: 1.0)

        records = driver.attempt_exchange_round(cycle=0, parity=0)

        assert records[0].accepted is False
        assert driver.replicas[0].pH[0] == 4.0
        assert driver.replicas[1].pH[0] == 6.0
        assert driver.replica_of_rung == [0, 1]

    @pytest.mark.unit
    def test_parity_selects_correct_adjacent_pairs(self) -> None:
        driver = object.__new__(PHREMDDriver)
        driver.pH_ladder = [1.0, 2.0, 3.0, 4.0]
        driver.replicas = [_fake_replica(pH, {1: 10}) for pH in driver.pH_ladder]
        driver.replica_of_rung = [0, 1, 2, 3]
        driver._rng = np.random.default_rng(0)

        even = driver.attempt_exchange_round(cycle=0, parity=0)
        assert {(r.rung_low, r.rung_high) for r in even} == {(0, 1), (2, 3)}

        driver.replica_of_rung = [0, 1, 2, 3]
        odd = driver.attempt_exchange_round(cycle=0, parity=1)
        assert {(r.rung_low, r.rung_high) for r in odd} == {(1, 2)}


class TestPHREMDDriverValidation:
    @pytest.mark.unit
    def test_requires_at_least_two_rungs(self) -> None:
        with pytest.raises(ValueError, match='at least 2 rungs'):
            PHREMDDriver(
                prmtop_file='x.prmtop',
                inpcrd_file='x.inpcrd',
                titratable={},
                pH_ladder=[5.0],
            )

    @pytest.mark.unit
    def test_cuda_requires_matching_device_ids(self) -> None:
        with pytest.raises(ValueError, match='device_ids'):
            PHREMDDriver(
                prmtop_file='x.prmtop',
                inpcrd_file='x.inpcrd',
                titratable={},
                pH_ladder=[4.0, 6.0],
                platform_name='CUDA',
                device_ids=[0],
            )


@pytest.mark.integration
class TestPHREMDDriverIntegration:
    """End-to-end on the real (CPU) ConstantPH engine, tiny fixture.

    Deliberately does NOT assert that a swap is accepted (that would tie a
    physics-dependent outcome to an RNG seed and risk CI flakiness) -- the
    swap *mechanism* is already proven deterministically in
    TestAttemptExchangeRound above. This test proves the real wiring (actual
    ConstantPH replicas, actual total_protons reads, actual CSV logging, and
    the UWHAM glue) works end-to-end without crashing.
    """

    def test_run_produces_valid_logs_and_uwham_consumes_them(
        self, real_amber_titratable_solvated_files, skip_without_openmm, tmp_path
    ) -> None:
        files = real_amber_titratable_solvated_files
        titratable = {
            '1': {'variants': ['LYN', 'LYS'], 'resname': 'LYS', 'pdb_resnum': 1},
            '2': {'variants': ['ASP', 'ASH'], 'resname': 'ASP', 'pdb_resnum': 2},
        }
        pH_ladder = [2.0, 5.0, 8.0]

        driver = PHREMDDriver(
            prmtop_file=files['prmtop'],
            inpcrd_file=files['inpcrd'],
            titratable=titratable,
            pH_ladder=pH_ladder,
            relaxationSteps=5,
            platform_name='CPU',
            seed=0,
        )

        logdir = tmp_path / 'remd_logs'
        driver.run(n_cycles=6, n_steps=5, exchange_interval=1, logdir=str(logdir))

        with open(logdir / 'exchanges.csv') as fh:
            exch_rows = list(csv.DictReader(fh))
        # 6 cycles, parity alternates 0,1,0,1,0,1; with 3 rungs each parity
        # only fits one adjacent pair, so 6 cycles -> 6 total attempts (3 of
        # each pair).
        assert len(exch_rows) == 6
        pairs_seen = {(row['rung_low'], row['rung_high']) for row in exch_rows}
        assert pairs_seen == {('0', '1'), ('1', '2')}
        for row in exch_rows:
            assert math.isfinite(float(row['delta']))
            assert row['accepted'] in ('True', 'False')

        import polars as pl

        rank_dfs = []
        for i in range(len(pH_ladder)):
            df = pl.read_csv(logdir / f'rank{i}.csv')
            assert set(df.columns) == {'cycle', 'rankid', 'current_pH', 'r1', 'r2'}
            assert len(df) == 6
            assert set(df['current_pH'].to_list()) <= set(pH_ladder)
            rank_dfs.append(df)

        full = pl.concat(rank_dfs)
        resid_cols = ['r1', 'r2']

        results = uwham_titration_curves(full, resid_cols, pH_grid=pH_ladder)
        assert set(results) == set(resid_cols)
        for pKa, _hill_n in results.values():
            assert math.isfinite(pKa)
