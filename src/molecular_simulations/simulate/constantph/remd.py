# ruff: noqa: N815
"""pH replica exchange (pH-REMD) on top of the discrete-state ConstantPH engine.

Single-process prototype: N ``ConstantPH`` replicas live in one process, one
per pH-ladder rung, and periodically attempt Metropolis exchanges of pH
*labels* between adjacent rungs. Swapping labels (not physical
configurations/positions) makes every potential-energy term cancel exactly
between the two replicas being considered, leaving only the proton-count/pH
coupling term -- see :func:`exchange_delta`. This means an exchange attempt
costs zero extra force evaluations: it only needs each replica's current
total proton count, which is already cheap plain-Python bookkeeping
(:func:`total_protons`).

This module intentionally does not modify :class:`ConstantPH` -- a replica
pinned to a single pH keeps ``self.pH`` as a length-1 list, and reassigning
its pH label is just ``replica.pH[0] = new_value``.
"""

import csv
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


def exchange_delta(pH_a: float, n_a: int, pH_b: float, n_b: int) -> float:
    """Reduced-potential difference for swapping the pH labels of two replicas.

    Swapping labels rather than configurations makes every potential-energy
    term cancel exactly, leaving only the proton/pH coupling term::

        Delta = ln(10) * (pH_b - pH_a) * (n_a - n_b)

    Accept the swap if ``Delta <= 0``, else with probability ``exp(-Delta)``
    (see :func:`accept_exchange`).
    """
    return np.log(10.0) * (pH_b - pH_a) * (n_a - n_b)


def accept_exchange(delta: float, rng: np.random.Generator | None = None) -> bool:
    """Metropolis accept/reject for an :func:`exchange_delta` value."""
    if delta <= 0.0:
        return True
    rng = rng if rng is not None else np.random.default_rng()
    return bool(rng.random() < np.exp(-delta))


def total_protons(replica) -> int:
    """Total proton count across a replica's titratable residues, current state.

    Same expression ``ConstantPH._attemptPHChange`` uses internally for its
    own (single-process, simulated-tempering) pH move -- plain Python ints,
    no OpenMM calls.
    """
    return sum(
        t.explicitStates[t.currentIndex].numHydrogens
        for t in replica.titrations.values()
    )


@dataclass
class ExchangeRecord:
    """One exchange *attempt* (not just accepted swaps) between adjacent rungs."""

    cycle: int
    rung_low: int
    rung_high: int
    replica_low: int
    replica_high: int
    pH_low: float
    pH_high: float
    n_low: int
    n_high: int
    delta: float
    accepted: bool


def _build_replica(
    prmtop_file, inpcrd_file, pH, titratable, relaxationSteps, platform, properties
):
    """Build one ConstantPH replica pinned to a single pH.

    Mirrors ``scripts/benchmark_hewl.py``'s ``_make_cph`` (same explicitArgs/
    implicitArgs/integrator conventions), extended with an explicit
    ``properties`` (e.g. ``{'DeviceIndex': ...}``) argument that ``_make_cph``
    does not support today.
    """
    from openmm import LangevinMiddleIntegrator
    from openmm.app import PME, CutoffNonPeriodic, HBonds
    from openmm.unit import amu, kelvin, kilojoules_per_mole, nanometers, picosecond

    from molecular_simulations.data import get_ref_energies
    from molecular_simulations.simulate.constantph.constantph import ConstantPH

    refs = get_ref_energies('amber19')
    variants = {int(k): v['variants'] for k, v in titratable.items()}
    reference = {
        int(k): [e * kilojoules_per_mole for e in refs[v['resname']]]
        for k, v in titratable.items()
    }
    return ConstantPH(
        prmtop_file=str(prmtop_file),
        inpcrd_file=str(inpcrd_file),
        pH=[pH],
        residueVariants=variants,
        referenceEnergies=reference,
        relaxationSteps=relaxationSteps,
        explicitArgs=dict(
            nonbondedMethod=PME,
            nonbondedCutoff=0.9 * nanometers,
            constraints=HBonds,
            hydrogenMass=1.5 * amu,
        ),
        implicitArgs=dict(
            nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=2.0 * nanometers,
            constraints=HBonds,
        ),
        integrator=LangevinMiddleIntegrator(
            300 * kelvin, 1.0 / picosecond, 0.004 * picosecond
        ),
        relaxationIntegrator=LangevinMiddleIntegrator(
            300 * kelvin, 10.0 / picosecond, 0.002 * picosecond
        ),
        platform=platform,
        properties=properties,
    )


class PHREMDDriver:
    """Single-process pH replica exchange over N ConstantPH replicas.

    Each replica owns its own OpenMM Context (and, for CUDA, its own device)
    for the lifetime of the run. Exchange attempts swap only the cheap pH
    *label* between two replica objects -- never positions, velocities, or
    protonation state -- so a replica's physical identity (and device) stays
    fixed while the pH it is currently simulating under wanders.
    """

    def __init__(
        self,
        prmtop_file,
        inpcrd_file,
        titratable: dict,
        pH_ladder: list[float],
        relaxationSteps: int = 500,
        temperature: float = 300.0,
        platform_name: str = 'CPU',
        device_ids: list[int] | None = None,
        seed: int | None = None,
    ):
        if len(pH_ladder) < 2:
            raise ValueError('pH_ladder must have at least 2 rungs to exchange between')
        if platform_name != 'CPU' and (
            device_ids is None or len(device_ids) != len(pH_ladder)
        ):
            got = None if device_ids is None else len(device_ids)
            raise ValueError(
                f'device_ids must have one entry per pH-ladder rung when '
                f'platform_name={platform_name!r} (got {got}, need {len(pH_ladder)})'
            )

        self.pH_ladder = sorted(pH_ladder)
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)

        from openmm import Platform

        platform = Platform.getPlatformByName(platform_name)

        self.replicas = []
        for i, pH in enumerate(self.pH_ladder):
            properties = (
                {'DeviceIndex': str(device_ids[i]), 'Precision': 'mixed'}
                if platform_name != 'CPU'
                else None
            )
            replica = _build_replica(
                prmtop_file,
                inpcrd_file,
                pH,
                titratable,
                relaxationSteps,
                platform,
                properties,
            )
            replica.simulation.minimizeEnergy()
            replica.simulation.context.setVelocitiesToTemperature(temperature)
            self.replicas.append(replica)

        # rung index -> current index into self.replicas (physical slot/device)
        self.replica_of_rung = list(range(len(self.pH_ladder)))
        self.resnums = [int(v['pdb_resnum']) for v in titratable.values()]
        self.resids = [int(k) for k in titratable]

    def attempt_exchange_round(self, cycle: int, parity: int) -> list[ExchangeRecord]:
        """Attempt exchanges between adjacent rungs at the given parity.

        parity=0 attempts (0,1),(2,3),...; parity=1 attempts (1,2),(3,4),...
        Every attempt is recorded, not just accepted ones.
        """
        records = []
        n_rungs = len(self.pH_ladder)
        for k in range(parity, n_rungs - 1, 2):
            i, j = self.replica_of_rung[k], self.replica_of_rung[k + 1]
            rep_lo, rep_hi = self.replicas[i], self.replicas[j]
            pH_lo, pH_hi = self.pH_ladder[k], self.pH_ladder[k + 1]
            n_lo, n_hi = total_protons(rep_lo), total_protons(rep_hi)
            delta = exchange_delta(pH_lo, n_lo, pH_hi, n_hi)
            accepted = accept_exchange(delta, self._rng)
            if accepted:
                rep_lo.pH[0], rep_hi.pH[0] = pH_hi, pH_lo
                self.replica_of_rung[k], self.replica_of_rung[k + 1] = j, i
            records.append(
                ExchangeRecord(
                    cycle, k, k + 1, i, j, pH_lo, pH_hi, n_lo, n_hi, delta, accepted
                )
            )
        return records

    def run(
        self,
        n_cycles: int,
        n_steps: int,
        exchange_interval: int = 1,
        logdir: str = 'remd_logs',
    ) -> None:
        """Run n_cycles of (step + MC) for every replica, with periodic exchanges.

        Writes one ``rank{i}.csv`` per replica slot (protonation states,
        current pH *after* that cycle's exchange attempt) and a shared
        ``exchanges.csv`` (one row per exchange attempt).
        """
        logdir_path = Path(logdir)
        logdir_path.mkdir(parents=True, exist_ok=True)

        with ExitStack() as stack:
            rank_files = [
                stack.enter_context(open(logdir_path / f'rank{i}.csv', 'w', newline=''))
                for i in range(len(self.replicas))
            ]
            rank_writers = [csv.writer(fh) for fh in rank_files]
            for w in rank_writers:
                w.writerow(
                    ['cycle', 'rankid', 'current_pH', *[f'r{n}' for n in self.resnums]]
                )

            exch_fh = stack.enter_context(
                open(logdir_path / 'exchanges.csv', 'w', newline='')
            )
            exch_writer = csv.writer(exch_fh)
            exch_writer.writerow([f.name for f in fields(ExchangeRecord)])

            parity = 0
            for cycle in range(n_cycles):
                for replica in self.replicas:
                    replica.simulation.step(n_steps)
                    replica.attemptMCStep(self.temperature)

                if exchange_interval > 0 and cycle % exchange_interval == 0:
                    records = self.attempt_exchange_round(cycle, parity)
                    parity = 1 - parity
                    for rec in records:
                        exch_writer.writerow(list(asdict(rec).values()))

                for i, replica in enumerate(self.replicas):
                    row = [cycle, i, replica.pH[0]]
                    for resid in self.resids:
                        t = replica.titrations[resid]
                        row.append(1 if t.currentIndex == t.protonatedIndex else 0)
                    rank_writers[i].writerow(row)

                if cycle % 100 == 0:
                    for fh in (*rank_files, exch_fh):
                        fh.flush()


def uwham_titration_curves(df, resid_cols: list[str], pH_grid) -> dict:
    """Reweight REMD samples across the pH ladder with UWHAM per residue.

    Naive per-pH curve fitting (as ``benchmark_hewl.py analyze`` does for
    independent fixed-pH runs) is wrong for REMD data, since exchanges
    correlate samples across pH. UWHAMSolver already exists for exactly this
    case but was unwired; this is the glue.

    Returns ``{resid_col: (pKa, hill_n)}``.
    """
    from molecular_simulations.analysis.constant_pH_analysis import UWHAMSolver
    from molecular_simulations.simulate.constantph.reference_energy import (
        fit_titration_midpoint,
    )

    solver = UWHAMSolver()
    solver.load_data(df, resid_cols)
    solver.solve()

    pH_grid = np.asarray(pH_grid, dtype=float)
    results = {}
    for resid in resid_cols:
        fractions = [
            solver.compute_expectation_at_pH(solver.states[resid], pH) for pH in pH_grid
        ]
        results[resid] = fit_titration_midpoint(
            pH_grid, fractions, pKa0=float(np.median(pH_grid))
        )
    return results
