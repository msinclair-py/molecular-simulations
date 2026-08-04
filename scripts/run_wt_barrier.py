"""Production driver: converged WT EVB 0->1 barrier + reaction free energy.

This is the GPU production analogue of ``scripts/benchmark_evb.py`` for the real
PQQ-GDH hydride transfer. It consumes a diabatic system already built by
:class:`~molecular_simulations.build.EVBBuilder` (an ``evb_meta.json`` pointing at
``reactant.prmtop``/``product.prmtop``/``system.inpcrd`` plus the 0-based
donor/acceptor/reactive indices) and runs the FULL 0->1 mapping schedule --
reaching BOTH the reactant and product basins so the diabatic reaction free
energy ``dG_rxn`` comes out, which is what P2 calibration (:func:`calibrate_evb`)
needs. It runs several independent replicas (distinct seeds) and aggregates them
into mean +/- SEM, the honest EVB error estimate.

The per-window ``window{i}.npz`` files (v1, v2) are persisted under each replica
directory, so the (alpha, H12) calibration can be run offline afterwards on the
same samples without re-simulating.

Distribution is one mapping window per Aurora GPU tile via parsl
(:class:`~molecular_simulations.utils.parsl_settings.AuroraSettings`, 12 tiles /
node). This driver owns the parsl DataFlowKernel; ``EVBMapping`` submits to it.

Environment-specific inputs -- the PBS account/queue/walltime/filesystems, the
``worker_init`` that activates your working OpenMM on the compute node, and the
OpenMM platform name -- are all command-line arguments; nothing about the machine
is hard-coded.

Example (from the repo root on an Aurora login node)::

    python scripts/run_wt_barrier.py \
        --meta   /flare/FRAME-IDP/msinclair/evb/wt/evb_meta.json \
        --out    /flare/FRAME-IDP/msinclair/evb/wt/run \
        --account FRAME-IDP --queue debug --walltime 01:00:00 --nodes 1 \
        --filesystems flare:home \
        --worker-init 'source /flare/FRAME-IDP/msinclair/envs/agent/bin/activate' \
        --platform OpenCL \
        --n-windows 24 --replicas 3 --n-prod 500000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Converged WT EVB 0->1 barrier + dG_rxn on Aurora GPUs.'
    )
    # -- system (from EVBBuilder) ------------------------------------------
    p.add_argument(
        '--meta',
        type=Path,
        required=True,
        help='evb_meta.json from EVBBuilder (prmtops, coords, transfer indices).',
    )
    p.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Output directory for replica windows + results.json.',
    )
    # -- lambda schedule ----------------------------------------------------
    p.add_argument(
        '--n-windows', type=int, default=24, help='Windows over [lam_min, lam_max].'
    )
    p.add_argument(
        '--lam-min',
        type=float,
        default=0.0,
        help='Reactant endpoint (need 0 for dG_rxn).',
    )
    p.add_argument(
        '--lam-max',
        type=float,
        default=1.0,
        help='Product endpoint (need 1 for dG_rxn).',
    )
    p.add_argument(
        '--uniform',
        action='store_true',
        help='Evenly spaced windows instead of the default endpoint-dense (cosine) schedule.',
    )
    # -- sampling -----------------------------------------------------------
    p.add_argument(
        '--replicas', type=int, default=3, help='Independent seeded replicas.'
    )
    p.add_argument(
        '--seed', type=int, default=1, help='Base seed; replica r uses seed+r.'
    )
    p.add_argument('--temperature', type=float, default=300.0)
    p.add_argument(
        '--n-equil', type=int, default=50000, help='Equilibration steps/window (1 fs).'
    )
    p.add_argument(
        '--n-prod', type=int, default=500000, help='Production steps/window (1 fs).'
    )
    p.add_argument('--sample-interval', type=int, default=100)
    p.add_argument(
        '--platform',
        default='OpenCL',
        help='OpenMM platform (Aurora Intel GPU: OpenCL).',
    )
    p.add_argument(
        '--no-soft-core',
        action='store_true',
        help='Disable reactive-atom endpoint soft-core.',
    )
    p.add_argument(
        '--sc-alpha',
        type=float,
        default=0.5,
        help='Soft-core offset (larger = softer).',
    )
    # -- Morse reactive bond (must match the build) -------------------------
    p.add_argument(
        '--D-e', type=float, default=460.0, help='Morse well depth (kJ/mol).'
    )
    p.add_argument(
        '--morse-alpha', type=float, default=22.0, help='Morse width (nm^-1).'
    )
    p.add_argument(
        '--r0', type=float, default=0.097, help='Morse equilibrium distance (nm).'
    )
    p.add_argument(
        '--nonbonded-cutoff',
        type=float,
        default=1.0,
        help='PME real-space cutoff (nm).',
    )
    # -- analysis -----------------------------------------------------------
    p.add_argument(
        '--n-bins', type=int, default=50, help='Energy-gap bins for the PMF.'
    )
    # -- Aurora / PBS -------------------------------------------------------
    p.add_argument('--account', required=True, help='PBS project/allocation.')
    p.add_argument('--queue', default='debug', help='PBS queue.')
    p.add_argument('--walltime', default='01:00:00', help='PBS walltime HH:MM:SS.')
    p.add_argument(
        '--nodes', type=int, default=1, help='Aurora nodes (12 GPU tiles each).'
    )
    p.add_argument(
        '--accelerators',
        type=int,
        default=12,
        help='GPU tiles (workers) per node. Lower it to bound per-node thread '
        'oversubscription from the Intel oneAPI runtime; pair with OMP_NUM_THREADS '
        'in --worker-init.',
    )
    p.add_argument(
        '--filesystems',
        default='flare:home',
        help='PBS filesystems resource -- Aurora rejects jobs that do not declare it.',
    )
    p.add_argument(
        '--worker-init',
        default='',
        help='Shell run on each worker before the app (activate your OpenMM env, module loads).',
    )
    return p.parse_args()


def make_schedule(args: argparse.Namespace) -> np.ndarray:
    """Full-range lambda schedule; endpoint-dense unless --uniform.

    A converged WT calibration run needs BOTH basins, so the default spans the
    whole [lam_min, lam_max] (0..1) rather than the reactant->TS-only window used
    for per-mutant screening.
    """
    from molecular_simulations.simulate.evb_mapping import (
        default_lambda_schedule,
        endpoint_dense_schedule,
    )

    if args.uniform:
        lams = default_lambda_schedule(args.n_windows)
        return args.lam_min + (args.lam_max - args.lam_min) * lams
    return endpoint_dense_schedule(args.n_windows, args.lam_min, args.lam_max)


def main() -> None:
    args = parse_args()
    from molecular_simulations.simulate.evb_mapping import (
        EVBMapping,
        aggregate_replicas,
    )

    meta = json.loads(Path(args.meta).read_text())
    reactant = meta['reactant_prmtop']
    product = meta['product_prmtop']
    coords = meta['coordinates']
    donor = meta.get('donor_atom')
    acceptor = meta.get('acceptor_atom')
    reactive = meta.get('reactive_atom')
    if None in (donor, acceptor, reactive):
        raise SystemExit(
            f'evb_meta.json is missing transfer indices '
            f'(donor={donor}, acceptor={acceptor}, reactive={reactive}); rebuild '
            'the system with donor/acceptor/reactive atom names.'
        )

    lambdas = make_schedule(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f'[wt-barrier] {len(lambdas)} windows lambda={lambdas[0]:.3f}..{lambdas[-1]:.3f} '
        f'x {args.replicas} replicas; donor={donor} acceptor={acceptor} reactive={reactive}',
        flush=True,
    )

    # This driver owns the parsl DFK (see EVBMapping docstring). One block of
    # `--nodes` Aurora nodes, 12 windows/node concurrently across the GPU tiles.
    import parsl

    from molecular_simulations.utils.parsl_settings import AuroraSettings

    settings = AuroraSettings(
        account=args.account,
        queue=args.queue,
        walltime=args.walltime,
        num_nodes=args.nodes,
        available_accelerators=[str(i) for i in range(args.accelerators)],
        # Aurora PBS requires an explicit filesystems resource on every job.
        scheduler_options=f'#PBS -l filesystems={args.filesystems}',
        worker_init=args.worker_init,
    )
    parsl_cfg = settings.config_factory(str(out / 'runinfo'))

    mappers: list[EVBMapping] = []
    with parsl.load(parsl_cfg):
        for r in range(args.replicas):
            evb = EVBMapping(
                reactant,
                product,
                coords,
                out_path=out / f'rep{r}',
                lambdas=lambdas,
                temperature=args.temperature,
                n_equil=args.n_equil,
                n_prod=args.n_prod,
                sample_interval=args.sample_interval,
                platform=args.platform,
                nonbonded_cutoff=args.nonbonded_cutoff,
                donor=donor,
                acceptor=acceptor,
                reactive=reactive,
                D_e=args.D_e,
                alpha=args.morse_alpha,
                r0=args.r0,
                soft_core=not args.no_soft_core,
                sc_alpha=args.sc_alpha,
            )
            evb.run(seed=args.seed + r)  # window i -> (seed+r)*1000 + i
            mappers.append(evb)
            print(f'[wt-barrier] replica {r} sampling complete', flush=True)
    parsl.clear()

    # Diabatic (h12=0) profile per replica -> upper-bound barrier + dG_rxn.
    # H12/alpha calibration is done offline afterwards from the saved npz windows.
    results = [evb.analyze(h12=0.0, n_bins=args.n_bins) for evb in mappers]
    agg = aggregate_replicas(results)

    summary = {
        'meta': str(args.meta),
        'n_windows': len(lambdas),
        'lambda_min': float(lambdas[0]),
        'lambda_max': float(lambdas[-1]),
        'replicas': int(args.replicas),
        'h12': 0.0,
        'aggregate': agg,
        'per_replica': [
            {'dG_rxn': float(r.dG_rxn), 'dG_barrier': float(r.dG_barrier)}
            for r in results
        ],
    }
    (out / 'results.json').write_text(json.dumps(summary, indent=2))

    print('[wt-barrier] === diabatic (h12=0) result ===', flush=True)
    print(
        f'  dG_rxn     = {agg["dG_rxn"]:.2f} +/- {agg["dG_rxn_sem"]:.2f} kJ/mol',
        flush=True,
    )
    print(
        f'  dG_barrier = {agg["dG_barrier"]:.2f} +/- {agg["dG_barrier_sem"]:.2f} kJ/mol '
        f'(upper bound; h12=0)',
        flush=True,
    )
    print(f'  n finite replicas = {agg["n"]}', flush=True)
    print(f'[wt-barrier] wrote {out / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
