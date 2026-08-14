#!/usr/bin/env python
"""pH replica exchange (pH-REMD) driver for the HEWL constant-pH benchmark.

Builds on ``scripts/benchmark_hewl.py``'s existing ``build`` stage (run that
first to produce ``hewl/{system.prmtop,system.inpcrd,titratable.json}``) and
adds a true pH-REMD ``run`` stage. Unlike ``benchmark_hewl.py run``, which
launches fully INDEPENDENT fixed-pH simulations, this holds every pH-ladder
rung as a live replica in one process and periodically attempts Metropolis
exchanges of pH *labels* between adjacent rungs (see
``molecular_simulations.simulate.constantph.remd.PHREMDDriver``).

Two stages (run ``scripts/benchmark_hewl.py build`` first):

  # 1. Run pH-REMD over a ladder, single process holding all replicas
  python scripts/ph_remd_hewl.py run --sys hewl --pH-ladder 2.0 2.5 3.0 3.5 4.0 \\
      --cycles 2000 --steps 500 --exchange-interval 1 \\
      --platform CUDA --device-ids 0 1 2 3 4 --out hewl/remd_logs

  # 2. UWHAM-reweight the correlated samples into per-residue pKa/Hill fits
  python scripts/ph_remd_hewl.py analyze --logdir hewl/remd_logs --discard 200 \\
      --nmr scripts/data/hewl_reference_pka.csv --out hewl

Statistical note: REMD samples ARE correlated across pH (a rank's pH wanders
via exchanges), so the naive per-pH curve fitting ``benchmark_hewl.py
analyze`` uses for independent runs is NOT valid here -- this script
UWHAM-reweights instead (``molecular_simulations...remd.uwham_titration_curves``).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def run(args):
    from molecular_simulations.simulate.constantph.remd import PHREMDDriver

    titratable = json.loads((Path(args.sys) / 'titratable.json').read_text())
    ladder = sorted(args.pH_ladder)

    device_ids = args.device_ids
    if args.platform != 'CPU' and device_ids is None:
        device_ids = list(range(len(ladder)))
        print(
            f'--device-ids not given; defaulting to {device_ids} for '
            f'{len(ladder)} rungs'
        )

    driver = PHREMDDriver(
        prmtop_file=Path(args.sys) / 'system.prmtop',
        inpcrd_file=Path(args.sys) / 'system.inpcrd',
        titratable=titratable,
        pH_ladder=ladder,
        relaxationSteps=args.relaxation_steps,
        platform_name=args.platform,
        device_ids=device_ids,
        seed=args.seed,
    )
    print(f'built {len(ladder)} replicas on {args.platform}: {ladder}')
    driver.run(
        n_cycles=args.cycles,
        n_steps=args.steps,
        exchange_interval=args.exchange_interval,
        logdir=args.out,
    )
    print(f'wrote {args.out}/rank*.csv and {args.out}/exchanges.csv')


def analyze(args):
    import polars as pl

    from molecular_simulations.simulate.constantph.remd import uwham_titration_curves

    rank_files = sorted(Path(args.logdir).glob('rank*.csv'))
    if not rank_files:
        raise SystemExit(f'no rank*.csv logs found under {args.logdir}')

    dfs = [pl.read_csv(f)[args.discard :] for f in rank_files]
    dfs = [df for df in dfs if len(df) > 0]
    if not dfs:
        raise SystemExit(
            f'--discard {args.discard} left no samples in any rank*.csv log'
        )
    full = pl.concat(dfs)

    resid_cols = [c for c in full.columns if c.startswith('r') and c[1:].isdigit()]
    pH_grid = sorted(full['current_pH'].unique().to_list())

    nmr = {}
    if args.nmr and Path(args.nmr).exists():
        lines = [
            ln
            for ln in Path(args.nmr).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith('#')
        ]
        for r in csv.DictReader(lines):
            nmr[str(r['resnum'])] = (r['resname'], float(r['pKa_exp']))

    fit_results = uwham_titration_curves(full, resid_cols, pH_grid)

    results = []
    for col in sorted(resid_cols, key=lambda c: int(c[1:])):
        resnum = col[1:]
        pKa, hill_n = fit_results[col]
        resname, exp = nmr.get(resnum, ('?', float('nan')))
        results.append(
            {
                'resnum': resnum,
                'resname': resname,
                'pKa_calc': pKa,
                'hill_n': hill_n,
                'pKa_exp': exp,
            }
        )

    print('\nresnum resname  pKa_calc  hill_n   pKa_exp   error')
    diffs = []
    for r in results:
        err = r['pKa_calc'] - r['pKa_exp']
        exp_s = f'{r["pKa_exp"]:>6.2f}' if np.isfinite(r['pKa_exp']) else '   n/a'
        err_s = f'{err:>6.2f}' if np.isfinite(err) else '   n/a'
        if np.isfinite(err):
            diffs.append(err)
        print(
            f'  {r["resnum"]:>4} {r["resname"]:>4}   {r["pKa_calc"]:>6.2f}   '
            f'{r["hill_n"]:>5.2f}   {exp_s}   {err_s}'
        )
    if diffs:
        d = np.array(diffs)
        rmse = float(np.sqrt(np.mean(d**2)))
        mae = float(np.mean(np.abs(d)))
        calc = np.array([r['pKa_calc'] for r in results if np.isfinite(r['pKa_exp'])])
        exp = np.array([r['pKa_exp'] for r in results if np.isfinite(r['pKa_exp'])])
        corr = float(np.corrcoef(calc, exp)[0, 1]) if len(calc) > 1 else float('nan')
        print(
            f'\nN={len(diffs)}  RMSE={rmse:.2f}  MAE={mae:.2f}  R={corr:.2f} '
            f'pKa units (UWHAM-reweighted)'
        )
    else:
        print('\nNo experimental values matched; provide --nmr to compute RMSE.')

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / 'pka_results_remd.json').write_text(json.dumps(results, indent=2))
    print(f'wrote {args.out}/pka_results_remd.json')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run', help='run pH-REMD (single process, N live replicas)')
    r.add_argument(
        '--sys', default='hewl', help='directory from benchmark_hewl.py build'
    )
    r.add_argument(
        '--pH-ladder', type=float, nargs='+', dest='pH_ladder', required=True
    )
    r.add_argument('--cycles', type=int, default=2000)
    r.add_argument('--steps', type=int, default=500)
    r.add_argument('--exchange-interval', type=int, default=1, dest='exchange_interval')
    r.add_argument('--relaxation-steps', type=int, default=500, dest='relaxation_steps')
    r.add_argument('--platform', default='CUDA')
    r.add_argument('--device-ids', type=int, nargs='+', dest='device_ids', default=None)
    r.add_argument('--seed', type=int, default=None)
    r.add_argument('--out', default='hewl/remd_logs')
    r.set_defaults(func=run)

    a = sub.add_parser('analyze', help='UWHAM-reweight + fit pKa, compare to NMR')
    a.add_argument('--logdir', default='hewl/remd_logs')
    a.add_argument(
        '--nmr', default=str(Path(__file__).parent / 'data' / 'hewl_reference_pka.csv')
    )
    a.add_argument(
        '--discard', type=int, default=200, help='cycles to discard per rank'
    )
    a.add_argument('--out', default='hewl')
    a.set_defaults(func=analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
