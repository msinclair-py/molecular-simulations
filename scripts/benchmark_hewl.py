#!/usr/bin/env python
"""Hen egg-white lysozyme (HEWL) constant-pH pKa benchmark.

The standard first-tier validation for a constant-pH implementation: titrate the
Asp/Glu/His sites of HEWL and compare the computed pKa values to experimental NMR
values (RMSE + correlation). HEWL is small, stable, disulfide-rich (its Cys are
all bonded -> CYX, so they drop out of the titration), and its Glu35 (upshifted)
and Asp52 (catalytic) are well-known environment-perturbed landmarks.

Three stages (run in order):

  # 1. Prepare + build a constant-pH-ready solvated system (needs AmberTools)
  python scripts/benchmark_hewl.py build --pdb 1AKI --out hewl

  # 2. Run one fixed-pH simulation per (pH, replica). Embarrassingly parallel:
  #    launch one process per GPU, e.g.
  #      CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_hewl.py run \
  #          --sys hewl --pH 3.0 --replica 0 --cycles 2000 --steps 500 --out hewl/logs
  #    (or --all to loop every pH/replica sequentially on one device)
  python scripts/benchmark_hewl.py run --sys hewl --all --cycles 2000 --steps 500 \
      --out hewl/logs

  # 3. Fit pKa + Hill per residue and compare to NMR
  python scripts/benchmark_hewl.py analyze --logdir hewl/logs \
      --nmr scripts/data/hewl_reference_pka.csv --out hewl

Statistical note: these are INDEPENDENT fixed-pH simulations (one pH per run), so
simple per-pH curve fitting is the correct analysis (no replica-exchange/UWHAM
reweighting needed). Discard the equilibration window (``--discard``) and use
several replicas per pH for error bars.
"""

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

# Titratable residue conventions (side-chain sites reachable in pH ~2-8).
# name in built topology -> ([deprotonated, protonated] variants, canonical key)
TITRATABLE = {
    'ASP': (['ASP', 'ASH'], 'ASP'),
    'ASH': (['ASP', 'ASH'], 'ASP'),
    'GLU': (['GLU', 'GLH'], 'GLU'),
    'GLH': (['GLU', 'GLH'], 'GLU'),
    'HIS': (['HID', 'HIP'], 'HIS'),
    'HIP': (['HID', 'HIP'], 'HIS'),
    'HID': (['HID', 'HIP'], 'HIS'),
    'HIE': (['HID', 'HIP'], 'HIS'),
}
DEFAULT_PH_LADDER = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]


# ---------------------------------------------------------------------------
# Stage 1: prepare structure + build
# ---------------------------------------------------------------------------


def fetch_pdb(pdb_id, dest):
    """Download a PDB from RCSB unless it already exists locally."""
    dest = Path(dest)
    if dest.exists():
        return dest
    url = f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb'
    print(f'fetching {url}')
    urllib.request.urlretrieve(url, dest)
    return dest


def clean_pdb(src, dest, chain=None):
    """Keep only protein ATOM records (drop HETATM/water/ligands/altlocs).

    Returns the ordered list of (resSeq, resName) for the retained residues so
    downstream code can recover the original PDB numbering after tleap rebuilds
    the topology.
    """
    residues = []
    seen = set()
    out_lines = []
    for line in Path(src).read_text().splitlines():
        if not line.startswith('ATOM'):
            if line.startswith('TER'):
                out_lines.append(line)
            continue
        altloc = line[16]
        if altloc not in (' ', 'A'):
            continue
        ch = line[21]
        if chain and ch != chain:
            continue
        resseq = line[22:27].strip()  # includes insertion code if any
        resname = line[17:20].strip()
        key = (ch, resseq)
        if key not in seen:
            seen.add(key)
            residues.append((resseq, resname))
        # normalise altloc to blank so tleap is happy
        out_lines.append(line[:16] + ' ' + line[17:])
    Path(dest).write_text('\n'.join(out_lines) + '\nEND\n')
    return residues


def build(args):
    from molecular_simulations.build.build_amber import ConstantPHSolvent

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = (
        fetch_pdb(args.pdb, out / f'{args.pdb}.pdb')
        if not Path(args.pdb).exists()
        else Path(args.pdb)
    )
    cleaned = out / 'cleaned.pdb'
    ordered = clean_pdb(raw, cleaned, chain=args.chain)
    print(f'cleaned structure: {len(ordered)} protein residues')

    builder = ConstantPHSolvent(
        path=str(out / 'build'),
        pdb=str(cleaned),
        padding=args.padding,
        delete_temp_file=True,
    )
    builder.build()
    prmtop = out / 'build' / 'system.prmtop'
    inpcrd = out / 'build' / 'system.inpcrd'
    for f, name in [(prmtop, 'system.prmtop'), (inpcrd, 'system.inpcrd')]:
        (out / name).write_bytes(Path(f).read_bytes())

    # Map built topology residues -> original PDB numbering (same order) and pick
    # the titratable sidechain sites, excluding the chain termini.
    from openmm.app import AmberPrmtopFile

    top = AmberPrmtopFile(str(out / 'system.prmtop')).topology
    # The built protein residues are in the same order as the cleaned PDB, so map
    # by position to recover original PDB numbering. Skip the chain termini.
    built = list(top.residues())
    titratable = {}
    n_prot = len(ordered)
    for i, res in enumerate(built[:n_prot]):
        pdb_resseq, _pdb_name = ordered[i]
        if res.name in TITRATABLE and 0 < i < n_prot - 1:
            variants, canonical = TITRATABLE[res.name]
            titratable[res.index] = {
                'pdb_resnum': pdb_resseq,
                'resname': canonical,
                'variants': variants,
            }
    if args.residues:
        keep = set(args.residues)
        titratable = {k: v for k, v in titratable.items() if v['resname'] in keep}

    (out / 'titratable.json').write_text(json.dumps(titratable, indent=2))
    counts = {}
    for v in titratable.values():
        counts[v['resname']] = counts.get(v['resname'], 0) + 1
    print(f'titratable sites: {counts} (total {len(titratable)})')
    print(f'wrote {out}/system.prmtop, system.inpcrd, titratable.json')


# ---------------------------------------------------------------------------
# Stage 2: run one fixed-pH simulation
# ---------------------------------------------------------------------------


def _make_cph(sysdir, pH, titratable, platform):
    from openmm import LangevinMiddleIntegrator, Platform
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
        prmtop_file=str(Path(sysdir) / 'system.prmtop'),
        inpcrd_file=str(Path(sysdir) / 'system.inpcrd'),
        pH=[pH],
        residueVariants=variants,
        referenceEnergies=reference,
        relaxationSteps=500,
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
        platform=Platform.getPlatformByName(platform),
    )


def run_single(sysdir, pH, replica, cycles, steps, platform, outdir):
    titratable = json.loads((Path(sysdir) / 'titratable.json').read_text())
    cph = _make_cph(sysdir, pH, titratable, platform)
    cph.simulation.minimizeEnergy()
    cph.simulation.context.setVelocitiesToTemperature(300.0)

    resnums = [titratable[k]['pdb_resnum'] for k in titratable]
    idxs = [int(k) for k in titratable]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = Path(outdir) / f'pH{pH:.2f}_rep{replica}.csv'
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['cycle', 'pH', *[f'r{n}' for n in resnums]])
        for c in range(cycles):
            cph.simulation.step(steps)
            cph.attemptMCStep(300.0)
            row = [c, pH]
            for ri in idxs:
                t = cph.titrations[ri]
                row.append(1 if t.currentIndex == t.protonatedIndex else 0)
            w.writerow(row)
            if c % 100 == 0:
                fh.flush()
    print(f'wrote {path} ({cycles} cycles)')


def run(args):
    ladder = args.pH_ladder if args.pH_ladder else DEFAULT_PH_LADDER
    if args.all:
        for rep in range(args.replicas):
            for pH in ladder:
                run_single(
                    args.sys, pH, rep, args.cycles, args.steps, args.platform, args.out
                )
    else:
        run_single(
            args.sys,
            args.pH,
            args.replica,
            args.cycles,
            args.steps,
            args.platform,
            args.out,
        )


# ---------------------------------------------------------------------------
# Stage 3: analyze -> pKa vs NMR
# ---------------------------------------------------------------------------


def analyze(args):
    from molecular_simulations.simulate.constantph.reference_energy import (
        fit_titration_midpoint,
    )

    # Aggregate fraction protonated per (residue, pH) across all csv logs.
    per_res = {}  # resnum -> {pH -> [samples]}
    for csvf in sorted(Path(args.logdir).glob('pH*_rep*.csv')):
        with open(csvf) as fh:
            rows = list(csv.DictReader(fh))
        rows = rows[args.discard :]
        if not rows:
            continue
        pH = float(rows[0]['pH'])
        rescols = [c for c in rows[0] if c.startswith('r')]
        for col in rescols:
            resnum = col[1:]
            vals = [int(r[col]) for r in rows]
            per_res.setdefault(resnum, {}).setdefault(pH, []).extend(vals)

    nmr = {}
    if args.nmr and Path(args.nmr).exists():
        lines = [
            ln
            for ln in Path(args.nmr).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith('#')
        ]
        for r in csv.DictReader(lines):
            nmr[str(r['resnum'])] = (r['resname'], float(r['pKa_exp']))

    results = []
    for resnum, phmap in sorted(per_res.items(), key=lambda kv: int(kv[0])):
        phs = sorted(phmap)
        frac = [float(np.mean(phmap[p])) for p in phs]
        if len(phs) < 3:
            continue
        pKa, n = fit_titration_midpoint(phs, frac, pKa0=5.0)
        resname, exp = nmr.get(resnum, ('?', float('nan')))
        results.append(
            {
                'resnum': resnum,
                'resname': resname,
                'pKa_calc': pKa,
                'hill_n': n,
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
            f'\nN={len(diffs)}  RMSE={rmse:.2f}  MAE={mae:.2f}  R={corr:.2f} pKa units'
        )
        print('Literature-quality explicit/hybrid CpHMD: RMSE ~0.8-1.0.')
    else:
        print('\nNo experimental values matched; provide --nmr to compute RMSE.')

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / 'pka_results.json').write_text(json.dumps(results, indent=2))
    _plot(results, per_res, Path(args.out))


def _plot(results, per_res, out):
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    have_exp = [r for r in results if np.isfinite(r['pKa_exp'])]
    if have_exp:
        fig, ax = plt.subplots(figsize=(5, 5))
        c = [r['pKa_calc'] for r in have_exp]
        e = [r['pKa_exp'] for r in have_exp]
        ax.scatter(e, c, color='C0')
        for r in have_exp:
            ax.annotate(
                f'{r["resname"]}{r["resnum"]}',
                (r['pKa_exp'], r['pKa_calc']),
                fontsize=7,
                alpha=0.7,
            )
        lims = [min(e + c) - 0.5, max(e + c) + 0.5]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlabel('experimental pKa (NMR)')
        ax.set_ylabel('computed pKa')
        ax.set_title('HEWL constant-pH benchmark')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        fig.tight_layout()
        fig.savefig(out / 'pka_vs_nmr.png', dpi=130)
        print(f'wrote {out}/pka_vs_nmr.png')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='fetch + build CpH-ready HEWL system')
    b.add_argument('--pdb', default='1AKI', help='PDB id or local .pdb path')
    b.add_argument('--chain', default='A')
    b.add_argument('--padding', type=float, default=12.0)
    b.add_argument(
        '--residues',
        nargs='+',
        default=['ASP', 'GLU', 'HIS'],
        help='titratable residue types to include',
    )
    b.add_argument('--out', default='hewl')
    b.set_defaults(func=build)

    r = sub.add_parser('run', help='run fixed-pH constant-pH simulation(s)')
    r.add_argument('--sys', default='hewl')
    r.add_argument('--pH', type=float, default=4.0)
    r.add_argument('--replica', type=int, default=0)
    r.add_argument('--replicas', type=int, default=3, help='(with --all)')
    r.add_argument('--pH-ladder', type=float, nargs='+', dest='pH_ladder')
    r.add_argument('--all', action='store_true', help='loop every pH/replica here')
    r.add_argument('--cycles', type=int, default=2000)
    r.add_argument('--steps', type=int, default=500)
    r.add_argument('--platform', default='CUDA')
    r.add_argument('--out', default='hewl/logs')
    r.set_defaults(func=run)

    a = sub.add_parser('analyze', help='fit pKa + compare to NMR')
    a.add_argument('--logdir', default='hewl/logs')
    a.add_argument(
        '--nmr', default=str(Path(__file__).parent / 'data' / 'hewl_reference_pka.csv')
    )
    a.add_argument('--discard', type=int, default=200, help='cycles to discard')
    a.add_argument('--out', default='hewl')
    a.set_defaults(func=analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
