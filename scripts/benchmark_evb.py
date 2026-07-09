#!/usr/bin/env python
"""EVB free-energy validation benchmark.

The analogue of ``benchmark_hewl.py`` for the EVB umbrella-sampling machinery
(``molecular_simulations.simulate.free_energy``). Where the constant-pH
benchmark titrates HEWL and compares computed pKa values to experimental NMR,
this benchmark drives the same reaction through umbrella windows, reconstructs
the potential of mean force (PMF) with ``EVBAnalyzer`` (MBAR, WHAM fallback),
and compares the recovered free-energy observables -- the activation barrier
``dG*`` and the reaction free energy ``dG_rxn`` -- to reference values.

There are two complementary validation tiers, mirroring the two things a
constant-pH benchmark checks (does the sampler reproduce a *known* landmark, and
is the machinery self-consistent):

  1. ``selftest`` -- ANALYTIC self-consistency (no GPU, deterministic, CI-safe).
     Synthetic umbrella data are drawn (1-D Metropolis) from a *known* underlying
     PMF -- flat, harmonic, and a symmetric double well -- and pushed through the
     real ``EVBAnalyzer`` pipeline. Because the underlying surface is exact, the
     recovered PMF has a known answer: a flat surface must come back flat, a
     harmonic surface must come back as its parabola, and the symmetric double
     well must return its analytic barrier height with dG_rxn = 0 and a PMF
     symmetric about rc = 0. This validates equilibration detection, overlap,
     and MBAR/WHAM reweighting against ground truth.

  2. ``build`` / ``run`` / ``analyze`` -- a symmetric PHYSICAL reaction
     (malonaldehyde intramolecular O-H...O proton transfer). The molecule is
     symmetric under swapping the two oxygens, so dG_rxn is exactly zero by
     symmetry; ``|dG_rxn|`` is therefore a parameter-free validation metric for
     a real MD run. ``build`` uses a symmetric double-Morse model (the proton is
     Morse-bonded to BOTH oxygens and nonbonded-excluded from each) with an
     overlap-matched umbrella window set, so the reactant and product wells are
     equivalent. The barrier is REPORTED but not asserted: without an
     off-diagonal H12 coupling the absolute barrier is parameter dependent, and
     a residual whole-PMF asymmetry remains from the frozen (asymmetric) GAFF
     Lewis structure.

Reference values live in ``scripts/data/evb_reference.csv``.

Typical use:

    # Tier 1 -- runs anywhere, no GPU/AmberTools needed
    python scripts/benchmark_evb.py selftest --out evb_bench

    # Tier 2 -- needs RDKit (+ AmberTools for build, a GPU for run)
    python scripts/benchmark_evb.py build   --out evb_bench
    python scripts/benchmark_evb.py run     --config evb_bench/evb_config.json
    python scripts/benchmark_evb.py analyze --config evb_bench/evb_config.json
"""

import argparse
import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Boltzmann constant in kJ/(mol*K) -- identical to free_energy.KB.
KB = 8.314462618e-3

# --- Analytic constants for the selftest surfaces (kJ/mol, nm units) ---------
# The double-well barrier height MUST match the `double_well,barrier` row of
# evb_reference.csv (that CSV is the source of truth the selftest asserts
# against; these constants define the surface the synthetic data are drawn from).
A_BARRIER = 20.0  # symmetric double-well barrier height (kJ/mol)
B_WELL = 0.04  # double-well minima at rc = +/- B_WELL (nm)
K0_HARMONIC = 8000.0  # curvature of the harmonic reference surface (kJ/mol/nm^2)

DEFAULT_REFERENCE = Path(__file__).parent / 'data' / 'evb_reference.csv'
SELFTEST_KINDS = ('flat', 'harmonic', 'double_well')


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------


def load_reference(path: Path) -> dict[tuple[str, str], dict]:
    """Load the EVB reference table keyed by ``(case, observable)``.

    Args:
        path: Path to the reference CSV (see scripts/data/evb_reference.csv).

    Returns:
        Mapping ``(case, observable) -> {'ref', 'tol', 'units', 'note'}`` where
        ``ref``/``tol`` are floats (``nan`` for reported-only rows).
    """
    lines = [
        ln
        for ln in Path(path).read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith('#')
    ]
    table = {}
    for r in csv.DictReader(lines):
        table[(r['case'], r['observable'])] = {
            'ref': float(r['ref_value']),
            'tol': float(r['tol']),
            'units': r['units'],
            'note': r['note'],
        }
    return table


# ---------------------------------------------------------------------------
# Analytic surfaces + synthetic umbrella sampling (selftest tier)
# ---------------------------------------------------------------------------


def analytic_u0(
    kind: str,
    rc: np.ndarray | float,
    A: float = A_BARRIER,
    b: float = B_WELL,
    K0: float = K0_HARMONIC,
) -> np.ndarray | float:
    """Underlying (unbiased) PMF the synthetic data are drawn from.

    Args:
        kind: One of ``'flat'``, ``'harmonic'``, ``'double_well'``.
        rc: Reaction coordinate value(s) in nm.
        A: Double-well barrier height in kJ/mol.
        b: Double-well minima position (+/- b) in nm.
        K0: Harmonic force constant in kJ/mol/nm^2.

    Returns:
        Free energy (kJ/mol) at ``rc`` for the requested surface.
    """
    if kind == 'flat':
        return np.zeros_like(rc) if isinstance(rc, np.ndarray) else 0.0
    if kind == 'harmonic':
        return 0.5 * K0 * np.asarray(rc) ** 2
    if kind == 'double_well':
        return A * ((np.asarray(rc) / b) ** 2 - 1.0) ** 2
    raise ValueError(f'unknown surface kind: {kind}')


def default_windows(rc_span: float, n_windows: int) -> np.ndarray:
    """Evenly spaced umbrella window centres in ``[-rc_span, +rc_span]`` (nm)."""
    return np.linspace(-rc_span, rc_span, n_windows)


def sample_window(
    u0: Callable[[float], float],
    rc0: float,
    k_umb: float,
    temperature: float,
    n_frames: int,
    burn: int,
    step: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw RC samples for one umbrella window via 1-D Metropolis Monte Carlo.

    Samples the biased distribution
    ``P(rc) ~ exp(-beta * [U0(rc) + 0.5 * k_umb * (rc - rc0)^2])`` where ``U0``
    is the (known) underlying PMF. This is exactly the distribution an EVB
    umbrella window would sample if the true PMF were ``U0``, so feeding these
    samples to ``EVBAnalyzer`` tests whether it reconstructs ``U0``.

    Args:
        u0: Callable returning the underlying PMF (kJ/mol) at a scalar rc.
        rc0: Window centre (nm).
        k_umb: Umbrella force constant (kJ/mol/nm^2).
        temperature: Temperature (K).
        n_frames: Number of production samples to return.
        burn: Number of burn-in steps to discard.
        step: Metropolis proposal std-dev (nm).
        rng: Seeded numpy random generator.

    Returns:
        Array of ``n_frames`` RC samples (nm).
    """
    beta = 1.0 / (KB * temperature)

    def utot(x: float) -> float:
        return u0(x) + 0.5 * k_umb * (x - rc0) ** 2

    x = rc0
    ux = utot(x)
    out = np.empty(n_frames)
    for i in range(burn + n_frames):
        xp = x + rng.normal(0.0, step)
        up = utot(xp)
        if up <= ux or rng.random() < np.exp(-beta * (up - ux)):
            x, ux = xp, up
        if i >= burn:
            out[i - burn] = x
    return out


def generate_umbrella_logs(
    kind: str,
    rc0_values: np.ndarray,
    k_umb: float,
    temperature: float,
    n_frames: int,
    out_dir: Path,
    prefix: str,
    seed: int,
    burn: int = 1000,
    A: float = A_BARRIER,
    b: float = B_WELL,
    K0: float = K0_HARMONIC,
) -> list[np.ndarray]:
    """Write synthetic ``{prefix}_{i}.log`` files EVBAnalyzer can read.

    Each log is a CSV with an ``rc`` column, matching ``RCReporter`` output.

    Returns:
        The list of per-window RC arrays (also written to disk).
    """
    import polars as pl

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma = np.sqrt(KB * temperature / k_umb)
    step = 1.5 * sigma

    rc_data = []
    for i, rc0 in enumerate(rc0_values):
        rng = np.random.default_rng(seed + i)
        samples = sample_window(
            lambda x: float(analytic_u0(kind, x, A=A, b=b, K0=K0)),
            float(rc0),
            k_umb,
            temperature,
            n_frames,
            burn,
            step,
            rng,
        )
        pl.DataFrame({'rc': samples}).write_csv(str(out_dir / f'{prefix}_{i}.log'))
        rc_data.append(samples)
    return rc_data


# ---------------------------------------------------------------------------
# PMF observables (shared by both tiers)
# ---------------------------------------------------------------------------


def pmf_range(pmf: np.ndarray) -> float:
    """Peak-to-trough span of a PMF, ignoring NaN bins (kJ/mol)."""
    valid = pmf[np.isfinite(pmf)]
    return float(np.max(valid) - np.min(valid)) if len(valid) else float('nan')


def extract_barrier_dg(bin_centers: np.ndarray, pmf: np.ndarray) -> dict:
    """Extract barrier and reaction free energy from a double-well PMF.

    The reactant basin is the PMF minimum at rc < 0, the product basin the
    minimum at rc > 0, and the transition state the maximum between them.

    Args:
        bin_centers: RC bin centres (nm), ascending.
        pmf: PMF values (kJ/mol); NaN bins are ignored.

    Returns:
        Dict with ``barrier`` (dG*, kJ/mol), ``dG_rxn`` (kJ/mol) and the
        ``rc_reactant``/``rc_product``/``rc_ts`` locations (nm). Barrier is NaN
        when the PMF is single-sided (only one basin present).
    """
    valid = np.isfinite(pmf)
    bc, p = bin_centers[valid], pmf[valid]
    left, right = bc < 0, bc > 0
    if not left.any() or not right.any():
        return {
            'barrier': float('nan'),
            'dG_rxn': float('nan'),
            'rc_reactant': float('nan'),
            'rc_product': float('nan'),
            'rc_ts': float('nan'),
        }

    ir = np.argmin(p[left])
    ip = np.argmin(p[right])
    rc_r, p_r = float(bc[left][ir]), float(p[left][ir])
    rc_p, p_p = float(bc[right][ip]), float(p[right][ip])

    between = (bc >= rc_r) & (bc <= rc_p)
    its = np.argmax(p[between])
    rc_ts, p_ts = float(bc[between][its]), float(p[between][its])

    return {
        'barrier': p_ts - p_r,
        'dG_rxn': p_p - p_r,
        'rc_reactant': rc_r,
        'rc_product': rc_p,
        'rc_ts': rc_ts,
    }


def pmf_asymmetry(bin_centers: np.ndarray, pmf: np.ndarray, n: int = 40) -> float:
    """RMS of ``PMF(rc) - PMF(-rc)`` over the symmetric RC range (kJ/mol).

    Zero for a perfectly symmetric PMF; used to validate symmetric reactions.
    """
    valid = np.isfinite(pmf)
    bc, p = bin_centers[valid], pmf[valid]
    if len(bc) < 2:
        return float('nan')
    p = p - p.min()
    rmax = min(-bc.min(), bc.max())
    if rmax <= 0:
        return float('nan')
    grid = np.linspace(0.0, rmax, n)
    ppos = np.interp(grid, bc, p)
    pneg = np.interp(-grid, bc, p)
    return float(np.sqrt(np.mean((ppos - pneg) ** 2)))


def pmf_rmse_vs_analytic(
    bin_centers: np.ndarray,
    pmf: np.ndarray,
    kind: str,
    A: float = A_BARRIER,
    b: float = B_WELL,
    K0: float = K0_HARMONIC,
) -> float:
    """RMSE between a recovered PMF and the analytic surface (kJ/mol).

    Both curves are shifted to a zero minimum before comparison so only the
    shape is compared (an overall additive constant is unobservable in a PMF).
    """
    valid = np.isfinite(pmf)
    bc, p = bin_centers[valid], pmf[valid]
    if len(bc) == 0:
        return float('nan')
    p0 = np.asarray(analytic_u0(kind, bc, A=A, b=b, K0=K0), dtype=float)
    p = p - p.min()
    p0 = p0 - p0.min()
    return float(np.sqrt(np.mean((p - p0) ** 2)))


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _check(value: float, ref: dict) -> tuple[bool | None, float]:
    """Compare a value to a reference row. Returns (passed, error).

    ``passed`` is None for reported-only rows (NaN ref/tol).
    """
    if not np.isfinite(ref['ref']) or not np.isfinite(ref['tol']):
        return None, float('nan')
    err = abs(value - ref['ref'])
    return err <= ref['tol'], err


def _print_results(title: str, rows: list[dict]) -> bool:
    """Print a results table; return True if every asserted row passed."""
    print(f'\n{title}')
    print(f'{"case":<16}{"observable":<16}{"value":>9}{"ref":>8}{"tol":>7}   result')
    all_ok = True
    for r in rows:
        passed, _ = _check(r['value'], r['ref'])
        if passed is None:
            result = 'reported'
        elif passed:
            result = 'PASS'
        else:
            result = 'FAIL'
            all_ok = False
        ref_s = (
            f'{r["ref"]["ref"]:>8.2f}' if np.isfinite(r['ref']['ref']) else '     n/a'
        )
        tol_s = (
            f'{r["ref"]["tol"]:>7.2f}' if np.isfinite(r['ref']['tol']) else '    n/a'
        )
        print(
            f'{r["case"]:<16}{r["observable"]:<16}{r["value"]:>9.2f}'
            f'{ref_s}{tol_s}   {result}'
        )
    return all_ok


# ---------------------------------------------------------------------------
# Stage 1: selftest (analytic self-consistency)
# ---------------------------------------------------------------------------


def selftest(args) -> int:
    from molecular_simulations.simulate.free_energy import EVBAnalyzer

    reference = load_reference(args.reference)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rc0_values = default_windows(args.rc_span, args.n_windows)

    rows = []
    recovered = {}  # kind -> (bin_centers, pmf)
    for kind in SELFTEST_KINDS:
        logdir = out / 'selftest' / kind
        generate_umbrella_logs(
            kind,
            rc0_values,
            args.k,
            args.temperature,
            args.n_frames,
            logdir,
            'selftest',
            args.seed,
        )
        analyzer = EVBAnalyzer(
            log_path=logdir,
            log_prefix='selftest',
            k_umbrella=args.k,
            rc0_values=rc0_values,
            output_path=logdir,
        )
        result = analyzer.run_full_analysis(
            temperature=args.temperature, n_bins=args.n_bins
        )
        bc, pmf = result.pmf.bin_centers, result.pmf.pmf
        recovered[kind] = (bc, pmf)

        if kind == 'flat':
            rows.append(
                {
                    'case': kind,
                    'observable': 'pmf_range',
                    'value': pmf_range(pmf),
                    'ref': reference[(kind, 'pmf_range')],
                }
            )
        elif kind == 'harmonic':
            rows.append(
                {
                    'case': kind,
                    'observable': 'pmf_rmse',
                    'value': pmf_rmse_vs_analytic(bc, pmf, kind),
                    'ref': reference[(kind, 'pmf_rmse')],
                }
            )
        elif kind == 'double_well':
            obs = extract_barrier_dg(bc, pmf)
            rows.append(
                {
                    'case': kind,
                    'observable': 'barrier',
                    'value': obs['barrier'],
                    'ref': reference[(kind, 'barrier')],
                }
            )
            rows.append(
                {
                    'case': kind,
                    'observable': 'dG_rxn',
                    'value': obs['dG_rxn'],
                    'ref': reference[(kind, 'dG_rxn')],
                }
            )
            rows.append(
                {
                    'case': kind,
                    'observable': 'pmf_asymmetry',
                    'value': pmf_asymmetry(bc, pmf),
                    'ref': reference[(kind, 'pmf_asymmetry')],
                }
            )

    ok = _print_results('EVB selftest (analytic self-consistency)', rows)
    print(
        f'\n{sum(1 for r in rows if _check(r["value"], r["ref"])[0])}'
        f'/{sum(1 for r in rows if _check(r["value"], r["ref"])[0] is not None)}'
        ' asserted checks passed'
    )

    _dump_json(out / 'selftest_results.json', rows)
    _plot_selftest(recovered, out)
    _parity_plot(
        rows, out / 'selftest_parity.png', 'EVB selftest: computed vs analytic'
    )
    if not ok:
        print('\nSELFTEST FAILED: recovered PMF did not match the analytic surface.')
    return 0 if ok else 1


def _plot_selftest(recovered: dict, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, len(recovered), figsize=(4 * len(recovered), 4))
    if len(recovered) == 1:
        axes = [axes]
    for ax, (kind, (bc, pmf)) in zip(axes, recovered.items(), strict=True):
        valid = np.isfinite(pmf)
        p = pmf.copy()
        if valid.any():
            p = p - np.nanmin(p)
        p0 = np.asarray(analytic_u0(kind, bc), dtype=float)
        p0 = p0 - p0.min()
        ax.plot(bc[valid], p[valid], 'o-', ms=3, label='recovered', color='C0')
        ax.plot(bc, p0, 'k--', lw=1, label='analytic')
        ax.set_title(kind)
        ax.set_xlabel('rc (nm)')
        ax.set_ylabel('PMF (kJ/mol)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'selftest_pmf.png', dpi=130)
    print(f'wrote {out}/selftest_pmf.png')


def _parity_plot(rows: list[dict], out_path: Path, title: str) -> None:
    """Scatter computed vs reference for every asserted observable (kJ/mol).

    The EVB analogue of the constant-pH ``pka_vs_nmr.png``: points on the y = x
    line agree with their reference; distance off the diagonal is the error.
    Rows without a finite reference (reported-only) are skipped.
    """
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pts = [
        (r['ref']['ref'], r['value'], f'{r["case"]}:{r["observable"]}')
        for r in rows
        if np.isfinite(r['ref']['ref']) and np.isfinite(r['value'])
    ]
    if not pts:
        return
    ref = [p[0] for p in pts]
    val = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ref, val, color='C0', zorder=3)
    for x, y, lab in pts:
        ax.annotate(lab, (x, y), fontsize=7, alpha=0.7)
    lims = [min(ref + val) - 1.0, max(ref + val) + 1.0]
    ax.plot(lims, lims, 'k--', lw=1, label='y = x')
    ax.set_xlabel('reference (kJ/mol)')
    ax.set_ylabel('computed (kJ/mol)')
    ax.set_title(title)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f'wrote {out_path}')


def _plot_pmf(bin_centers: np.ndarray, pmf: np.ndarray, out_path: Path) -> None:
    """Plot a recovered PMF with its mirror image to show reaction symmetry."""
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    valid = np.isfinite(pmf)
    bc, p = bin_centers[valid], pmf[valid]
    if len(bc) == 0:
        return
    p = p - p.min()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(bc, p, 'o-', ms=3, color='C0', label='PMF')
    ax.plot(-bc, p, ':', lw=1, color='C3', label='mirror (rc -> -rc)')
    ax.set_xlabel('reaction coordinate (nm)')
    ax.set_ylabel('PMF (kJ/mol)')
    ax.set_title('EVB PMF (symmetric reaction)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f'wrote {out_path}')


def _dump_json(path: Path, rows: list[dict]) -> None:
    payload = [
        {
            'case': r['case'],
            'observable': r['observable'],
            'value': r['value'],
            'ref_value': r['ref']['ref'],
            'tol': r['ref']['tol'],
            'passed': _check(r['value'], r['ref'])[0],
        }
        for r in rows
    ]
    Path(path).write_text(json.dumps(payload, indent=2))
    print(f'wrote {path}')


# ---------------------------------------------------------------------------
# Stage 2: build the symmetric physical reaction (malonaldehyde)
# ---------------------------------------------------------------------------


def build_malonaldehyde_structure(out_dir: Path) -> dict:
    """Embed the malonaldehyde cis-enol and locate the transfer atoms.

    Malonaldehyde's enol (OHC-CH=CH-OH, Z) has a symmetric intramolecular
    O-H...O hydrogen bond. The donor/acceptor are the two oxygens and the
    transferring atom is the hydroxyl hydrogen, giving the difference-of-
    distances reaction coordinate d(O_donor, H) - d(O_acceptor, H).

    Writes an SDF (with explicit bonds, used for parameterization) and a PDB
    (for viewing). The SDF matters: parameterizing from a PDB would make
    antechamber distance-perceive a spurious O...H bond across the short
    H-bond contact and abort, so connectivity must be carried explicitly.

    Args:
        out_dir: Directory to write ``malonaldehyde.sdf`` and
            ``malonaldehyde.pdb`` into.

    Returns:
        Dict with 0-based ``donor``/``acceptor``/``reactive`` atom indices (the
        RDKit ordering, an initial guess -- the authoritative indices are
        re-derived from the built topology by :func:`derive_transfer_indices`),
        plus ``sdf``/``pdb`` paths and the ``acceptor_h_angstrom`` contact.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # cis-enol of malonaldehyde: the hydroxyl H and carbonyl O should form a
    # 6-membered O-H...O ring so the proton can transfer.
    mol = Chem.MolFromSmiles('O=C/C=C\\O')
    mol = Chem.AddHs(mol)

    # Donor O = the hydroxyl oxygen (bonded to H); acceptor O = carbonyl oxygen.
    reactive = donor = acceptor = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'O':
            continue
        h_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'H']
        if h_neighbors:
            donor = atom.GetIdx()
            reactive = h_neighbors[0].GetIdx()
        else:
            acceptor = atom.GetIdx()
    if None in (donor, acceptor, reactive):
        raise RuntimeError('failed to identify O-H...O transfer atoms')

    # Embed several conformers and keep the intramolecular H-bonded one (the
    # conformer that brings the acceptor O closest to the transferring H).
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=25, params=params))
    AllChem.MMFFOptimizeMoleculeConfs(mol)

    def acceptor_h_dist(conf_id: int) -> float:
        conf = mol.GetConformer(conf_id)
        a = conf.GetAtomPosition(acceptor)
        h = conf.GetAtomPosition(reactive)
        return a.Distance(h)

    best = min(cids, key=acceptor_h_dist)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf = out_dir / 'malonaldehyde.sdf'
    pdb = out_dir / 'malonaldehyde.pdb'
    Chem.MolToMolFile(mol, str(sdf), confId=best)
    Chem.MolToPDBFile(mol, str(pdb), confId=best)
    return {
        'donor': donor,
        'acceptor': acceptor,
        'reactive': reactive,
        'sdf': str(sdf),
        'pdb': str(pdb),
        'acceptor_h_angstrom': round(acceptor_h_dist(best), 3),
    }


def solvate_ligand_system(
    out: Path, ligand_file: Path, pad: float = 12.0
) -> tuple[Path, Path]:
    """GAFF2-parameterize a ligand and solvate it into an OPC water box.

    ``LigandBuilder`` (antechamber/parmchk2) generates the ligand ``.mol2``,
    ``.frcmod`` and ``.lib``; a small tleap script then solvates the single
    molecule and writes AMBER ``prmtop``/``inpcrd``. A periodic box is required
    because the EVB run path builds its OpenMM system with PME.

    Args:
        out: Benchmark output directory (a ``build/`` subdir is created).
        ligand_file: Ligand SDF written by
            :func:`build_malonaldehyde_structure` (SDF carries explicit bonds).
        pad: Solvent padding in Angstroms.

    Returns:
        ``(prmtop, inpcrd)`` paths for the solvated system.

    Raises:
        RuntimeError: If AMBERHOME is unset or tleap fails to write outputs.
    """
    import os
    import shutil
    import subprocess

    from molecular_simulations.build import LigandBuilder

    if 'AMBERHOME' not in os.environ:
        raise RuntimeError('AMBERHOME is not set; cannot run antechamber/tleap.')
    tleap = str(Path(os.environ['AMBERHOME']) / 'bin' / 'tleap')

    build_dir = out / 'build'
    build_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ligand_file, build_dir / ligand_file.name)
    stem = ligand_file.stem
    prmtop = build_dir / f'{stem}.prmtop'
    inpcrd = build_dir / f'{stem}.inpcrd'

    cwd = Path.cwd()
    os.chdir(build_dir)  # antechamber writes intermediates to the cwd
    try:
        # LigandBuilder (lig_number=0) writes {stem}.mol2/.frcmod/.lib and names
        # the unit LG0. Instantiate from the .lib off unit -- NOT loadmol2, which
        # re-perceives connectivity and aborts on the short acceptor..H contact.
        LigandBuilder(path=str(build_dir), lig=ligand_file.name).parameterize_ligand()

        leap_in = build_dir / 'solvate.leap'
        leap_in.write_text(
            'source leaprc.gaff2\n'
            'source leaprc.water.opc\n'
            f'loadamberparams {stem}.frcmod\n'
            f'loadoff {stem}.lib\n'
            'mol = LG0\n'
            f'solvateBox mol OPCBOX {pad:.1f}\n'
            f'saveAmberParm mol {prmtop.name} {inpcrd.name}\n'
            'quit\n'
        )
        log = build_dir / 'solvate.log'
        with open(log, 'w') as fh:
            subprocess.run(
                [tleap, '-f', leap_in.name], check=True, stdout=fh, stderr=fh
            )
        if not (prmtop.exists() and inpcrd.exists()):
            raise RuntimeError(f'tleap did not write {prmtop.name}; see {log}')
        return prmtop, inpcrd
    finally:
        os.chdir(cwd)


def derive_transfer_indices(prmtop: Path, inpcrd: Path) -> dict:
    """Read the donor/acceptor/reactive atom indices from the built topology.

    Rather than trust that RDKit/antechamber preserved atom order, locate the
    transfer atoms by connectivity in the final (solvated) topology: within the
    ligand residue, the reactive H is the hydrogen bonded to an oxygen, the
    donor is that oxygen, and the acceptor is the other oxygen. This is robust
    to any atom reordering during parameterization.

    Args:
        prmtop: Solvated AMBER topology.
        inpcrd: Matching coordinates.

    Returns:
        Dict with 0-based ``donor``/``acceptor``/``reactive`` atom indices.

    Raises:
        RuntimeError: If the O-H...O motif cannot be located.
    """
    import MDAnalysis as mda

    u = mda.Universe(
        str(prmtop), str(inpcrd), format='INPCRD', topology_format='PRMTOP'
    )
    lig = u.residues[0].atoms  # ligand (LG0) is the first residue
    oxygens = [a for a in lig if 14.0 < a.mass < 17.0]
    hydrogens = [a for a in lig if a.mass < 2.0]
    for h in hydrogens:
        o_bonded = [b for b in h.bonded_atoms if 14.0 < b.mass < 17.0]
        if o_bonded:
            donor = int(o_bonded[0].index)
            reactive = int(h.index)
            acceptor = int(next(o.index for o in oxygens if o.index != donor))
            return {'donor': donor, 'acceptor': acceptor, 'reactive': reactive}
    raise RuntimeError('could not locate the O-H...O transfer motif in topology')


def build(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atoms = build_malonaldehyde_structure(out)
    print(
        f'wrote {atoms["sdf"]} + {atoms["pdb"]} '
        f'(acceptor..H = {atoms["acceptor_h_angstrom"]} A)'
    )

    # Symmetric reaction coordinate bracketing the transfer. At the built
    # geometry the proton sits on the donor: rc = d(donor,H) - d(acceptor,H)
    # ~ 0.097 nm - acceptor..H. We span +/- ~1.6x that so both the reactant
    # (-) and product (+) basins are well inside the window set, and choose the
    # increment so neighbouring umbrellas overlap at ~1.4 sigma for the default
    # k (sigma = sqrt(kB*T/k)); poor overlap is what makes WHAM/MBAR unstable.
    rc_reactant = 0.097 - atoms['acceptor_h_angstrom'] / 10.0  # nm
    rc_bound = round(1.6 * abs(rc_reactant), 2)
    increment = round(2.0 * rc_bound / (args.n_windows - 1), 4)
    sigma = (KB * args.temperature / args.k) ** 0.5
    print(
        f'reaction coordinate: +/-{rc_bound} nm, {args.n_windows} windows, '
        f'increment {increment} nm (~{increment / sigma:.1f} sigma overlap)'
    )

    config = {
        'system': 'malonaldehyde_proton_transfer',
        'structure': atoms['sdf'],
        'pdb': atoms['pdb'],
        'topology': str(out / 'build' / 'malonaldehyde.prmtop'),
        'coordinates': str(out / 'build' / 'malonaldehyde.inpcrd'),
        # Initial guess (RDKit order); overwritten with topology-derived indices
        # once the system is built.
        'donor_atom': f'index {atoms["donor"]}',
        'acceptor_atom': f'index {atoms["acceptor"]}',
        'reactive_atom': f'index {atoms["reactive"]}',
        'log_prefix': 'malonaldehyde',
        'n_windows': args.n_windows,
        # Symmetric [min, max, increment] window set (nm), overlap-matched to k.
        'reaction_coordinate': [-rc_bound, rc_bound, increment],
        # Symmetric double-Morse: Morse-bond the proton to BOTH donor and
        # acceptor (and nonbonded-exclude the acceptor pair) so a symmetric
        # transfer has symmetric reactant/product wells and dG_rxn ~ 0.
        'second_morse': True,
        # O-H Morse parameters (kJ/mol, nm). D_e ~ O-H BDE; alpha, r0 typical.
        'D_e': 460.0,
        'alpha': 22.0,
        'r0': 0.097,
        'k': args.k,
        'k_path': 100.0,
        # 1 fs timestep: the stiff Morse + umbrella forces are unstable at 2 fs.
        'dt': args.dt,
        'temperature': args.temperature,
    }

    # Parameterize + solvate if AmberTools is available; otherwise leave the
    # structure + config for the user to build themselves.
    try:
        prmtop, inpcrd = solvate_ligand_system(
            out, Path(atoms['sdf']), pad=args.padding
        )
        idx = derive_transfer_indices(prmtop, inpcrd)
        config['topology'] = str(prmtop)
        config['coordinates'] = str(inpcrd)
        config['donor_atom'] = f'index {idx["donor"]}'
        config['acceptor_atom'] = f'index {idx["acceptor"]}'
        config['reactive_atom'] = f'index {idx["reactive"]}'
        print(f'built solvated system: {prmtop}, {inpcrd}')
        print(
            f'transfer atoms (from topology) donor/acceptor/reactive = '
            f'{idx["donor"]}/{idx["acceptor"]}/{idx["reactive"]}'
        )
    except Exception as e:  # build is best-effort without AMBERHOME
        print(f'skipped parameterization/solvation ({type(e).__name__}: {e}).')
        print(
            'Set AMBERHOME and re-run `build`, or supply your own solvated '
            'prmtop/inpcrd and edit topology/coordinates in evb_config.json.'
        )

    (out / 'evb_config.json').write_text(json.dumps(config, indent=2))
    print(f'wrote {out}/evb_config.json')
    return 0


# ---------------------------------------------------------------------------
# Stage 3: run umbrella windows for the physical reaction (needs a GPU)
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def run(args) -> int:
    from molecular_simulations.simulate.free_energy import EVB
    from molecular_simulations.utils.parsl_settings import LocalSettings

    cfg = _load_config(args.config)
    top = Path(cfg['topology'])
    coord = Path(cfg['coordinates'])
    if not top.exists() or not coord.exists():
        print(f'ERROR: topology/coordinates not found ({top}, {coord}).')
        print('Finish `build` (parameterize + solvate) before `run`.')
        return 1

    log_path = Path(args.out)
    log_path.mkdir(parents=True, exist_ok=True)
    # On a plain single GPU box the default ALCF-style MpiExecLauncher fails
    # (stock mpiexec rejects --cpu-bind); --local swaps in a SimpleLauncher and
    # pins one worker per local GPU.
    settings = LocalSettings(
        mpi_launcher=not args.local, available_accelerators=args.n_gpus
    )
    parsl_config = settings.config_factory(str(log_path / 'runinfo'))

    evb = EVB(
        topology=top,
        coordinates=coord,
        donor_atom=cfg['donor_atom'],
        acceptor_atom=cfg['acceptor_atom'],
        reactive_atom=cfg['reactive_atom'],
        parsl_config=parsl_config,
        log_path=log_path,
        log_prefix=cfg['log_prefix'],
        steps=args.steps,
        dt=cfg.get('dt', 0.002),
        k=cfg['k'],
        k_path=cfg['k_path'],
        D_e=cfg['D_e'],
        alpha=cfg['alpha'],
        r0=cfg['r0'],
        n_windows=cfg['n_windows'],
        reaction_coordinate=cfg.get('reaction_coordinate'),
        second_morse=cfg.get('second_morse', False),
        platform=args.platform,
    )
    evb.save_metadata()
    evb.run_evb()
    print(f'EVB windows complete; RC logs in {log_path}')
    return 0


# ---------------------------------------------------------------------------
# Stage 4: analyze the physical run -> dG_rxn vs symmetric reference
# ---------------------------------------------------------------------------


def analyze(args) -> int:
    from molecular_simulations.simulate.free_energy import EVBAnalyzer

    reference = load_reference(args.reference)
    cfg = _load_config(args.config)
    logdir = Path(args.logdir)

    analyzer = EVBAnalyzer.from_metadata(logdir / 'evb_metadata.toml')
    result = analyzer.run_full_analysis(
        temperature=cfg.get('temperature', 300.0), n_bins=args.n_bins
    )
    bc, pmf = result.pmf.bin_centers, result.pmf.pmf
    obs = extract_barrier_dg(bc, pmf)
    asym = pmf_asymmetry(bc, pmf)

    rows = [
        {
            'case': 'proton_transfer',
            'observable': 'dG_rxn',
            'value': obs['dG_rxn'],
            'ref': reference[('proton_transfer', 'dG_rxn')],
        },
        {
            'case': 'proton_transfer',
            'observable': 'barrier',
            'value': obs['barrier'],
            'ref': reference[('proton_transfer', 'barrier')],
        },
    ]
    ok = _print_results('EVB physical benchmark (malonaldehyde transfer)', rows)
    print(f'\nPMF asymmetry (RMS PMF(rc)-PMF(-rc)): {asym:.2f} kJ/mol')
    print(
        'Barrier is reported only (double-Morse PMF, no H12 coupling; not '
        'force-field-free). Residual PMF asymmetry reflects the frozen GAFF '
        'Lewis structure.'
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _dump_json(out / 'physical_results.json', rows)
    _parity_plot(rows, out / 'evb_parity.png', 'EVB physical: computed vs reference')
    _plot_pmf(bc, pmf, out / 'evb_pmf.png')
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest='cmd', required=True)

    common_ref = dict(default=str(DEFAULT_REFERENCE))

    s = sub.add_parser('selftest', help='analytic self-consistency (no GPU)')
    s.add_argument('--out', default='evb_bench')
    s.add_argument('--reference', **common_ref)
    s.add_argument('--n-windows', type=int, default=25, dest='n_windows')
    s.add_argument('--rc-span', type=float, default=0.06, dest='rc_span')
    s.add_argument('--k', type=float, default=100000.0, help='umbrella k (kJ/mol/nm^2)')
    s.add_argument('--n-frames', type=int, default=4000, dest='n_frames')
    s.add_argument('--n-bins', type=int, default=60, dest='n_bins')
    s.add_argument('--temperature', type=float, default=300.0)
    s.add_argument('--seed', type=int, default=2024)
    s.set_defaults(func=selftest)

    b = sub.add_parser('build', help='build the symmetric malonaldehyde system')
    b.add_argument('--out', default='evb_bench')
    b.add_argument('--n-windows', type=int, default=36, dest='n_windows')
    b.add_argument('--k', type=float, default=80000.0, help='umbrella k (kJ/mol/nm^2)')
    b.add_argument('--dt', type=float, default=0.001, help='timestep (ps)')
    b.add_argument('--padding', type=float, default=12.0, help='solvent pad (A)')
    b.add_argument('--temperature', type=float, default=300.0)
    b.set_defaults(func=build)

    r = sub.add_parser('run', help='run EVB umbrella windows (needs a GPU)')
    r.add_argument('--config', default='evb_bench/evb_config.json')
    r.add_argument('--out', default='evb_bench/logs')
    r.add_argument('--steps', type=int, default=250000)
    r.add_argument('--platform', default='CUDA')
    r.add_argument(
        '--local',
        action='store_true',
        help='use a SimpleLauncher (plain single GPU node, no MPI/scheduler)',
    )
    r.add_argument('--n-gpus', type=int, default=4, dest='n_gpus')
    r.set_defaults(func=run)

    a = sub.add_parser('analyze', help='reduce physical run -> dG_rxn vs reference')
    a.add_argument('--config', default='evb_bench/evb_config.json')
    a.add_argument('--logdir', default='evb_bench/logs')
    a.add_argument('--reference', **common_ref)
    a.add_argument('--n-bins', type=int, default=60, dest='n_bins')
    a.add_argument('--out', default='evb_bench')
    a.set_defaults(func=analyze)

    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
