#!/usr/bin/env python
"""Recalibrate constant-pH reference energies from model compounds.

For each titratable residue this builds a capped ``Ace-X-Nme`` model compound in
its fully protonated form (:class:`ConstantPHSolvent`), constructs a
single-residue :class:`ConstantPH`, and runs
:meth:`ReferenceEnergyFinder.findReferenceEnergiesIterative` to find the
reference energy that reproduces the experimental pKa in an independent
titration. Results (converged reference energy, validation midpoint, Hill
coefficient) are written to JSON and printed as a table.

Requires AmberTools (tleap) on PATH or ``AMBERHOME`` set. Run:

    python scripts/calibrate_reference_energies.py --out refs.json
    python scripts/calibrate_reference_energies.py --residues ASP GLU

Reference energies are force-field specific; regenerate them whenever the
implicit-solvent energy function changes (e.g. after the CustomGBForce fix).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# residue -> ([deprotonated, protonated] variants, experimental pKa)
# Cys uses CYX (not the thiolate CYM) as its deprotonated form: OpenMM's
# addHydrogens only supports the CYS/CYX pair, and this matches the convention in
# ConstantPHEnsemble.build_dicts. The reference energy is calibrated on the same
# CYX<->CYS pair, so it stays self-consistent with production titrations.
MODELS = {
    'ASP': (['ASP', 'ASH'], 3.9),
    'GLU': (['GLU', 'GLH'], 4.3),
    'HIS': (['HID', 'HIP'], 6.5),
    'LYS': (['LYN', 'LYS'], 10.5),
    'CYS': (['CYX', 'CYS'], 8.3),
    # TYR requires custom TYD (tyrosinate) template not in standard AMBER/OpenMM
    # 'TYR': (['TYD', 'TYR'], 10.1),
}


def amberhome():
    home = os.environ.get('AMBERHOME')
    if home and (Path(home) / 'bin' / 'tleap').exists():
        return Path(home)
    import shutil

    tleap = shutil.which('tleap')
    if tleap:
        return Path(tleap).resolve().parent.parent
    raise SystemExit('AmberTools (tleap) not found; set AMBERHOME.')


def build_source_pdb(resname, workdir, home):
    """tleap sequence-build a capped Ace-X-Nme peptide."""
    workdir.mkdir(parents=True, exist_ok=True)
    src = (workdir / f'{resname}_source.pdb').resolve()
    leap = workdir / 'seq.in'
    leap.write_text(
        'source leaprc.protein.ff19SB\n'
        f'mol = sequence {{ ACE {resname} NME }}\n'
        f'savepdb mol {src}\n'
        'quit\n'
    )
    subprocess.run(
        [str(home / 'bin' / 'tleap'), '-f', str(leap.resolve())],
        cwd=str(workdir),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return src


def build_model_compound(resname, workdir, home, padding=10.0):
    from molecular_simulations.build.build_amber import ConstantPHSolvent

    src = build_source_pdb(resname, workdir, home)
    out = workdir / 'build'
    ConstantPHSolvent(
        path=str(out), pdb=str(src), padding=padding, amberhome=str(home), debug=True
    ).build()
    return out / 'system.prmtop', out / 'system.inpcrd'


def make_cph(prmtop, inpcrd, variants):
    from openmm import LangevinMiddleIntegrator, Platform
    from openmm.app import PME, CutoffNonPeriodic, HBonds
    from openmm.unit import amu, kelvin, kilojoules_per_mole, nanometers, picosecond

    from molecular_simulations.simulate.constantph.constantph import ConstantPH

    return ConstantPH(
        prmtop_file=str(prmtop),
        inpcrd_file=str(inpcrd),
        pH=[7.0],
        residueVariants={1: variants},
        referenceEnergies={1: [0.0 * kilojoules_per_mole, 0.0 * kilojoules_per_mole]},
        relaxationSteps=100,
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
        platform=Platform.getPlatformByName('CUDA'),
    )


def calibrate(resname, workdir, home, iterations, substeps):
    from openmm.unit import kelvin, kilojoules_per_mole

    from molecular_simulations.simulate.constantph.reference_energy import (
        ReferenceEnergyFinder,
    )

    variants, pKa = MODELS[resname]
    prmtop, inpcrd = build_model_compound(resname, workdir, home)
    cph = make_cph(prmtop, inpcrd, variants)
    finder = ReferenceEnergyFinder(cph, pKa, 300.0 * kelvin)
    history = finder.findReferenceEnergiesIterative(
        iterations=iterations, substeps=substeps
    )
    ref_kJ = cph.titrations[1].referenceEnergies[1].value_in_unit(kilojoules_per_mole)
    _final_ref, final_mid, final_n = history[-1]
    return {
        'residue': resname,
        'variants': variants,
        'pKa_target': pKa,
        'reference_energy_kJ': round(ref_kJ, 3),
        'validation_pKa': round(final_mid, 3),
        'hill_n': round(final_n, 3),
        'rounds': len(history),
        'history': [[round(a, 3), round(b, 3), round(c, 3)] for a, b, c in history],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--residues', nargs='+', default=list(MODELS), choices=list(MODELS))
    ap.add_argument('--iterations', type=int, default=6000)
    ap.add_argument('--substeps', type=int, default=20)
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--out', default='reference_energies.json')
    args = ap.parse_args()

    home = amberhome()
    root = Path(args.workdir) if args.workdir else Path('cph_calibration')
    results = {}
    for res in args.residues:
        print(f'=== calibrating {res} (pKa {MODELS[res][1]}) ===', flush=True)
        results[res] = calibrate(res, root / res, home, args.iterations, args.substeps)
        r = results[res]
        print(
            f'  {res}: ref={r["reference_energy_kJ"]} kJ/mol  '
            f'validation pKa={r["validation_pKa"]} (target {r["pKa_target"]})  '
            f'n={r["hill_n"]}  rounds={r["rounds"]}',
            flush=True,
        )
        Path(args.out).write_text(json.dumps(results, indent=2))  # incremental save

    print('\nresidue  ref_energy(kJ/mol)  validation_pKa  target  hill_n')
    for res, r in results.items():
        print(
            f'  {res:4s}   {r["reference_energy_kJ"]:>10.2f}       '
            f'{r["validation_pKa"]:>6.2f}      {r["pKa_target"]:>4.1f}   {r["hill_n"]:.2f}'
        )
    print(f'\nwritten to {args.out}')


if __name__ == '__main__':
    sys.exit(main())
