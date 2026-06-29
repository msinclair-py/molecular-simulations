---
name: build-systems
description: Build AMBER molecular systems (topology + coordinates) for OpenMM simulations with molecular-simulations. Use when preparing a protein, protein-ligand complex, or implicit-solvent system from a PDB/CIF — solvating, neutralizing/ionizing to 150 mM NaCl, parameterizing small molecules with GAFF2, or assigning chains/disulfides before running MD.
---

# Building molecular systems

The `molecular_simulations.build` module wraps AmberTools (`tleap`, `cpptraj`,
`antechamber`, `parmchk2`) to turn a structure file into the `.prmtop` topology
and `.inpcrd` coordinate files that the `run-simulations` workflow expects.

## Prerequisites

- **AmberTools must be installed and `AMBERHOME` set** in the environment. Every
  builder resolves `tleap`/`cpptraj`/`antechamber` under `$AMBERHOME/bin`. If
  `AMBERHOME` is unset, construction raises `ValueError`. You can also pass
  `amberhome='/path/to/amber'` explicitly.
- Ligand support (`ComplexBuilder`, `LigandBuilder`) needs the optional extras:
  `pip install molecular-simulations[ligand]` (RDKit + OpenBabel). These are
  imported lazily — if missing, only the protein-only builders are available.

## Output convention

Builders write `system.prmtop` and `system.inpcrd` (plus `system.pdb`) into the
output directory, and move all intermediate files into a `build/` subdirectory.
Point the `Simulator` at that directory — it looks for `system.prmtop` /
`system.inpcrd` by default.

## Explicit solvent (most common)

Cubic OPC water box, neutralized, then ionized to ~150 mM NaCl.

```python
from pathlib import Path
from molecular_simulations.build import ExplicitSolvent

builder = ExplicitSolvent(
    path=Path('/path/to/outputs'),   # where system.prmtop/.inpcrd land
    pdb=Path('/path/to/protein.pdb'),
    padding=10.0,                    # Å of water padding on the longest axis
)
builder.build()
```

Useful options:
- `disulfide_residues=[23, 88]` — force those residues to `CYX` and let tleap
  form the bonds (handy for binder design where distance-based detection is
  unwanted). Otherwise `cpptraj prepareforleap` keeps only `existingdisulfides`.
- `glycans`, `rna`, `dna`, `phos_protein`, `mod_protein` — booleans that toggle
  extra `leaprc` force fields (defaults to protein-only ff19SB).
- `polarizable=True` — switch to ff15ipq / SPC-Eb water.
- `debug=True` — write the `tleap.in`/cpptraj inputs to disk instead of a temp
  file so you can inspect them when a build fails.

## Implicit solvent

No water box — produces topology/coords with `mbondi3` radii for GB simulations
(pair with `ImplicitSimulator`). Note: defaults differ from explicit
(`glycans=True` by default here).

```python
from molecular_simulations.build import ImplicitSolvent

builder = ImplicitSolvent(path='/path/to/outputs', pdb='protein.pdb', protein=True)
builder.build()
```

## Protein-ligand complex

`ComplexBuilder` extends `ExplicitSolvent`: it parameterizes the ligand(s) with
GAFF2 (AM1-BCC charges via antechamber/SQM), then combines + solvates.

```python
from molecular_simulations.build import ComplexBuilder

builder = ComplexBuilder(
    path='/path/to/outputs',
    pdb='protein.pdb',
    lig='ligand.sdf',          # or a list: ['lig0.sdf', 'lig1.sdf']
    padding=12.0,
)
builder.build()
```

- Ligand residues are named `LG0`, `LG1`, … in the resulting topology.
- Pass `lig_param_prefix='/path/to/params/ligand'` to reuse pre-computed
  `.frcmod`/`.lib`/`.mol2` files instead of re-running antechamber.
- A `LigandError` is raised if antechamber or SQM fails (check the
  `*_sqm.out` file in `build/` for "Calculation Completed").

To parameterize a ligand on its own, use `LigandBuilder(path, lig, lig_number=0)`
and call `parameterize_ligand()`.

## Helpers (`molecular_simulations.build`)

- `convert_cif_with_gemmi(cif)` / `convert_cif_with_biopython(cif)` — CIF → PDB.
- `add_chains(pdb, first_res, last_res)` — assign chain A/B IDs by residue range
  (writes `*_withchains.pdb`); needed before chain-based MDAnalysis selections in
  analysis.

## Tips

- Box size is estimated from the longest coordinate axis + `2 * padding`; it is
  intentionally approximate but fine for periodic boxes.
- Builds run `tleap`/`cpptraj` with stdout/stderr suppressed. If a system fails
  silently, re-run with `debug=True` and execute the written `tleap.in` by hand
  to see the leap log.
- For HPC fan-out (build many systems in parallel), wrap `builder.build()` in a
  Parsl `python_app` — see the `parsl-hpc` skill.
