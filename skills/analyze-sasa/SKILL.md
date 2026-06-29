---
name: analyze-sasa
description: Compute solvent-accessible surface area on a structure or MD trajectory with molecular-simulations — absolute per-residue SASA (Shrake-Rupley) and RelativeSASA fractional exposure (0=buried, 1=exposed). Use when measuring residue burial/exposure, finding solvent-exposed surfaces or buried interfaces, or quantifying conformational changes in accessibility over a trajectory.
---

# SASA analysis

`molecular_simulations.analysis.SASA` and `RelativeSASA` implement the
Shrake-Rupley algorithm as MDAnalysis `AnalysisBase` classes, so they iterate a
trajectory and accept the standard `run(start=, stop=, step=)` slicing. Results
are **per-residue** and **averaged over the analyzed frames**.

## Absolute SASA

```python
import MDAnalysis as mda
from molecular_simulations.analysis import SASA

u = mda.Universe('system.prmtop', 'prod.dcd')
sasa = SASA(u.select_atoms('protein'), probe_radius=1.4, n_points=256)
sasa.run()                       # or run(start=100, stop=500, step=10)
print(sasa.results.sasa)         # ndarray, one value per residue (Å²)
```

Arguments:
- `ag` — an MDAnalysis `AtomGroup` (not a whole Universe).
- `probe_radius=1.4` — water probe radius in Å (standard).
- `n_points=256` — points per atomic sphere; higher = more accurate but slower.

Requirements / gotchas:
- The Universe **must expose an `elements` topology attribute** — atomic radii
  come from MDAnalysis `vdwradii` keyed on element. AMBER `.prmtop` provides
  elements; a bare PDB may not (`ValueError` if missing).
- The AtomGroup must **not** be an `UpdatingAtomGroup` (e.g. avoid
  `select_atoms(..., updating=True)`) — raises `TypeError`.
- SASA is summed per residue using `resid` (1-indexed); make sure your selection
  is contiguous/sensible for the residue indexing you expect.

## Relative SASA (fractional exposure)

`RelativeSASA` adds `results.relative_area` — each residue's SASA divided by its
maximum accessible area, computed in a **tripeptide context** so the amide
linkage and neighbors aren't over-counted. Values run 0 (buried) → 1 (exposed).

```python
from molecular_simulations.analysis import RelativeSASA

u = mda.Universe('system.prmtop', 'prod.dcd')
rsasa = RelativeSASA(u.select_atoms('protein'))
rsasa.run()
print(rsasa.results.sasa)            # absolute SASA (Å²)
print(rsasa.results.relative_area)   # fractional exposure, 0..1
```

Extra requirement: the Universe needs a **`bonds`** attribute (the tripeptide
reference uses a `byres (bonded resindex ...)` selection). AMBER topologies
include bonds. If you loaded a structure without bonds, add them first, e.g.
`u.guess_bonds()` or load alongside a topology that defines connectivity.

## Single structure (no trajectory)

Load just the topology/coordinate pair and `run()` over the single frame:

```python
u = mda.Universe('system.prmtop', 'system.inpcrd')
SASA(u.select_atoms('protein')).run()
```

## Tips
- Subsample long trajectories with `run(step=N)` — SASA is the per-frame
  bottleneck (KDTree over `n_points` per atom).
- To map values back to residues, zip `results.sasa` with
  `ag.residues.resids` / `.resnames`.
- For interface burial, compute SASA of each partner alone vs. the complex and
  take the difference (buried surface area).
- Parallelize across many trajectories with the `parsl-hpc` skill — wrap the
  `SASA(...).run()` call in a CPU `@python_app`.

See also: `analyze-ipsae` (interface scoring of predicted complexes).
