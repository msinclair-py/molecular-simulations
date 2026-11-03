---
name: molsim-analyze-ipsae
description: Score interfaces of predicted complexes (Chai-1, Boltz, AlphaFold3) with molecular-simulations' ipSAE module — ipSAE2 (headline; also scores ligands, glycans and ions) plus the legacy ipSAE, ipTM, pDockQ, pDockQ2 and LIS — per chain pair from the predicted structure and its PAE/pLDDT arrays. Use when ranking or filtering predicted complexes or binder designs by interface confidence, or batch-scoring many models.
---

# ipSAE2 / ipSAE interface scoring

`molecular_simulations.analysis.ipSAE` scores every chain pair of a predicted
structure from the predictor's PAE (and pLDDT) and returns one Polars row per
pair, also written to `<out_path>/ipSAE_scores.parquet`.

**ipSAE2 is the headline score.** It is a native port of
[DunbrackLab/IPSAE2](https://github.com/DunbrackLab/IPSAE2) and scores
proteins, nucleic acids, ligands, glycans and ions. The legacy ipSAE, ipTM,
pDockQ, pDockQ2 and LIS are kept for comparison with older campaigns; they
only cover protein and nucleic acid chains.

## Usage

```python
from molecular_simulations.analysis.ipSAE import ipSAE

scorer = ipSAE(
    structure_file='pred.model_idx_0.cif',  # .cif or .pdb
    plddt_file='scores.model_idx_0.npz',    # .npz with 'plddt'; None for ipSAE2 only
    pae_file='scores.model_idx_0.npz',      # .npz with 'pae' (N x N tokens)
    out_path='ipsae/',                      # optional; defaults to the npz folder
)
scorer.run()
print(scorer.scores.select('chain1', 'chain2', 'type1', 'type2', 'ipSAE2', 'ipSAE'))
```

- **Score selection:** `scores=` takes any of `'ipsae2'`, `'ipsae'`, `'pdockq'`,
  `'pdockq2'`, `'lis'` (case-insensitive); the default is all of them.
  `scores=['ipsae2']` with `plddt_file=None` needs only the PAE.
- **One file for both arrays:** Chai-1 (`scores.model_idx_{i}.npz` with
  full data) and others store `pae` and `plddt` together; pass the same path
  twice. Boltz writes `pae_*.npz` and `plddt_*.npz` separately.
- **ipSAE2 options:** `ipsae2_kwargs={'pae_cutoff': 10, 'contact_cutoff': 5,
  'all_pairs': False, ...}`.
- **`skip_chains`:** chain IDs left out of every score. Without `'ipsae2'` in
  `scores`, ligand, glycan and ion chains must be listed here; with it, they
  are dropped from the legacy scores automatically.

## How tokens are matched

Predictors tokenize ligands and modified residues (e.g. phosphoserine) per
atom, so the PAE is not one row per residue. The module rebuilds the
predictor's token order from the structure and checks it against the PAE
size; tokens Chai-1 keeps without coordinates (pLDDT exactly 0, e.g. glycan
leaving atoms) are dropped. If no token order fits, it raises
`TokenMismatchError` rather than guessing — check that the structure and the
npz come from the same model.

## Output

One row per unordered chain pair, ordered by molecule type (prot, nuc, lig,
ion) and then file order:

- `chain1`, `chain2`, `type1`, `type2` (`prot`, `nuc`, `lig` or `ion`).
- `ipSAE2` and its directional detail: `ipSAE2_12` aligns on `chain1` and
  scores `chain2`, `ipSAE2_21` the reverse, plus `ipSAE2_iPAE_*`,
  `ipSAE2_n0_*`, `ipSAE2_d0_*`, `ipSAE2_n_contact_*`, `ipSAE2_align_token_*`.
  `ipSAE2_source` says which direction `ipSAE2` took: the larger one for
  same-type pairs, otherwise the one aligned on the more complex molecule.
- Legacy `pDockQ`, `pDockQ2`, `LIS`, `ipTM`, `ipSAE` (null for pairs involving
  a ligand, glycan or ion). They report the direction with the larger ipSAE,
  recorded in `ipSAE_source` (`12` or `21`).

## Interpreting

- Rank by **ipSAE2** of the pair you care about (higher is better).
- ipSAE2 only scores pairs with a heavy-atom contact within 5 Å: a pair
  missing from ipSAE2 means no interface, not missing data.
- Never mix ipSAE and ipSAE2 in one ranking; they are on different scales.
- Look a pair up in either order:
  ```python
  import polars as pl
  pair = scorer.scores.filter(
      pl.concat_list('chain1', 'chain2').list.sort() == sorted(['A', 'B'])
  )
  ```

## Batch scoring

```python
from pathlib import Path
import polars as pl
from molecular_simulations.analysis.ipSAE import ipSAE

rows = []
for cif in Path('outputs').glob('*/pred.model_idx_*.cif'):
    npz = cif.with_name(cif.name.replace('pred.', 'scores.')).with_suffix('.npz')
    scorer = ipSAE(cif, npz, npz, out_path=cif.parent / f'ipsae_{cif.stem}')
    scorer.run()
    rows.append(scorer.scores.with_columns(pl.lit(str(cif)).alias('model')))
scores = pl.concat(rows, how='diagonal')
```

Scoring is CPU-only and takes well under a second per model. For thousands of
models, wrap `ipSAE(...).run()` in a Parsl `python_app` with the
`molsim-parsl-hpc` skill.

See also: `molsim-analyze-sasa` (per-residue solvent accessibility and
interface burial).
