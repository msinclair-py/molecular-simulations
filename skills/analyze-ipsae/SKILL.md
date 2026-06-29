---
name: analyze-ipsae
description: Score predicted protein complex interfaces with molecular-simulations' ipSAE — computes ipSAE, ipTM, pDockQ, pDockQ2, and LIS per chain pair from a predicted structure plus its pLDDT/PAE confidence arrays. Use when ranking or filtering AlphaFold-Multimer / Boltz / Chai docking predictions by interface confidence, or batch-scoring many predicted models.
---

# ipSAE interface scoring

`molecular_simulations.analysis.ipSAE` scores a predicted multi-chain structure
using the predictor's confidence outputs. For each chain pair it computes pDockQ,
pDockQ2, LIS, ipTM, and ipSAE, returns a Polars DataFrame, and writes
`ipSAE_scores.parquet`.

## Usage

```python
from molecular_simulations.analysis import ipSAE

scorer = ipSAE(
    structure_file='model.pdb',     # predicted complex; .pdb or .cif
    plddt_file='plddt.npz',         # numpy .npz, key 'plddt'
    pae_file='pae.npz',             # numpy .npz, key 'pae'  (N x N)
    out_path='scores/',             # optional; defaults to plddt_file's parent
    skip_chains={'C'},              # optional; exclude ligand/glycan chains
)
scorer.run()
print(scorer.scores)                # per-chain-pair Polars DataFrame
# also persisted to <out_path>/ipSAE_scores.parquet
```

## Inputs

- **`structure_file`** — the predicted model (`.pdb`/`.cif`). Chains are parsed
  and classified (protein vs. nucleic) internally.
- **`plddt_file`** — `.npz` containing a `plddt` array (length = number of
  tokens/residues). Values are auto-scaled: if they look like 0–1 fractions they
  are multiplied by 100, so either scale works.
- **`pae_file`** — `.npz` containing a `pae` array, the residue×residue predicted
  aligned error matrix (N×N).
- **`out_path`** — output directory (created if absent). Defaults to the parent
  of `plddt_file`.
- **`skip_chains`** — set/list of chain IDs to drop before scoring (e.g. ligand
  or glycan chains). Their pLDDT/PAE tokens are inferred from the remaining
  tokens and removed so the matrices stay aligned to the scored chains.

Token-count alignment matters: the `plddt` length and `pae` dimensions must
correspond to the structure's residues (after `skip_chains` removal). Mismatches
produce wrong scores or indexing errors — confirm your predictor exports pLDDT
and PAE in the same residue/token order as the structure file.

## Output

`scorer.scores` is a Polars DataFrame with one row per ordered chain pair plus a
max-value summary (ipSAE is asymmetric, so A→B and B→A both appear). Persist it
yourself with `scorer.scores.write_parquet(...)`, or read the auto-written
`ipSAE_scores.parquet`.

Interpreting the metrics (higher = more confident interface):
- **ipSAE** — interface score from aligned errors (the headline metric here).
- **ipTM** — interface predicted TM-score.
- **pDockQ / pDockQ2** — predicted DockQ-style interface quality.
- **LIS** — local interaction score.

## Batch scoring

Loop over prediction directories and collect the parquet outputs:

```python
from pathlib import Path
from molecular_simulations.analysis import ipSAE

for d in Path('/path/to/predictions').glob('model_*'):
    ipSAE(d / 'model.pdb', d / 'plddt.npz', d / 'pae.npz', out_path=d).run()
```

For many models, wrap `ipSAE(...).run()` in a CPU `@python_app` and fan out with
the `parsl-hpc` skill, then concatenate the per-model parquet files with Polars
for ranking/filtering.

See also: `analyze-sasa` (per-residue solvent accessibility / interface burial).
