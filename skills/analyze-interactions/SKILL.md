---
name: analyze-interactions
description: Analyze protein-protein/ligand interactions from MD trajectories with molecular-simulations — per-residue interaction-energy fingerprinting (electrostatic + Lennard-Jones), linear interaction energy (static structure or dynamic trajectory), and KMeans clustering of per-frame feature data. Use when characterizing binding interfaces, footprinting which residues drive an interaction, computing chain-chain interaction energies, or clustering conformations/fingerprints.
---

# Interaction & clustering analyses

Three tools in `molecular_simulations.analysis` for characterizing how parts of a
system interact and for grouping the resulting per-frame data:

- **`Fingerprinter`** — per-residue interaction-energy fingerprints.
- **`StaticInteractionEnergy` / `DynamicInteractionEnergy`** — linear interaction
  energy between a chain and the rest of the system.
- **`AutoKMeans`** — automatic KMeans++ clustering (with PCA) of `.npy` features.

For surface accessibility see `analyze-sasa`; for predicted-complex interface
scoring see `analyze-ipsae`.

## Fingerprinter — per-residue interaction energy

Computes per-residue Lennard-Jones and electrostatic interaction energy between a
**target** selection and a **binder** selection across every trajectory frame.
Reads nonbonded parameters (charge/sigma/epsilon) straight from an AMBER `.prmtop`.

```python
from molecular_simulations.analysis import Fingerprinter

fp = Fingerprinter(
    'complex.prmtop',
    trajectory='prod.dcd',          # if None: looks for <stem>.inpcrd then .rst7
    target_selection='segid A',     # MDAnalysis selection language
    binder_selection=None,          # default: 'not <target_selection>'
)
fp.run()
fp.save()                           # -> fingerprint.npz (next to topology)
```

Outputs (`np.savez` to `fingerprint.npz`):
- `target` — array `(n_frames, n_target_residues, 2)`
- `binder` — array `(n_frames, n_binder_residues, 2)`

The last axis is `[LJ, electrostatic]` energy per residue per frame. Average over
axis 0 to get a per-residue footprint of which residues drive the interaction.

Notes:
- Requires an AMBER `.prmtop` (an OpenMM `System` is built to read nonbonded
  params). A PDB-only topology won't have force-field parameters.
- `out_path` / `out_name` override the default output location/filename.
- Add chain IDs/segids first with `molecular_simulations.build.add_chains` if your
  structure lacks them, or use a residue-based selection like `'resid 1 to 100'`.

## Linear interaction energy

Splits the nonbonded energy of a selected chain vs. everything else into
Lennard-Jones and Coulomb components using an OpenMM implicit-solvent (GBn2)
system.

### Static — single structure

```python
from molecular_simulations.analysis import StaticInteractionEnergy

ie = StaticInteractionEnergy(
    'complex.pdb',
    chain='B',                      # energy between chain B and the rest
    platform='CUDA',                # 'CUDA' | 'CPU' | 'OpenCL'
    first_residue=None,             # optional resid range within the chain
    last_residue=None,
)
ie.compute()
print(ie.lj, ie.coulomb)            # kcal/mol
```

Use whitespace `chain=' '` if the PDB has no chain IDs.

### Dynamic — over a trajectory

```python
from molecular_simulations.analysis import DynamicInteractionEnergy

die = DynamicInteractionEnergy(
    'system.prmtop',
    'prod.dcd',
    stride=1,                       # subsample frames
    chain='A',
    platform='CUDA',
    progress_bar=True,
)
die.compute_energies()
print(die.energies.shape)           # (n_frames, 2) -> columns [LJ, Coulomb]
```

## AutoKMeans — clustering per-frame features

Auto-selects cluster count (silhouette sweep up to `max_clusters`), reduces
dimensionality (PCA by default), and assigns each frame to a cluster. It loads a
**directory of `.npy` files** (one per system/replica), so first save your
features (e.g. averaged fingerprints, distances, CVs) as `.npy`.

```python
from molecular_simulations.analysis import AutoKMeans

clusterer = AutoKMeans(
    data_directory='features/',     # globs '<pattern>*.npy'
    pattern='',                     # filename prefix filter, optional
    max_clusters=10,
    stride=1,                       # step size of the cluster-count sweep
    reduction_algorithm='PCA',
    reduction_kws={'n_components': 2},
)
clusterer.run()
clusterer.save_labels()             # -> cluster_assignments.parquet (system, frame, cluster)
clusterer.save_centers()            # -> cluster_centers.json ({cluster: (replica, frame)})
```

After `run()`: `clusterer.labels` (per-frame assignments), `clusterer.reduced`
(reduced coords), `clusterer.centers`, and `clusterer.cluster_centers` (maps each
cluster to the representative `(replica, frame)`).

Dataloaders (pass as `dataloader=`):
- `GenericDataloader` (default) — stacks `.npy` arrays as-is.
- `PeriodicDataloader` — for periodic features (e.g. dihedral angles); encodes
  periodicity before clustering.

Note: `Fingerprinter.save()` writes an **`.npz`**, whereas `AutoKMeans` reads
**`.npy`**. To cluster fingerprints, extract the array you care about (e.g.
per-residue mean LJ+ES) and `np.save` it per system into the features directory.

## Scaling out
Each of these is per-system/per-trajectory work. To process many trajectories,
wrap the `.run()` / `.compute_energies()` call in a `@python_app` (GPU for the
OpenMM-based interaction energies, CPU for fingerprinting/clustering) and fan out
with the `parsl-hpc` skill, then aggregate the outputs.
