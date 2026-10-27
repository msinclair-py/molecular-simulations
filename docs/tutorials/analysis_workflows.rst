Analysis Workflows
==================

This tutorial demonstrates comprehensive analysis pipelines for characterizing 
protein-protein or protein-ligand binding interfaces.

Complete Interface Analysis
---------------------------

Combine multiple analysis tools to fully characterize a binding interface:

.. code-block:: python

   from pathlib import Path
   from molecular_simulations.analysis import (
       Fingerprinter,
       PPInteractions,
       SASA,
       RelativeSASA,
   )
   import MDAnalysis as mda

   # Paths
   top = Path("system.prmtop")
   traj = Path("prod.dcd")
   out = Path("analysis_results")
   out.mkdir(exist_ok=True)

   # Load universe for MDAnalysis-based analyses
   u = mda.Universe(str(top), str(traj))

   # AMBER prmtop files carry no chain or segment IDs, so select each
   # partner by residue range (adjust to your system)
   target_sel = "resid 1-85"
   binder_sel = "resid 86-98"

Step 1: Interaction Energy Fingerprinting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identify residues with strong energetic contributions:

.. code-block:: python

   fp = Fingerprinter(
       topology=str(top),
       trajectory=str(traj),
       target_selection=target_sel,
       binder_selection=binder_sel,
   )
   fp.run()
   fp.save(out / "fingerprint.npz")

   # Analyze results
   import numpy as np
   data = np.load(out / "fingerprint.npz")

   # Find residues with strongest interactions
   total_energy = data["electrostatic"] + data["vdw"]
   mean_energy = total_energy.mean(axis=0)  # Average over frames
   hotspots = np.argsort(mean_energy)[:10]  # Top 10 contributors

   print("Top interaction hotspots:")
   for i in hotspots:
       print(f"  Residue {data['residues'][i]}: {mean_energy[i]:.2f} kcal/mol")

Step 2: Contact Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Characterize specific interaction types:

.. code-block:: python

   ppi = PPInteractions(
       top=str(top),
       traj=str(traj),
       out=str(out / "ppi_results.json"),
       sel1=target_sel,
       sel2=binder_sel,
       hbond_cutoff=3.5,
       sb_cutoff=6.0,
       hydrophobic_cutoff=8.0,
   )
   ppi.run()

   # Results saved as JSON with:
   # - Hydrogen bond frequencies
   # - Salt bridge occupancies
   # - Hydrophobic contact probabilities

Step 3: Burial Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Calculate interface burial upon complex formation:

.. code-block:: python

   # SASA of the complex
   complex_sasa = SASA(u, selection="protein")
   complex_sasa.run()

   # For interface residues, compare to isolated chains
   chain_a = u.select_atoms(target_sel)
   chain_b = u.select_atoms(binder_sel)

   # Buried surface area = SASA(A) + SASA(B) - SASA(complex)

Step 4: Relative Solvent Accessibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identify buried vs exposed residues:

.. code-block:: python

   rsasa = RelativeSASA(u, selection=target_sel)
   rsasa.run()

   # Residues with relative SASA < 0.25 are typically buried
   buried = rsasa.results < 0.25

Batch Processing Multiple Systems
---------------------------------

Process multiple trajectories efficiently:

.. code-block:: python

   from natsort import natsorted
   from tqdm import tqdm
   from concurrent.futures import ProcessPoolExecutor

   def analyze_replica(replica_dir):
       """Analyze a single replica."""
       top = replica_dir / "system.prmtop"
       traj = replica_dir / "prod.dcd"
       out = replica_dir / "analysis"
       out.mkdir(exist_ok=True)

       fp = Fingerprinter(
           topology=str(top),
           trajectory=str(traj),
           target_selection=target_sel,
       )
       fp.run()
       fp.save(out / "fingerprint.npz")
       return replica_dir

   # Process replicas in parallel
   replica_dirs = natsorted(Path("./").glob("replica_*"))

   with ProcessPoolExecutor(max_workers=4) as executor:
       results = list(tqdm(
           executor.map(analyze_replica, replica_dirs),
           total=len(replica_dirs),
           desc="Analyzing replicas"
       ))

Aggregating Results
-------------------

Combine results from multiple replicas:

.. code-block:: python

   import numpy as np
   from pathlib import Path

   # Collect fingerprints from all replicas
   fingerprints = []
   for rep in Path("./").glob("replica_*/analysis/fingerprint.npz"):
       data = np.load(rep)
       fingerprints.append(data["electrostatic"] + data["vdw"])

   # Stack and compute statistics
   all_fp = np.concatenate(fingerprints, axis=0)
   mean_fp = all_fp.mean(axis=0)
   std_fp = all_fp.std(axis=0)

   print(f"Analyzed {all_fp.shape[0]} total frames")
   print(f"Mean binding energy: {mean_fp.sum():.2f} ± {std_fp.sum():.2f} kcal/mol")

Visualization
-------------

Create publication-quality figures:

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns

   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # Energy fingerprint heatmap
   ax = axes[0]
   sns.heatmap(
       all_fp.mean(axis=0).reshape(-1, 1),
       cmap="RdBu_r",
       center=0,
       ax=ax,
       cbar_kws={"label": "Interaction Energy (kcal/mol)"}
   )
   ax.set_title("Per-Residue Interaction Energies")

   # Contact frequency bar plot
   ax = axes[1]
   # ... load and plot PPI results

   plt.tight_layout()
   plt.savefig("interface_analysis.png", dpi=300)

Interface Scoring for AlphaFold Models
--------------------------------------

For predicted structures with AlphaFold confidence metrics:

.. code-block:: python

   from molecular_simulations.analysis import ipSAE

   scorer = ipSAE(
       structure_file="alphafold_complex.pdb",
       pae_file="predicted_aligned_error.json",
       plddt_file="plddt_scores.json",
   )
   scores = scorer.run()

   print(f"ipTM: {scores['ipTM']:.3f}")
   print(f"ipSAE: {scores['ipSAE']:.3f}")
   print(f"pDockQ: {scores['pDockQ']:.3f}")
   print(f"pDockQ2: {scores['pDockQ2']:.3f}")

   # Interpretation:
   # pDockQ > 0.5: Likely correct interface
   # pDockQ 0.23-0.5: Possible interface
   # pDockQ < 0.23: Unlikely correct
