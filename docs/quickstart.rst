Quickstart
==========

This guide walks through the core workflow: building a molecular system, running a 
simulation, and analyzing the results.

Building a Solvated System
--------------------------

The :class:`~molecular_simulations.build.ExplicitSolvent` class creates AMBER-compatible 
topology and coordinate files from a PDB structure:

.. code-block:: python

   from molecular_simulations.build import ExplicitSolvent
   from pathlib import Path

   pdb_file = Path("/path/to/protein.pdb")
   output_dir = Path("/path/to/simulation/inputs")

   builder = ExplicitSolvent(output_dir, pdb_file)
   builder.build()

   # Outputs:
   # - system.prmtop (topology)
   # - system.inpcrd (coordinates)

The builder uses ff19SB for proteins and OPC water by default. For systems with 
non-standard residues or small molecules, see :doc:`tutorials/ligand_systems`.

Running a Simulation
--------------------

The :class:`~molecular_simulations.simulate.Simulator` class handles equilibration and 
production MD with OpenMM:

.. code-block:: python

   from molecular_simulations.simulate import Simulator

   sim = Simulator(
       path=output_dir,
       equil_steps=500_000,    # 1 ns equilibration (2 fs timestep)
       prod_steps=25_000_000,  # 100 ns production (4 fs timestep)
   )
   sim.run()

   # Outputs:
   # - prod.dcd (trajectory)
   # - prod.log (energies, temperature, etc.)

Running on HPC Clusters
~~~~~~~~~~~~~~~~~~~~~~~

For running multiple replicas on a cluster with PBS either submit a normal job
and call `python script.py` using the LocalSettings or a few prepared Settings
configs are provided for ALCF Polaris and ALCF Aurora:

.. code-block:: python

   import parsl
   from molecular_simulations.simulate import AuroraSettings

   settings = AuroraSettings.from_yaml("parsl_config.yaml")
   config = settings.config_factory("/path/to/run_dir")
   parsl.load(config)

   @parsl.python_app
   def run_md(path, steps=25_000_000):
       from molecular_simulations.simulate import Simulator
       Simulator(path, prod_steps=steps).run()

   # Submit jobs for each replica
   futures = [run_md(p) for p in Path("./").glob("replica_*")]
   results = [f.result() for f in futures]

The above code, if executed on a login node of ALCF Aurora would run each simulation
until completion. This includes resubmitting if walltime is hit until each job is finished.
For more information about Parsl, please read the Parsl Documentation.

Analyzing Trajectories
----------------------

Interaction Energy Fingerprinting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate per-residue electrostatic and van der Waals interaction energies:

.. code-block:: python

   from molecular_simulations.analysis import Fingerprinter

   fp = Fingerprinter(
       topology="/path/to/system.prmtop",
       trajectory="/path/to/prod.dcd",
       target_selection="segid A",      # Receptor
       binder_selection="segid B",      # Ligand/binder
   )
   fp.run()
   fp.save()  # Saves fingerprint.npz

Automatic Clustering
~~~~~~~~~~~~~~~~~~~~

Cluster simulation frames using KMeans++ with automatic k selection:

.. code-block:: python

   from molecular_simulations.analysis import AutoKMeans

   clusterer = AutoKMeans(
       data_directory="/path/to/features/",
       max_clusters=10,
       reduction_algorithm="PCA",
       reduction_kws={"n_components": 2},
   )
   clusterer.run()
   clusterer.save_labels()
   clusterer.save_centers()

The AutoKMeans is a great first-pass clustering and is sometimes sufficient
without relying on testing various clustering methods. If you find that it is
doing a poor job for your data, it may require custom clustering analyses.

Protein-Protein Interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze hydrogen bonds, salt bridges, and hydrophobic contacts:

.. code-block:: python

   from molecular_simulations.analysis import PPInteractions

   ppi = PPInteractions(
       top="/path/to/topology.prmtop",
       traj="/path/to/trajectory.dcd",
       out="/path/to/outputs/ppi_results.json",
       sel1="resid 1-85",  # prmtop has no chain IDs; select by residue
       sel2="resid 86-98",
   )
   ppi.run()

MM-PBSA Binding Free Energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate binding free energies with MM-PBSA:

.. code-block:: python

   from molecular_simulations.simulate.mmpbsa import MMPBSA

   mmpbsa = MMPBSA(
       top="/path/to/system.prmtop",
       dcd="/path/to/prod.dcd",
       selections=[":1-100", ":101-200"],  # receptor, ligand
       n_cpus=4,
       amberhome="/path/to/amber",
       parallel_mode="frame",
   )
   results = mmpbsa.run()

The amberhome kwarg is not necessary but is provided as way for virtual environments,
such as those created with `uv`, can still take advantage of this code by providing a
path to another location which has ambertools installed (which cannot be done via uv/pip).

Interface Scoring
~~~~~~~~~~~~~~~~~

Score interfaces of predicted complexes from pLDDT and PAE:

.. code-block:: python

   from molecular_simulations.analysis import ipSAE

   scorer = ipSAE(
       structure_file="/path/to/complex.cif",
       plddt_file="/path/to/plddt.npz",
       pae_file="/path/to/pae.npz",
   )
   scorer.run()
   print(scorer.scores)  # ipTM, ipSAE, ipSAE2, pDockQ, pDockQ2, LIS

Next Steps
----------

* :doc:`tutorials/index` - Detailed tutorials for specific workflows
* :doc:`api/overview` - Overview of the package architecture
* :doc:`api/modules` - Complete API reference
