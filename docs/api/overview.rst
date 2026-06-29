API Overview
============

molecular-simulations is organized into three main subpackages corresponding to the 
MD workflow: building systems, running simulations, and analyzing results.

Package Structure
-----------------

.. code-block:: text

   molecular_simulations/
   ├── build/          # System preparation
   │   ├── build_amber     # AMBER system building
   │   ├── build_interface # Interface utilities
   │   └── build_ligand    # Small molecule parameterization
   ├── simulate/       # Simulation execution
   │   ├── omm_simulator   # OpenMM simulation runner
   │   ├── mmpbsa          # MM-PBSA calculations
   │   └── parsl_settings  # HPC configuration
   └── analysis/       # Trajectory analysis
       ├── autocluster         # Automatic clustering
       ├── cov_ppi             # Protein-protein interactions
       ├── fingerprinter       # Interaction fingerprinting
       ├── interaction_energy  # Energy calculations
       ├── ipSAE               # Interface scoring
       ├── sasa                # Solvent accessible surface area
       └── utils               # Shared utilities

Building Systems (``molecular_simulations.build``)
--------------------------------------------------

Classes for preparing molecular systems with AMBER force fields.

.. autosummary::
   :nosignatures:

   molecular_simulations.build.ExplicitSolvent
   molecular_simulations.build.ImplicitSolvent
   molecular_simulations.build.LigandBuilder

:class:`~molecular_simulations.build.ExplicitSolvent`
   Build explicitly solvated systems with OPC water and counterions. Supports 
   ff19SB (proteins), OL21 (DNA), and OL3 (RNA) force fields.

:class:`~molecular_simulations.build.ImplicitSolvent`
   Build systems for implicit solvent simulations using GB models.

:class:`~molecular_simulations.build.LigandBuilder`
   Parameterize small molecules with GAFF2. Requires RDKit and OpenBabel.

Running Simulations (``molecular_simulations.simulate``)
--------------------------------------------------------

Classes for executing MD simulations and related calculations.

.. autosummary::
   :nosignatures:

   molecular_simulations.simulate.Simulator
   molecular_simulations.simulate.mmpbsa.MMPBSA
   molecular_simulations.utils.parsl_settings.LocalSettings
   molecular_simulations.utils.parsl_settings.PolarisSettings

:class:`~molecular_simulations.simulate.Simulator`
   Run OpenMM simulations with automatic equilibration (NVT → NPT) and 
   production dynamics. Supports GPU acceleration and hydrogen mass repartitioning.

:class:`~molecular_simulations.simulate.mmpbsa.MMPBSA`
   Calculate MM-PBSA binding free energies with frame-level parallelization.

:class:`~molecular_simulations.utils.parsl_settings.LocalSettings`
   Parsl configuration for local workstations and small clusters.

:class:`~molecular_simulations.utils.parsl_settings.PolarisSettings`
   Parsl configuration for ALCF Polaris supercomputer.

Analyzing Trajectories (``molecular_simulations.analysis``)
-----------------------------------------------------------

Classes for analyzing MD trajectories and structures.

.. autosummary::
   :nosignatures:

   molecular_simulations.analysis.AutoKMeans
   molecular_simulations.analysis.Fingerprinter
   molecular_simulations.analysis.cov_ppi.PPInteractions
   molecular_simulations.analysis.ipSAE
   molecular_simulations.analysis.SASA
   molecular_simulations.analysis.RelativeSASA

:class:`~molecular_simulations.analysis.AutoKMeans`
   Automatic KMeans clustering with elbow method for optimal k. Supports 
   PCA and other dimensionality reduction methods.

:class:`~molecular_simulations.analysis.Fingerprinter`
   Calculate per-residue interaction energy fingerprints (electrostatic + LJ).

:class:`~molecular_simulations.analysis.cov_ppi.PPInteractions`
   Analyze protein-protein interactions using covariance-based contact detection. 
   Identifies hydrogen bonds, salt bridges, and hydrophobic contacts.

:class:`~molecular_simulations.analysis.ipSAE`
   Score protein interfaces using AlphaFold-derived metrics: ipTM, ipSAE, 
   pDockQ, and pDockQ2.

:class:`~molecular_simulations.analysis.SASA`
   Calculate absolute solvent accessible surface area per residue.

:class:`~molecular_simulations.analysis.RelativeSASA`
   Calculate relative SASA normalized by maximum accessible area.

Common Patterns
---------------

Most analysis classes follow a consistent interface:

.. code-block:: python

   # Initialize with input files
   analyzer = AnalysisClass(topology, trajectory, **options)

   # Run the analysis
   analyzer.run()

   # Save results
   analyzer.save()  # or save_labels(), save_centers(), etc.

Selection Language
~~~~~~~~~~~~~~~~~~

Classes that accept selections use MDAnalysis selection syntax:

.. code-block:: python

   # Chain selection
   sel = "chainID A"

   # Segment selection
   sel = "segid PROA"

   # Residue range
   sel = "resid 1-100"

   # Combined selections
   sel = "chainID A and name CA"

See the `MDAnalysis selection documentation 
<https://docs.mdanalysis.org/stable/documentation_pages/selections.html>`_ 
for the full syntax.

Full API Reference
------------------

For complete documentation of all classes and methods, see :doc:`modules`.
