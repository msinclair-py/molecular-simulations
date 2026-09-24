Protein-Ligand Systems
======================

This tutorial covers working with systems containing small molecule ligands.

.. note::
   
   This tutorial requires the ``ligand`` optional dependencies:
   
   .. code-block:: console
   
      $ pip install "molecular-simulations[ligand]"

Prerequisites
-------------

* RDKit for molecule handling
* OpenBabel for format conversion
* A protein structure and ligand (SDF/MOL2 format)

Parameterizing Small Molecules
------------------------------

The :class:`~molecular_simulations.build.build_ligand.LigandBuilder` class handles GAFF2 
parameterization:

.. code-block:: python

   from molecular_simulations.build.build_ligand import LigandBuilder
   from pathlib import Path

   ligand_file = Path("ligand.sdf")
   output_dir = Path("./ligand_params")

   builder = LigandBuilder(
       ligand_file=ligand_file,
       output_dir=output_dir,
   )
   builder.build()

   # Outputs:
   # - ligand.mol2 (with charges)
   # - ligand.frcmod (GAFF2 parameters)

Building the Complex
--------------------

Combine the parameterized ligand with your protein:

.. code-block:: python

   from molecular_simulations.build.build_ligand import ComplexBuilder

   builder = ComplexBuilder(
       path=Path("./complex_sim"),                 # Path for output files
       pdb=Path("protein.pdb"),                    # Path to protein input PDB
       ligand_param_prefix=output_dir / "ligand",  # Prefix of .frcmod, .lib files; if None compute params
       lig=output_dir / "ligand.mol2",             # Path to mol2 file
   )
   builder.build()

Running and Analysis
--------------------

Simulation and analysis proceed as with protein-only systems. Interaction energy analysis
is forthcoming, stay tuned!

Common Issues
-------------

**Ligand parameterization fails**
   Check that the ligand has correct protonation state and no unusual 
   functional groups. GAFF2 may not cover all chemistries.
