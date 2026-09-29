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
* AmberTools with ``AMBERHOME`` set
* A ligand in SDF, MOL2, or PDB format

Parameterizing Small Molecules
------------------------------

The :class:`~molecular_simulations.build.build_ligand.LigandBuilder` class handles GAFF2 
parameterization:

.. code-block:: python

   from molecular_simulations.build.build_ligand import LigandBuilder
   from pathlib import Path

   ligand_file = Path("ligand.sdf")

   builder = LigandBuilder(path=ligand_file.parent, lig=ligand_file.name)
   builder.parameterize_ligand()

   # Outputs:
   # - ligand.mol2 (with charges)
   # - ligand.frcmod (GAFF2 parameters)
   # - ligand.lib (tleap library)

Ligand in Solution
------------------

Build a periodic OPC-water box containing only a ligand. The generated
``system.prmtop`` and ``system.inpcrd`` use the standard simulator filenames:

.. code-block:: python

   from molecular_simulations.build import LigandSolutionBuilder
   from molecular_simulations.simulate import Simulator

   builder = LigandSolutionBuilder(
       path=Path("./ligand_solution"),
       lig=Path("ligand.sdf"),
       padding=10.0,
   )
   builder.build()

   simulator = Simulator(path=builder.path)
   simulator.run()

Pass ``lig_param_prefix=Path("./ligand_params/ligand")`` to reuse the
parameter files produced above instead of parameterizing again.

Building the Complex
--------------------

Combine the parameterized ligand with your protein:

.. code-block:: python

   from molecular_simulations.build import ComplexBuilder

   builder = ComplexBuilder(
       path=Path("./complex_sim"),
       pdb=Path("protein.pdb"),
       lig=Path("ligand.sdf"),
       lig_param_prefix=Path("./ligand_params/ligand"),
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
