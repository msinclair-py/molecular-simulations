Installation
============

Requirements
------------

molecular-simulations requires Python 3.10 or later and depends on several scientific 
computing packages:

* OpenMM ≥ 8.0 (for molecular dynamics simulations)
* MDAnalysis ≥ 2.7 (for trajectory analysis)
* Parsl ≥ 2024.1.29 (for HPC workflow management)
* AmberTools (for system building with tleap)
* NumPy, SciPy, scikit-learn, Polars

Installing from PyPI
--------------------

The simplest way to install molecular-simulations is via pip:

.. code-block:: console

   $ pip install molecular-simulations

This installs the core package with all standard dependencies.

Optional Dependencies
---------------------

**Small Molecule Support**

For parameterizing small molecules with GAFF2, you'll need RDKit and OpenBabel:

.. code-block:: console

   $ pip install "molecular-simulations[ligand]"

**Development Installation**

For development and testing:

.. code-block:: console

   $ pip install "molecular-simulations[dev]"

Installing with Conda
---------------------

For complex dependencies like OpenMM and AmberTools, conda is often easier.
To install on a CUDA-enabled machine:

.. code-block:: console

   $ conda create -n molsim python=3.11
   $ conda activate molsim
   $ conda install -c conda-forge "openmm[cuda]" ambertools
   $ pip install molecular-simulations

Installing from Source
----------------------

To install the latest development version:

.. code-block:: console

   $ git clone https://github.com/msinclair-py/molecular-simulations.git
   $ cd molecular-simulations
   $ pip install -e ".[dev]"

Verifying Installation
----------------------

You can verify your installation by importing the package:

.. code-block:: python

   import molecular_simulations
   from molecular_simulations.build import ExplicitSolvent
   from molecular_simulations.simulate import Simulator
   from molecular_simulations.analysis import Fingerprinter

   print("Installation successful!")

To verify OpenMM GPU support:

.. code-block:: python

   import openmm
   print(openmm.Platform.getNumPlatforms())
   for i in range(openmm.Platform.getNumPlatforms()):
       print(openmm.Platform.getPlatform(i).getName())

Known Issues
------------

* OpenMM versions 8.0-8.1 may exhibit slower integration times for larger systems 
  due to a known bug. Consider using OpenMM ≥ 8.2 if available.
* On some HPC systems, you may need to set ``OMP_NUM_THREADS=1`` to avoid OpenBLAS 
  threading conflicts for parallel MM-PBSA calculations.
