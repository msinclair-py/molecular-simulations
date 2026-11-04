Installation
============

Requirements
------------

molecular-simulations supports Python 3.10–3.13 and depends on several scientific
computing packages:

* OpenMM ≥ 8.0 (for molecular dynamics simulations)
* MDAnalysis ≥ 2.7 (for trajectory analysis)
* Parsl ≥ 2024.1.29 (for HPC workflow management)
* AmberTools from conda-forge (for system building with tleap; see below)
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

Which AmberTools to Install
---------------------------

AmberTools is not distributed on PyPI. Install ``ambertools`` from
**conda-forge**: it provides everything molecular-simulations calls
(``tleap``, ``antechamber``, ``parmchk2``, ``sqm``, ``pdb4amber``,
``cpptraj``, ``sander``, ``mmpbsa_py_energy``, ``parmed``) and the force fields
it loads, and it is the conda-forge package that
`Amber's install page <https://ambermd.org/GetAmber.php>`_ lists as official.

* **It works with NumPy 2.** molecular-simulations 0.5 and later requires
  ``numpy>=2.0``, and conda-forge's AmberTools allows it (``numpy>=1.23,<3``).
  An AmberTools build that pins ``numpy<2`` cannot share an environment with
  the current package, and pip and pixi do not report that as an error: they
  quietly install molecular-simulations 0.4.x, the last release without the
  NumPy 2 requirement. If ``pip show molecular-simulations`` reports 0.4.x,
  check which NumPy your AmberTools requires.
* **No CUDA or MPI is needed.** The conda-forge build has neither. MD runs on
  OpenMM; AmberTools is only used to build systems and for serial tools, and
  MM-PBSA is parallelized across processes with Parsl.
* **Use Python 3.10–3.13.** ``rust-simulation-tools``, a dependency, has no
  Python 3.14 wheels yet, hence the ``<3.14`` bound below.

pip may warn that ``ndfes``, ``edgembar`` and ``proprep`` require
``numpy<2``. These are free-energy analysis tools bundled inside the
conda-forge AmberTools package; molecular-simulations does not use them, so
the warning is safe to ignore.

If AmberTools is already installed elsewhere (a cluster module, another
environment, a source build), point to it with the ``AMBERHOME`` environment
variable or the ``amberhome=`` argument of the builders instead.

Installing with Conda
---------------------

.. code-block:: console

   $ conda create -n molsim -c conda-forge "python>=3.10,<3.14" ambertools openmm
   $ conda activate molsim
   $ pip install molecular-simulations

``mamba`` is a drop-in replacement for ``conda`` and resolves environments
faster.

Installing with Pixi
--------------------

.. code-block:: console

   $ pixi init molsim
   $ cd molsim
   $ pixi add "python>=3.10,<3.14" ambertools openmm
   $ pixi add --pypi molecular-simulations

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

Check that you got the current release rather than 0.4.x (see
`Which AmberTools to Install`_):

.. code-block:: console

   $ pip show molecular-simulations | grep Version

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
