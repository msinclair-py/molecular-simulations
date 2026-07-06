molecular-simulations
=====================

.. image:: https://github.com/msinclair-py/molecular-simulations/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/msinclair-py/molecular-simulations/actions
   :alt: CI/CD

.. image:: https://codecov.io/gh/msinclair-py/molecular-simulations/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/msinclair-py/molecular-simulations
   :alt: codecov

.. image:: https://img.shields.io/pypi/v/molecular-simulations
   :target: https://pypi.org/project/molecular-simulations/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10+

A comprehensive Python toolkit for building, running, and analyzing molecular dynamics 
simulations using the AMBER force field ecosystem and OpenMM.

Features
--------

**System Building**
   Build explicitly solvated systems with OPC water, implicit solvent systems, and 
   parameterize small molecules via GAFF2.

**Simulation Engine**
   Run simulations with OpenMM (v8.0+) with GPU acceleration. Supports advanced methods 
   like constant-pH MD and Empirical Valence Bond simulations. Deploy to HPC clusters 
   via Parsl.

**Analysis Tools**
   Automatic clustering with KMeans++, protein-protein interaction analysis, interaction 
   energy fingerprinting, MM-PBSA binding free energy calculations, SASA calculations, 
   and interface scoring (ipTM, ipSAE, pDockQ).

Quick Example
-------------

.. code-block:: python

   from molecular_simulations.build import ExplicitSolvent
   from molecular_simulations.simulate import Simulator
   from molecular_simulations.analysis import Fingerprinter
   from pathlib import Path

   # Build a solvated system
   builder = ExplicitSolvent(Path("./outputs"), Path("protein.pdb"))
   builder.build()

   # Run simulation
   sim = Simulator(builder.out.parent)
   sim.run()

   # Analyze interactions
   fp = Fingerprinter("system.prmtop", trajectory="prod.dcd", target_selection="segid A")
   fp.run()
   fp.save()

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   usage
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/overview
   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   contributing

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
