Contributing
============

Thank you for your interest in contributing to molecular-simulations! This document 
provides guidelines for contributing code, documentation, and bug reports.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: console

      $ git clone https://github.com/YOUR_USERNAME/molecular-simulations.git
      $ cd molecular-simulations

3. Install development dependencies:

   .. code-block:: console

      $ pip install -e ".[dev]"

4. Create a branch for your changes:

   .. code-block:: console

      $ git checkout -b feature/your-feature-name

Development Workflow
--------------------

Running Tests
~~~~~~~~~~~~~

Run the test suite with pytest:

.. code-block:: console

   $ pytest

For tests requiring specific dependencies:

.. code-block:: console

   $ pytest -m "not requires_openmm"  # Skip OpenMM tests
   $ pytest -m "not requires_amber"   # Skip AmberTools tests

Check coverage:

.. code-block:: console

   $ pytest --cov=molecular_simulations --cov-report=html

Code Style
~~~~~~~~~~

We use standard Python conventions:

* Follow PEP 8 for code style
* Use type hints for function signatures
* Write docstrings in Google format

Example docstring:

.. code-block:: python

   def calculate_energy(atoms, cutoff=10.0):
       """Calculate pairwise interaction energy.

       Args:
           atoms (MDAnalysis.AtomGroup): Atoms to include in calculation.
           cutoff (float): Distance cutoff in Angstroms. Default is 10.0.

       Returns:
           (float): Total interaction energy in kcal/mol.

       Raises:
           ValueError: If atoms contains fewer than 2 atoms.

       Examples:
           >>> u = mda.Universe("system.prmtop", "traj.dcd")
           >>> energy = calculate_energy(u.select_atoms("protein"))
       """

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Build the docs locally:

.. code-block:: console

   $ cd docs
   $ make html

View at ``docs/_build/html/index.html``.

Submitting Changes
------------------

1. Ensure all tests pass
2. Update documentation if needed
3. Add changelog entry if applicable
4. Commit your changes:

   .. code-block:: console

      $ git add .
      $ git commit -m "Add feature X"

5. Push to your fork:

   .. code-block:: console

      $ git push origin feature/your-feature-name

6. Open a Pull Request on GitHub

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

* Provide a clear description of the changes
* Reference any related issues
* Include tests for new functionality
* Update documentation as needed
* Keep commits focused and atomic

Reporting Bugs
--------------

Please report bugs by opening a GitHub issue with:

* A clear, descriptive title
* Steps to reproduce the problem
* Expected vs actual behavior
* System information (Python version, OS, package versions)
* Any relevant error messages or logs

Feature Requests
----------------

Feature requests are welcome! Please open an issue describing:

* The use case for the feature
* How it would benefit users
* Any implementation ideas you have

Questions
---------

For questions about using the package, please:

1. Check the documentation
2. Search existing issues
3. Open a new issue if needed

License
-------

By contributing, you agree that your contributions will be licensed under the 
MIT License.
