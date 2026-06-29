# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'molecular-simulations'
copyright = '2025, Matt Sinclair'
author = 'Matt Sinclair'
try:
    release = _pkg_version('molecular_simulations')
except PackageNotFoundError:
    release = '0.0.0'
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',      # Add source code links
    'sphinx.ext.todo',          # Support TODO notes
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',        # Copy button for code blocks (add to deps)
    'sphinx_wagtail_theme',
]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',  # Keep source order
    'special-members': '__init__',
    'exclude-members': '__weakref__',
}
autoclass_content = 'both'  # Include __init__ docstring
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Napoleon settings (for NumPy/Google style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# Mock only optional deps not installed during the docs build. The docs CI job
# installs .[docs], which pulls in all core runtime deps, so only the `ligand`
# extras (openbabel, rdkit) are absent. Mocking installed packages such as numpy
# breaks runtime-evaluated annotations (e.g. ``np.ndarray | None``) and makes
# autosummary report spurious import failures.
autodoc_mock_imports = [
    'openbabel',
    'rdkit',
    'calvados',
]

# Templates
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_wagtail_theme'
html_show_sphinx = False
html_static_path = ['_static']

html_theme_options = {
    'project_name': 'molecular-simulations',
    'header_links': 'Documentation|/index',
}

# Custom CSS (create _static/custom.css if needed)
# html_css_files = ['custom.css']

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'MDAnalysis': ('https://docs.mdanalysis.org/stable/', None),
    'parsl': ('https://parsl.readthedocs.io/en/stable/', None),
    'polars': ('https://docs.pola.rs/api/python/stable/', None),
}

# -- Extension configuration -------------------------------------------------

# sphinx_copybutton settings
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True

# TODO extension
todo_include_todos = True

# -- Suppress specific warnings ----------------------------------------------
suppress_warnings = ['autodoc.import']
