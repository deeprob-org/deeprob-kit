import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'DeeProb-kit'
author = 'The Deep Probabilistic Modeling Organization'
copyright = '2022, {}'.format(author)
release = version = '1.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser'
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# -- Options for Intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ("https://docs.scipy.org/doc/numpy/", None),
    'torch': ("https://pytorch.org/docs/master/", None),
}
intersphinx_disabled_domains = ['std']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'display_version': False,
    'navigation_depth': 6,
    'collapse_navigation': True,
}
html_logo = 'deeprob-logo.svg'
html_only = True

# -- Autodoc settings --------------------------------------------------------
autoclass_content = 'init'
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__call__',
}

# -- MyST settings -----------------------------------------------------------
myst_footnote_transition = False
