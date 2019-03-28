# -*- coding: utf-8 -*-
#
# sknet documentation build configuration file, created by sphinx-quickstart
#

import os
import sys
sys.path.insert(0, os.path.abspath('../../../'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.linkcode',  # link to github, see linkcode_resolve() below
    'numpydoc',
]

# See https://github.com/rtfd/readthedocs.org/issues/283
mathjax_path = ('https://cdn.mathjax.org/mathjax/latest/MathJax.js?'
                'config=TeX-AMS-MML_HTMLorMML')

# see http://stackoverflow.com/q/12206334/562769
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Sknet'
copyright = u'2019, Sknet contributors'
author = 'Randall Balestriero'


import sknet
import tensorflow
version = tensorflow.__version__
release = sknet.__version__


exclude_patterns = ['build']

# Resolve function for the linkcode extension.
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        import inspect
        import os
        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(lasagne.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != 'py' or not info['module']:
        return None
    try:
        filename = 'sknet/%s#L%d-L%d' % find_source()
    except Exception:
        filename = info['module'].replace('.', '/') + '.py'
    tag = 'master' if 'dev' in release else ('v' + release)
    return "https://github.com/RandallBalestriero/Sknet/%s/%s" % (tag, filename)

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']



language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

autodoc_default_options = {
    'autodoc_member_order': 'groupwise'
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'Sknetdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'Sknet.tex', 'Sknet Documentation',
     'Randall Balestriero', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'Sknet', 'Sknet Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Sknet', 'Sknet Documentation',
     author, 'Sknet', 'One line description of project.',
     'Miscellaneous'),
]


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
