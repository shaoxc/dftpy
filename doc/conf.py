import sys
import sphinx_rtd_theme

project = 'dftpy'
copyright = '2019, Pavanello Research Group'
author = 'Pavanello Research Group'
release = '0.0.1'

source_suffix = '.rst'
master_doc = 'index'

nbsphinx_execute = 'never'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'nbsphinx']

templates_path = ['templates']
exclude_patterns = ['build']

# html_theme = 'default'
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_favicon = 'static/dftpy.ico'
html_logo = 'static/dftpy.png'
html_style = 'custom.css'
html_static_path = ['static']
html_last_updated_fmt = '%A, %d %b %Y %H:%M:%S'

latex_show_urls = 'inline'
latex_show_pagerefs = True
latex_documents = [('index', not True)]
