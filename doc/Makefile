# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PYTHON = python3

# Put it first so that "make" without argument is like "make help".
#
html: config
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(SPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

config:
	@echo "Base on configentries.json generate config.rst"
	$(PYTHON) gen_doc.py


.PHONY: help Makefile html browse

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

browse:
	x-www-browser build/html/index.html

upload: html
	scp -r ./build/html/* web@dftpy.rutgers.edu:/home/web/dftpy/html/
