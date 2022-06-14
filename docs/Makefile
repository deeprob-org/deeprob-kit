# Minimal makefile for Sphinx documentation
#
# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS   ?=
SPHINXBUILD  ?= sphinx-build
SPHINXAPIDOC ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
APIDOCDIR     = apidoc
PYLIBDIR      = ../deeprob

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Generate HTML documentation
html: apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Generate API docs
apidoc:
	@$(SPHINXAPIDOC) -o "$(SOURCEDIR)/$(APIDOCDIR)" "$(PYLIBDIR)"


.PHONY: help clean Makefile

# Clean all
clean:
	rm -rf "$(SOURCEDIR)/$(APIDOCDIR)"
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	echo $@
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
