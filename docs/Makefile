# Minimal Sphinx Makefile

SPHINXBUILD ?= sphinx-build
SOURCEDIR = .
BUILDDIR = _build

.PHONY: help clean html

help:
	@echo "make html - build HTML docs"

clean:
	rm -rf $(BUILDDIR)

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" -W --keep-going