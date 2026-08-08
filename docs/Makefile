# Minimal makefile for Sphinx documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

# Catch-all target: route everything else to sphinx-build.
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
