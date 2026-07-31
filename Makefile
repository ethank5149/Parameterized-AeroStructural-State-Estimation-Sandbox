PYTHON ?= /config/miniconda3/envs/.conda-venv/bin/python

# ------------------------------------------------------------------ papers
papers: passes-updated.pdf passes-hgv-updated.pdf

passes-updated.pdf: passes-updated.tex passes-references.bib
	latexmk -pdf passes-updated.tex

passes-hgv-updated.pdf: passes-hgv-updated.tex passes-hgv-references.bib
	latexmk -pdf passes-hgv-updated.tex

clean-papers:
	latexmk -C

# -------------------------------------------------------------------- code
.PHONY: papers clean-papers install test lint typecheck verify check

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy

verify:
	$(PYTHON) -m passes.verification --output results

check: lint typecheck test verify
