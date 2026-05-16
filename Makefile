.PHONY: help install smoke all

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  make install  - install package in editable mode with dev deps"
	@echo "  make smoke    - fast reproducibility smoke check (no large data)"
	@echo "  make all      - full article pipeline (requires prepared datasets)"

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .[dev]

smoke:
	$(PYTHON) -m pytest tests/test_memberships.py tests/test_rules.py tests/test_serialization.py -q
	$(PYTHON) -c "import ruanfis; print('ruanfis import ok')"

all:
	@echo "make all requires locally prepared datasets. See DATA_PREPARATION.md"
	$(PYTHON) compact_stable_kafn/scripts/run_budget_sweep.py --selection-method budget_prune
	$(PYTHON) compact_stable_kafn/scripts/collect_tables.py
