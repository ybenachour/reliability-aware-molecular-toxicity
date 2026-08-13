SHELL := /bin/bash
PYTHON ?= python
NOTEBOOKS := $(sort $(wildcard notebooks/*.ipynb))

.PHONY: install test lint format notebooks notebooks-smoke clean provenance verify-sources synthetic-smoke manifest

install:
	$(PYTHON) -m pip install -e ".[ml,deep,data,dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

notebooks:
	TOX_SCREEN_PROFILE=$${TOX_SCREEN_PROFILE:-full} bash scripts/run_notebooks.sh

notebooks-smoke:
	TOX_SCREEN_PROFILE=smoke bash scripts/run_notebooks.sh

provenance:
	$(PYTHON) -c "from toxicity_screening.utils import environment_snapshot; environment_snapshot('reports/environment_snapshot.json')"

clean:
	find results figures tables reports models -type f ! -name '.gitkeep' -delete
	find data/interim data/processed -type f ! -name '.gitkeep' -delete

verify-sources:
	PYTHONPATH=src $(PYTHON) scripts/verify_sources.py

synthetic-smoke:
	PYTHONPATH=src $(PYTHON) scripts/synthetic_smoke_test.py

manifest:
	PYTHONPATH=src $(PYTHON) scripts/build_manifest.py
