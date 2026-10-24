PYTHON ?= python3

.PHONY: test reproduce-smash reproduce-paper
test:
	$(PYTHON) -m pytest -q tests

# Compact canonical tables; raw prediction scoring is a separate supplied-data step.
reproduce-smash:
	$(PYTHON) scripts/export_smash_results.py

reproduce-paper:
	$(MAKE) -C ../HGNDPaper PYTHON="$(PYTHON)"
