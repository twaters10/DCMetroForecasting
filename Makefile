# Pipeline entry points.
#
# The chain has a fixed order and one step whose omission is SILENT: skipping `publish`
# leaves the live recent-conditions baselines stale, so `recent_deviation` gets measured
# against out-of-date medians. No error, just quietly worse predictions. That is the
# footgun this file exists to remove.
#
# Every recipe runs as ONE shell with `set -e`, so a failure mid-chain aborts rather than
# carrying on to the next command with a broken input.

.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
SHELL := /bin/bash

PY      := .venv/bin/python
PROFILE := metro-pulse
START   := 2026-08-07

.DEFAULT_GOAL := help
.PHONY: help etl features train publish retrain monitor transfers check register-hint ui

help:
	@echo "make etl        - process outstanding service days, sync to S3"
	@echo "make features   - rebuild the feature and journey tables"
	@echo "make train      - retrain both models and compare them"
	@echo "make publish    - refresh the serving inputs (NEVER skip this)"
	@echo "make retrain    - etl -> features -> train -> publish, failing fast"
	@echo "make monitor    - score yesterday against ground truth, publish metrics"
	@echo "make transfers  - score composed two-leg predictions on real transfers"
	@echo "make check      - tests, lint, format"
	@echo "make ui         - launch the local Streamlit app over the endpoint"

etl:
	./scripts/run_etl_daily.sh

features:
	AWS_PROFILE=$(PROFILE) $(PY) -m src.features.build --start $(START)
	$(PY) -m src.journeys.build

train:
	AWS_PROFILE=$(PROFILE) $(PY) -m src.models.train
	$(PY) -m src.journeys.train
	$(PY) -m src.journeys.compare

# `stations` first: `publish` reads station_index.json and fails loudly if it is absent,
# but a stale one would be accepted silently after a timetable rollover.
publish:
	$(PY) -m src.serving.stations
	AWS_PROFILE=$(PROFILE) $(PY) -m src.serving.publish

retrain: etl features train publish register-hint

# Deliberately NOT part of `retrain`. Promoting a model is a judgement call that has
# already gone the other way once: a retrain moved every absolute number in the wrong
# direction while the ranking held, because the validation window changed. Read
# `journeys.compare` before shipping anything.
register-hint:
	@run=$$($(PY) -c "import json,pathlib; print(json.load(open(pathlib.Path('data/models/journey_duration/latest').resolve()/'manifest.json'))['run_id'])")
	@echo ""
	@echo "run $$run is built but NOT registered."
	@echo "review the compare output above, then:"
	@echo "  AWS_PROFILE=$(PROFILE) $(PY) -m src.serving.register"

monitor:
	AWS_PROFILE=$(PROFILE) $(PY) -m src.monitoring.report

# Local only, and no AWS: it reads the journey table and the model straight from
# disk. Run it after `train`, because it scores whatever `latest` points at.
transfers:
	$(PY) -m src.journeys.transfers \
		--output data/models/journey_duration/latest/transfer_validation.json

check:
	JAVA_HOME=$${JAVA_HOME:-/opt/homebrew/opt/openjdk@17} $(PY) -m pytest tests/ -q
	$(PY) -m ruff check src/ tests/ infra/
	$(PY) -m black --check src/ tests/ infra/

# Local, not hosted: a SageMaker endpoint requires SigV4 and a static page cannot sign a
# request. boto3 signs from here, so no API Gateway or proxy Lambda is needed and the
# project's "only managed compute" claim stays true.
ui:
	AWS_PROFILE=$(PROFILE) .venv/bin/streamlit run src/ui/app.py
