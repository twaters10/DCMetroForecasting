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

# Derived, gitignored under `data/`, and rebuildable by `mlflow-sync` from the run
# manifests — so it is deliberately not backed up or synced to S3. Artifacts are.
MLFLOW_URI  := sqlite:///data/mlflow/tracking.db

# NOT 5000, MLflow's default: macOS ControlCenter binds it for the AirPlay Receiver, so
# `mlflow ui` dies with "Address already in use" on a stock Mac and the reason is not
# obvious from the error.
MLFLOW_PORT := 5555

.DEFAULT_GOAL := help
.PHONY: help etl features train publish retrain monitor dashboard transfers check register-hint compare registry mlflow mlflow-sync ui install-cron uninstall-cron cron-log

help:
	@echo "make etl        - process outstanding service days, sync to S3"
	@echo "make features   - rebuild the feature and journey tables"
	@echo "make train      - retrain both models and compare them"
	@echo "make publish    - refresh the serving inputs (NEVER skip this)"
	@echo "make retrain    - etl -> features -> train -> publish, failing fast"
	@echo "make monitor    - score unpublished service days, publish metrics"
	@echo "make dashboard  - (re)create the CloudWatch dashboard from code"
	@echo "make transfers  - score composed two-leg predictions on real transfers"
	@echo "make compare    - this run vs the previous ones, on identical rows"
	@echo "make registry   - list the registered versions and how they scored"
	@echo "make mlflow     - browse every run in the MLflow UI (syncs first)"
	@echo "make check      - tests, lint, format"
	@echo "make ui         - launch the local Streamlit app over the endpoint"
	@echo "make install-cron   - schedule the daily ETL locally (replaces CI)"
	@echo "make cron-log       - tail the scheduled run log"

etl:
	./scripts/run_etl_daily.sh

features:
	AWS_PROFILE=$(PROFILE) $(PY) -m src.features.build --start $(START)
	$(PY) -m src.journeys.build

train:
	AWS_PROFILE=$(PROFILE) $(PY) -m src.models.train
	$(PY) -m src.journeys.train
	# The p80 "arrive by" model, retrained WITH the median rather than by hand. It was
	# by hand, and `register` bundles whatever `journey_duration_p80/latest` points at,
	# so the median improved every retrain while the quantile beside it silently aged.
	# Both are user-facing: the app shows them as the trip time and "Budget for".
	$(PY) -m src.journeys.train --quantile
	$(PY) -m src.journeys.compare

# `stations` first: `publish` reads station_index.json and fails loudly if it is absent,
# but a stale one would be accepted silently after a timetable rollover.
publish:
	$(PY) -m src.serving.stations
	# Map geometry for the dashboard. Local only -- it is not packaged into
	# model.tar.gz, because the inference container has no use for coordinates.
	$(PY) -m src.serving.geometry
	AWS_PROFILE=$(PROFILE) $(PY) -m src.serving.publish

retrain: etl features train publish register-hint

# Deliberately NOT part of `retrain`. Promoting a model is a judgement call that has
# already gone the other way once: a retrain moved every absolute number in the wrong
# direction while the ranking held, because the validation window changed. Read
# `journeys.compare` before shipping anything.
register-hint: compare
	@run=$$($(PY) -c "import json,pathlib; print(json.load(open(pathlib.Path('data/models/journey_duration/latest').resolve()/'manifest.json'))['run_id'])")
	@echo ""
	@echo "run $$run is built but NOT registered."
	@echo "review the compare output above, then:"
	@echo "  AWS_PROFILE=$(PROFILE) $(PY) -m src.serving.register"

# Run over run, NOT approach vs approach — `journeys.compare` already does the latter.
# `--rescore` is the whole point: each run graded itself on its own validation window,
# so the recorded numbers are not comparable to each other. This re-scores the last
# three runs on one common set of rows. Three, not all, because every extra run means
# loading another booster over ~677k rows.
compare:
	$(PY) -m src.models.compare_runs --rescore --limit 3

registry:
	AWS_PROFILE=$(PROFILE) $(PY) -m src.models.compare_runs --registry

# Sync before launching, always: the UI reading a store that is missing the run you just
# trained is the one failure mode that would make anyone distrust it. The sync is
# idempotent and skips everything already logged, so this costs seconds.
#
# `mlflow-sync` is separately callable, but there is deliberately no way to launch the
# UI *without* syncing.
mlflow: mlflow-sync
	$(PY) -m mlflow ui --backend-store-uri $(MLFLOW_URI) --port $(MLFLOW_PORT)

mlflow-sync:
	AWS_PROFILE=$(PROFILE) $(PY) -m src.models.mlflow_sync --registry

monitor:
	AWS_PROFILE=$(PROFILE) $(PY) -m src.monitoring.report --catchup

# Recreates the CloudWatch dashboard from src/monitoring/dashboard.py. Safe to re-run:
# `put_dashboard` replaces the document wholesale, so this is also how you undo a
# console edit someone made by hand.
dashboard:
	AWS_PROFILE=$(PROFILE) $(PY) -m src.monitoring.dashboard

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

# The launcher is COPIED outside ~/Documents because macOS TCC will not let
# /bin/bash read a script stored in the repo. Re-run this after editing
# scripts/etl_cron.sh, or the installed copy keeps running the old version.
install-cron:
	./scripts/install_etl_cron.sh

uninstall-cron:
	./scripts/install_etl_cron.sh --uninstall

cron-log:
	tail -n 40 "$$HOME/Library/Logs/metro-pulse/etl-$$(date +%Y-%m).log"
