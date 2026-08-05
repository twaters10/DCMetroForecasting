# metro-pulse-forecast

Forecasting WMATA (Washington DC Metro rail and bus) trip duration and congestion
from a self-built historical archive.

WMATA publishes real-time data but no historical archive of actual trip times, so
this project starts by building its own dataset: a collector Lambda polls the
GTFS-realtime feeds every minute and archives raw snapshots to S3, which
accumulate into a training set.

## Status

| Stage | Component | State |
| --- | --- | --- |
| 1 | Collection — `infra/lambda_collector/` | in progress |
| 2 | Processing — local PySpark ETL | not started |
| 3 | Training — LightGBM/XGBoost, local | not started |
| 4 | Registry & serving — SageMaker Serverless Inference | not started |
| 5 | Monitoring — `evidently` drift reports | not started |
| 6 | CI — lint, tests, model-quality gate | not started |

## Quick start (collector)

```bash
cp .env.example .env          # then paste your WMATA API key in
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r infra/lambda_collector/requirements-dev.txt
python infra/lambda_collector/local_run.py
```

Full collector documentation lands in `infra/lambda_collector/` as that stage
completes.
