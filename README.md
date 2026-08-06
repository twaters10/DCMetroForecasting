# metro-pulse-forecast

Forecasting WMATA (Washington DC Metro rail and bus) trip duration and congestion
from a self-built historical archive.

WMATA publishes real-time data but no historical archive of actual trip times, so
this project starts by building its own dataset: a collector Lambda polls the
GTFS-realtime feeds every minute and archives raw snapshots to S3, which
accumulate into a training set. The same function archives the static GTFS
timetable daily — WMATA serves only the current bundle, so the scheduled half of
every actual-vs-scheduled delta has to be captured as it goes by. See
[`docs/static-gtfs.md`](docs/static-gtfs.md).

## Status

| Stage | Component | State |
| --- | --- | --- |
| 1 | Collection — `infra/lambda_collector/` (realtime + static GTFS) | in progress |
| 2 | Processing — local PySpark ETL (`src/etl/`) | rail pipeline running; DQ report, tests and docs outstanding |
| 3 | Training — LightGBM/XGBoost, local | not started |
| 4 | Registry & serving — SageMaker Serverless Inference | not started |
| 5 | Monitoring — `evidently` drift reports | not started |
| 6 | CI — lint, tests, model-quality gate | not started |

## Quick start (collector)

```bash
cp .env.example .env          # then paste your WMATA API key in
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r infra/lambda_collector/requirements-dev.txt
pip install -r requirements-dev.txt        # test + ETL tooling

python infra/lambda_collector/local_run.py                    # realtime feeds
python infra/lambda_collector/local_run.py --task static-gtfs  # static GTFS
```

Both write to `LOCAL_OUTPUT_DIR` and need no AWS credentials.

AWS work uses the `metro-pulse` profile — the `default` profile in this account is
stale and will fail with `InvalidClientTokenId`:

```bash
export AWS_PROFILE=metro-pulse
```

Full collector documentation lands in `infra/lambda_collector/` as that stage
completes.

## Quick start (ETL)

PySpark needs a JVM. The ETL locates a Homebrew JDK automatically and otherwise
fails with install instructions rather than a stack trace:

```bash
brew install openjdk@17
export JAVA_HOME=/opt/homebrew/opt/openjdk@17   # only if auto-detection misses
```

Explore what a window of the archive actually contains — run this before trusting
any derived field:

```bash
python -m src.etl.explore --mode rail --start 2026-08-05T11 --end 2026-08-05T14
```

Build the trip-segment table. Takes a UTC hour range or one local service day, and
re-running a range replaces only those `service_date` partitions:

```bash
python -m src.etl.pipeline --mode rail --date 2026-08-05
python -m src.etl.pipeline --mode rail --start 2026-08-05T11 --end 2026-08-05T14
python -m src.etl.pipeline --mode rail --date 2026-08-05 --skip-decode  # iterate fast
```

`--skip-decode` reuses the staged observations instead of re-reading S3, which is the
loop to use while changing derivation logic. See [`docs/static-gtfs.md`](docs/static-gtfs.md)
for why the schedule join keys on `scheduled_trip_id`.

### Unattended

A GitHub Actions workflow ([`.github/workflows/etl-daily.yml`](.github/workflows/etl-daily.yml))
runs `src.etl.catchup` daily at 07:00 UTC. It processes every *complete* service day
missing from S3 rather than "yesterday", so a skipped or delayed run costs nothing and
the next one catches up. One-time AWS OIDC setup is in
[`docs/ci-setup.md`](docs/ci-setup.md).

Locally, whenever you want:

```bash
./scripts/run_etl_daily.sh --dry-run   # what is outstanding
./scripts/run_etl_daily.sh             # process it, then sync to S3
```

The wrapper sets `AWS_PROFILE` and `JAVA_HOME` itself, so nothing needs exporting.
A local `launchd` schedule is *not* used — macOS TCC blocks a LaunchAgent from reading
anything under `~/Documents`, so it cannot execute the pipeline at all
([`docs/ci-setup.md`](docs/ci-setup.md) has the detail).

### Reading the segment table

S3 is the authoritative copy — `raw/` expires after 90 days, after which the segments
are the only surviving record of those service days.

```python
from src.etl.config import EtlConfig
from src.etl.processed import read_segments

table = read_segments(EtlConfig.from_env())          # every service date
table = read_segments(config, ["2026-08-05"])        # one date
```

See [`docs/etl.md`](docs/etl.md) for how arrivals are derived and what the data cannot
tell you.
