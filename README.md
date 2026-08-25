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

A cron entry on this machine runs `src.etl.catchup` four times a day — 09:00, 13:00,
17:00 and 21:00 local. It processes every *complete* service day missing from S3 rather
than "yesterday", so a skipped or delayed run costs nothing and the next one catches up.

```bash
make install-cron     # (re)install the schedule
make cron-log         # tail the scheduled run log
make uninstall-cron
```

Four firings rather than one because cron, unlike launchd, does not catch up a run
missed while the laptop was asleep — and a no-op run costs about three seconds.

The launcher is installed to `~/.metro-pulse/run-etl.sh`, deliberately **outside**
`~/Documents`. macOS TCC grants filesystem access per *binary*: `/bin/bash` is denied
everything under `~/Documents`, so cron cannot read a launcher stored in this repo, but
`.venv/bin/python` resolves to `/Library/Frameworks/Python.framework`, which holds Full
Disk Access and reads the repo fine. The launcher therefore hands every repo path to
Python and none to the shell, and writes its log to `~/Library/Logs/metro-pulse` because
the shell performs that redirect. `scripts/etl_cron.sh` is the source of truth; re-run
`make install-cron` after editing it.

The GitHub Actions workflow
([`.github/workflows/etl-daily.yml`](.github/workflows/etl-daily.yml)) is kept for
on-demand `workflow_dispatch` runs from a machine that is not this laptop. Its schedule
has been removed. One-time AWS OIDC setup is in [`docs/ci-setup.md`](docs/ci-setup.md).

Locally, whenever you want:

```bash
./scripts/run_etl_daily.sh --dry-run   # what is outstanding
./scripts/run_etl_daily.sh             # process it, then sync to S3
```

The wrapper sets `AWS_PROFILE` and `JAVA_HOME` itself, so nothing needs exporting. It is
for **interactive** use — a terminal holds Documents access, so it can read itself. cron
cannot, which is why the scheduled path uses a separate installed launcher.

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
tell you, and [`docs/polling-cadence.md`](docs/polling-cadence.md) for what a finer polling
cadence would buy, what it costs, and why the cheap approach does not work.
