"""The unattended entry point: process every service day that is still outstanding.

    python -m src.etl.catchup                 # process whatever is missing
    python -m src.etl.catchup --dry-run       # say what would run, do nothing

This is what the daily schedule invokes. It is deliberately **not** "process yesterday",
because that design loses data here for three reasons:

1. A service day only closes at 04:00 UTC the next day — trips after local midnight
   belong to the previous service day — so "yesterday" is ambiguous around the boundary.
2. The compute is a laptop. `cron` does not fire while a Mac is asleep, and even
   launchd's wake-catch-up only runs a missed job once.
3. `raw/` survives 90 days and the pipeline is idempotent per `service_date`.

So the schedule is only a trigger. Each run asks which complete service days are absent
from S3 and processes those. A week of downtime then costs nothing, and the real
deadline is the 90-day retention rather than any particular cron firing.

Exit status is non-zero if any date failed, so launchd records a failure.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .archive import earliest_snapshot
from .config import FEEDS_BY_MODE, SERVICE_TZ, EtlConfig
from .processed import completed_service_dates, sync_partitions
from .schedule import service_day_end, service_day_start

logger = logging.getLogger("catchup")

# How far back a run looks for gaps. Wide enough to absorb a holiday or a stretch of
# laptop downtime; narrow enough that the nightly run is not listing the whole archive.
# Anything older is a deliberate backfill — see the runbook.
DEFAULT_LOOKBACK_DAYS = 14

# Staged observations are disposable once a date's segments are written and synced, and
# they accrue ~5 MB per service day.
DEFAULT_STAGING_RETENTION_DAYS = 7


def pending_service_dates(
    now: datetime,
    done: set[str],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    force: bool = False,
    archive_start: datetime | None = None,
) -> list[str]:
    """Which service dates should be processed, oldest first.

    Pure — no clock, no S3 — because this is the scheduling logic and it has to be
    testable without either. A date qualifies when all of these hold:

    - it falls inside the lookback window
    - its service day has **closed**: `now` is at or past 04:00 UTC the following day.
      Processing a day still in progress would write a partition that looks complete but
      is missing its evening, and nothing downstream could tell.
    - the archive covers the **whole** of it: the service day begins at or after
      `archive_start`. Without this a 14-day lookback proposes dates from before the
      collector existed — 12 of them on first run — each starting a Spark session only
      to fail on missing input. It also excludes the collector's first, partial day,
      which would otherwise be written with most of its hours empty.
    - it is not already in `done` (unless `force`)
    """
    today_local = now.astimezone(SERVICE_TZ).date()
    candidates = [
        today_local - timedelta(days=offset) for offset in range(lookback_days, -1, -1)
    ]

    pending: list[str] = []
    for candidate in candidates:
        if now < service_day_end(candidate):
            continue  # the day has not finished yet — it runs past local midnight
        if archive_start is not None and service_day_start(candidate) < archive_start:
            continue  # the archive does not go back this far
        iso = candidate.isoformat()
        if force or iso not in done:
            pending.append(iso)
    return pending


def run_one_date(service_date: str, mode: str, staging: Path, output: Path) -> bool:
    """Run the pipeline for one service date. Returns whether it succeeded.

    A subprocess rather than an in-process call: one date failing must not abort the
    rest of the catch-up, and each run gets a clean JVM rather than trying to restart a
    stopped SparkSession in the same interpreter.
    """
    command = [
        sys.executable,
        "-m",
        "src.etl.pipeline",
        "--mode",
        mode,
        "--date",
        service_date,
        "--staging",
        str(staging),
        "--output",
        str(output),
        "--fail-on-quality",
    ]
    logger.info("processing %s", service_date)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        logger.error("%s FAILED (exit %d)", service_date, result.returncode)
        return False
    return True


def prune_staging(staging: Path, mode: str, keep_days: int, now: datetime) -> int:
    """Delete staged observations older than `keep_days`. Returns partitions removed.

    Staging is a working artefact, fully reproducible from `raw/` for 90 days, and it is
    the only thing here that grows without bound.
    """
    if keep_days <= 0 or not staging.exists():
        return 0

    cutoff = now.astimezone(SERVICE_TZ).date() - timedelta(days=keep_days)
    removed = 0
    for feed in FEEDS_BY_MODE[mode]:
        feed_dir = staging / feed
        if not feed_dir.is_dir():
            continue
        for partition in feed_dir.glob("service_date=*"):
            try:
                stamp = date.fromisoformat(partition.name.split("=", 1)[1])
            except ValueError:
                continue
            if stamp < cutoff:
                shutil.rmtree(partition)
                removed += 1
    if removed:
        logger.info("pruned %d staging partition(s) older than %s", removed, cutoff)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(FEEDS_BY_MODE), default="rail")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--staging", default="data/staging")
    parser.add_argument("--output", default="data/processed/segments")
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would run, change nothing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="reprocess dates already present in S3 (use after changing the logic)",
    )
    parser.add_argument(
        "--prune-staging-days", type=int, default=DEFAULT_STAGING_RETENTION_DAYS
    )
    parser.add_argument(
        "--no-sync", action="store_true", help="skip the upload to S3 (local only)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    import boto3

    config = EtlConfig.from_env()
    now = datetime.now(UTC)
    staging = Path(args.staging)
    output = Path(args.output)

    s3 = boto3.client("s3", region_name=config.aws_region)
    vp_feed = FEEDS_BY_MODE[args.mode][0]
    archive_start = earliest_snapshot(s3, config, vp_feed)

    logger.info("mode=%s lookback=%dd", args.mode, args.lookback_days)
    logger.info("archive begins %s", archive_start or "(empty)")

    # An empty archive is not the same as an unbounded one. `pending_service_dates`
    # reads `archive_start=None` as "no lower bound", which is right for tests but
    # wrong here: with nothing in raw/ it would propose every date in the lookback
    # window and start a Spark session for each, only to fail on missing input. This
    # is reachable in practice — it is exactly the state right after wiping raw/.
    if archive_start is None:
        logger.info("raw archive is empty — nothing to process yet")
        return 0

    done = set() if args.force else completed_service_dates(config, s3)
    pending = pending_service_dates(
        now, done, args.lookback_days, args.force, archive_start
    )

    logger.info("already in S3: %d date(s)", len(done))
    if not pending:
        logger.info("nothing outstanding — up to date")
        return 0
    logger.info("outstanding: %s", ", ".join(pending))

    if args.dry_run:
        logger.info("dry run — nothing processed")
        return 0

    succeeded: list[str] = []
    failed: list[str] = []
    for service_date in pending:
        if run_one_date(service_date, args.mode, staging, output):
            succeeded.append(service_date)
        else:
            failed.append(service_date)

    # Sync only what succeeded. A date whose pipeline failed may have written a partial
    # partition locally, and copying that to S3 would mark it done and stop the next run
    # from retrying it.
    if succeeded and not args.no_sync:
        sync_partitions(config, output, succeeded)

    prune_staging(staging, args.mode, args.prune_staging_days, now)

    logger.info("processed %d, failed %d", len(succeeded), len(failed))
    if failed:
        logger.error("failed dates: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
