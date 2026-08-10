"""The ETL entrypoint: raw snapshots to a trip-segment table.

    python -m src.etl.pipeline --mode rail --start 2026-08-05T11 --end 2026-08-05T14
    python -m src.etl.pipeline --mode rail --date 2026-08-05 --output data/processed

Runs identically on one hour or the whole archive — the window is an argument, and
nothing in the path handling changes with its size. Two stages:

**A. Decode** (`decode.py`) — threaded protobuf decode of the requested UTC window
into observation Parquet. Skipped with `--skip-decode` when iterating on derivation
logic against an already-decoded window, which is the common case while the segment
definition is still moving.

**B. Derive** (`arrivals.py`, `segments.py`) — Spark reads the observations, derives
arrivals, joins the schedule, and writes segments partitioned by `service_date`.

Both stages are idempotent per partition: re-running a date replaces that date and
leaves every other one alone.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.parquet as pq

from .archive import (
    modal_interval_seconds,
    resolve_static_bundle,
    snapshot_keys_by_hour,
)
from .config import FEEDS_BY_MODE, SERVICE_TZ, EtlConfig
from .decode import decode_feed
from .quality import QualityReport, check_segments, check_snapshot_coverage
from .schedule import (
    build_match_report,
    scheduled_stop_times,
    service_day_end,
    service_day_start,
)
from .spark import build_session

logger = logging.getLogger("etl")


def parse_hour(value: str) -> datetime:
    """Parse `YYYY-MM-DDTHH` or a fuller ISO stamp as UTC."""
    text = value if len(value) > 13 else f"{value}:00:00"
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def fully_covered_service_dates(start: datetime, end: datetime) -> list[str]:
    """Service dates whose whole local service day lies inside the requested window.

    Stage B reads everything in staging, not just the window that was decoded, because
    staging accumulates. Without this filter a run for one date also rewrites its
    neighbours from whatever slivers happen to be staged: `--date 2026-08-06` produced a
    250-row `service_date=2026-08-04` partition out of a 1,099-row sliver, which is real
    data but a fraction of that day, and nothing in the output distinguishes it from a
    complete one.

    A date is only written when this window is authoritative for the whole of it. Dates
    seen but not covered are skipped and logged, so they can be run deliberately.
    """
    covered: list[str] = []
    cursor = start.astimezone(SERVICE_TZ).date() - timedelta(days=1)
    last = end.astimezone(SERVICE_TZ).date()
    while cursor <= last:
        # `service_day_end` runs past local midnight, so a window must cover that
        # overhang to be authoritative for the day. Using local midnight here was the
        # bug: it declared a day fully covered while the last hours of it sat outside
        # the decoded window.
        if start <= service_day_start(cursor) and service_day_end(cursor) <= end:
            covered.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return covered


def service_day_bounds(service_date: str) -> tuple[datetime, datetime]:
    """UTC window covering one America/New_York service day, overhang included.

    A local day is not a UTC day, and a GTFS *service* day is not a local day either. It
    starts at noon minus twelve hours and runs past the following midnight, so this uses
    `service_day_start`/`service_day_end` rather than constructing local midnights.

    The previous version built naive local midnights, which was wrong twice over: it
    disagreed with `service_day_start` on DST boundaries, and it stopped at midnight and
    so excluded the late-night tail — 424 vehicle-records for one service date.
    """
    day = datetime.fromisoformat(service_date).date()
    return service_day_start(day), service_day_end(day)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(FEEDS_BY_MODE), default="rail")
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument("--date", help="one America/New_York service day, YYYY-MM-DD")
    window.add_argument("--start", type=parse_hour, help="UTC hour, e.g. 2026-08-05T11")
    parser.add_argument("--end", type=parse_hour, help="UTC hour, exclusive")
    parser.add_argument(
        "--staging",
        default="data/staging",
        help="where decoded observation Parquet lands (stage A output)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/segments",
        help="where the segment table is written (stage B output)",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="reuse existing staged observations instead of re-reading S3",
    )
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--fail-on-quality",
        action="store_true",
        help=(
            "exit non-zero if the data-quality report finds a blocking issue. Off by "
            "default: a backfill over a partially-collected day may legitimately want "
            "the rows anyway, flagged. On for anything automated."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    if args.date:
        start, end = service_day_bounds(args.date)
    else:
        if args.end is None:
            parser.error("--end is required with --start")
        start, end = args.start, args.end

    config = EtlConfig.from_env()
    s3 = boto3.client("s3")
    vp_feed, tu_feed = FEEDS_BY_MODE[args.mode]
    staging = Path(args.staging)

    local_start = start.astimezone(SERVICE_TZ)
    logger.info("mode=%s", args.mode)
    logger.info(
        "window %s .. %s UTC (%s %s local)",
        f"{start:%Y-%m-%d %H:%M}",
        f"{end:%Y-%m-%d %H:%M}",
        f"{local_start:%Y-%m-%d %H:%M}",
        f"{local_start:%Z}",
    )

    # ---- Stage A: decode ----
    decode_summaries: list[dict[str, object]] = []
    if args.skip_decode:
        logger.info("stage A skipped, reusing %s", staging)
        # Snapshot coverage is only visible from the archive listing — once decoded, a
        # missing hour is indistinguishable from an hour with no service. So the listing
        # still happens even when the decode is skipped, or the DQ report would quietly
        # lose its only downtime check. Listing is cheap; decoding is not.
        for feed in (vp_feed, tu_feed):
            keys_by_hour = snapshot_keys_by_hour(s3, config, feed, start, end)
            all_keys = [key for keys in keys_by_hour.values() for key in keys]
            decode_summaries.append(
                {
                    "feed": feed,
                    "snapshots_by_hour": {k: len(v) for k, v in keys_by_hour.items()},
                    "interval_seconds": modal_interval_seconds(all_keys),
                }
            )
    else:
        for feed, kind in ((vp_feed, "vehicle_positions"), (tu_feed, "trip_updates")):
            summary = decode_feed(
                s3, config, feed, kind, start, end, staging, args.max_workers
            )
            decode_summaries.append(summary)
            logger.info(
                "decoded feed=%s snapshots=%d rows=%d dates=%s",
                summary["feed"],
                summary["snapshots"],
                summary["rows"],
                ",".join(summary["service_dates"]),
            )

    # ---- Static schedule and match rate (step 2) ----
    service_date = local_start.date().isoformat()
    bundle = resolve_static_bundle(s3, config, args.mode, service_date)
    logger.info("static bundle %s (version %s)", bundle.key, bundle.schedule_version)

    observations_for_report = pq.read_table(
        staging / tu_feed, columns=["trip_id", "route_id"]
    ).to_pandas()
    match_report = build_match_report(observations_for_report, bundle)

    schedule_frame = scheduled_stop_times(bundle)
    # Bridged through Parquet rather than spark.createDataFrame(pandas_df): PySpark
    # warns it does not yet fully support pandas >= 3.0, and this sidesteps that
    # conversion path entirely while also caching the parsed schedule.
    schedule_path = (
        config.cache_dir / "schedule" / args.mode / f"{bundle.feed_start_date}"
    )
    schedule_path.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(schedule_frame, preserve_index=False),
        schedule_path / "stop_times.parquet",
    )

    # ---- Stage B: derive ----
    spark = build_session(app_name=f"metro-pulse-etl-{args.mode}")
    try:
        from pyspark.sql import functions as F

        from .arrivals import combine_arrivals, derive_tu_arrivals, derive_vp_arrivals
        from .segments import add_scheduled_timestamps, build_segments, write_segments

        # Deduplicate on read. Stage A writes each UTC window to its own files without
        # clobbering neighbouring windows (see decode.write_window), which means two
        # overlapping decodes leave the shared snapshots staged twice. Every duplicate
        # is a byte-identical re-decode of one snapshot, so dropping on the natural key
        # is safe, and doing it here keeps stage A from having to reason about overlap.
        # Left un-deduplicated, a repeated observation becomes a repeated arrival and
        # then a zero-duration segment.
        vp_obs = spark.read.parquet(str(staging / vp_feed)).dropDuplicates(
            ["captured_at", "trip_id"]
        )
        tu_obs = spark.read.parquet(str(staging / tu_feed)).dropDuplicates(
            ["captured_at", "trip_id", "stop_sequence"]
        )

        vp_arrivals = derive_vp_arrivals(vp_obs)
        tu_arrivals = derive_tu_arrivals(tu_obs)
        arrivals = combine_arrivals(vp_arrivals, tu_arrivals).cache()

        by_source = {
            row["arrival_source"]: row["count"]
            for row in arrivals.groupBy("arrival_source").count().collect()
        }
        logger.info("arrivals derived: %s", by_source)

        schedule = spark.read.parquet(str(schedule_path))
        # The schedule is date-independent (offsets in seconds); the segments are not.
        # Expanding it across the service dates present in this run resolves each
        # offset to an absolute instant per date, and `service_date` stays a join key
        # all the way through so the expansion cannot fan out the segment table.
        dates = [
            row["service_date"]
            for row in arrivals.select("service_date").distinct().collect()
        ]
        logger.info("resolving schedule for service dates: %s", dates)
        schedule_with_dates = add_scheduled_timestamps(
            schedule.crossJoin(
                spark.createDataFrame([(d,) for d in dates], "service_date string")
            ),
            "service_date",
        ).select(
            "scheduled_trip_id",
            "stop_sequence",
            "service_date",
            "scheduled_arrival_ts",
            "scheduled_departure_ts",
        )

        all_segments = build_segments(
            arrivals,
            schedule_with_dates,
            mode=args.mode,
            static_gtfs_version=bundle.version_label,
            bundle_schedule_version=bundle.schedule_version,
        )

        # Write only the dates this window is authoritative for. Staging accumulates
        # across runs, so without this a run for one day rewrites its neighbours from
        # whatever slivers are staged.
        targets = fully_covered_service_dates(start, end)
        skipped = sorted(
            {
                str(row["service_date"])
                for row in all_segments.select("service_date").distinct().collect()
            }
            - set(targets)
        )
        if skipped:
            logger.info(
                "skipping %d service date(s) only partially covered by this window: %s",
                len(skipped),
                ", ".join(skipped),
            )
        logger.info("writing service dates: %s", ", ".join(targets) or "(none)")

        segments = all_segments.filter(
            F.col("service_date").isin(targets) if targets else F.lit(False)
        ).cache()

        total = segments.count()
        logger.info("segments built: %d", total)

        # ---- Step 5: data quality ----
        # Computed before the write, so a report is produced even if the write fails,
        # and printed after it so it is the last thing on screen rather than scrolled
        # away by Spark's progress output.
        quality = QualityReport(
            coverage=[check_snapshot_coverage(s) for s in decode_summaries],
            match_rate=match_report,
            segments=check_segments(segments),
        )

        if total > 0:
            write_segments(segments, args.output)
            logger.info("wrote %s partitioned by service_date", args.output)

        print(quality.format())

        issues = quality.blocking_issues
        if issues:
            for issue in issues:
                logger.error("data quality: %s", issue)
            if args.fail_on_quality:
                return 1
        if total == 0:
            return 1
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
