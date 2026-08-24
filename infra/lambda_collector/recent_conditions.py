"""Rebuild the live recent-conditions table from the last hour of snapshots.

    {"task": "recent_conditions"}   — every 5 minutes

## Why this task exists

`recent_deviation` is the strongest feature the journey model has (+0.255 against the
residual, against -0.010 for upstream delay), and `recent_conditions_lookup.parquet` was
built to serve it. But that lookup comes from the batch pipeline, which only processes
**completed service days**: a day closes at 04:00 UTC the next day, so the freshest
entry is already hours old when written. Measured against the live endpoint: 0 of 1,052
entries fell inside the 3600s staleness window, so every `recent_*` feature was null on
every request. The endpoint was running on schedule and calendar features alone.

Batch cannot fix this at any cadence; it is the wrong data source. This task reads the
raw feed directly.

## What it does and does not compute

Derives arrivals from the last ~90 minutes of VehiclePositions, builds segments from
consecutive arrivals, and compares each to the batch-published baselines. Ninety
minutes rather than sixty so a traversal completing near the boundary still has both
bracketing snapshots available.

**Arrival derivation is imported, not reimplemented.** `metro_etl.arrivals_live` is
the pandas port of the Spark logic, and `tests/test_arrivals_parity.py` fails if the
two ever disagree by a single row. A third implementation here would defeat that
entirely.

The slow-moving halves — segment x hour medians and scheduled durations — are
published daily by `src.serving.publish` and simply read here."""

from __future__ import annotations

import gzip
import io
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger("collector.recent_conditions")

LOOKBACK_MINUTES = 90
SERVING_PREFIX = "models/serving/"
OUTPUT_KEY = f"{SERVING_PREFIX}recent_conditions_live.csv"

# Every write is ALSO archived under a timestamped key. The live object is overwritten
# every five minutes, so without this a past prediction can never be faithfully
# reconstructed — `recent_deviation` depends on conditions at a moment, and scoring a
# historical prediction against conditions it did not have would flatter the model.
#
# ~12 KB per write, ~3.5 MB/day, and it CANNOT be backfilled: every interval this is not
# running is an interval of monitoring history that will never exist.
ARCHIVE_PREFIX = "models/serving/history/recent_conditions/"
MEDIANS_KEY = f"{SERVING_PREFIX}segment_hour_medians.csv"
SCHEDULED_KEY = f"{SERVING_PREFIX}segment_scheduled.csv"

FEED = "rail_vehicle_positions"
SERVICE_TZ_OFFSET_HOURS = -4  # EDT; only used for the hour key on the median join


def _snapshot_keys(s3: Any, bucket: str, prefix: str, since: datetime) -> list[str]:
    """Keys for the hours the lookback window touches.

    Listing per hour-partition rather than scanning the feed: the archive holds 90 days,
    and a prefix-wide list would page through ~130,000 objects to find 90.
    """
    keys: list[str] = []
    hour = since.replace(minute=0, second=0, microsecond=0)
    end = datetime.now(UTC)
    while hour <= end:
        partition = (
            f"{prefix}{FEED}/year={hour:%Y}/month={hour:%m}"
            f"/day={hour:%d}/hour={hour:%H}/"
        )
        response = s3.list_objects_v2(Bucket=bucket, Prefix=partition)
        keys.extend(item["Key"] for item in response.get("Contents", []))
        hour += timedelta(hours=1)
    return sorted(keys)


def _decode(s3: Any, bucket: str, keys: list[str]) -> Any:
    """Decode snapshots into the staged-observation shape `arrivals_live` expects."""
    import pandas as pd
    from google.transit import gtfs_realtime_pb2

    rows: list[dict[str, Any]] = []
    for key in keys:
        payload = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        if key.endswith(".gz"):
            payload = gzip.decompress(payload)
        message = gtfs_realtime_pb2.FeedMessage()
        message.ParseFromString(payload)

        # The capture time is in the object name, not the payload: it is the time WE
        # polled, which is what brackets an arrival. The feed's own header timestamp is
        # the agency's and drifts.
        captured_epoch = int(key.rsplit("-", 1)[-1].split(".")[0])
        captured_at = datetime.fromtimestamp(captured_epoch, UTC).replace(tzinfo=None)

        for entity in message.entity:
            if not entity.HasField("vehicle"):
                continue
            vehicle = entity.vehicle
            trip_id = vehicle.trip.trip_id or None
            if not trip_id or not vehicle.HasField("current_stop_sequence"):
                continue
            scheduled_trip_id, _, version = trip_id.rpartition("_")
            rows.append(
                {
                    "captured_at": captured_at,
                    "trip_id": trip_id,
                    "scheduled_trip_id": scheduled_trip_id or trip_id,
                    "schedule_version": version or "",
                    "route_id": vehicle.trip.route_id or "",
                    "direction_id": int(vehicle.trip.direction_id),
                    "vehicle_id": vehicle.vehicle.id or "",
                    "stop_id": vehicle.stop_id or "",
                    "stop_sequence": int(vehicle.current_stop_sequence),
                    "status": "",
                    "vehicle_ts": captured_at,
                    "service_date": f"{captured_at:%Y-%m-%d}",
                }
            )
    logger.info("decoded %d observation(s) from %d snapshot(s)", len(rows), len(keys))
    return pd.DataFrame(rows)


def _segments_from_arrivals(arrivals: Any) -> Any:
    """Consecutive arrivals on one trip become one segment traversal.

    Mirrors `etl/segments.py`: `actual_departure_ts` is the UPSTREAM ARRIVAL, not a true
    departure, so dwell at the upstream stop is inside the duration. Getting this wrong
    would make every live duration disagree with every trained one.
    """
    ordered = arrivals.sort_values(["trip_id", "stop_sequence"], kind="stable")
    grouped = ordered.groupby("trip_id", sort=False)

    ordered = ordered.assign(
        from_stop_id=grouped["stop_id"].shift(1),
        actual_departure_ts=grouped["actual_arrival_ts"].shift(1),
    )
    usable = ordered[ordered["from_stop_id"].notna()].copy()
    usable["actual_duration_sec"] = (
        usable["actual_arrival_ts"] - usable["actual_departure_ts"]
    ).dt.total_seconds()
    usable = usable.rename(columns={"stop_id": "to_stop_id"})

    # A segment must actually go somewhere. A vehicle re-reported at the same stop with
    # an intervening sequence change produces a from == to "traversal" that joins to no
    # baseline, carries a NaN deviation, and can never match a real journey's first leg.
    # Harmless downstream, but it is noise in a published table, and noise that looks
    # deliberate is worse than noise that is filtered.
    real = usable["from_stop_id"] != usable["to_stop_id"]
    dropped = int((~real).sum())
    if dropped:
        logger.info("dropped %d same-stop pseudo-segment(s)", dropped)

    return usable[real & (usable["actual_duration_sec"] > 0)]


def build_recent_conditions(config: Any, s3: Any) -> dict[str, Any]:
    """Read snapshots, derive segments, compare to baselines, write the live table."""
    import pandas as pd
    from metro_etl.arrivals_live import derive_vp_arrivals

    since = datetime.now(UTC) - timedelta(minutes=LOOKBACK_MINUTES)
    # `s3_prefix`, not `raw_prefix` — that is EtlConfig's name for the same thing.
    # CollectorConfig is a separate class with its own vocabulary, and testing against a
    # stand-in object rather than the real one is what let this reach the Lambda.
    keys = _snapshot_keys(s3, config.s3_bucket, config.s3_prefix, since)
    if not keys:
        return {"ok": False, "reason": "no snapshots in the lookback window"}

    observations = _decode(s3, config.s3_bucket, keys)
    if observations.empty:
        return {"ok": False, "reason": "no usable observations"}

    arrivals = derive_vp_arrivals(observations)
    segments = _segments_from_arrivals(arrivals)
    if segments.empty:
        return {"ok": False, "reason": "no completed traversals"}

    def read_csv(key: str) -> pd.DataFrame:
        body = s3.get_object(Bucket=config.s3_bucket, Key=key)["Body"].read()
        return pd.read_csv(io.BytesIO(body))

    medians = read_csv(MEDIANS_KEY)
    scheduled = read_csv(SCHEDULED_KEY)

    segments["local_hour"] = (
        segments["actual_departure_ts"] + timedelta(hours=SERVICE_TZ_OFFSET_HOURS)
    ).dt.hour
    joined = segments.merge(
        medians, on=["from_stop_id", "to_stop_id", "local_hour"], how="left"
    ).merge(scheduled, on=["from_stop_id", "to_stop_id"], how="left")

    # The definition that must match training: deviation is the residual against the
    # segment x hour median, not against the schedule.
    joined["deviation"] = joined["actual_duration_sec"] - joined["baseline_sec"]
    joined["delay"] = joined["actual_duration_sec"] - joined["scheduled_sec"]

    lookup = (
        joined.groupby(["from_stop_id", "to_stop_id"], as_index=False)
        .agg(
            completed_at=("actual_arrival_ts", "max"),
            recent_duration_median=("actual_duration_sec", "median"),
            recent_duration_mean=("actual_duration_sec", "mean"),
            recent_delay_mean=("delay", "mean"),
            recent_traversals=("actual_duration_sec", "size"),
            recent_deviation=("deviation", "mean"),
        )
        .round(3)
    )

    buffer = io.StringIO()
    lookup.to_csv(buffer, index=False)
    payload = buffer.getvalue().encode()
    s3.put_object(
        Bucket=config.s3_bucket,
        Key=OUTPUT_KEY,
        Body=payload,
        ContentType="text/csv",
    )

    # Hive-partitioned by date so a monitoring job can read one service day without
    # listing the whole history.
    now = datetime.now(UTC)
    archive_key = f"{ARCHIVE_PREFIX}date={now:%Y-%m-%d}/{now:%Y%m%dT%H%M%S}Z.csv"
    s3.put_object(
        Bucket=config.s3_bucket,
        Key=archive_key,
        Body=payload,
        ContentType="text/csv",
    )
    logger.info(
        "wrote %d segment(s) to %s and %s", len(lookup), OUTPUT_KEY, archive_key
    )
    return {
        "ok": True,
        "segments": int(len(lookup)),
        "snapshots": len(keys),
        "traversals": int(len(segments)),
        "key": OUTPUT_KEY,
        "archive_key": archive_key,
    }
