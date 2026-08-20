"""Stage A: decode raw protobuf snapshots into flat observation Parquet.

## Why decoding happens here and not inside Spark

The brief asked for this tradeoff to be assessed rather than assumed, so:

GTFS-realtime decoding is a pure-Python protobuf parse. It cannot vectorise, so a
Spark UDF over `binaryFile` buys no compute advantage — it only moves the same
per-record Python work onto executors, paying serialisation on the way.

The decisive factor is object shape, not compute. A month is ~43k objects per feed,
each 6 KB–1 MB gzipped. Spark's `binaryFile` source assigns roughly one task per
file, so that is 43k tasks whose scheduling overhead exceeds the work in each. The
job is also almost entirely S3 round-trip latency, and a 16-thread pool saturates S3
far better on one machine than a handful of local executors each doing blocking reads.

So the decode is a threaded Python pre-pass, and its output is Parquet. Three
concrete wins:

1. **The small-file problem is solved before Spark sees the data.** 43k tiny objects
   become a handful of columnar files per service date.
2. **Derivation becomes re-runnable without re-downloading.** The brief says this
   logic will be re-run many times as it evolves; re-decoding 43k objects each time
   would dominate the loop. Stage A is cached, stage B is cheap.
3. **Spark is left doing what it is actually good at** — the windowed joins and
   aggregations in `arrivals.py` and `segments.py`, over a compact columnar input.

The cost is an intermediate dataset on disk. That is a straightforwardly good trade
here, and it is the same reason production pipelines land raw feeds before modelling
them.

Output layout, mirroring the raw archive so the same pruning logic applies:

    {out}/{feed}/service_date=YYYY-MM-DD/part-*.parquet
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from google.transit import gtfs_realtime_pb2

from .archive import (
    Snapshot,
    modal_interval_seconds,
    read_snapshots,
    snapshot_keys_by_hour,
)
from .config import MIN_PLAUSIBLE_EPOCH_SEC, SERVICE_TZ, EtlConfig
from .progress import Progress

logger = logging.getLogger(__name__)

_VEHICLE_STATUS = gtfs_realtime_pb2.VehiclePosition.VehicleStopStatus
_SCHEDULE_RELATIONSHIP = (
    gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship
)

# Explicit schemas rather than letting Arrow infer. Inference on a short window can
# type an all-null column as null and then fail to merge with a later window where it
# is populated — a failure that only appears once the archive grows.
VEHICLE_POSITION_SCHEMA = pa.schema(
    [
        ("captured_at", pa.timestamp("s", tz="UTC")),
        ("service_date", pa.string()),
        ("trip_id", pa.string()),
        ("scheduled_trip_id", pa.string()),
        ("schedule_version", pa.string()),
        ("route_id", pa.string()),
        ("direction_id", pa.int32()),
        ("vehicle_id", pa.string()),
        ("stop_id", pa.string()),
        ("stop_sequence", pa.int32()),
        ("status", pa.string()),
        ("vehicle_ts", pa.timestamp("s", tz="UTC")),
    ]
)

TRIP_UPDATE_SCHEMA = pa.schema(
    [
        ("captured_at", pa.timestamp("s", tz="UTC")),
        ("service_date", pa.string()),
        ("trip_id", pa.string()),
        ("scheduled_trip_id", pa.string()),
        ("schedule_version", pa.string()),
        ("route_id", pa.string()),
        ("direction_id", pa.int32()),
        ("vehicle_id", pa.string()),
        ("stop_id", pa.string()),
        ("stop_sequence", pa.int32()),
        ("arrival_ts", pa.timestamp("s", tz="UTC")),
        ("departure_ts", pa.timestamp("s", tz="UTC")),
        ("relationship", pa.string()),
    ]
)


def _split_trip_id(trip_id: str) -> tuple[str, str | None]:
    """`12345678_20670` -> (`12345678`, `20670`); bare ids keep their whole value.

    Split at decode time so both halves are columns in the observation table. The
    base is the join key; the version is what gets compared against the bundle to
    detect a stale-timetable join. Deriving them later would mean re-parsing strings
    inside every Spark stage.
    """
    head, sep, tail = trip_id.rpartition("_")
    if not sep or not head:
        return trip_id, None
    return head, tail


def _service_date(trip: Any, captured_at: datetime) -> str:
    """The GTFS service date for an observation.

    Prefers the feed's own `trip.start_date`, which is the service date WMATA
    assigned — authoritative, and correct for after-midnight trips that belong to the
    previous service day. Falls back to the local calendar date of capture only when
    the field is absent (vehicles not on a scheduled trip), which cannot be right for
    a post-midnight trip but is the best available for a row that has no trip anyway.
    """
    if trip.start_date:
        raw = trip.start_date
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    return captured_at.astimezone(SERVICE_TZ).date().isoformat()


def _epoch_or_none(value: int) -> int | None:
    """Reject sentinel timestamps.

    WMATA sometimes sets a time field to 0 rather than omitting it. Zero is a valid
    protobuf int but not a valid 2026 instant, and it poisons any min/max over
    predictions.
    """
    return value if value >= MIN_PLAUSIBLE_EPOCH_SEC else None


def vehicle_position_rows(snapshot: Snapshot) -> Iterator[dict[str, Any]]:
    """One row per vehicle in a VehiclePositions snapshot."""
    for entity in snapshot.message.entity:
        v = entity.vehicle
        if not v.trip.trip_id:
            # No trip means deadheading or between assignments. It cannot join to a
            # schedule and cannot contribute a segment, so it is dropped here rather
            # than carried through four stages to be filtered at the end.
            continue
        base, version = _split_trip_id(v.trip.trip_id)
        yield {
            "captured_at": snapshot.captured_at,
            "service_date": _service_date(v.trip, snapshot.captured_at),
            "trip_id": v.trip.trip_id,
            "scheduled_trip_id": base,
            "schedule_version": version,
            "route_id": v.trip.route_id or None,
            "direction_id": (
                int(v.trip.direction_id) if v.trip.HasField("direction_id") else None
            ),
            "vehicle_id": v.vehicle.id or None,
            "stop_id": v.stop_id or None,
            "stop_sequence": (
                int(v.current_stop_sequence)
                if v.HasField("current_stop_sequence")
                else None
            ),
            "status": (
                _VEHICLE_STATUS.Name(v.current_status)
                if v.HasField("current_status")
                else None
            ),
            "vehicle_ts": (
                _epoch_or_none(v.timestamp) if v.HasField("timestamp") else None
            ),
        }


def trip_update_rows(snapshot: Snapshot) -> Iterator[dict[str, Any]]:
    """One row per stop_time_update, carrying its parent trip's context."""
    for entity in snapshot.message.entity:
        tu = entity.trip_update
        if not tu.trip.trip_id:
            continue
        base, version = _split_trip_id(tu.trip.trip_id)
        for stu in tu.stop_time_update:
            relationship = _SCHEDULE_RELATIONSHIP.Name(stu.schedule_relationship)
            # SKIPPED stops are stops the vehicle will not serve. They carry no times
            # (verified: every SKIPPED row had neither arrival nor departure) and must
            # never become a segment row, so they are dropped at the source.
            if relationship == "SKIPPED":
                continue
            yield {
                "captured_at": snapshot.captured_at,
                "service_date": _service_date(tu.trip, snapshot.captured_at),
                "trip_id": tu.trip.trip_id,
                "scheduled_trip_id": base,
                "schedule_version": version,
                "route_id": tu.trip.route_id or None,
                "direction_id": (
                    int(tu.trip.direction_id)
                    if tu.trip.HasField("direction_id")
                    else None
                ),
                "vehicle_id": tu.vehicle.id or None,
                "stop_id": stu.stop_id or None,
                "stop_sequence": (
                    int(stu.stop_sequence) if stu.HasField("stop_sequence") else None
                ),
                "arrival_ts": (
                    _epoch_or_none(stu.arrival.time)
                    if stu.arrival.HasField("time")
                    else None
                ),
                "departure_ts": (
                    _epoch_or_none(stu.departure.time)
                    if stu.departure.HasField("time")
                    else None
                ),
                "relationship": relationship,
            }


_ROW_BUILDERS = {
    "vehicle_positions": (vehicle_position_rows, VEHICLE_POSITION_SCHEMA),
    "trip_updates": (trip_update_rows, TRIP_UPDATE_SCHEMA),
}


def window_id(start: datetime, end: datetime) -> str:
    """A filename-safe identifier for the UTC window a decode covers."""
    return f"{start:%Y%m%dT%H}-{end:%Y%m%dT%H}"


def write_window(
    table: pa.Table, destination: Path, start: datetime, end: datetime
) -> None:
    """Write one window's observations, replacing only that window's own files.

    The obvious choice, `existing_data_behavior="delete_matching"`, is wrong here and
    the failure is silent. A UTC window almost always straddles two service dates —
    trips running after local midnight belong to the *previous* service day — so
    decoding 2026-08-05 also emits a sliver of 2026-08-04. `delete_matching` would
    delete the whole existing 2026-08-04 partition and leave only that sliver. Running
    the documented per-day backfill loop would therefore truncate each day as it
    processed the next, and the output would look entirely normal.

    Instead each write is scoped to its own window: files are named for the window that
    produced them, that window's previous files are removed first (so a re-run replaces
    itself even if it now yields fewer files), and other windows' files in the same
    partition are left alone.

    Windows that *overlap* each other still duplicate the shared snapshots. Stage B
    deduplicates on read rather than this stage trying to reason about overlap, because
    only the reader knows the natural key.
    """
    marker = window_id(start, end)

    # Remove this window's previous output before rewriting. `overwrite_or_ignore`
    # alone would replace same-named files but orphan any extras from a run that
    # produced more fragments than this one does.
    if destination.exists():
        for stale in destination.rglob(f"{marker}-*.parquet"):
            stale.unlink()

    ds.write_dataset(
        table,
        destination,
        format="parquet",
        partitioning=ds.partitioning(
            pa.schema([("service_date", pa.string())]), flavor="hive"
        ),
        basename_template=f"{marker}-part-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )


def decode_feed(
    s3: Any,
    config: EtlConfig,
    feed: str,
    kind: str,
    start: datetime,
    end: datetime,
    output_root: Path,
    max_workers: int = 16,
) -> dict[str, Any]:
    """Decode one feed over a UTC window and write observation Parquet.

    Returns a summary including the per-hour snapshot counts, which the data-quality
    report uses to detect collector downtime — an hour holding 12 files instead of 60
    is missing data, and only this stage can see it.
    """
    builder, schema = _ROW_BUILDERS[kind]

    keys_by_hour = snapshot_keys_by_hour(s3, config, feed, start, end)
    keys = [key for hour_keys in keys_by_hour.values() for key in hour_keys]
    if not keys:
        raise FileNotFoundError(
            f"no snapshots under s3://{config.s3_bucket}/{config.raw_prefix}{feed}/ "
            f"for {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} UTC"
        )

    logger.info("decoding feed=%s snapshots=%d", feed, len(keys))
    rows: list[dict[str, Any]] = []
    # This loop is the longest silent stretch in the whole ETL — ~2,880 S3 round trips
    # per rail service day. Without progress a working run and a hung one look alike.
    progress = Progress(logger, len(keys), f"{feed} snapshots")
    for snapshot in read_snapshots(s3, config, keys, max_workers):
        rows.extend(builder(snapshot))
        progress.advance()
    progress.done()
    logger.info("  %s decoded %d observation row(s)", feed, len(rows))

    table = pa.Table.from_pylist(rows, schema=schema)
    destination = output_root / feed
    write_window(table, destination, start, end)

    service_dates = sorted(set(table.column("service_date").to_pylist()))
    return {
        "feed": feed,
        "snapshots": len(keys),
        "rows": table.num_rows,
        "service_dates": service_dates,
        "snapshots_by_hour": {k: len(v) for k, v in keys_by_hour.items()},
        # Measured, not assumed. The data-quality report needs it to know what a
        # complete hour looks like, and the collector's cadence has already changed
        # once (60s -> 30s) so a constant would describe the wrong era.
        "interval_seconds": modal_interval_seconds(keys),
        "path": str(destination),
    }


def read_observations(root: Path, feed: str) -> pa.Table:
    """Read decoded observations back, for tests and inspection."""
    return pq.read_table(root / feed)
