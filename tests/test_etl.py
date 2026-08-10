"""Unit tests for the ETL, against synthetic fixtures only.

No network, no S3, no `.env`. The GTFS-realtime messages are built in
`fixtures_gtfs_rt.py` and go through the same `decode.py` path as production data, so
these exercise the real parsing rather than a stand-in.

Most of these tests exist because the situation they describe actually broke the
pipeline on real data. Those are marked REGRESSION.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from fixtures_gtfs_rt import (
    at,
    epoch,
    progressing_vehicle,
    trip_updates,
    vehicle_positions,
)

from src.etl.archive import Snapshot
from src.etl.config import SERVICE_TZ
from src.etl.decode import (
    TRIP_UPDATE_SCHEMA,
    VEHICLE_POSITION_SCHEMA,
    trip_update_rows,
    vehicle_position_rows,
)
from src.etl.quality import check_snapshot_coverage
from src.etl.schedule import (
    base_trip_id,
    gtfs_time_to_seconds,
    schedule_version,
    scheduled_timestamp,
    service_day_start,
)

# --------------------------------------------------------------------------
# schedule.py — pure functions, no Spark
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("trip_id", "expected_base", "expected_version"),
    [
        ("11970073_20660", "11970073", "20660"),
        ("NR170", "NR170", None),  # non-revenue moves carry no version suffix
        ("a_b_c", "a_b", "c"),  # split at the LAST underscore, not the first
    ],
)
def test_trip_id_is_split_at_the_version_suffix(
    trip_id, expected_base, expected_version
):
    assert base_trip_id(trip_id) == expected_base
    assert schedule_version(trip_id) == expected_version


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00:00:00", 0),
        ("08:30:00", 30600),
        ("23:59:59", 86399),
        ("24:15:00", 87300),  # past midnight — a trip that started the previous evening
        ("25:30:00", 91800),
        ("not a time", None),
    ],
)
def test_gtfs_times_past_midnight_are_offsets_not_clock_times(value, expected):
    """GTFS encodes after-midnight stops as HH >= 24, which must not wrap or raise."""
    assert gtfs_time_to_seconds(value) == expected


def test_service_day_starts_at_noon_minus_twelve_on_a_normal_day():
    """On a date with no transition, the rule agrees with local midnight."""
    start = service_day_start(date(2026, 8, 5))

    assert start == datetime(2026, 8, 5, 4, 0, tzinfo=UTC)  # 00:00 EDT
    assert start.astimezone(SERVICE_TZ).hour == 0


def test_service_day_start_on_spring_forward_is_not_local_midnight():
    """REGRESSION: timedelta arithmetic on an aware datetime is wall-clock.

    2026-03-08 is the spring-forward date. Noon is unambiguously EDT (UTC-4), so
    noon-12h is 04:00 UTC. Naive local midnight that day is 00:00 EST = 05:00 UTC.
    They differ by exactly the DST gap, and the original implementation returned the
    naive answer while the Spark path returned the correct one — so one stop time
    resolved to two different instants depending on the code path.
    """
    start = service_day_start(date(2026, 3, 8))

    assert start == datetime(2026, 3, 8, 4, 0, tzinfo=UTC)
    assert start != datetime(2026, 3, 8, 5, 0, tzinfo=UTC)


def test_service_day_start_on_fall_back_is_not_local_midnight():
    """2026-11-01 is fall-back; noon is EST (UTC-5), so noon-12h is 05:00 UTC."""
    assert service_day_start(date(2026, 11, 1)) == datetime(
        2026, 11, 1, 5, 0, tzinfo=UTC
    )


def test_after_midnight_stop_time_resolves_past_the_service_date():
    """A 25:30 stop on Aug 5 is 01:30 local on Aug 6, still service_date Aug 5."""
    resolved = scheduled_timestamp(date(2026, 8, 5), "25:30:00")

    assert resolved is not None
    local = resolved.astimezone(SERVICE_TZ)
    assert (local.date(), local.hour, local.minute) == (date(2026, 8, 6), 1, 30)


# --------------------------------------------------------------------------
# decode.py — protobuf to rows
# --------------------------------------------------------------------------


def snapshot(message, moment=None) -> Snapshot:
    moment = moment or at(0)
    return Snapshot(captured_at=moment, key="test", message=message)


def test_vehicle_without_a_trip_is_dropped():
    """A deadheading vehicle cannot join a schedule, so it never enters the pipeline."""
    message = vehicle_positions([{"trip_id": "T1", "stop_sequence": 1}])
    message.entity[0].vehicle.trip.ClearField("trip_id")

    assert list(vehicle_position_rows(snapshot(message))) == []


def test_decode_splits_trip_id_and_derives_service_date_from_the_feed():
    message = vehicle_positions(
        [{"trip_id": "11970073_20660", "stop_sequence": 5, "start_date": "20260804"}]
    )

    row = next(iter(vehicle_position_rows(snapshot(message))))

    assert row["scheduled_trip_id"] == "11970073"
    assert row["schedule_version"] == "20660"
    # From trip.start_date, not the capture date — an after-midnight trip belongs to the
    # previous service day and only the feed knows which.
    assert row["service_date"] == "2026-08-04"


def test_skipped_stops_never_become_rows():
    """A SKIPPED stop is one the vehicle will not serve; it carries no times."""
    message = trip_updates(
        [
            {
                "trip_id": "T1",
                "stops": [
                    {"stop_sequence": 1, "arrival_ts": at(5)},
                    {"stop_sequence": 2, "arrival_ts": None, "relationship": "SKIPPED"},
                ],
            }
        ]
    )

    rows = list(trip_update_rows(snapshot(message)))

    assert [r["stop_sequence"] for r in rows] == [1]


def test_zero_timestamps_are_treated_as_missing_not_as_1970():
    """REGRESSION: WMATA sets arrival.time to 0 rather than omitting it (473/629,030).

    Zero is a valid protobuf int but not a valid instant, and one of them destroyed a
    min/max over predictions and reported drift as 56 years.
    """
    message = trip_updates([{"trip_id": "T1", "stops": [{"stop_sequence": 1}]}])
    message.entity[0].trip_update.stop_time_update[0].arrival.time = 0

    row = next(iter(trip_update_rows(snapshot(message))))

    assert row["arrival_ts"] is None


# --------------------------------------------------------------------------
# Spark-backed derivation
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spark():
    from src.etl.spark import build_session

    session = build_session(app_name="etl-tests", shuffle_partitions=1, cores="1")
    yield session
    session.stop()


def stage(spark, tmp_path, snapshots, kind: str):
    """Decode snapshots to Parquet and read them back as Spark would in production."""
    builder, schema = (
        (vehicle_position_rows, VEHICLE_POSITION_SCHEMA)
        if kind == "vp"
        else (trip_update_rows, TRIP_UPDATE_SCHEMA)
    )
    rows = [
        row
        for moment, message in snapshots
        for row in builder(snapshot(message, moment))
    ]
    destination = tmp_path / kind
    ds.write_dataset(
        pa.Table.from_pylist(rows, schema=schema), destination, format="parquet"
    )
    return spark.read.parquet(str(destination))


def arrival_epochs(frame):
    """Collect arrival instants as unix seconds.

    Timestamps are compared as epochs, not as datetime objects: PySpark's `collect()`
    renders a TimestampType into a *naive* Python datetime in the driver's local zone,
    so an assertion against a UTC-aware datetime fails even when the stored instant is
    correct. Comparing epochs asserts the instant and is immune to where the test runs.
    """
    from pyspark.sql import functions as F

    rows = (
        frame.withColumn("epoch", F.unix_timestamp("actual_arrival_ts"))
        .orderBy("stop_sequence")
        .collect()
    )
    return [r["epoch"] for r in rows]


def test_vp_arrival_is_the_midpoint_of_the_bracketing_snapshots(spark, tmp_path):
    from src.etl.arrivals import derive_vp_arrivals

    # Stops 1, 2, 3 at one-minute intervals: two arrival transitions.
    observations = stage(
        spark, tmp_path, progressing_vehicle("T1", [(0, 1), (1, 2), (2, 3)]), "vp"
    )

    derived = derive_vp_arrivals(observations)
    arrivals = derived.orderBy("stop_sequence").collect()

    assert [r["stop_sequence"] for r in arrivals] == [2, 3]
    assert [r["arrival_bracket_sec"] for r in arrivals] == [60, 60]
    assert all(r["arrival_confident"] for r in arrivals)
    # Arrival at stop 2 is bracketed by minutes 0 and 1, so the estimate is minute 0.5.
    assert arrival_epochs(derived) == [epoch(at(0)) + 30, epoch(at(1)) + 30]


def test_vp_feed_dropout_produces_a_wide_bracket_flagged_not_dropped(spark, tmp_path):
    """REGRESSION: 1.3% of real transitions span minutes of absence, up to 44 min."""
    from src.etl.arrivals import derive_vp_arrivals

    observations = stage(
        spark, tmp_path, progressing_vehicle("T1", [(0, 1), (10, 2)]), "vp"
    )

    arrivals = derive_vp_arrivals(observations).collect()

    assert len(arrivals) == 1, "the row is kept, not discarded"
    assert arrivals[0]["arrival_bracket_sec"] == 600
    assert arrivals[0]["arrival_confident"] is False


def test_vp_records_stops_passed_between_polls(spark, tmp_path):
    """A jump of 2 means one stop was never observed; the span must be visible."""
    from src.etl.arrivals import derive_vp_arrivals

    observations = stage(
        spark, tmp_path, progressing_vehicle("T1", [(0, 1), (1, 3)]), "vp"
    )

    arrivals = derive_vp_arrivals(observations).collect()

    assert arrivals[0]["sequence_jump"] == 2


def test_tu_uses_the_last_forward_looking_prediction_not_the_drifted_one(
    spark, tmp_path
):
    """REGRESSION: the whole TripUpdates estimator.

    Real feed behaviour: the stop never leaves the list, and after the train passes the
    prediction keeps drifting later (median +53s). Taking the final value lands +58s
    late; taking the last still-in-the-future value is unbiased (median +2s).

    Stop 1 is predicted to arrive at minute 3. Snapshots at minutes 0-2 see that as
    upcoming. At minute 4 it is already past, and at minute 5 the value has drifted
    on to minute 4 — still in the past at capture, which is what the real feed does.
    The estimate must be minute 3, the last forward-looking value, not minute 4.
    """
    from src.etl.arrivals import derive_tu_arrivals

    # (capture minute, predicted arrival minute). The prediction must actually cross
    # into the past for the arrival to be detectable — a "drifted" value still in the
    # future is just a later prediction, and yields no arrival at all.
    timeline = [(0, 3), (1, 3), (2, 3), (4, 3), (5, 4)]
    snapshots = [
        (
            at(minute),
            trip_updates(
                [
                    {
                        "trip_id": "T1",
                        "stops": [{"stop_sequence": 1, "arrival_ts": at(predicted)}],
                    }
                ],
                captured_at=at(minute),
            ),
        )
        for minute, predicted in timeline
    ]

    derived = derive_tu_arrivals(stage(spark, tmp_path, snapshots, "tu"))

    assert derived.count() == 1
    assert arrival_epochs(derived) == [epoch(at(3))]


def test_tu_ignores_stops_never_observed_going_stale(spark, tmp_path):
    """A stop still in the future when the window ends has not been arrived at.

    Without this, every in-flight trip contributes a fabricated arrival at the edge —
    which is what an absolute window cutoff did, inflating TripUpdates rows from ~1,600
    to 4,073 on real data.
    """
    from src.etl.arrivals import derive_tu_arrivals

    snapshots = [
        (
            at(minute),
            trip_updates(
                [
                    {
                        "trip_id": "T1",
                        "stops": [{"stop_sequence": 1, "arrival_ts": at(99)}],
                    }
                ],
                captured_at=at(minute),
            ),
        )
        for minute in (0, 1, 2)
    ]

    assert derive_tu_arrivals(stage(spark, tmp_path, snapshots, "tu")).count() == 0


def test_reused_trip_id_is_split_into_separate_runs(spark, tmp_path):
    """REGRESSION: the −4,800 second durations.

    WMATA reuses a trip_id within a service day — trip 12072038_20660 ran its whole stop
    sequence twice in one 3-hour window. Keying on trip_id alone paired stop 11 of the
    second run with stop 10 of the first.
    """
    from src.etl.arrivals import combine_arrivals, derive_vp_arrivals

    # Stops 1,2,3 then 1,2,3 again half an hour later.
    sequences = [(0, 1), (1, 2), (2, 3), (30, 1), (31, 2), (32, 3)]
    observations = stage(spark, tmp_path, progressing_vehicle("T1", sequences), "vp")

    arrivals = derive_vp_arrivals(observations)
    combined = combine_arrivals(arrivals, arrivals.limit(0)).collect()

    runs = {(r["trip_run"], r["stop_sequence"]) for r in combined}
    assert {r for r, _ in runs} == {0, 1}, "both journeys must be present, numbered"
    # No (run, stop) key appears twice — the condition that failed before.
    assert len(runs) == len(combined)


def test_segments_never_pair_across_two_runs(spark, tmp_path):
    """The consequence of the fix: no negative durations from interleaved runs."""
    from src.etl.arrivals import combine_arrivals, derive_vp_arrivals
    from src.etl.segments import build_segments

    sequences = [(0, 1), (1, 2), (2, 3), (30, 1), (31, 2), (32, 3)]
    observations = stage(spark, tmp_path, progressing_vehicle("T1", sequences), "vp")
    arrivals = derive_vp_arrivals(observations)
    combined = combine_arrivals(arrivals, arrivals.limit(0))

    empty_schedule = combined.select(
        "scheduled_trip_id",
        "service_date",
        "stop_sequence",
        combined["actual_arrival_ts"].alias("scheduled_arrival_ts"),
    ).limit(0)

    segments = build_segments(
        combined,
        empty_schedule,
        mode="rail",
        static_gtfs_version="test",
        bundle_schedule_version=None,
    ).collect()

    assert segments, "segments should be produced"
    assert all(s["actual_duration_sec"] >= 0 for s in segments)
    assert all(s["stop_span"] == 1 for s in segments)


# --------------------------------------------------------------------------
# quality.py
# --------------------------------------------------------------------------


def test_hours_that_have_not_elapsed_are_not_reported_as_downtime():
    """REGRESSION: a same-day run flagged every remaining hour as collector downtime.

    An hour cannot be judged complete before it has finished, and a blocking issue
    raised on a healthy archive trains the reader to ignore the report.
    """
    summary = {
        "feed": "rail_vehicle_positions",
        "snapshots_by_hour": {
            "year=2026/month=08/day=05/hour=10/": 60,
            "year=2026/month=08/day=05/hour=11/": 0,  # future at `now` below
        },
    }

    coverage = check_snapshot_coverage(
        summary, now=datetime(2026, 8, 5, 11, 30, tzinfo=UTC)
    )

    assert coverage.empty_hours == []
    assert len(coverage.pending_hours) == 1
    assert coverage.complete_hours == 1
    assert coverage.missing_snapshots == 0


def test_an_elapsed_empty_hour_is_reported_as_downtime():
    summary = {
        "feed": "rail_vehicle_positions",
        "snapshots_by_hour": {
            "year=2026/month=08/day=05/hour=10/": 0,
            "year=2026/month=08/day=05/hour=11/": 31,
        },
    }

    coverage = check_snapshot_coverage(
        summary, now=datetime(2026, 8, 5, 20, 0, tzinfo=UTC)
    )

    assert len(coverage.empty_hours) == 1
    assert coverage.short_hours == [("year=2026/month=08/day=05/hour=11/", 31)]
    assert coverage.missing_snapshots == 89  # 120 expected, 31 observed


# --------------------------------------------------------------------------
# decode.write_window — staging layout
# --------------------------------------------------------------------------


def test_adjacent_windows_do_not_delete_each_others_partitions(tmp_path):
    """REGRESSION: the per-day backfill loop silently truncated the previous day.

    A UTC window straddles two service dates, because trips after local midnight belong
    to the previous service day. With `delete_matching`, decoding Aug 5 (which emits a
    sliver of Aug 4) deleted the whole existing Aug 4 partition and left only the
    sliver — and the output looked entirely normal.
    """
    from src.etl.decode import VEHICLE_POSITION_SCHEMA, write_window

    def table(service_dates):
        return pa.Table.from_pylist(
            [
                {"captured_at": at(0), "service_date": d, "trip_id": f"T{i}"}
                for i, d in enumerate(service_dates)
            ],
            schema=VEHICLE_POSITION_SCHEMA,
        )

    destination = tmp_path / "vp"
    # Day one: a full Aug 4, plus the usual sliver of Aug 3.
    write_window(
        table(["2026-08-03", "2026-08-04", "2026-08-04"]),
        destination,
        datetime(2026, 8, 4, 4, tzinfo=UTC),
        datetime(2026, 8, 5, 4, tzinfo=UTC),
    )
    # Day two: Aug 5, emitting a sliver of Aug 4 as every real window does.
    write_window(
        table(["2026-08-04", "2026-08-05"]),
        destination,
        datetime(2026, 8, 5, 4, tzinfo=UTC),
        datetime(2026, 8, 6, 4, tzinfo=UTC),
    )

    counts = (
        pq.read_table(destination).to_pandas().groupby("service_date").size().to_dict()
    )
    # Aug 4 keeps day one's two rows AND gains day two's sliver: 3, not 1.
    assert counts["2026-08-04"] == 3
    assert counts["2026-08-03"] == 1
    assert counts["2026-08-05"] == 1


def test_rerunning_the_same_window_replaces_its_own_output(tmp_path):
    """Idempotency has to survive the fix: same window twice must not double."""
    from src.etl.decode import VEHICLE_POSITION_SCHEMA, write_window

    rows = [
        {"captured_at": at(0), "service_date": "2026-08-05", "trip_id": "T1"},
        {"captured_at": at(1), "service_date": "2026-08-05", "trip_id": "T2"},
    ]
    table = pa.Table.from_pylist(rows, schema=VEHICLE_POSITION_SCHEMA)
    destination = tmp_path / "vp"
    bounds = (datetime(2026, 8, 5, 4, tzinfo=UTC), datetime(2026, 8, 6, 4, tzinfo=UTC))

    write_window(table, destination, *bounds)
    write_window(table, destination, *bounds)

    assert pq.read_table(destination).num_rows == 2


def test_only_fully_covered_service_dates_are_written():
    """REGRESSION: a run for one day wrote a partial partition for its neighbour.

    A --date window is 04:00 UTC to 04:00 UTC next day. It contains exactly one whole
    local service day; the neighbouring dates it touches are slivers and must not be
    written, or a 250-row Aug 4 sits in the output looking like a complete day.
    """
    from src.etl.pipeline import fully_covered_service_dates, service_day_bounds

    start, end = service_day_bounds("2026-08-06")

    assert fully_covered_service_dates(start, end) == ["2026-08-06"]


def test_a_multi_day_window_covers_every_whole_day_inside_it():
    from src.etl.pipeline import fully_covered_service_dates, service_day_bounds

    start, _ = service_day_bounds("2026-08-04")
    _, end = service_day_bounds("2026-08-06")

    assert fully_covered_service_dates(start, end) == [
        "2026-08-04",
        "2026-08-05",
        "2026-08-06",
    ]


def test_a_partial_window_covers_no_service_date():
    """Three hours in the middle of a day is authoritative for nothing."""
    from src.etl.pipeline import fully_covered_service_dates

    start = datetime(2026, 8, 5, 11, tzinfo=UTC)
    end = datetime(2026, 8, 5, 14, tzinfo=UTC)

    assert fully_covered_service_dates(start, end) == []


# --------------------------------------------------------------------------
# catchup.py — the scheduling logic
# --------------------------------------------------------------------------

ARCHIVE_START = datetime(2026, 8, 4, 22, tzinfo=UTC)


def pending(now, done=frozenset(), lookback=14, force=False, archive=ARCHIVE_START):
    from src.etl.catchup import pending_service_dates

    return pending_service_dates(now, set(done), lookback, force, archive)


def test_a_service_day_is_not_pending_until_it_closes():
    """REGRESSION: a service day runs PAST local midnight, and the window must too.

    Aug 6's service day starts at 04:00 UTC Aug 6 (local midnight) and ends four hours
    after the *next* local midnight — 08:00 UTC Aug 7 — because trips carrying
    `start_date = 2026-08-06` are still running 3.5 hours into Aug 7 local time.

    Closing at local midnight, as this originally did, left 424 vehicle-records for a
    single service date outside the window. The symptom was subtle: the segment table
    simply had no rows at local hours 0-3, which reads as "service stops at midnight"
    rather than as a bug.
    """
    from src.etl.schedule import service_day_end

    assert service_day_end(date(2026, 8, 6)) == datetime(2026, 8, 7, 8, tzinfo=UTC)

    # One second before the day closes, and exactly at it.
    assert "2026-08-06" not in pending(datetime(2026, 8, 7, 7, 59, tzinfo=UTC))
    assert "2026-08-06" in pending(datetime(2026, 8, 7, 8, 0, tzinfo=UTC))

    # The old boundary must no longer be treated as closing time.
    assert "2026-08-06" not in pending(datetime(2026, 8, 7, 4, 0, tzinfo=UTC))


def test_service_day_window_includes_the_post_midnight_overhang():
    """`--date D` must decode past local midnight, or it clips D's late-night tail."""
    from src.etl.pipeline import service_day_bounds

    start, end = service_day_bounds("2026-08-07")

    assert start == datetime(2026, 8, 7, 4, tzinfo=UTC)  # local midnight
    assert end == datetime(2026, 8, 8, 8, tzinfo=UTC)  # +4h past the next one
    assert (end - start).total_seconds() / 3600 == 28


def test_window_spans_28_hours_across_both_dst_transitions():
    """The overhang must not silently become 27 or 29 hours twice a year."""
    from src.etl.pipeline import service_day_bounds

    for day in ("2026-03-08", "2026-11-01", "2026-08-07"):
        start, end = service_day_bounds(day)
        assert (end - start).total_seconds() / 3600 == 28, day


def test_coverage_requires_the_overhang_not_just_midnight():
    """A window stopping at local midnight is NOT authoritative for that service day."""
    from src.etl.pipeline import fully_covered_service_dates
    from src.etl.schedule import service_day_start

    start = service_day_start(date(2026, 8, 7))
    to_midnight = service_day_start(date(2026, 8, 8))  # the old, wrong end
    assert fully_covered_service_dates(start, to_midnight) == []

    _, proper_end = __import__(
        "src.etl.pipeline", fromlist=["service_day_bounds"]
    ).service_day_bounds("2026-08-07")
    assert fully_covered_service_dates(start, proper_end) == ["2026-08-07"]


def test_dates_before_the_archive_begins_are_never_proposed():
    """REGRESSION: the first dry run proposed 14 dates, 12 of them with no data.

    Each would have started a Spark session only to fail on missing input, so the
    scheduled job would have reported failure every night. Aug 4 is excluded too — the
    collector started at 22:00 UTC that day, so that service day is only partly covered.
    """
    result = pending(datetime(2026, 8, 7, 12, tzinfo=UTC))

    assert result == ["2026-08-05", "2026-08-06"]
    assert all(d >= "2026-08-05" for d in result)


def test_dates_already_in_s3_are_skipped_and_force_overrides():
    now = datetime(2026, 8, 7, 12, tzinfo=UTC)

    assert pending(now, done={"2026-08-05"}) == ["2026-08-06"]
    assert pending(now, done={"2026-08-05"}, force=True) == ["2026-08-05", "2026-08-06"]


def test_a_gap_in_the_middle_is_picked_up():
    """The catch-up property itself: downtime in the middle is not skipped forever."""
    now = datetime(2026, 8, 12, 12, tzinfo=UTC)

    result = pending(now, done={"2026-08-05", "2026-08-06", "2026-08-09"})

    assert result == ["2026-08-07", "2026-08-08", "2026-08-10", "2026-08-11"]


def test_lookback_bounds_the_candidate_list():
    now = datetime(2026, 9, 1, 12, tzinfo=UTC)

    assert pending(now, lookback=3) == [
        "2026-08-29",
        "2026-08-30",
        "2026-08-31",
    ]


def test_sync_uploads_only_parquet_not_sparks_crc_sidecars(tmp_path):
    """Spark writes .crc checksums beside each part file; they are Hadoop-internal."""
    from src.etl.config import EtlConfig
    from src.etl.processed import sync_partitions

    partition = tmp_path / "service_date=2026-08-05"
    partition.mkdir(parents=True)
    (partition / "part-00000.snappy.parquet").write_bytes(b"data")
    (partition / ".part-00000.snappy.parquet.crc").write_bytes(b"crc")
    (partition / "_SUCCESS").write_bytes(b"")

    class StubS3:
        def __init__(self):
            self.uploaded = []

        def get_paginator(self, _):
            return type("P", (), {"paginate": lambda *a, **k: iter([{}])})()

        def upload_file(self, _local, _bucket, key):
            self.uploaded.append(key)

    client = StubS3()
    config = EtlConfig.from_env({"S3_BUCKET": "b", "S3_PROCESSED_PREFIX": "processed/"})
    result = sync_partitions(config, tmp_path, ["2026-08-05"], s3=client)

    assert result == {"2026-08-05": 1}
    assert client.uploaded == [
        "processed/segments/service_date=2026-08-05/part-00000.snappy.parquet"
    ]


# --------------------------------------------------------------------------
# Polling cadence — measured, not assumed
# --------------------------------------------------------------------------


def keys_at(seconds: list[int], feed: str = "rail_vehicle_positions") -> list[str]:
    """Snapshot keys with the given unix timestamps, in the collector's layout."""
    return [
        f"raw/{feed}/year=2026/month=08/day=05/hour=12/{feed}-{t}.pb.gz"
        for t in seconds
    ]


def test_cadence_is_measured_from_the_keys():
    """60s and 30s eras are both read correctly from the data itself."""
    from src.etl.archive import modal_interval_seconds

    assert modal_interval_seconds(keys_at([0, 60, 120, 180])) == 60
    assert modal_interval_seconds(keys_at([0, 30, 60, 90, 120])) == 30


def test_an_outage_gap_does_not_skew_the_measured_cadence():
    """The mode, not the mean — one 40-minute hole would drag an average far off."""
    from src.etl.archive import modal_interval_seconds

    stamps = [0, 30, 60, 90, 2490, 2520, 2550]  # 40-minute outage in the middle
    assert modal_interval_seconds(keys_at(stamps)) == 30


@pytest.mark.parametrize("stamps", [[], [0]])
def test_cadence_is_unknown_when_there_is_no_gap_to_measure(stamps):
    from src.etl.archive import modal_interval_seconds

    assert modal_interval_seconds(keys_at(stamps)) is None


def test_duplicate_keys_do_not_vote_for_a_zero_second_cadence():
    from src.etl.archive import modal_interval_seconds

    # Three copies of one timestamp plus two real 30s gaps.
    assert modal_interval_seconds(keys_at([0, 0, 0, 30, 60])) == 30


def test_coverage_uses_the_measured_cadence_not_a_constant():
    """REGRESSION: the only collector-downtime detector, silently broken by 30s polling.

    With EXPECTED_SNAPSHOTS_PER_HOUR hardcoded to 60, a 30s-cadence hour holding 120
    files trivially clears a 54-file bar — and so does an hour holding 60, which is
    *half the data missing*. Measuring the cadence keeps the check meaningful.
    """
    summary = {
        "feed": "rail_vehicle_positions",
        "interval_seconds": 30,
        "snapshots_by_hour": {
            "year=2026/month=08/day=05/hour=10/": 120,  # complete at 30s
            "year=2026/month=08/day=05/hour=11/": 60,  # half missing at 30s
        },
    }

    coverage = check_snapshot_coverage(
        summary, now=datetime(2026, 8, 5, 20, tzinfo=UTC)
    )

    assert coverage.expected_per_hour == 120
    assert coverage.complete_hours == 1
    assert coverage.short_hours == [("year=2026/month=08/day=05/hour=11/", 60)]
    assert coverage.missing_snapshots == 60


def test_coverage_falls_back_to_the_constant_when_cadence_is_unmeasurable():
    from src.etl.config import EXPECTED_SNAPSHOTS_PER_HOUR

    summary = {
        "feed": "rail_vehicle_positions",
        "interval_seconds": None,
        "snapshots_by_hour": {"year=2026/month=08/day=05/hour=10/": 60},
    }

    coverage = check_snapshot_coverage(
        summary, now=datetime(2026, 8, 5, 20, tzinfo=UTC)
    )

    assert coverage.expected_per_hour == EXPECTED_SNAPSHOTS_PER_HOUR
    assert coverage.complete_hours == 1


def test_negative_duration_is_bounded_by_the_rows_own_bracket(spark, tmp_path):
    """REGRESSION: a fixed 60s bound is twice as permissive as it should be at 30s.

    A -45s duration is a plausible rounding artefact when the arrival was bracketed by
    snapshots 60s apart, and impossible when they were 30s apart. The bound has to come
    from the row, because the archive spans both cadences.
    """
    from src.etl.segments import build_segments

    def segment_with(bracket_sec: int, duration_sec: int):
        rows = [
            # Stop 1 then stop 2; the lag() pairing turns these into one segment whose
            # duration is the gap between the two arrival timestamps.
            {
                "trip_id": "T1",
                "trip_run": 0,
                "scheduled_trip_id": "T1",
                "schedule_version": "1",
                "route_id": "RED",
                "direction_id": 0,
                "service_date": "2026-08-05",
                "stop_id": f"PF_{seq}",
                "stop_sequence": seq,
                "actual_arrival_ts": at(0) + timedelta(seconds=offset),
                "arrival_bracket_sec": bracket_sec,
                "arrival_confident": True,
                "arrival_source": "vehicle_position",
                "observed_at_utc": at(0),
            }
            for seq, offset in ((1, 0), (2, duration_sec))
        ]
        arrivals = spark.createDataFrame(rows)
        empty_schedule = arrivals.select(
            "scheduled_trip_id",
            "service_date",
            "stop_sequence",
            arrivals["actual_arrival_ts"].alias("scheduled_arrival_ts"),
        ).limit(0)
        built = build_segments(arrivals, empty_schedule, "rail", "test", None).collect()
        return built[0]

    # -45s duration: within a 60s bracket, outside a 30s one.
    assert segment_with(60, -45)["duration_plausible"] is True
    assert segment_with(30, -45)["duration_plausible"] is False
