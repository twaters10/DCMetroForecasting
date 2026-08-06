"""Step 4: arrival events into a trip-segment table.

One row per observed traversal of a segment — stop A to stop B on one trip — with the
scheduled and actual duration side by side and their difference.

Two modelling decisions are visible in the schema and both are consequences of what
the feeds actually contain rather than preferences:

**`actual_departure_ts` is the upstream arrival.** WMATA populates
`departure.time` on only 4.5% of stop_time_updates, and exclusively at
`stop_sequence` 1 — never on an intermediate stop, and never alongside an arrival on
the same stop. So a true departure is unavailable for every segment except a trip's
first. The upstream *arrival* is used instead, which means **dwell time at the
upstream stop is included in `actual_duration_sec`**. This is a real limitation, not a
rounding detail: for a rail segment of two to three minutes, a 20–30 second dwell is
a meaningful share. It is recorded honestly rather than hidden, and
`scheduled_duration_sec` is computed the same way (upstream scheduled arrival to
downstream scheduled arrival) so the two sides are comparable and `delay_sec` is not
contaminated by the asymmetry.

**Segments are built from consecutive *observed* arrivals, not consecutive scheduled
stops.** Where a stop was passed between polls, the segment spans the gap and
`stop_span > 1` records it. Dropping those rows would silently bias the dataset
against fast traversals; keeping them unlabelled would corrupt per-segment duration.
"""

from __future__ import annotations

from typing import Final

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .config import SERVICE_TZ

# A segment whose actual duration exceeds this is almost certainly a derivation
# artefact rather than a delayed train — a vehicle that vanished from the feed and
# returned, or a trip_id reused across service days. Flagged in the output and counted
# in the DQ report rather than dropped, so the rate stays visible.
IMPLAUSIBLE_DURATION_SEC: Final[int] = 60 * 60

# Ratio of actual to scheduled duration beyond which a row is suspect. A train can
# genuinely take three times as long as scheduled; ten times means something else.
IMPLAUSIBLE_DURATION_RATIO: Final[float] = 10.0

SEGMENT_COLUMNS: Final[tuple[str, ...]] = (
    "trip_id",
    "trip_run",
    "scheduled_trip_id",
    "route_id",
    "direction_id",
    "mode",
    "from_stop_id",
    "to_stop_id",
    "stop_sequence",
    "from_stop_sequence",
    "stop_span",
    "scheduled_departure_ts",
    "scheduled_arrival_ts",
    "scheduled_duration_sec",
    "actual_departure_ts",
    "actual_arrival_ts",
    "actual_duration_sec",
    "delay_sec",
    "arrival_source",
    "arrival_bracket_sec",
    "arrival_confident",
    "duration_plausible",
    "schedule_version_agrees",
    "static_gtfs_version",
    "service_date",
    "observed_at_utc",
)


def build_segments(
    arrivals: DataFrame,
    scheduled: DataFrame,
    mode: str,
    static_gtfs_version: str,
    bundle_schedule_version: str | None,
) -> DataFrame:
    """Pair consecutive arrivals into segments and attach the schedule.

    `arrivals` comes from `arrivals.combine_arrivals`; `scheduled` is
    `schedule.scheduled_stop_times` with absolute timestamps already resolved for the
    service date (see `add_scheduled_timestamps`).
    """
    # Partition by (trip_id, trip_run), not trip_id alone: WMATA reuses a trip_id for
    # more than one journey in a service day, and pairing across two runs produces
    # durations of minus an hour. See arrivals.assign_trip_runs.
    #
    # Order by stop_sequence, not by timestamp. Sequence is the trip's own topology;
    # arrival timestamps can tie or invert slightly when two stops fall inside one
    # polling interval, which would silently reorder a segment.
    trip_order = Window.partitionBy("trip_id", "trip_run").orderBy("stop_sequence")

    paired = (
        arrivals.withColumn("from_stop_id", F.lag("stop_id").over(trip_order))
        .withColumn("from_stop_sequence", F.lag("stop_sequence").over(trip_order))
        .withColumn("actual_departure_ts", F.lag("actual_arrival_ts").over(trip_order))
        .withColumn("from_bracket_sec", F.lag("arrival_bracket_sec").over(trip_order))
        .withColumn("from_confident", F.lag("arrival_confident").over(trip_order))
        .filter(F.col("from_stop_sequence").isNotNull())
    )

    segments = paired.select(
        "trip_id",
        "trip_run",
        "scheduled_trip_id",
        "schedule_version",
        "route_id",
        "direction_id",
        "service_date",
        "observed_at_utc",
        "arrival_source",
        F.col("from_stop_id"),
        F.col("stop_id").alias("to_stop_id"),
        F.col("from_stop_sequence").cast("int").alias("from_stop_sequence"),
        F.col("stop_sequence").cast("int").alias("stop_sequence"),
        (F.col("stop_sequence") - F.col("from_stop_sequence"))
        .cast("int")
        .alias("stop_span"),
        "actual_departure_ts",
        "actual_arrival_ts",
        (
            F.unix_timestamp("actual_arrival_ts")
            - F.unix_timestamp("actual_departure_ts")
        )
        .cast("int")
        .alias("actual_duration_sec"),
        # The segment's uncertainty is driven by whichever endpoint is vaguer, so take
        # the wider bracket rather than either one alone.
        F.greatest(
            F.coalesce(F.col("arrival_bracket_sec"), F.lit(0)),
            F.coalesce(F.col("from_bracket_sec"), F.lit(0)),
        )
        .cast("int")
        .alias("arrival_bracket_sec"),
        (F.col("arrival_confident") & F.col("from_confident")).alias(
            "arrival_confident"
        ),
    )

    # Attach the schedule at both endpoints. Joining twice on the version-stable
    # scheduled_trip_id — never on the versioned trip_id, which matches 0%.
    #
    # `service_date` is a join key, not just a passenger. The same scheduled_trip_id
    # runs on many dates, and its scheduled *timestamps* differ per date, so a schedule
    # frame spanning N dates would match each segment N times and multiply the output.
    # With one date in the run the bug is invisible; it appears the first time a range
    # is processed, which is exactly when nobody is watching row counts.
    schedule_keys = ["scheduled_trip_id", "service_date"]
    from_schedule = scheduled.select(
        *schedule_keys,
        F.col("stop_sequence").alias("from_stop_sequence"),
        F.col("scheduled_arrival_ts").alias("scheduled_departure_ts"),
    )
    to_schedule = scheduled.select(
        *schedule_keys,
        F.col("stop_sequence"),
        F.col("scheduled_arrival_ts"),
    )

    joined = segments.join(
        from_schedule, on=[*schedule_keys, "from_stop_sequence"], how="left"
    ).join(to_schedule, on=[*schedule_keys, "stop_sequence"], how="left")

    scheduled_duration = F.unix_timestamp("scheduled_arrival_ts") - F.unix_timestamp(
        "scheduled_departure_ts"
    )

    enriched = (
        joined.withColumn("scheduled_duration_sec", scheduled_duration.cast("int"))
        .withColumn(
            "delay_sec",
            (F.col("actual_duration_sec") - F.col("scheduled_duration_sec")).cast(
                "int"
            ),
        )
        .withColumn("mode", F.lit(mode))
        .withColumn("static_gtfs_version", F.lit(static_gtfs_version))
        .withColumn(
            # Whether the timetable joined against is the one the realtime feed was
            # emitting. The join works either way (scheduled_trip_id is version-stable)
            # but a mismatch means scheduled times may not have been in force, so every
            # row carries the verdict rather than the run carrying it once.
            "schedule_version_agrees",
            F.col("schedule_version").eqNullSafe(F.lit(bundle_schedule_version)),
        )
        .withColumn(
            "duration_plausible",
            F.col("actual_duration_sec").isNotNull()
            # A small negative duration is the two endpoints landing inside one
            # polling interval; anything beyond that cannot be real.
            #
            # Bounded by the row's OWN bracket rather than a global constant. The
            # collector's cadence changed from 60s to 30s, so the archive spans both
            # and a fixed 60 would be twice as permissive as it should be on newer
            # rows while being exactly right on older ones. `arrival_bracket_sec` is
            # measured per row, so it is correct in every era by construction.
            & (F.col("actual_duration_sec") >= -F.col("arrival_bracket_sec"))
            & (F.col("actual_duration_sec") <= IMPLAUSIBLE_DURATION_SEC)
            & (
                F.col("scheduled_duration_sec").isNull()
                | (F.col("scheduled_duration_sec") <= 0)
                | (
                    F.col("actual_duration_sec")
                    <= F.col("scheduled_duration_sec") * IMPLAUSIBLE_DURATION_RATIO
                )
            ),
        )
    )

    return enriched.select(*SEGMENT_COLUMNS)


def add_scheduled_timestamps(
    scheduled: DataFrame, service_date_column: str
) -> DataFrame:
    """Resolve GTFS second-offsets to absolute instants for a service date.

    This is the *only* place local time enters the pipeline, and it implements the GTFS
    service-day rule: the day starts at noon minus twelve hours in America/New_York,
    not at midnight. On a normal date the two agree. On a DST boundary they do not —
    local midnight is ambiguous in autumn and nonexistent in spring, while noon is
    always well defined — so anchoring at noon keeps a `25:30:00` stop time on a real
    instant. Everything downstream is UTC.
    """
    service_day_start = F.unix_timestamp(
        F.to_utc_timestamp(
            F.concat(F.col(service_date_column), F.lit(" 12:00:00")), SERVICE_TZ.key
        )
    ) - F.lit(12 * 3600)

    return (
        scheduled.withColumn("service_day_start_utc", service_day_start)
        .withColumn(
            "scheduled_arrival_ts",
            F.from_unixtime(
                F.col("service_day_start_utc") + F.col("scheduled_arrival_offset_sec")
            ).cast("timestamp"),
        )
        .withColumn(
            "scheduled_departure_ts",
            F.from_unixtime(
                F.col("service_day_start_utc") + F.col("scheduled_departure_offset_sec")
            ).cast("timestamp"),
        )
    )


def write_segments(segments: DataFrame, output_path: str) -> None:
    """Write the segment table as Parquet, partitioned by service_date.

    Partition overwrite is dynamic (set on the session in `spark.build_session`), so
    re-running a date range replaces exactly those `service_date` partitions and
    leaves the rest of the archive intact. `mode("overwrite")` with static
    partitioning would delete every previously processed day — the difference between
    a re-runnable pipeline and a destructive one.

    `repartition("service_date")` collapses each day to a single file. Without it every
    Spark partition writes its own fragment — eight ~130 KB files for one day of rail,
    so ~720 objects across a 90-day retention. That is precisely the small-file problem
    the decode pre-pass exists to avoid, and it would be reintroduced at the output. A
    service day is around 1 MB; one file per day is the right grain.
    """
    segments.repartition("service_date").write.mode("overwrite").partitionBy(
        "service_date"
    ).parquet(output_path)
