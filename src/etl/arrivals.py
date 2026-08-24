"""Step 3: turning snapshots into arrival events.

The archive holds *state*, sampled every 60 seconds. It contains no "vehicle arrived
at stop X at time T" record — that has to be derived, and the derivation is the part
of this pipeline most likely to be quietly wrong.

Two independent derivations, both measured on a real 3-hour rail window (2026-08-05
11:00–14:00 UTC, 360 snapshots):

**VehiclePositions — primary.** When a vehicle's `current_stop_sequence` increments
between consecutive snapshots, it arrived at the new stop inside that interval. The
arrival is *bracketed* by the two capture times, so the midpoint is the estimate and
the bracket width is a real error bar rather than an assumption. 6,119 transitions
found, 98.7% with a bracket inside 180s, median bracket 60s, 98.8% advancing exactly
one stop.

That the increment marks arrival and not departure was verified rather than assumed,
because the two differ by an entire segment: at the newly-incremented sequence the
reported status is `STOPPED_AT` 92.3% of the time, and 98.9% of (trip, stop) pairs are
already `STOPPED_AT` at first sighting. Note this means WMATA uses `IN_TRANSIT_TO`
with sequence N to mean "departed N", not the spec's "heading to N" — it is the last
status seen at a sequence 54.1% of the time. Reading it the spec's way would shift
every arrival by one segment.

**TripUpdates — fallback.** The obvious estimator here is "the last prediction before
the stop disappears from the feed". **That estimator does not exist for this feed**, and
assuming it does is the single biggest trap in this stage. Measured over 511 trips:

- the minimum `stop_sequence` in a trip's stop list **never advances** (0 of 511 trips)
- stop lists do not shrink as stops are served (2% shrink, and those are trips
  truncated at the far end)
- 28.6% of rows carry a predicted arrival that is already in the past
- once passed, the value keeps changing — only 7.5% freeze — drifting a further
  median +53s

So WMATA never marks a stop as served; it just keeps re-predicting, and the prediction
decays after the fact. Taking the *final* observed value therefore reads a stale drift,
not an arrival: it lands a median **+58s** after the observed truth, only 53.7% within
60s.

The estimator that does work is the **last prediction issued while the stop was still
in the future**, for stops observed crossing their own predicted arrival. Against
VehiclePositions' observed arrivals (6,036 overlapping pairs) that is essentially
unbiased — median **+2s**, p10 −25s, p90 +49s, **91.3% within 60s**. No bias correction
is applied or needed; the earlier apparent bias was an artifact of reading the wrong
value, not a property of WMATA's predictor.

This also yields a genuine bracket rather than a point guess: the arrival sits between
the last snapshot that predicted it forward and the first that predicted it past, so
`arrival_bracket_sec` means the same thing for both sources.

Every output row records which source it came from and how wide its bracket was, so
downstream modelling can weight or filter on provenance instead of trusting a blend.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .config import (
    MAX_ARRIVAL_BRACKET_SEC,
    SOURCE_TRIP_UPDATE,
    SOURCE_VEHICLE_POSITION,
)


def derive_vp_arrivals(observations: DataFrame) -> DataFrame:
    """Derive arrival events from VehiclePositions stop_sequence transitions.

    One row per observed arrival: (trip, stop_sequence) with a bracketed timestamp.
    """
    usable = observations.filter(
        F.col("stop_sequence").isNotNull() & F.col("trip_id").isNotNull()
    )

    # Partitioned by trip rather than vehicle: a vehicle serves many trips in a day,
    # and a trip is the unit whose stop_sequence progresses.
    #
    # Note `trip_id` alone is NOT unique per run — see `assign_trip_runs`. It is the
    # right partition here, where only consecutive-pair differences matter, but the
    # segment builder needs the finer key.
    timeline = Window.partitionBy("trip_id").orderBy("captured_at")

    with_previous = (
        usable.withColumn("prev_sequence", F.lag("stop_sequence").over(timeline))
        .withColumn("prev_captured_at", F.lag("captured_at").over(timeline))
        .withColumn("prev_stop_id", F.lag("stop_id").over(timeline))
    )

    advanced = with_previous.filter(
        F.col("prev_sequence").isNotNull()
        & (F.col("stop_sequence") > F.col("prev_sequence"))
    )

    bracket = F.unix_timestamp("captured_at") - F.unix_timestamp("prev_captured_at")

    return advanced.select(
        "trip_id",
        "scheduled_trip_id",
        "schedule_version",
        "route_id",
        "direction_id",
        "vehicle_id",
        "service_date",
        "stop_id",
        "stop_sequence",
        # Midpoint of the bracket. Unbiased if arrival is uniform within the interval,
        # which is the best available assumption at a fixed poll rate.
        F.from_unixtime(
            (F.unix_timestamp("prev_captured_at") + F.unix_timestamp("captured_at")) / 2
        )
        .cast("timestamp")
        .alias("actual_arrival_ts"),
        bracket.cast("int").alias("arrival_bracket_sec"),
        (F.col("stop_sequence") - F.col("prev_sequence"))
        .cast("int")
        .alias("sequence_jump"),
        F.lit(SOURCE_VEHICLE_POSITION).alias("arrival_source"),
        F.col("captured_at").alias("observed_at_utc"),
    ).withColumn(
        # A wide bracket means the vehicle dropped out of the feed and reappeared, so
        # the estimate is vague rather than wrong. Flagged, not dropped: the caller
        # decides, and the DQ report counts them.
        "arrival_confident",
        F.col("arrival_bracket_sec") <= MAX_ARRIVAL_BRACKET_SEC,
    )


def derive_tu_arrivals(observations: DataFrame) -> DataFrame:
    """Derive arrival events from the last forward-looking TripUpdates prediction.

    The arrival signal is a (trip, stop) *crossing its own predicted arrival*: observed
    at least once with the prediction still in the future, and at least once with it in
    the past. The estimate is the last prediction taken while it was still in the
    future, which measures as essentially unbiased against observed arrivals (median
    +2s, 91.3% within 60s).

    Two tempting alternatives are both wrong for this feed, and the module docstring has
    the evidence: the stop never disappears from the list, so "last value before
    disappearance" has no trigger; and the final observed value keeps drifting after the
    train has passed, landing a median +58s late.
    """
    usable = observations.filter(
        F.col("arrival_ts").isNotNull()
        & F.col("stop_sequence").isNotNull()
        & F.col("trip_id").isNotNull()
    ).withColumn(
        # How far ahead the predicted arrival was at the moment of capture. Positive
        # means still upcoming, negative means the prediction is already stale.
        "horizon_sec",
        F.unix_timestamp("arrival_ts") - F.unix_timestamp("captured_at"),
    )

    lifetime = Window.partitionBy("trip_id", "stop_sequence")

    # Only stops we watched cross their own predicted arrival: seen at least once
    # forward-looking and at least once stale. That crossing is the arrival signal —
    # it replaces the disappearance that this feed never provides.
    crossed = (
        usable.withColumn("max_horizon", F.max("horizon_sec").over(lifetime))
        .withColumn("min_horizon", F.min("horizon_sec").over(lifetime))
        .withColumn("observation_count", F.count("*").over(lifetime))
        .filter((F.col("max_horizon") > 0) & (F.col("min_horizon") < 0))
    )

    # The last still-forward-looking prediction. Ordering descending by capture time
    # among the forward-looking rows only, so the drifted post-arrival values — which
    # carry a +58s bias — are never selected.
    pre_arrival = Window.partitionBy("trip_id", "stop_sequence").orderBy(
        F.col("captured_at").desc()
    )
    selected = (
        crossed.filter(F.col("horizon_sec") > 0)
        .withColumn("recency", F.row_number().over(pre_arrival))
        .filter(F.col("recency") == 1)
    )

    return selected.select(
        "trip_id",
        "scheduled_trip_id",
        "schedule_version",
        "route_id",
        "direction_id",
        "vehicle_id",
        "service_date",
        "stop_id",
        "stop_sequence",
        F.col("arrival_ts").alias("actual_arrival_ts"),
        # The bracket is how stale the prediction was allowed to get before we caught
        # the crossing: the remaining horizon at the last forward-looking sighting. It
        # is an upper bound on the error, in the same units and meaning as the
        # VehiclePositions bracket, so `arrival_confident` compares like with like.
        F.col("horizon_sec").cast("int").alias("arrival_bracket_sec"),
        F.lit(1).cast("int").alias("sequence_jump"),
        F.lit(SOURCE_TRIP_UPDATE).alias("arrival_source"),
        F.col("captured_at").alias("observed_at_utc"),
        F.col("observation_count"),
    ).withColumn(
        "arrival_confident",
        F.col("arrival_bracket_sec") <= MAX_ARRIVAL_BRACKET_SEC,
    )


def combine_arrivals(vp_arrivals: DataFrame, tu_arrivals: DataFrame) -> DataFrame:
    """Prefer the observed arrival; fall back to the corrected prediction.

    A left anti-join rather than a union-then-deduplicate: the intent is "TripUpdates
    only where VehiclePositions saw nothing", and expressing that directly means a
    future change to the preference rule is a one-line edit rather than a subtle
    change to sort keys inside a window.
    """
    keys = ["trip_id", "stop_sequence"]
    columns = [
        "trip_id",
        "scheduled_trip_id",
        "schedule_version",
        "route_id",
        "direction_id",
        "vehicle_id",
        "service_date",
        "stop_id",
        "stop_sequence",
        "actual_arrival_ts",
        "arrival_bracket_sec",
        "sequence_jump",
        "arrival_source",
        "observed_at_utc",
        "arrival_confident",
    ]

    gap_filling = tu_arrivals.join(vp_arrivals.select(*keys), on=keys, how="left_anti")
    combined = vp_arrivals.select(*columns).unionByName(gap_filling.select(*columns))
    return assign_trip_runs(combined)


def assign_trip_runs(arrivals: DataFrame) -> DataFrame:
    """Split each trip_id into separate runs, and number them.

    **A `trip_id` is not unique per journey.** WMATA reuses one within a service day —
    trip `12072038_20660` traverses its whole stop sequence twice in a single 3-hour
    window, once around 12:11–12:38 and again around 13:29–13:56. Treating `trip_id` as
    the journey key interleaves the two runs when ordering by `stop_sequence`, which
    pairs stop 11 of the second run with stop 10 of the first and yields durations like
    −4,800 seconds. On a real sample that was 276 duplicated (trip, stop) keys and 284
    segments spanning zero stops.

    So runs are separated by sessionisation: order a trip's arrivals by time, and start
    a new run wherever `stop_sequence` fails to advance. Downstream keys on
    (`trip_id`, `trip_run`).

    A caveat this exposes rather than solves: static GTFS holds one scheduled timetable
    per `trip_id`, so a second run has no schedule of its own and its `delay_sec` is
    measured against the first run's times. `trip_run > 0` is carried into the output so
    those rows are identifiable.
    """
    ordering = Window.partitionBy("trip_id").orderBy(
        "actual_arrival_ts", "stop_sequence"
    )
    running = (
        Window.partitionBy("trip_id")
        .orderBy("actual_arrival_ts", "stop_sequence")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    return (
        arrivals.withColumn("prev_sequence", F.lag("stop_sequence").over(ordering))
        .withColumn(
            "run_boundary",
            F.when(
                F.col("prev_sequence").isNotNull()
                & (F.col("stop_sequence") <= F.col("prev_sequence")),
                1,
            ).otherwise(0),
        )
        .withColumn("trip_run", F.sum("run_boundary").over(running).cast("int"))
        .drop("prev_sequence", "run_boundary")
    )
