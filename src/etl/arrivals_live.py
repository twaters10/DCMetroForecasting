"""Arrival derivation in pandas, for the live path that cannot run Spark.

## Why this file is a liability, and what makes it acceptable

This is a **second implementation of rules that already exist** in `arrivals.py`. The
project avoids that everywhere else — `features/build.py` puts batch and serving
through one function precisely because "a Spark implementation and a separate Python
reimplementation drift silently, and you find out from production predictions, not
from a test."

It exists because the recent-conditions lookup has to be refreshed every few minutes
from live snapshots, that has to run in a Lambda, and a Lambda cannot run Spark. One
hour of rail vehicle positions is 60 objects and 323 KiB, so pandas is more than
sufficient — the problem was never performance.

What makes it acceptable is `tests/test_arrivals_parity.py`, which runs both
implementations over the same fixture and fails if a single row differs. **If that
test is ever skipped or marked xfail, this module becomes exactly the hazard it was
written to avoid.**

## Spark is the reference

`arrivals.py` produced the data every model was trained on. When the two disagree, this
file is wrong by definition and conforms to Spark — never the reverse. "Fixing" a parity
failure by changing the Spark side would silently redefine what the model was trained
against.

Only the VehiclePositions path is ported. It supplies **99.3%** of arrivals; the
TripUpdates fallback is 0.7% and is not worth a second implementation of its own."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import MAX_ARRIVAL_BRACKET_SEC, SOURCE_VEHICLE_POSITION

logger = logging.getLogger("etl.arrivals_live")

# The columns `derive_vp_arrivals` emits, in Spark's order. Kept explicit so a parity
# failure points at a column rather than at a shape mismatch.
ARRIVAL_COLUMNS: tuple[str, ...] = (
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
)


def derive_vp_arrivals(observations: pd.DataFrame) -> pd.DataFrame:
    """Derive arrival events from VehiclePositions stop_sequence transitions.

    Mirrors `arrivals.derive_vp_arrivals`. One row per observed arrival: (trip,
    stop_sequence) with a bracketed timestamp.
    """
    usable = observations[
        observations["stop_sequence"].notna() & observations["trip_id"].notna()
    ]
    if usable.empty:
        return pd.DataFrame(columns=list(ARRIVAL_COLUMNS))

    # Partitioned by trip rather than vehicle: a vehicle serves many trips in a day, and
    # a trip is the unit whose stop_sequence progresses. `trip_id` alone is not unique
    # per run, but only consecutive-pair differences matter here.
    ordered = usable.sort_values(["trip_id", "captured_at"], kind="stable")
    grouped = ordered.groupby("trip_id", sort=False)

    previous_sequence = grouped["stop_sequence"].shift(1)
    previous_captured = grouped["captured_at"].shift(1)

    advanced = previous_sequence.notna() & (
        ordered["stop_sequence"] > previous_sequence
    )
    if not advanced.any():
        return pd.DataFrame(columns=list(ARRIVAL_COLUMNS))

    rows = ordered[advanced]
    start = previous_captured[advanced]
    end = rows["captured_at"]

    # Timedelta arithmetic, NOT epoch conversion. An earlier version did
    # `astype("int64") // 10**9`, which silently assumes NANOSECOND datetime resolution.
    # pandas 3 constructs datetimes from Python objects at microsecond resolution and
    # pandas 2 at nanosecond, so that divisor is wrong by 1000x on one of them — the
    # Lambda wrote timestamps in 1970 and durations of 1 second while the same code was
    # correct locally. The parity test could not catch it: both sides ran one pandas.
    #
    # `.dt.floor("s")` preserves Spark's truncation: from_unixtime takes an integral
    # argument, so a .5 midpoint floors rather than rounds.
    delta = end - start
    bracket = delta.dt.total_seconds().astype("int64")
    midpoint = (start + delta / 2).dt.floor("s")

    result = pd.DataFrame(
        {
            "trip_id": rows["trip_id"].to_numpy(),
            "scheduled_trip_id": rows["scheduled_trip_id"].to_numpy(),
            "schedule_version": rows["schedule_version"].to_numpy(),
            "route_id": rows["route_id"].to_numpy(),
            "direction_id": rows["direction_id"].to_numpy(),
            "vehicle_id": rows["vehicle_id"].to_numpy(),
            "service_date": rows["service_date"].to_numpy(),
            "stop_id": rows["stop_id"].to_numpy(),
            "stop_sequence": rows["stop_sequence"].to_numpy(),
            "actual_arrival_ts": midpoint.to_numpy(),
            "arrival_bracket_sec": bracket.to_numpy().astype(np.int32),
            "sequence_jump": (
                rows["stop_sequence"].to_numpy()
                - previous_sequence[advanced].to_numpy()
            ).astype(np.int32),
            "arrival_source": SOURCE_VEHICLE_POSITION,
            "observed_at_utc": rows["captured_at"].to_numpy(),
            # A wide bracket means the vehicle dropped out and reappeared, so the
            # estimate is vague rather than wrong. Flagged, not dropped.
            "arrival_confident": bracket.to_numpy() <= MAX_ARRIVAL_BRACKET_SEC,
        }
    )
    logger.info(
        "derived %d arrival(s) from %d observation(s)", len(result), len(usable)
    )
    return result[list(ARRIVAL_COLUMNS)]
