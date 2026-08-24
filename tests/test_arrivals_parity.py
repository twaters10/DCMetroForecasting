"""Spark and pandas arrival derivation must agree, row for row.

`src/etl/arrivals_live.py` is a second implementation of rules that already exist in
`src/etl/arrivals.py`. That is a hazard the rest of this project deliberately avoids —
serving and batch share one function everywhere else — and it exists only because the
live path runs in a Lambda, which cannot run Spark.

**This file is the entire safety argument for that duplication.** Two copies of the
same rules are fine while something proves they agree; the failure mode is six months
from now, when one side gets a bug fix and the other does not. Nothing errors. The
model is trained on arrivals computed one way and served arrivals computed slightly
differently, and the predictions quietly get worse with no log line to explain it.

**Spark is the reference.** `arrivals.py` produced the data every model was trained
on, so when the two disagree, the pandas side is wrong by definition and conforms to
Spark — never the reverse. Fixing a parity failure by editing the Spark side would
silently redefine what the model was trained against.

**If this test is ever skipped or marked xfail, delete `arrivals_live.py` too.** The
module is only defensible while this passes; keeping it with the test disabled is
strictly worse than not having the live path at all, because the comment claiming
protection would still be there.

The fixtures below are chosen to carry the nasty cases, not a happy path: feed
dropouts, sequence jumps, out-of-order snapshots, a reused trip_id, and single-
observation trips that produce no arrival at all. A parity test only proves agreement
on what it exercises."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.etl.arrivals import derive_vp_arrivals as spark_derive
from src.etl.arrivals_live import ARRIVAL_COLUMNS
from src.etl.arrivals_live import derive_vp_arrivals as pandas_derive

BASE = datetime(2026, 8, 24, 12, 0, tzinfo=UTC).replace(tzinfo=None)


def observations(trip_id: str, sequences: list[tuple[int, int]], vehicle: str = "V1"):
    """Staged VehiclePosition rows, matching the real observation schema.

    Built directly rather than via `fixtures_gtfs_rt.progressing_vehicle`, which emits
    protobuf snapshots that would need the whole decode path to become rows. A parity
    test should exercise the derivation, not the decoder.

    `sequences` is `(minute_offset, stop_sequence)`. Skipping a minute produces a feed
    dropout; skipping a sequence produces a stop passed between polls.
    """
    return [
        {
            "captured_at": BASE + timedelta(minutes=minute),
            "trip_id": trip_id,
            "scheduled_trip_id": trip_id.split("_")[0],
            "schedule_version": "20670",
            "route_id": "RED",
            "direction_id": 0,
            "vehicle_id": vehicle,
            "stop_id": f"PF_A{sequence:02d}_C",
            "stop_sequence": sequence,
            "status": "STOPPED_AT",
            "vehicle_ts": BASE + timedelta(minutes=minute),
            "service_date": "2026-08-21",
        }
        for minute, sequence in sequences
    ]


def _observations_frame(rows: list[dict]) -> pd.DataFrame:
    """The staged-observation shape both implementations consume."""
    frame = pd.DataFrame(rows)
    frame["captured_at"] = pd.to_datetime(frame["captured_at"])
    frame["vehicle_ts"] = pd.to_datetime(frame["vehicle_ts"])
    return frame


def _spark_frame(spark, rows: list[dict]):
    return spark.createDataFrame(_observations_frame(rows))


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Put either implementation's output into one comparable shape."""
    if frame.empty:
        return pd.DataFrame(columns=list(ARRIVAL_COLUMNS))
    out = frame[list(ARRIVAL_COLUMNS)].copy()
    out["actual_arrival_ts"] = pd.to_datetime(out["actual_arrival_ts"])
    out["observed_at_utc"] = pd.to_datetime(out["observed_at_utc"])
    for column in ("arrival_bracket_sec", "sequence_jump", "stop_sequence"):
        out[column] = out[column].astype("int64")
    out["arrival_confident"] = out["arrival_confident"].astype(bool)
    return out.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)


def _assert_parity(spark, rows: list[dict]) -> pd.DataFrame:
    spark_result = _normalise(spark_derive(_spark_frame(spark, rows)).toPandas())
    pandas_result = _normalise(pandas_derive(_observations_frame(rows)))

    pd.testing.assert_frame_equal(
        spark_result,
        pandas_result,
        check_dtype=False,  # Spark int32 vs pandas int64 is not a semantic difference
        obj="pandas arrivals must match Spark exactly",
    )
    return spark_result


# --------------------------------------------------------------------------- fixtures


def _clean_run() -> list[dict]:
    """Three stops at one-minute intervals — two arrival transitions."""
    return observations("T1", [(0, 1), (1, 2), (2, 3)])


def _dropout() -> list[dict]:
    """A ten-minute gap: a wide bracket, flagged rather than dropped."""
    return observations("T1", [(0, 1), (10, 2)])


def _sequence_jump() -> list[dict]:
    """Stop 2 passed between polls, so the transition spans two stops."""
    return observations("T1", [(0, 1), (1, 3), (2, 4)])


def _reused_trip_id() -> list[dict]:
    """WMATA reuses a trip_id within a service day — ~19% of real rows."""
    return observations("T1", [(0, 1), (1, 2)]) + observations(
        "T1", [(200, 1), (201, 2)]
    )


def _single_observation() -> list[dict]:
    """One snapshot cannot produce a transition, so neither side may emit a row."""
    return observations("T1", [(0, 1)])


def _stationary() -> list[dict]:
    """The sequence never advances: no arrivals, and no crash on an empty result."""
    return observations("T1", [(0, 1), (1, 1), (2, 1)])


# ------------------------------------------------------------------------------ tests


@pytest.mark.parametrize(
    "name, rows",
    [
        ("clean run", _clean_run()),
        ("feed dropout", _dropout()),
        ("sequence jump", _sequence_jump()),
        ("reused trip_id", _reused_trip_id()),
        ("single observation", _single_observation()),
        ("stationary vehicle", _stationary()),
    ],
)
def test_pandas_matches_spark(spark, name, rows):
    _assert_parity(spark, rows)


@pytest.mark.parametrize("extra_seconds", [1, 3, 5, 7, 9, 11])
def test_parity_holds_on_odd_length_brackets(spark, extra_seconds):
    """The midpoint must TRUNCATE, not round.

    Spark computes `from_unixtime((prev + cur) / 2)` and from_unixtime takes an integral
    argument, so a .5 midpoint floors. Rounding in pandas would sit one second off Spark
    on some brackets — plausible-looking, never erroring, and wrong.

    Parametrised over several odd brackets on purpose. A single case is not enough:
    numpy rounds half to EVEN, so a 61s bracket (midpoint 30.5 -> 30) agrees with floor
    by luck, and a test using only that one passes even when the implementation rounds.
    A 63s bracket (31.5 -> 32) is what actually separates them. Verified by mutation —
    swapping the floor for a round must fail this test.
    """
    rows = observations("T1", [(0, 1), (1, 2)])
    rows[-1]["captured_at"] = rows[-1]["captured_at"] + timedelta(seconds=extra_seconds)

    result = _assert_parity(spark, rows)
    assert result["arrival_bracket_sec"].iat[0] == 60 + extra_seconds


def test_multiple_trips_do_not_bleed_into_each_other(spark):
    """Partitioned by trip: one trip's last stop must not bracket another's first."""
    rows = observations("T1", [(0, 1), (1, 2)]) + observations("T2", [(0, 5), (1, 6)])
    result = _assert_parity(spark, rows)
    assert set(result["trip_id"]) == {"T1", "T2"}
    assert len(result) == 2


@pytest.mark.parametrize("resolution", ["ns", "us", "ms", "s"])
def test_derivation_is_datetime_resolution_independent(resolution):
    """The live path must not care what resolution pandas hands it.

    REGRESSION: the first version converted to epoch seconds with
    `astype("int64") // 10**9`, which silently assumes NANOSECOND resolution. pandas 3
    builds datetimes from Python objects at microsecond resolution and pandas 2 at
    nanosecond, so the divisor was wrong by 1000x on one of them. Locally it was
    correct; in the Lambda, on the layer's pandas, it wrote timestamps in 1970 and
    durations of 1 second.

    The parity test could not catch this: both implementations ran under one pandas.
    Only varying the resolution exposes it, which is what this does.
    """
    frame = _observations_frame(observations("T1", [(0, 1), (1, 2), (3, 3)]))
    frame["captured_at"] = frame["captured_at"].astype(f"datetime64[{resolution}]")

    result = pandas_derive(frame)

    assert len(result) == 2
    assert list(result["arrival_bracket_sec"]) == [60, 120]
    # Midpoint of minutes 0 and 1 is 12:00:30 — not 1970, and not a 1-second duration.
    assert str(result["actual_arrival_ts"].iat[0]) == "2026-08-24 12:00:30"
