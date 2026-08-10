"""Tests for the feature layer, against synthetic fixtures only.

No network, no S3, no `.env`. Every fixture is small enough to reason about by hand,
which is the point: an as-of join that is off by one still produces plausible-looking
numbers on real data, and only a fixture where you know the right answer catches it.

The leakage tests are the ones that matter. A model trained on a leaked feature
validates beautifully then fails in production, and by then the culprit is buried under
forty other features.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    LOOKUP_FEATURE_COLUMNS,
    RecentConditionsLookup,
    build_recent_conditions_lookup,
    compute_features,
)
from src.features.config import FeatureConfig
from src.features.historical import (
    build_historical_features,
    headway,
    rolling_segment_conditions,
    upstream_delay,
)
from src.features.safe import station_code, temporal_features
from src.features.split import temporal_split

T0 = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def segment_row(
    *,
    depart: datetime,
    arrive: datetime,
    from_stop: str = "PF_A01_C",
    to_stop: str = "PF_A02_C",
    trip: str = "T1",
    run: int = 0,
    seq: int = 2,
    scheduled: float = 120.0,
    route: str = "RED",
) -> dict:
    """One segment traversal. Duration and delay derived so fixtures stay consistent."""
    duration = (arrive - depart).total_seconds()
    return {
        "service_date": depart.date().isoformat(),
        "trip_id": trip,
        "trip_run": run,
        "route_id": route,
        "direction_id": 0,
        "from_stop_id": from_stop,
        "to_stop_id": to_stop,
        "from_stop_sequence": seq - 1,
        "stop_sequence": seq,
        "stop_span": 1,
        "actual_departure_ts": depart,
        "actual_arrival_ts": arrive,
        "actual_duration_sec": duration,
        "scheduled_duration_sec": scheduled,
        "delay_sec": duration - scheduled,
    }


def frame(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    for column in ("actual_departure_ts", "actual_arrival_ts"):
        out[column] = pd.to_datetime(out[column], utc=True)
    return out


# --------------------------------------------------------------------------
# As-of boundaries — the leakage-critical logic
# --------------------------------------------------------------------------


def test_a_traversal_completing_exactly_at_T_is_excluded():
    """Strict `<`, not `<=`.

    A traversal finishing at exactly T is not information a dispatcher had *at* T. This
    is the classic off-by-one, and `merge_asof`'s default `allow_exact_matches=True`
    would silently include it.
    """
    rows = frame(
        [
            # Completes at exactly T=12:10, the query row's departure.
            segment_row(depart=at(5), arrive=at(10), trip="EARLIER"),
            segment_row(depart=at(10), arrive=at(12), trip="QUERY"),
        ]
    )
    result = rolling_segment_conditions(rows)
    query = result.loc[rows.trip_id == "QUERY"]

    assert (
        query["recent_duration_median"].isna().all()
    ), "a traversal completing at exactly T must not be visible to a prediction at T"


def test_a_traversal_completing_one_second_before_T_is_included():
    """The other side of the same boundary — it must not be over-strict either."""
    rows = frame(
        [
            segment_row(
                depart=at(5), arrive=at(10) - timedelta(seconds=1), trip="EARLIER"
            ),
            segment_row(depart=at(10), arrive=at(12), trip="QUERY"),
        ]
    )
    result = rolling_segment_conditions(rows)
    query = result.loc[rows.trip_id == "QUERY"]

    assert query["recent_duration_median"].notna().all()
    assert float(query["recent_duration_median"].iloc[0]) == pytest.approx(299.0)


def test_a_traversal_still_in_flight_at_T_contributes_nothing():
    """Departed before T, arrives after T. Its duration is not knowable at T.

    Keying the as-of join on departure time instead of completion time is the mistake
    this catches, and it is a tempting one — departure is the column everything else is
    ordered by.
    """
    rows = frame(
        [
            # Departs 12:05 (before T), arrives 12:20 (after T=12:10).
            segment_row(depart=at(5), arrive=at(20), trip="IN_FLIGHT"),
            segment_row(depart=at(10), arrive=at(12), trip="QUERY"),
        ]
    )
    result = rolling_segment_conditions(rows)
    query = result.loc[rows.trip_id == "QUERY"]

    assert query["recent_duration_median"].isna().all()


def test_leakage_canary_a_future_traversal_must_not_move_any_feature():
    """The catch-all: add an extreme FUTURE row and assert nothing changes.

    This is the test that survives a refactor. A specific boundary assertion checks the
    logic you thought about; this checks the logic you did not.
    """
    base = [
        segment_row(depart=at(0), arrive=at(2), trip="A"),
        segment_row(depart=at(5), arrive=at(7), trip="B"),
        segment_row(depart=at(10), arrive=at(12), trip="QUERY"),
    ]
    # A wildly slow traversal that starts and finishes after the query row.
    future = segment_row(depart=at(20), arrive=at(60), trip="FUTURE")

    without = build_historical_features(frame(base))
    with_future = build_historical_features(frame([*base, future]))

    query_index = 2
    for column in without.columns:
        before = without[column].iloc[query_index]
        after = with_future[column].iloc[query_index]
        assert (
            pd.isna(before) and pd.isna(after)
        ) or before == after, (
            f"{column} changed when a future row was added — it is reading ahead of T"
        )


def test_recent_conditions_ignore_traversals_older_than_the_window():
    """A traversal from two hours ago says nothing about now.

    Without the age cap the feature always finds *some* prior traversal, so it looks
    dense and informative in a backtest while being unavailable at serving time.
    """
    config = FeatureConfig(rolling_max_age_sec=600)
    rows = frame(
        [
            segment_row(depart=at(-200), arrive=at(-198), trip="ANCIENT"),
            segment_row(depart=at(10), arrive=at(12), trip="QUERY"),
        ]
    )
    result = rolling_segment_conditions(rows, config)

    assert pd.isna(result["recent_duration_median"].iloc[1])


# --------------------------------------------------------------------------
# Upstream delay and the trip_run bug
# --------------------------------------------------------------------------


def test_upstream_delay_accumulates_only_prior_segments():
    """Hand-computed: three segments delayed 30s, 60s, 90s in sequence."""
    rows = frame(
        [
            segment_row(depart=at(0), arrive=at(2.5), seq=1, scheduled=120),  # +30
            segment_row(depart=at(3), arrive=at(6), seq=2, scheduled=120),  # +60
            segment_row(depart=at(7), arrive=at(10.5), seq=3, scheduled=120),  # +90
        ]
    )
    result = upstream_delay(rows)

    # First segment has nothing before it; then 30; then 30+60.
    assert pd.isna(result["upstream_delay_sec"].iloc[0])
    assert result["upstream_delay_sec"].iloc[1] == pytest.approx(30.0)
    assert result["upstream_delay_sec"].iloc[2] == pytest.approx(90.0)
    assert list(result["segments_completed"]) == [0, 1, 2]


def test_upstream_delay_does_not_leak_across_trip_runs():
    """REGRESSION: 18.9% of real rows are a repeat run of a reused trip_id.

    Keyed on `trip_id` alone, run 1's first segment would inherit run 0's accumulated
    delay — inventing delay that never happened, on nearly a fifth of the dataset.
    """
    rows = frame(
        [
            segment_row(
                depart=at(0), arrive=at(5), seq=1, run=0, scheduled=120
            ),  # +180
            segment_row(depart=at(6), arrive=at(8), seq=2, run=0, scheduled=120),
            # Same trip_id, second journey. Must start clean.
            segment_row(depart=at(60), arrive=at(62), seq=1, run=1, scheduled=120),
            segment_row(depart=at(63), arrive=at(65), seq=2, run=1, scheduled=120),
        ]
    )
    result = upstream_delay(rows)

    assert pd.isna(
        result["upstream_delay_sec"].iloc[2]
    ), "run 1 must start with no history"
    assert result["upstream_delay_sec"].iloc[3] == pytest.approx(0.0)
    assert list(result["segments_completed"]) == [0, 1, 0, 1]


# --------------------------------------------------------------------------
# Headway
# --------------------------------------------------------------------------


def test_headway_measures_the_gap_to_the_previous_departure():
    rows = frame(
        [
            segment_row(depart=at(0), arrive=at(2), trip="A"),
            segment_row(depart=at(4), arrive=at(6), trip="B"),
            segment_row(depart=at(9), arrive=at(11), trip="C"),
        ]
    )
    result = headway(rows)

    assert pd.isna(result["headway_sec"].iloc[0])  # nothing before the first
    assert result["headway_sec"].iloc[1] == pytest.approx(240.0)
    assert result["headway_sec"].iloc[2] == pytest.approx(300.0)


def test_headway_is_per_route_and_direction():
    """A Red Line train does not set the headway for a Blue Line one at that stop."""
    rows = frame(
        [
            segment_row(depart=at(0), arrive=at(2), route="RED", trip="R1"),
            segment_row(depart=at(1), arrive=at(3), route="BLUE", trip="B1"),
            segment_row(depart=at(5), arrive=at(7), route="RED", trip="R2"),
        ]
    )
    result = headway(rows)

    # R2's headway is measured from R1 at minute 0, not from the Blue train at minute 1.
    assert result["headway_sec"].iloc[2] == pytest.approx(300.0)


# --------------------------------------------------------------------------
# Safe features
# --------------------------------------------------------------------------


def test_cyclical_encoding_makes_23_and_0_adjacent():
    """The point of sin/cos: midnight sits next to 23:00, not maximally far from it."""
    rows = frame(
        [
            segment_row(
                depart=datetime(2026, 8, 7, 3, 0, tzinfo=UTC), arrive=at(0)
            ),  # 23:00 local
            segment_row(
                depart=datetime(2026, 8, 7, 4, 0, tzinfo=UTC), arrive=at(0)
            ),  # 00:00 local
            segment_row(
                depart=datetime(2026, 8, 7, 16, 0, tzinfo=UTC), arrive=at(0)
            ),  # 12:00 local
        ]
    )
    out = temporal_features(rows)
    assert list(out["local_hour"]) == [23, 0, 12]

    def distance(a: int, b: int) -> float:
        return float(
            np.hypot(
                out["hour_sin"].iloc[a] - out["hour_sin"].iloc[b],
                out["hour_cos"].iloc[a] - out["hour_cos"].iloc[b],
            )
        )

    assert distance(0, 1) < 0.3, "23:00 and 00:00 must be close"
    assert distance(0, 2) > 1.5, "23:00 and 12:00 must be far"


def test_departure_hour_is_local_not_utc():
    """A UTC hour is four or five hours wrong and invents a lunchtime rush."""
    rows = frame(
        [segment_row(depart=datetime(2026, 8, 7, 12, 0, tzinfo=UTC), arrive=at(2))]
    )
    assert temporal_features(rows)["local_hour"].iloc[0] == 8  # 12:00 UTC = 08:00 EDT


@pytest.mark.parametrize(
    ("stop_id", "expected"),
    [
        ("PF_A08_C", "A08"),
        ("PF_B35_C", "B35"),
        ("PF_D02_2", "D02"),
        ("STN_N06", "STN_N06"),  # not a platform id; passes through unchanged
    ],
)
def test_station_code_extracted_from_platform_id(stop_id, expected):
    """Several platform ids map to one station, so station features need the code."""
    assert station_code(pd.Series([stop_id])).iloc[0] == expected


# --------------------------------------------------------------------------
# Temporal split
# --------------------------------------------------------------------------


def split_fixture(days: int) -> pd.DataFrame:
    """One traversal per hour for `days` consecutive service days."""
    rows = []
    for day in range(days):
        for hour in range(24):
            depart = datetime(2026, 8, 1, tzinfo=UTC) + timedelta(days=day, hours=hour)
            rows.append(
                segment_row(depart=depart, arrive=depart + timedelta(minutes=2))
            )
    return frame(rows)


def test_embargo_leaves_a_gap_no_training_row_can_reach_across():
    """No training row may sit within `embargo_sec` of the earliest validation row.

    Rolling features look back `rolling_max_age_sec`; a validation row that starts less
    than that after the last training row shares history with it, and the validation
    score inherits the training set's information. The gap is what makes the two sets
    genuinely disjoint in time, not merely ordered.
    """
    rows = split_fixture(days=20)
    config = FeatureConfig(embargo_sec=3600, validation_fraction=0.2)

    train_mask, validation_mask, report = temporal_split(rows, config=config)
    times = rows["actual_departure_ts"]

    assert train_mask.sum() and validation_mask.sum()
    assert not (train_mask & validation_mask).any(), "a row cannot be in both sets"

    gap = times[validation_mask].min() - times[train_mask].max()
    assert gap.total_seconds() >= config.embargo_sec

    # Embargoed rows belong to neither set — that is the whole point, and it means the
    # two masks must NOT partition the frame.
    assert (
        report.embargoed_rows == len(rows) - report.train_rows - report.validation_rows
    )
    assert report.embargoed_rows > 0


def test_split_is_ordered_in_time_never_shuffled():
    """Every training row strictly precedes every validation row."""
    rows = split_fixture(days=20)
    train_mask, validation_mask, _ = temporal_split(rows)
    times = rows["actual_departure_ts"]

    assert times[train_mask].max() < times[validation_mask].min()


def test_split_reports_why_three_days_cannot_be_trusted():
    """The report has to say so out loud, or a meaningless score reads as a real one."""
    _, _, report = temporal_split(split_fixture(days=3))

    assert not report.is_trustworthy
    assert any("service day(s) available" in w for w in report.warnings)
    assert report.as_metadata()["trustworthy"] is False


def test_split_on_ample_data_raises_no_warnings():
    """The warnings must clear on their own once there is enough data — otherwise they
    are noise that gets ignored, including on the day they matter."""
    _, _, report = temporal_split(split_fixture(days=30))

    assert report.warnings == []
    assert report.is_trustworthy


# --------------------------------------------------------------------------
# Batch / serving parity
# --------------------------------------------------------------------------


def test_serving_reproduces_the_batch_features_for_the_same_row():
    """Same row, two code paths, identical numbers.

    Batch derives recent conditions from a frame containing the segment's history;
    serving has one request and reads a published lookup. If those disagree, the model
    is scored offline on values it will never see in production, and nothing in the
    training metrics can reveal it.
    """
    history = [
        segment_row(depart=at(0), arrive=at(3), trip="A"),
        segment_row(depart=at(10), arrive=at(12), trip="B"),
        segment_row(depart=at(20), arrive=at(24), trip="C"),
    ]
    published_at = at(25)
    lookup = RecentConditionsLookup(
        table=build_recent_conditions_lookup(frame(history)),
        published_at=published_at,
    )

    # A request arriving after the lookup was published.
    query = segment_row(depart=at(30), arrive=at(33), trip="QUERY")

    batch = compute_features(frame([*history, query]))
    served = compute_features(
        frame([query]), lookup=lookup, now=query["actual_departure_ts"]
    )

    query_row = batch.iloc[-1]
    for column in LOOKUP_FEATURE_COLUMNS:
        assert float(served[column].iloc[0]) == pytest.approx(
            float(query_row[column])
        ), f"{column} differs between the batch and serving paths"


def test_serving_emits_the_same_columns_as_batch():
    """A model fed a different column set than it was trained on fails obscurely.

    Serving genuinely cannot compute some features from one request — upstream delay,
    headway. Those must come back as nulls under their real names rather than being
    absent.
    """
    rows = frame(
        [
            segment_row(depart=at(0), arrive=at(3), trip="A"),
            segment_row(depart=at(10), arrive=at(12), trip="B"),
        ]
    )
    lookup = RecentConditionsLookup(
        table=build_recent_conditions_lookup(rows), published_at=at(13)
    )

    batch = compute_features(rows)
    served = compute_features(rows, lookup=lookup, now=at(15))

    assert set(served.columns) == set(batch.columns)


def test_a_stale_lookup_yields_nulls_not_a_confident_wrong_number():
    """The staleness rule has to hold on the serving side too.

    Batch nulls recent conditions when the last traversal is older than
    `rolling_max_age_sec`. If serving did not apply the same cutoff, a lookup published
    hours ago would feed the endpoint a value batch would have refused to produce —
    invisible in every offline metric.
    """
    config = FeatureConfig(rolling_max_age_sec=600)
    rows = frame([segment_row(depart=at(0), arrive=at(3), trip="A")])
    lookup = RecentConditionsLookup(
        table=build_recent_conditions_lookup(rows, config), published_at=at(4)
    )

    fresh = lookup.for_segment("PF_A01_C", "PF_A02_C", at(8), config)
    stale = lookup.for_segment("PF_A01_C", "PF_A02_C", at(120), config)

    assert fresh["recent_duration_median"] == pytest.approx(180.0)
    assert stale["recent_duration_median"] is None


def test_an_unknown_segment_returns_nulls_rather_than_failing():
    """A segment absent from the lookup is a cold start, not an error. The endpoint
    still has to answer — the model handles the nulls."""
    rows = frame([segment_row(depart=at(0), arrive=at(3), trip="A")])
    lookup = RecentConditionsLookup(
        table=build_recent_conditions_lookup(rows), published_at=at(4)
    )

    result = lookup.for_segment("PF_Z99_C", "PF_Z98_C", at(5), FeatureConfig())

    assert set(result) == set(LOOKUP_FEATURE_COLUMNS)
    assert all(value is None for value in result.values())


def test_the_lookup_holds_one_row_per_segment():
    rows = frame(
        [
            segment_row(depart=at(0), arrive=at(3), trip="A"),
            segment_row(depart=at(10), arrive=at(12), trip="B"),
            segment_row(
                depart=at(5), arrive=at(8), from_stop="PF_B01_C", to_stop="PF_B02_C"
            ),
        ]
    )
    table = build_recent_conditions_lookup(rows)

    assert len(table) == 2
    assert not table.duplicated(subset=["from_stop_id", "to_stop_id"]).any()
    # `completed_at` is the segment's LAST completion, not its first.
    entry = table[table["from_stop_id"] == "PF_A01_C"].iloc[0]
    assert pd.Timestamp(entry["completed_at"]) == pd.Timestamp(at(12))
