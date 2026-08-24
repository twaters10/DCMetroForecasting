"""Tests for the monitoring layer, against synthetic data only.

The parts worth testing are the ones that would report a healthy model while it is
failing: a snapshot lookup that quietly uses conditions from after the journey, an
accuracy metric that hides a regression at one horizon, a threshold that never trips.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from src.monitoring.config import MonitoringConfig
from src.monitoring.report import check_thresholds
from src.monitoring.shadow import (
    MAX_SNAPSHOT_AGE_SEC,
    null_rate,
    realised_accuracy,
    service_dates_to_score,
    snapshot_at,
)

NOON = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _snapshots():
    return {
        NOON - timedelta(minutes=10): pd.DataFrame({"marker": ["old"]}),
        NOON - timedelta(minutes=5): pd.DataFrame({"marker": ["current"]}),
        NOON + timedelta(minutes=5): pd.DataFrame({"marker": ["future"]}),
    }


# ------------------------------------------------------------------ snapshot lookup


def test_snapshot_is_the_latest_one_at_or_before_the_departure():
    """Using a later snapshot would hand the model conditions from after the journey."""
    picked = snapshot_at(pd.Timestamp(NOON), _snapshots())
    assert picked["marker"].iat[0] == "current"


def test_future_snapshots_are_never_used():
    """A journey at 11:50 must not see the 11:55 snapshot — that is leakage."""
    picked = snapshot_at(pd.Timestamp(NOON - timedelta(minutes=8)), _snapshots())
    assert picked["marker"].iat[0] == "old"


def test_no_snapshot_before_the_departure_returns_none():
    """Dropped rather than scored — scoring with null conditions measures a different
    model than the one deployed."""
    assert snapshot_at(pd.Timestamp(NOON - timedelta(hours=5)), _snapshots()) is None


def test_a_stale_snapshot_does_not_count_as_coverage():
    """Beyond the staleness window the snapshot is not what serving would have used."""
    old = {
        NOON - timedelta(seconds=MAX_SNAPSHOT_AGE_SEC + 60): pd.DataFrame({"m": [1]})
    }
    assert snapshot_at(pd.Timestamp(NOON), old) is None


# --------------------------------------------------------------------- accuracy


def _journeys(lengths_and_errors):
    rows = []
    for n, actual, predicted, scheduled in lengths_and_errors:
        rows.append(
            {
                "n_segments": n,
                "journey_duration_sec": actual,
                "scheduled_total_sec": scheduled,
                "_pred": predicted,
            }
        )
    frame = pd.DataFrame(rows)
    return frame, frame["_pred"]


def test_accuracy_is_reported_per_length_not_aggregated():
    """An aggregate hides a regression at one horizon — error is length-dependent."""
    frame, preds = _journeys(
        [(1, 120, 118, 130), (1, 120, 122, 130), (17, 2220, 1500, 2300)]
    )
    out = realised_accuracy(frame, preds, (1, 17)).set_index("segments")

    assert out.loc[1, "mae_model_sec"] == 2.0
    assert out.loc[17, "mae_model_sec"] == 720.0  # would vanish in an average


def test_schedule_baseline_is_scored_on_the_same_rows():
    """'Beats the timetable by X%' only means anything like-for-like."""
    frame, preds = _journeys([(4, 540, 530, 500)])
    out = realised_accuracy(frame, preds, (4,)).iloc[0]

    assert out["mae_model_sec"] == 10.0
    assert out["mae_schedule_sec"] == 40.0
    assert out["beats_schedule_pct"] == 75.0


def test_null_rate_of_a_missing_column_is_total():
    """A feature that is absent entirely is 100% unavailable, not 0%."""
    assert null_rate(pd.DataFrame({"other": [1]})) == 100.0
    assert null_rate(pd.DataFrame({"recent_deviation": [1.0, None]})) == 50.0


# ------------------------------------------------------------------- thresholds


def _accuracy(model_mae, schedule_mae, segments=4):
    return pd.DataFrame(
        [
            {
                "segments": segments,
                "journeys": 5000,
                "mae_model_sec": model_mae,
                "mae_schedule_sec": schedule_mae,
                "beats_schedule_pct": 100 * (1 - model_mae / schedule_mae),
            }
        ]
    )


def _context(null_pct=1.0, scored=5000):
    return {"null_rate_pct": null_pct, "scored_journeys": scored, "model_run": "r"}


def test_healthy_run_breaches_nothing():
    assert check_thresholds(_accuracy(40, 60), _context(), MonitoringConfig()) == []


def test_stale_conditions_breach_the_null_rate():
    """The failure that already happened once, silently, for days."""
    breaches = check_thresholds(
        _accuracy(40, 60), _context(null_pct=95), MonitoringConfig()
    )
    assert any("null on 95.0%" in b for b in breaches)


def test_losing_to_the_schedule_is_a_breach():
    """If the model stops beating the timetable it has no reason to exist."""
    breaches = check_thresholds(_accuracy(70, 60), _context(), MonitoringConfig())
    assert any("no better than the schedule" in b for b in breaches)


def test_too_few_journeys_is_flagged_as_noise_not_health():
    breaches = check_thresholds(
        _accuracy(40, 60), _context(scored=10), MonitoringConfig()
    )
    assert any("noisy" in b for b in breaches)


# ------------------------------------------------------------------ date selection


def test_only_complete_service_dates_are_scored():
    """Today is never complete — a service day closes at 04:00 UTC the next day."""
    features = pd.DataFrame(
        {"service_date": ["2026-08-22", "2026-08-23", "2026-08-24"]}
    )
    dates = service_dates_to_score(
        features, days=2, now=datetime(2026, 8, 24, 12, tzinfo=UTC)
    )

    assert dates == ["2026-08-22", "2026-08-23"]
    assert "2026-08-24" not in dates
