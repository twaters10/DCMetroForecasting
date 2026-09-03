"""Tests for the monitoring layer, against synthetic data only.

The parts worth testing are the ones that would report a healthy model while it is
failing: a snapshot lookup that quietly uses conditions from after the journey, an
accuracy metric that hides a regression at one horizon, a threshold that never trips.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.monitoring import dashboard, report
from src.monitoring.config import MonitoringConfig
from src.monitoring.report import check_thresholds, metric_timestamp, publish
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


# ----------------------------------------------------------------- publish timestamp


class _FakeCloudWatch:
    def __init__(self):
        self.batches = []

    def put_metric_data(self, Namespace, MetricData):  # noqa: N803 — boto3's spelling
        self.batches.append(MetricData)

    def datapoints(self):
        return [point for batch in self.batches for point in batch]


def _publish(monkeypatch, service_date, context_extra=None):
    import boto3

    fake = _FakeCloudWatch()
    monkeypatch.setattr(boto3, "client", lambda service: fake)
    context = {
        "service_date": service_date,
        "model_run": "run-1",
        "scored_journeys": 5000,
        "null_rate_pct": 1.0,
        **(context_extra or {}),
    }
    publish(_accuracy(40, 60), context, MonitoringConfig())
    return fake


def test_noon_utc_is_inside_the_service_day_whichever_end_you_measure_from():
    """A service day runs 04:00 UTC D to 04:00 UTC D+1, so midnight is ambiguous."""
    assert metric_timestamp("2026-08-25") == datetime(2026, 8, 25, 12, tzinfo=UTC)


def test_every_datapoint_is_stamped_with_the_service_date(monkeypatch):
    """Without this CloudWatch stamps ingestion time: day D is plotted at D+1, and a
    backfill collapses onto the single instant it was run."""
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    points = _publish(monkeypatch, yesterday).datapoints()

    assert points, "nothing was published"
    assert all(p["Timestamp"] == metric_timestamp(yesterday) for p in points)


def test_a_metric_added_later_cannot_escape_the_stamp(monkeypatch):
    """The stamp is applied to the whole batch, not at each construction site."""
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    points = _publish(monkeypatch, yesterday).datapoints()

    assert {p["MetricName"] for p in points} >= {
        "RecentDeviationNullRate",
        "ScoredJourneys",
        "RealisedMAE",
        "ScheduleBaselineMAE",
        "BeatsSchedulePct",
    }
    assert all("Timestamp" in p for p in points)


def test_publishing_past_the_backdating_window_fails_loudly(monkeypatch):
    """CloudWatch would reject the batch opaquely; the operator needs to know why."""
    stale = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
    with pytest.raises(ValueError, match="14 days"):
        _publish(monkeypatch, stale)


# ------------------------------------------------------------------------- catch-up


def _marker(tmp_path, entries):
    path = tmp_path / "published.json"
    path.write_text(json.dumps(entries))
    return path


def test_a_published_date_is_not_scored_twice(tmp_path, monkeypatch):
    """Cron fires four times a day; only the firing that finds new data does work."""
    marker = _marker(
        tmp_path, [{"service_date": "2026-08-25", "model_run": "r", "published_at": ""}]
    )
    monkeypatch.setattr(
        report, "service_dates_to_score", lambda *a, **k: ["2026-08-25", "2026-08-26"]
    )
    monkeypatch.setattr(
        pd, "read_parquet", lambda *a, **k: pd.DataFrame({"service_date": []})
    )

    assert report.outstanding_dates(
        marker=marker, now=datetime(2026, 8, 27, 12, tzinfo=UTC)
    ) == ["2026-08-26"]


def test_days_past_the_backdating_window_are_dropped_not_retried(tmp_path, monkeypatch):
    """They can never be plotted, and failing on them every run buries the days that
    still can be."""
    marker = _marker(tmp_path, [])
    monkeypatch.setattr(
        report, "service_dates_to_score", lambda *a, **k: ["2026-08-01", "2026-08-26"]
    )
    monkeypatch.setattr(
        pd, "read_parquet", lambda *a, **k: pd.DataFrame({"service_date": []})
    )

    assert report.outstanding_dates(
        marker=marker, now=datetime(2026, 8, 27, 12, tzinfo=UTC)
    ) == ["2026-08-26"]


def test_the_marker_records_the_run_so_a_retrain_can_rescore(tmp_path):
    """Skipping is keyed on the date, but which model produced it has to survive."""
    marker = tmp_path / "published.json"
    report.record_settled("2026-08-26", "run-9", marker=marker)
    report.record_settled("2026-08-25", "run-9", marker=marker)

    entries = json.loads(marker.read_text())
    assert [e["service_date"] for e in entries] == ["2026-08-25", "2026-08-26"]
    assert entries[0]["model_run"] == "run-9"


def test_republishing_a_date_replaces_its_entry(tmp_path):
    """A deliberate re-score must not leave two rows claiming the same day."""
    marker = tmp_path / "published.json"
    report.record_settled("2026-08-26", "run-9", marker=marker)
    report.record_settled("2026-08-26", "run-10", marker=marker)

    entries = json.loads(marker.read_text())
    assert len(entries) == 1
    assert entries[0]["model_run"] == "run-10"


# ----------------------------------------------------------------------- exit codes


def _run_catchup(monkeypatch, outcome, dates=("2026-08-26",)):
    """Drive main() in catch-up mode with score_one stubbed to `outcome`."""
    monkeypatch.setattr(report, "outstanding_dates", lambda *a, **k: list(dates))
    monkeypatch.setattr(report, "score_one", outcome)
    return report.main(["--catchup"])


def test_a_healthy_run_exits_zero(monkeypatch):
    assert _run_catchup(monkeypatch, lambda *a, **k: []) == 0


def test_a_threshold_breach_is_not_a_failure(monkeypatch):
    """Cron folds a non-zero status into its own log line. A model that is merely
    underperforming must not read as a broken pipeline, so a breach gets its own
    code."""
    assert _run_catchup(monkeypatch, lambda *a, **k: ["losing to the schedule"]) == 2
    assert report.EXIT_BREACH == 2


def test_a_broken_run_exits_one(monkeypatch):
    """Reserved for a traceback — the thing that actually needs a human."""

    def explode(*args, **kwargs):
        raise RuntimeError("s3 unreachable")

    assert _run_catchup(monkeypatch, explode, dates=("2026-08-25", "2026-08-26")) == 1


def test_one_bad_day_does_not_strand_the_rest(monkeypatch):
    """A date with no snapshot coverage would otherwise block every later date, every
    firing, forever."""
    scored = []

    def flaky(date, *args, **kwargs):
        if date == "2026-08-25":
            raise ValueError("no snapshot coverage")
        scored.append(date)
        return []

    assert _run_catchup(monkeypatch, flaky, dates=("2026-08-25", "2026-08-26")) == 1
    assert scored == ["2026-08-26"]


def test_nothing_outstanding_is_a_clean_no_op(monkeypatch):
    """Three of the four daily firings land here and must cost nothing."""
    monkeypatch.setattr(report, "outstanding_dates", lambda *a, **k: [])
    monkeypatch.setattr(
        report, "score_one", lambda *a, **k: pytest.fail("scored with nothing to do")
    )

    assert report.main(["--catchup"]) == 0


# ------------------------------------------------------------------------ dashboard


def test_no_widget_pins_a_model_run():
    """The old console dashboard hardcoded a run id in five widgets and went blank on
    every register. Nothing in the generated body may name a run."""
    body = json.dumps(dashboard.build_body("us-east-1"))

    assert "SEARCH(" in body
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z", body)


def test_model_widgets_select_by_search_and_keep_the_run_dimension():
    """SEARCH removes the need to retype the run id; it must not drop the dimension,
    which is what stops one series spanning two models across a redeploy."""
    widgets = dashboard.build_body("us-east-1")["widgets"]
    expressions = [
        entry[0]["expression"]
        for widget in widgets
        for entry in widget["properties"].get("metrics", [])
        if isinstance(entry[0], dict)
    ]

    assert expressions
    assert all("ModelRun" in e for e in expressions)
    assert all(dashboard.CLOUDWATCH_NAMESPACE in e for e in expressions)


def test_the_endpoint_widget_is_left_alone():
    """SageMaker's own metrics, legitimately near-empty. Not ours to rewrite."""
    widgets = dashboard.build_body("us-east-1")["widgets"]
    endpoint = [
        w for w in widgets if "Endpoint health" in w["properties"].get("title", "")
    ]

    assert len(endpoint) == 1
    assert all(
        entry[0] == "AWS/SageMaker" for entry in endpoint[0]["properties"]["metrics"]
    )


# ------------------------------------------------------- historical conditions join


def _day_with_one_long_journey():
    """One 8-segment journey: its first leg is A->B, its destination is far away."""
    return pd.DataFrame(
        [
            {
                "origin_stop_id": "PF_A01_1",
                "first_leg_to": "PF_A02_1",
                "destination_stop_id": "PF_A09_1",
                "origin_departure_ts": pd.Timestamp(NOON),
                "n_segments": 8,
                "recent_deviation": None,
            }
        ]
    )


def _conditions_snapshot():
    return {
        NOON
        - timedelta(minutes=2): pd.DataFrame(
            [
                {
                    "from_stop_id": "PF_A01_1",
                    "to_stop_id": "PF_A02_1",
                    "recent_deviation": 42.0,
                }
            ]
        )
    }


def test_conditions_are_keyed_on_the_origin_segment_not_the_destination():
    """The conditions table is indexed by adjacent stop pairs, and a journey's
    destination is not adjacent to its origin once it is longer than one segment.
    Keying on it matched only 1-segment journeys and reported recent_deviation — the
    model's strongest feature — as null on the other 84%."""
    out = report._apply_historical_conditions(
        _day_with_one_long_journey(), _conditions_snapshot()
    )

    assert out["_snapshot_covered"].all()
    assert out["recent_deviation"].iat[0] == 42.0


def test_a_journeys_table_without_first_leg_to_is_refused():
    """Falling back to the destination produced plausible-looking numbers from a
    silently broken join. Better to stop and say the table needs rebuilding."""
    stale = _day_with_one_long_journey().drop(columns=["first_leg_to"])

    with pytest.raises(ValueError, match="first_leg_to"):
        report._apply_historical_conditions(stale, _conditions_snapshot())


# ------------------------------------------------------------ fallback is not silent


def test_a_day_with_no_archived_conditions_is_refused_by_default(monkeypatch):
    """The journey table's own conditions are computed with perfect knowledge of every
    traversal up to departure — better information than serving had. Publishing that
    beside honestly-scored days puts a flattered number on the chart with nothing to
    mark it."""
    monkeypatch.setattr(report, "load_snapshots", lambda *a, **k: {})
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *a, **k: pd.DataFrame({"service_date": ["2026-08-20"]}),
    )

    with pytest.raises(report.NoSnapshotsError, match="2026-08-20"):
        report.score_service_date("2026-08-20", MonitoringConfig(), None)


def test_a_skipped_day_is_recorded_so_it_is_not_retried_every_firing(monkeypatch):
    """The collector cannot go back and write conditions for a day already past, so
    retrying forever would bury the days that can still be published."""
    marker_entries = []
    monkeypatch.setattr(report, "outstanding_dates", lambda *a, **k: ["2026-08-20"])
    monkeypatch.setattr(
        report,
        "score_one",
        lambda *a, **k: (_ for _ in ()).throw(report.NoSnapshotsError("none")),
    )
    monkeypatch.setattr(
        report,
        "record_settled",
        lambda date, run, **kw: marker_entries.append((date, run, kw.get("status"))),
    )

    assert report.main(["--catchup"]) == 0
    assert marker_entries == [("2026-08-20", None, "skipped_no_snapshots")]


def test_catchup_refuses_to_mix_in_a_fallback_day():
    """--allow-fallback is for a single deliberate --date, never the shared series."""
    with pytest.raises(SystemExit):
        report.main(["--catchup", "--allow-fallback"])


def test_the_publishable_horizon_agrees_with_the_publish_guard(tmp_path, monkeypatch):
    """A date that passes selection but fails publish() fails on every single firing.
    Both must test the datapoint's own timestamp, not the calendar date."""
    marker = _marker(tmp_path, [])
    boundary = "2026-08-19"
    monkeypatch.setattr(report, "service_dates_to_score", lambda *a, **k: [boundary])
    monkeypatch.setattr(
        pd, "read_parquet", lambda *a, **k: pd.DataFrame({"service_date": []})
    )
    # Noon on 08-19 is more than 14 days before this moment, so publish() would reject.
    now = datetime(2026, 9, 2, 18, tzinfo=UTC)

    assert report.outstanding_dates(marker=marker, now=now) == []


def test_the_dashboard_opens_on_a_range_that_can_contain_a_datapoint():
    """The console default is the last three hours. These metrics are one point per
    service day at noon UTC, so that window is empty except for the minutes after a
    publish — indistinguishable from a dashboard with no data at all."""
    body = dashboard.build_body("us-east-1")

    assert body["start"] == "-P14D"
    assert body["periodOverride"] == "inherit"
