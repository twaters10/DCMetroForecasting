"""Score a completed service day and publish the result to CloudWatch.

    python -m src.monitoring.report                 # yesterday
    python -m src.monitoring.report --date 2026-08-23
    python -m src.monitoring.report --catchup       # every unpublished complete day
    python -m src.monitoring.report --dry-run       # score, print, publish nothing

Runs after the ETL, because it needs actuals. A service day closes at 04:00 UTC the
following day, so the earliest this can score day D is D+1.

`--catchup` is what cron runs. It is driven by which days are unpublished rather than
by when it happens to fire, so the four daily firings do real work at most once and a
day that landed while the laptop was asleep is picked up on the next one instead of
being lost. Exit codes exist to keep that legible from a log: `0` healthy, `2` a
threshold breach, `1` something actually broke.

Publishes to the `MetroPulse/Model` namespace, dimensioned by journey length so a
regression at one horizon is visible. CloudWatch is the system of record; this module
only computes and pushes.

**Every metric carries the model run id as a dimension.** Without it a metric series
silently spans two different models across a redeploy, and the step change looks like
drift rather than a deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from ..etl.config import EtlConfig
from ..models.artifacts import resolve_run
from .config import REPORT_LENGTHS, MonitoringConfig
from .shadow import (
    load_snapshots,
    null_rate,
    realised_accuracy,
    service_dates_to_score,
)

logger = logging.getLogger("monitoring.report")

# CloudWatch refuses a datapoint older than two weeks. Scoring still works beyond that —
# the numbers just cannot be plotted — so this bounds publishing, never computation.
MAX_BACKDATE_DAYS = 14

# Which service dates have already been pushed. A local file rather than a CloudWatch
# probe: the skip decision then costs nothing and works with no network, which matters
# because this runs from cron on a laptop that is often offline.
PUBLISHED_MARKER = Path("data/monitoring/published.json")

# Breaching a threshold is a signal, not a failure. Cron needs to tell the two apart:
# `1` is reserved for a traceback, so a model that is merely underperforming cannot make
# a healthy ETL run look broken.
EXIT_BREACH = 2


class NoSnapshotsError(Exception):
    """No archived conditions cover this date, so it cannot be scored honestly."""


def score_service_date(
    date: str,
    config: MonitoringConfig,
    etl_config: EtlConfig,
    allow_fallback: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Reconstruct predictions for one service day and score them against actuals.

    Without archived snapshots the only conditions available are the journey table's
    own, which are computed with perfect knowledge of every traversal up to departure —
    better information than serving actually had, so the model scores better than it
    deserves. `allow_fallback` opts into that for a one-off investigation; it is off by
    default because such a day is not comparable with the rest of the series and, on a
    shared chart, is indistinguishable from one that is.
    """
    import lightgbm as lgb

    from ..journeys.train import build_matrix
    from ..models.encode import CategoricalEncoder

    journeys = pd.read_parquet(Path("data/processed/journeys") / "table")
    journeys["service_date"] = journeys["service_date"].astype(str)
    day = journeys[journeys["service_date"] == date].copy()
    if day.empty:
        raise ValueError(f"no journeys for {date} — has the ETL processed it?")

    run = resolve_run(config.model_root)
    booster = lgb.Booster(model_file=str(run / "model.txt"))
    encoder = CategoricalEncoder.load(run / "encoder.json")
    columns = json.loads((run / "feature_columns.json").read_text())
    run_id = json.loads((run / "manifest.json").read_text())["run_id"]

    # Replace each journey's recent-conditions with the snapshot in force at its
    # departure. Without this the model is scored using information from after the
    # journey began, which flatters it — see shadow.py.
    snapshots = load_snapshots(date, etl_config)
    if snapshots:
        day = _apply_historical_conditions(day, snapshots)
        covered = day["_snapshot_covered"].sum()
        logger.info(
            "%d of %d journeys covered by an archived snapshot", covered, len(day)
        )
        day = day[day["_snapshot_covered"]].drop(columns=["_snapshot_covered"])
    elif allow_fallback:
        logger.warning(
            "no archived snapshots for %s — scoring with the conditions recorded in "
            "the journey table. Valid for a one-off backfill, never for live "
            "monitoring, and NOT comparable with the published series.",
            date,
        )
    else:
        raise NoSnapshotsError(
            f"no archived conditions for {date}; the collector began writing them on "
            "2026-08-24. Pass --allow-fallback to score it anyway, but the result "
            "flatters the model and does not belong on the same chart."
        )

    if day.empty:
        raise ValueError(f"no journeys for {date} had snapshot coverage")

    predictions = pd.Series(
        booster.predict(build_matrix(day, encoder, columns)), index=day.index
    )
    accuracy = realised_accuracy(day, predictions, REPORT_LENGTHS)
    context = {
        "service_date": date,
        "model_run": run_id,
        "scored_journeys": int(len(day)),
        "null_rate_pct": null_rate(day),
    }
    return accuracy, context


def _apply_historical_conditions(day: pd.DataFrame, snapshots: dict) -> pd.DataFrame:
    """Overwrite recent_* with the archived snapshot in force at each departure.

    Keyed on (origin_stop_id, first_leg_to) — the origin SEGMENT, which is what the
    conditions table is indexed by and what serving looks up. Keying on the journey's
    destination instead silently misses every journey longer than one segment and
    reports the model's strongest feature as null on ~84% of rows.
    """
    if "first_leg_to" not in day.columns:
        raise ValueError(
            "journeys table has no `first_leg_to` column — rebuild it with "
            "`python -m src.journeys.build`. Scoring without it would match only "
            "single-segment journeys and null out recent_deviation on the rest."
        )
    columns = [
        "recent_duration_median",
        "recent_duration_mean",
        "recent_delay_mean",
        "recent_traversals",
        "recent_deviation",
    ]
    out = day.copy()
    out["_snapshot_covered"] = False

    departures = pd.to_datetime(out["origin_departure_ts"], utc=True)
    for moment, table in sorted(snapshots.items()):
        # Rows whose departure this snapshot is the latest one at or before.
        later = [t for t in snapshots if t > moment]
        upper = min(later) if later else None
        window = departures >= moment
        if upper is not None:
            window &= departures < upper
        if not window.any():
            continue

        indexed = table.set_index(["from_stop_id", "to_stop_id"])
        keys = list(
            zip(
                out.loc[window, "origin_stop_id"],
                out.loc[window, "first_leg_to"],
                strict=True,
            )
        )
        for column in columns:
            if column in indexed.columns:
                out.loc[window, column] = [
                    indexed[column].get(k, float("nan")) for k in keys
                ]
        out.loc[window, "_snapshot_covered"] = True
    return out


def metric_timestamp(service_date: str) -> datetime:
    """Noon UTC on the service date.

    Not midnight: a service day runs 04:00 UTC D to 04:00 UTC D+1, so noon is the one
    hour unambiguously inside it whichever end you measure from. Without an explicit
    timestamp CloudWatch stamps ingestion time, which plots day D at D+1 and collapses
    a whole backfill onto the instant it was run.
    """
    return datetime.fromisoformat(service_date).replace(hour=12, tzinfo=UTC)


def publish(accuracy: pd.DataFrame, context: dict, config: MonitoringConfig) -> int:
    """Push metrics to CloudWatch. Returns how many datapoints were written."""
    import boto3

    stamp = metric_timestamp(context["service_date"])
    age = datetime.now(UTC) - stamp
    if age > timedelta(days=MAX_BACKDATE_DAYS):
        raise ValueError(
            f"{context['service_date']} is {age.days} days old and CloudWatch refuses "
            f"datapoints older than {MAX_BACKDATE_DAYS} days. It can still be scored "
            "with --dry-run, but the result cannot be plotted."
        )

    client = boto3.client("cloudwatch")
    common = [
        {"Name": "ModelRun", "Value": context["model_run"]},
    ]
    data = [
        {
            "MetricName": "RecentDeviationNullRate",
            "Value": context["null_rate_pct"],
            "Unit": "Percent",
            "Dimensions": common,
        },
        {
            "MetricName": "ScoredJourneys",
            "Value": float(context["scored_journeys"]),
            "Unit": "Count",
            "Dimensions": common,
        },
    ]
    for _, row in accuracy.iterrows():
        length = [{"Name": "JourneySegments", "Value": str(int(row["segments"]))}]
        data += [
            {
                "MetricName": "RealisedMAE",
                "Value": row["mae_model_sec"],
                "Unit": "Seconds",
                "Dimensions": common + length,
            },
            {
                "MetricName": "ScheduleBaselineMAE",
                "Value": row["mae_schedule_sec"],
                "Unit": "Seconds",
                "Dimensions": common + length,
            },
            {
                "MetricName": "BeatsSchedulePct",
                "Value": row["beats_schedule_pct"],
                "Unit": "Percent",
                "Dimensions": common + length,
            },
        ]

    # Stamped in one place rather than at each construction site, so a metric added
    # later cannot silently fall back to ingestion time.
    for point in data:
        point["Timestamp"] = stamp

    # CloudWatch caps a PutMetricData call at 1000 datapoints; chunked so adding lengths
    # later cannot silently start dropping metrics.
    for start in range(0, len(data), 1000):
        client.put_metric_data(
            Namespace=config.namespace, MetricData=data[start : start + 1000]
        )
    return len(data)


def check_thresholds(accuracy: pd.DataFrame, context: dict, config: MonitoringConfig):
    """Which thresholds this run breaches. Empty means healthy."""
    breaches: list[str] = []
    if context["null_rate_pct"] > config.max_null_rate_pct:
        breaches.append(
            f"recent_deviation null on {context['null_rate_pct']:.1f}% of journeys "
            f"(limit {config.max_null_rate_pct:.0f}%) — the live conditions table is "
            "probably stale"
        )
    if context["scored_journeys"] < config.min_scored_journeys:
        breaches.append(
            f"only {context['scored_journeys']:,} journeys scored "
            f"(min {config.min_scored_journeys:,}) — treat these metrics as noisy"
        )
    losing = accuracy[
        accuracy["mae_model_sec"]
        > accuracy["mae_schedule_sec"] * config.max_mae_vs_schedule_ratio
    ]
    for _, row in losing.iterrows():
        breaches.append(
            f"at {int(row['segments'])} segments the model "
            f"({row['mae_model_sec']:.1f}s) is no better than the schedule "
            f"({row['mae_schedule_sec']:.1f}s)"
        )
    return breaches


def settled_dates(marker: Path = PUBLISHED_MARKER) -> set[str]:
    """Dates needing no further attempt — published, or deliberately skipped."""
    if not marker.exists():
        return set()
    return {entry["service_date"] for entry in json.loads(marker.read_text())}


def record_settled(
    service_date: str,
    model_run: str | None,
    marker: Path = PUBLISHED_MARKER,
    status: str = "published",
) -> None:
    """Note one service date as dealt with, replacing any earlier entry for it.

    Skips are recorded too, with the reason: a day the collector never covered will
    never become scoreable, and retrying it on all four firings forever would bury the
    days that can still be published. The model run is kept so a future retrain can
    decide to re-score a day deliberately; the skip decision only looks at the date.
    """
    entries = json.loads(marker.read_text()) if marker.exists() else []
    entries = [e for e in entries if e["service_date"] != service_date]
    entries.append(
        {
            "service_date": service_date,
            "model_run": model_run,
            "status": status,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    entries.sort(key=lambda e: e["service_date"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(entries, indent=2) + "\n")


def outstanding_dates(
    marker: Path = PUBLISHED_MARKER, now: datetime | None = None
) -> list[str]:
    """Complete service days that are still publishable and not yet published.

    Driven by what is missing rather than by when this happens to run, so a day the ETL
    landed while nobody was watching is picked up on the next firing instead of lost.
    Days past the backdating window are dropped rather than retried forever: they can no
    longer be plotted, and failing on them every run would bury the days that can.
    """
    journeys = pd.read_parquet(
        Path("data/processed/journeys") / "table", columns=["service_date"]
    )
    journeys["service_date"] = journeys["service_date"].astype(str)
    complete = service_dates_to_score(journeys, days=MAX_BACKDATE_DAYS, now=now)

    done = settled_dates(marker)
    # Tested against the datapoint's own timestamp, not the date, so this agrees
    # exactly with publish()'s guard. Comparing dates instead let a boundary day
    # through here and fail there, on every single firing.
    moment = now or datetime.now(UTC)

    def publishable(date: str) -> bool:
        return moment - metric_timestamp(date) <= timedelta(days=MAX_BACKDATE_DAYS)

    outstanding = [d for d in complete if d not in done and publishable(d)]

    expired = [d for d in complete if d not in done and not publishable(d)]
    if expired:
        logger.warning(
            "%d service date(s) are past the %d-day backdating window and will never "
            "be published: %s",
            len(expired),
            MAX_BACKDATE_DAYS,
            ", ".join(expired),
        )
    return outstanding


def score_one(
    date: str,
    config: MonitoringConfig,
    etl_config: EtlConfig,
    dry_run: bool,
    allow_fallback: bool = False,
) -> list[str]:
    """Score one service date, print the result, publish it. Returns its breaches."""
    accuracy, context = score_service_date(date, config, etl_config, allow_fallback)

    print(f"\nservice date {date} | model run {context['model_run']}")
    print(
        f"scored {context['scored_journeys']:,} journeys | "
        f"recent_deviation null on {context['null_rate_pct']:.1f}%"
    )
    print(accuracy.round(2).to_string(index=False))

    breaches = check_thresholds(accuracy, context, config)
    if breaches:
        print("\nTHRESHOLD BREACHES")
        for breach in breaches:
            print(f"  !! {breach}")
    else:
        print("\nno threshold breaches")

    if dry_run:
        print("\ndry run — nothing published")
        return breaches

    written = publish(accuracy, context, config)
    record_settled(date, context["model_run"])
    print(f"\npublished {written} datapoint(s) to {config.namespace}")
    return breaches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="service date; default yesterday")
    parser.add_argument(
        "--catchup",
        action="store_true",
        help="score every complete service day not yet published, oldest first",
    )
    parser.add_argument("--dry-run", action="store_true", help="publish nothing")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="score a day with no archived conditions; flatters the model, and the "
        "result is not comparable with the published series",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    config = MonitoringConfig()
    etl_config = EtlConfig.from_env()

    if args.catchup:
        if args.date:
            parser.error("--catchup and --date are mutually exclusive")
        if args.allow_fallback:
            # Catch-up feeds the shared chart. Mixing in a day scored on the fallback
            # would put a flattered number beside honest ones with nothing to tell them
            # apart, which is the failure this whole module exists to catch.
            parser.error("--allow-fallback is for a single --date, never a catch-up")
        dates = outstanding_dates()
        if not dates:
            print("nothing outstanding — every complete service day is published")
            return 0
        print(f"outstanding: {', '.join(dates)}")
    else:
        dates = [
            args.date or (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
        ]

    breaches: list[str] = []
    failed: list[str] = []
    for date in dates:
        try:
            breaches += score_one(
                date, config, etl_config, args.dry_run, args.allow_fallback
            )
        except NoSnapshotsError as skip:
            # Recorded, not retried. The collector cannot go back and write conditions
            # for a day that has already happened, so this will never succeed.
            logger.warning("%s skipped: %s", date, skip)
            if not args.dry_run:
                record_settled(date, None, status="skipped_no_snapshots")
        except Exception:
            # One bad day must not strand the rest of a catch-up.
            if len(dates) == 1:
                raise
            logger.exception("%s FAILED", date)
            failed.append(date)

    if failed:
        logger.error("failed dates: %s", ", ".join(failed))
        return 1
    return EXIT_BREACH if breaches else 0


if __name__ == "__main__":
    raise SystemExit(main())
