"""Score a completed service day and publish the result to CloudWatch.

    python -m src.monitoring.report                 # yesterday
    python -m src.monitoring.report --date 2026-08-23
    python -m src.monitoring.report --dry-run       # score, print, publish nothing

Runs after the ETL, because it needs actuals. A service day closes at 04:00 UTC the
following day, so the earliest this can score day D is D+1.

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
from .shadow import load_snapshots, null_rate, realised_accuracy

logger = logging.getLogger("monitoring.report")


def score_service_date(
    date: str, config: MonitoringConfig, etl_config: EtlConfig
) -> tuple[pd.DataFrame, dict]:
    """Reconstruct predictions for one service day and score them against actuals."""
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
    else:
        logger.warning(
            "no archived snapshots for %s — scoring with the conditions recorded in "
            "the journey table. Valid for a backfill, never for live monitoring.",
            date,
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
    """Overwrite recent_* with the archived snapshot in force at each departure."""
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
                (
                    out.loc[window, "first_leg_to"]
                    if "first_leg_to" in out.columns
                    else out.loc[window, "destination_stop_id"]
                ),
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


def publish(accuracy: pd.DataFrame, context: dict, config: MonitoringConfig) -> int:
    """Push metrics to CloudWatch. Returns how many datapoints were written."""
    import boto3

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="service date; default yesterday")
    parser.add_argument("--dry-run", action="store_true", help="publish nothing")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    config = MonitoringConfig()
    date = args.date or (datetime.now(UTC).date() - timedelta(days=1)).isoformat()

    accuracy, context = score_service_date(date, config, EtlConfig.from_env())

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

    if args.dry_run:
        print("\ndry run — nothing published")
        return 0

    written = publish(accuracy, context, config)
    print(f"\npublished {written} datapoint(s) to {config.namespace}")
    return 1 if breaches else 0


if __name__ == "__main__":
    raise SystemExit(main())
