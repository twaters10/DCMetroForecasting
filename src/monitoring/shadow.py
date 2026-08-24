"""Score the model against ground truth on journeys that actually ran.

## Why shadow scoring rather than captured traffic

The endpoint takes `(origin, destination, departure_ts)` — a rider's question, not a
journey that happened — so a live prediction has no ground truth attached to it. The
truth for such a request is **the actual duration of the first train departing A after
T**, which is well-defined and matchable, but organic traffic is a handful of requests
a day. Far too thin to detect anything.

So the primary source is the reverse: take journeys the ETL has confirmed *did* run,
and reconstruct what the model would have predicted at their departure moment.
Thousands of scored predictions per day across the whole network, rather than whatever
anyone happened to ask about.

## The honesty condition

A reconstructed prediction is only meaningful if the model is given the inputs it
would actually have had. `recent_deviation` depends on live conditions at that moment,
and the live table is overwritten every five minutes — so this reads the **archived**
snapshots written by the collector, picking the one in force at each journey's
departure.

Scoring against today's conditions instead would hand the model information from after
the journey began. The number would look better and mean nothing. Where no snapshot
covers a journey's departure, the row is **excluded** rather than scored with null
conditions, because that would measure a different model than the one deployed."""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime

import pandas as pd

from ..etl.config import EtlConfig
from .config import HISTORY_PREFIX, MonitoringConfig

logger = logging.getLogger("monitoring.shadow")

# A snapshot is written every 5 minutes; anything older than this was not the view in
# force at departure. Matches the serving staleness rule.
MAX_SNAPSHOT_AGE_SEC = 3600


def load_snapshots(
    service_date: str, config: EtlConfig, s3=None
) -> dict[datetime, pd.DataFrame]:
    """Every archived recent-conditions snapshot for one service date, by write time."""
    import boto3

    client = s3 or boto3.client("s3", region_name=config.aws_region)
    prefix = f"{HISTORY_PREFIX}date={service_date}/"
    pages = client.get_paginator("list_objects_v2").paginate(
        Bucket=config.s3_bucket, Prefix=prefix
    )

    snapshots: dict[datetime, pd.DataFrame] = {}
    for page in pages:
        for item in page.get("Contents", []):
            stamp = item["Key"].rsplit("/", 1)[-1].removesuffix(".csv")
            written = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            body = client.get_object(Bucket=config.s3_bucket, Key=item["Key"])[
                "Body"
            ].read()
            snapshots[written] = pd.read_csv(
                io.BytesIO(body), parse_dates=["completed_at"]
            )
    logger.info("loaded %d snapshot(s) for %s", len(snapshots), service_date)
    return snapshots


def snapshot_at(
    when: pd.Timestamp, snapshots: dict[datetime, pd.DataFrame]
) -> pd.DataFrame | None:
    """The snapshot in force at `when` — the latest written at or before it.

    Returns None when nothing covers that moment — the honest outcome, since the row is
    then dropped rather than scored against conditions the model would not have had.
    """
    eligible = [t for t in snapshots if t <= when.to_pydatetime()]
    if not eligible:
        return None
    written = max(eligible)
    if (when.to_pydatetime() - written).total_seconds() > MAX_SNAPSHOT_AGE_SEC:
        return None
    return snapshots[written]


def realised_accuracy(
    journeys: pd.DataFrame,
    predictions: pd.Series,
    lengths: tuple[int, ...],
    target: str = "journey_duration_sec",
) -> pd.DataFrame:
    """MAE per journey length, against the model AND the published schedule.

    The schedule column is the point. "The model beat the timetable by X%" is the claim
    this project makes, and reporting it on the same rows makes the metric
    self-normalising: a hard week degrades both, so the ratio stays honest where a bare
    MAE would look like a regression.
    """
    scored = journeys.assign(prediction=predictions.to_numpy())
    rows = []
    for length in lengths:
        part = scored[scored["n_segments"] == length]
        if part.empty:
            continue
        model_error = (part[target] - part["prediction"]).abs()
        schedule_error = (part[target] - part["scheduled_total_sec"]).abs()
        rows.append(
            {
                "segments": length,
                "journeys": len(part),
                "mae_model_sec": float(model_error.mean()),
                "mae_schedule_sec": float(schedule_error.mean()),
                "bias_model_sec": float((part[target] - part["prediction"]).mean()),
                "median_duration_sec": float(part[target].median()),
                "beats_schedule_pct": (
                    float(100 * (1 - model_error.mean() / schedule_error.mean()))
                    if schedule_error.mean()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def null_rate(frame: pd.DataFrame, column: str = "recent_deviation") -> float:
    """Share of rows where the model's strongest feature was unavailable."""
    if column not in frame.columns or frame.empty:
        return 100.0
    return float(100 * frame[column].isna().mean())


def service_dates_to_score(
    features: pd.DataFrame, days: int = 1, now: datetime | None = None
) -> list[str]:
    """The most recent complete service dates, newest last.

    Yesterday by default: a service day closes at 04:00 UTC the next day, so today is
    never complete and scoring it would report a partial day as a full one.
    """
    today = (now or datetime.now(UTC)).date()
    available = sorted(features["service_date"].astype(str).unique())
    return [d for d in available if d < today.isoformat()][-days:]


def config_defaults() -> MonitoringConfig:
    return MonitoringConfig()
