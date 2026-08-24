"""Publish the slow-moving half of the serving inputs to S3.

    python -m src.serving.publish

`recent_deviation` is a **live duration measured against a slow-moving baseline**. Only
the live half needs to be fresh, and splitting them is what makes the whole live path
tractable:

| input | changes | produced here | consumed by |
| --- | --- | --- | --- |
| segment x hour medians | slowly | daily, from the feature table | the live Lambda |
| scheduled duration per segment | on timetable rollover | daily | the live Lambda |
| station index | on timetable rollover | daily | the endpoint |

The Lambda then only has to derive *this hour's* traversals and compare them to these
baselines — a few thousand rows — rather than recomputing history every five minutes.

**CSV, not Parquet.** The Lambda and the inference container both read these, and
neither should need pyarrow: it is 126 MB installed against LightGBM's 5.5 MB, and in a
Lambda it competes with a hard package-size limit. pandas reads CSV natively.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from ..etl.config import EtlConfig
from ..features.config import SEGMENT_KEY, TARGET

logger = logging.getLogger("serving.publish")

SERVING_PREFIX = "models/serving/"
DEFAULT_FEATURES = "data/processed/features/table"
DEFAULT_LOCAL = "data/processed/serving"


def segment_hour_medians(features: pd.DataFrame) -> pd.DataFrame:
    """Median duration per (segment, local hour) — the baseline a deviation is against.

    This is the same quantity `models.evaluate.fitted_segment_hour_median` uses as the
    honest baseline, and the same one `historical.py` measures `recent_deviation`
    against. Computed once here so the live path never has to reconstruct it.
    """
    keys = [*SEGMENT_KEY, "local_hour"]
    medians = (
        features.groupby(keys, observed=True)[TARGET]
        .agg(baseline_sec="median", traversals="size")
        .reset_index()
    )
    medians["baseline_sec"] = medians["baseline_sec"].round().astype("int32")
    logger.info("segment x hour medians: %d row(s)", len(medians))
    return medians


def segment_scheduled(features: pd.DataFrame) -> pd.DataFrame:
    """Scheduled duration per segment, for `recent_delay_mean`.

    Modal rather than mean: scheduled durations are a small set of discrete values, and
    an average across a timetable change would produce a number the timetable never
    contained.
    """
    scheduled = (
        features.groupby(list(SEGMENT_KEY), observed=True)["scheduled_duration_sec"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.median())
        .reset_index()
        .rename(columns={"scheduled_duration_sec": "scheduled_sec"})
    )
    scheduled["scheduled_sec"] = scheduled["scheduled_sec"].round().astype("int32")
    logger.info("segment scheduled durations: %d row(s)", len(scheduled))
    return scheduled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--local", default=DEFAULT_LOCAL)
    parser.add_argument("--dry-run", action="store_true", help="write locally only")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    features = pd.read_parquet(args.features)
    features["service_date"] = features["service_date"].astype(str)
    logger.info(
        "read %d segment(s) over %d service date(s)",
        len(features),
        features["service_date"].nunique(),
    )

    local = Path(args.local)
    local.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "segment_hour_medians.csv": segment_hour_medians(features),
        "segment_scheduled.csv": segment_scheduled(features),
    }
    for name, frame in artifacts.items():
        frame.to_csv(local / name, index=False)

    station_index = local / "station_index.json"
    if not station_index.exists():
        raise FileNotFoundError(
            f"{station_index} missing — run `python -m src.serving.stations` first"
        )

    if args.dry_run:
        logger.info("dry run — wrote locally to %s, nothing uploaded", local)
        return 0

    import boto3

    config = EtlConfig.from_env()
    client = boto3.client("s3", region_name=config.aws_region)
    for name in [*artifacts, station_index.name]:
        key = f"{SERVING_PREFIX}{name}"
        client.upload_file(str(local / name), config.s3_bucket, key)
        logger.info("uploaded s3://%s/%s", config.s3_bucket, key)

    print(
        f"\npublished {len(artifacts) + 1} serving input(s) to "
        f"s3://{config.s3_bucket}/{SERVING_PREFIX}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
