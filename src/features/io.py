"""Loading the segment table and applying the quality filter.

Separated from the feature functions so that everything in `temporal`, `schedule` and
`historical` is a pure DataFrame transformation with no S3 in its import graph. That is
what makes the serving path possible: the endpoint imports the feature functions without
dragging boto3 and a bucket name along with them.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..etl.config import EtlConfig
from ..etl.processed import read_segments
from .config import FeatureConfig

logger = logging.getLogger("features.io")


def load_segments(
    start: str | None = None,
    end: str | None = None,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Read the segment table from S3 and apply the quality filter.

    `start`/`end` are inclusive `service_date` bounds. Both None reads everything.
    """
    settings = config or FeatureConfig()
    etl_config = EtlConfig.from_env()

    table = read_segments(etl_config)
    frame = table.to_pandas()
    frame["service_date"] = frame["service_date"].astype(str)

    if start:
        frame = frame[frame["service_date"] >= start]
    if end:
        frame = frame[frame["service_date"] <= end]

    return apply_quality_filter(frame, settings)


def apply_quality_filter(
    segments: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Drop rows the ETL flagged as untrustworthy, and report what went.

    Filtering rather than weighting for the hard failures — a segment whose duration is
    implausible is not a low-confidence observation, it is a derivation artefact. The
    *soft* signal, how precisely the arrival was bracketed, is kept as a column and
    surfaced as a sample weight instead.

    Every drop is logged with its count. A filter that quietly removes a third of the
    data is the kind of thing that only becomes visible when the model underperforms.
    """
    settings = config or FeatureConfig()
    before = len(segments)
    frame = segments

    for column in settings.filter_mask_columns():
        dropped = int((~frame[column].astype(bool)).sum())
        if dropped:
            logger.info("filter %s: dropping %d row(s)", column, dropped)
        frame = frame[frame[column].astype(bool)]

    if settings.require_observed_arrival:
        dropped = int((frame["arrival_source"] != "vehicle_position").sum())
        if dropped:
            logger.info("filter observed-only: dropping %d predicted row(s)", dropped)
        frame = frame[frame["arrival_source"] == "vehicle_position"]

    # Rows with no schedule cannot produce scheduled-duration features and cannot be
    # scored against the schedule baseline.
    dropped = int(frame["scheduled_duration_sec"].isna().sum())
    if dropped:
        logger.info("filter missing-schedule: dropping %d row(s)", dropped)
    frame = frame[frame["scheduled_duration_sec"].notna()]

    kept = len(frame)
    logger.info(
        "quality filter kept %d/%d rows (%.1f%%)",
        kept,
        before,
        100 * kept / max(before, 1),
    )
    return frame.reset_index(drop=True)
