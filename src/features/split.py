"""Temporal train/validation split with an embargo gap.

Built now, deliberately not trusted yet. With three service days there is no honest
split available — an 80/20 temporal cut puts Friday and Saturday in training and *only
Sunday evening* in validation, so any score measures "did the model learn Sunday
evenings" rather than generalisation. Day-of-week is itself a feature, which makes the
confound circular.

So this module implements the split correctly and **refuses to let you quietly trust
it**: `SplitReport.warnings` names every reason the split is not yet meaningful, and the
caller decides. The boundaries are configurable so the same code becomes useful once
there are weeks of data, without an edit.

Two rules that are not negotiable whenever it *is* used:

**No shuffling.** A random split leaks the future into the past through every rolling
feature — `recent_deviation` for a training row can be computed from a traversal that
sits in the validation set.

**An embargo wide enough to cover the lookback.** Rolling features look back
`rolling_max_age_sec`; a validation row starting less than that after the boundary can
see training data through its own history. The embargo drops those rows from *training*,
not from validation, because shrinking training is the conservative direction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from .config import FeatureConfig

logger = logging.getLogger("features.split")

# Below this a temporal split cannot separate day-of-week from time, so any validation
# score is measuring the wrong thing. Two full weeks give each weekday at least twice
# on either side of a boundary.
MIN_DAYS_FOR_MEANINGFUL_SPLIT = 14


@dataclass(frozen=True, slots=True)
class SplitReport:
    """Where the split fell, and every reason not to believe it yet."""

    boundary: datetime
    embargo_sec: int
    train_rows: int
    validation_rows: int
    embargoed_rows: int
    train_days: list[str]
    validation_days: list[str]
    unseen_segments: int
    warnings: list[str] = field(default_factory=list)

    @property
    def is_trustworthy(self) -> bool:
        return not self.warnings

    def as_metadata(self) -> dict[str, object]:
        """Recorded beside the feature table so a model traces back to its split."""
        return {
            "boundary_utc": self.boundary.isoformat(),
            "embargo_sec": self.embargo_sec,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "embargoed_rows": self.embargoed_rows,
            "train_days": self.train_days,
            "validation_days": self.validation_days,
            "unseen_segments": self.unseen_segments,
            "warnings": self.warnings,
            "trustworthy": self.is_trustworthy,
        }

    def format(self) -> str:
        lines = [
            "temporal split",
            f"  boundary            {self.boundary:%Y-%m-%d %H:%M} UTC",
            f"  embargo             {self.embargo_sec}s",
            f"  train               {self.train_rows:,} rows over "
            f"{len(self.train_days)} day(s)",
            f"  validation          {self.validation_rows:,} rows over "
            f"{len(self.validation_days)} day(s)",
            f"  embargoed from train {self.embargoed_rows:,} rows",
            f"  segments unseen in training {self.unseen_segments}",
        ]
        if self.warnings:
            lines.append("  !! THIS SPLIT IS NOT YET MEANINGFUL")
            lines.extend(f"     - {w}" for w in self.warnings)
        else:
            lines.append("  no warnings")
        return "\n".join(lines)


def temporal_split(
    features: pd.DataFrame,
    time_column: str = "actual_departure_ts",
    config: FeatureConfig | None = None,
    boundary: datetime | None = None,
) -> tuple[pd.Series, pd.Series, SplitReport]:
    """Split by time, with an embargo. Returns (train_mask, validation_mask, report).

    The two masks do **not** cover every row: embargoed rows belong to neither, which is
    the point. A three-way outcome is easy to get wrong as `~train_mask`, so both masks
    are returned explicitly rather than one being inferred.
    """
    settings = config or FeatureConfig()
    times = pd.to_datetime(features[time_column], utc=True)

    if boundary is None:
        boundary = times.quantile(1 - settings.validation_fraction)
    boundary = pd.Timestamp(boundary).tz_convert("UTC")

    validation_mask = times >= boundary
    embargo_start = boundary - timedelta(seconds=settings.embargo_sec)
    embargoed = (times >= embargo_start) & (times < boundary)
    train_mask = times < embargo_start

    train_days = sorted(features.loc[train_mask, "service_date"].astype(str).unique())
    validation_days = sorted(
        features.loc[validation_mask, "service_date"].astype(str).unique()
    )

    segment_pairs = list(
        zip(features["from_stop_id"], features["to_stop_id"], strict=True)
    )
    train_segments = {
        p for p, keep in zip(segment_pairs, train_mask, strict=True) if keep
    }
    unseen = {
        p for p, keep in zip(segment_pairs, validation_mask, strict=True) if keep
    } - train_segments

    report = SplitReport(
        boundary=boundary.to_pydatetime(),
        embargo_sec=settings.embargo_sec,
        train_rows=int(train_mask.sum()),
        validation_rows=int(validation_mask.sum()),
        embargoed_rows=int(embargoed.sum()),
        train_days=train_days,
        validation_days=validation_days,
        unseen_segments=len(unseen),
        warnings=_warnings(features, train_days, validation_days, len(unseen)),
    )

    if report.warnings:
        logger.warning("split is not yet meaningful:\n%s", report.format())
    return train_mask, validation_mask, report


def _warnings(
    features: pd.DataFrame,
    train_days: list[str],
    validation_days: list[str],
    unseen: int,
) -> list[str]:
    """Every reason this split should not be used to make a claim.

    Returned rather than raised: producing the split is still useful for exercising the
    plumbing, and a backfill may legitimately want it. What must not happen is a number
    coming out of it that reads like a generalisation estimate.
    """
    warnings: list[str] = []
    all_days = sorted(features["service_date"].astype(str).unique())

    if len(all_days) < MIN_DAYS_FOR_MEANINGFUL_SPLIT:
        warnings.append(
            f"only {len(all_days)} service day(s) available; "
            f"{MIN_DAYS_FOR_MEANINGFUL_SPLIT} are needed before a temporal split can "
            "separate day-of-week from time"
        )

    if validation_days:
        weekdays = {pd.Timestamp(d).day_name() for d in validation_days}
        if len(weekdays) == 1:
            warnings.append(
                f"validation covers only {weekdays.pop()} — day-of-week is a "
                "feature, so the split confounds what it is meant to measure"
            )

    if len(validation_days) == 1:
        warnings.append(
            "validation spans a single service date; a score from it reflects that "
            "day, not generalisation"
        )

    if unseen:
        warnings.append(
            f"{unseen} segment(s) appear in validation but never in training — "
            "cold-start categoricals the model has no basis to predict"
        )

    return warnings
