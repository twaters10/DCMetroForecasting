"""Block-aware temporal split for journey rows.

`features.split.temporal_split` splits row by row on time, which is right for
segments — each row is one independent observation. It is **wrong for journeys**,
because one block of ~14 consecutive segments generates up to ~90 overlapping journey
rows that share the same underlying traversals. A row-wise boundary would put the A->C
journey in training and the A->D journey in validation, and those two are the same
train over almost the same track. The score would be measuring memorisation.

So the unit of assignment is the **block**, not the row: every journey originating in a
block goes to the same side. Boundary and embargo keep the same meaning as the segment
split, and the embargo still shrinks *training* rather than validation, because
shrinking training is the conservative direction.

The embargo matters more here. A journey runs up to ~40 minutes, so one starting shortly
before the boundary is still going well after it, and its label is partly determined by
conditions on the validation side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from ..features.config import FeatureConfig
from ..features.split import MIN_DAYS_FOR_MEANINGFUL_SPLIT

logger = logging.getLogger("journeys.split")


@dataclass(frozen=True, slots=True)
class JourneySplitReport:
    """What the split did, and every reason not to trust it."""

    boundary: datetime
    embargo_sec: int
    train_rows: int
    validation_rows: int
    embargoed_rows: int
    train_blocks: int
    validation_blocks: int
    train_days: list[str]
    validation_days: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def is_trustworthy(self) -> bool:
        return not self.warnings

    def as_metadata(self) -> dict[str, object]:
        return {
            "boundary_utc": self.boundary.isoformat(),
            "embargo_sec": self.embargo_sec,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "embargoed_rows": self.embargoed_rows,
            "train_blocks": self.train_blocks,
            "validation_blocks": self.validation_blocks,
            "train_days": self.train_days,
            "validation_days": self.validation_days,
            "warnings": self.warnings,
            "trustworthy": self.is_trustworthy,
        }

    def format(self) -> str:
        lines = [
            "journey split (grouped by block)",
            f"  boundary            {self.boundary:%Y-%m-%d %H:%M} UTC",
            f"  embargo             {self.embargo_sec}s",
            f"  train               {self.train_rows:,} rows / "
            f"{self.train_blocks:,} blocks over {len(self.train_days)} day(s)",
            f"  validation          {self.validation_rows:,} rows / "
            f"{self.validation_blocks:,} blocks over "
            f"{len(self.validation_days)} day(s)",
            f"  embargoed from train {self.embargoed_rows:,} rows",
        ]
        if self.warnings:
            lines.append("  !! THIS SPLIT IS NOT YET MEANINGFUL")
            lines += [f"     - {w}" for w in self.warnings]
        return "\n".join(lines)


def split_journeys(
    journeys: pd.DataFrame,
    config: FeatureConfig | None = None,
    boundary: datetime | None = None,
) -> tuple[pd.Series, pd.Series, JourneySplitReport]:
    """Split by time, assigning whole blocks. Returns (train, validation, report)."""
    settings = config or FeatureConfig()
    times = pd.to_datetime(journeys["origin_departure_ts"], utc=True)

    # A block's position in time is its EARLIEST origin. Using the latest would let a
    # block whose first journey is deep in training be assigned to validation.
    block_time = times.groupby(journeys["block"]).transform("min")

    if boundary is None:
        # Quantile over blocks, not rows: long blocks generate more rows and would
        # otherwise drag the boundary toward themselves.
        unique = block_time.groupby(journeys["block"]).first()
        boundary = unique.quantile(1 - settings.validation_fraction)
    boundary = pd.Timestamp(boundary).tz_convert("UTC")

    embargo_start = boundary - timedelta(seconds=settings.embargo_sec)
    validation_mask = block_time >= boundary
    embargoed = (block_time >= embargo_start) & (block_time < boundary)
    train_mask = block_time < embargo_start

    train_days = sorted(journeys.loc[train_mask, "service_date"].astype(str).unique())
    validation_days = sorted(
        journeys.loc[validation_mask, "service_date"].astype(str).unique()
    )

    warnings: list[str] = []
    all_days = sorted(set(train_days) | set(validation_days))
    if len(all_days) < MIN_DAYS_FOR_MEANINGFUL_SPLIT:
        warnings.append(
            f"only {len(all_days)} service day(s) available; "
            f"{MIN_DAYS_FOR_MEANINGFUL_SPLIT} are needed before a temporal split can "
            "separate day-of-week from time"
        )
    if len(validation_days) < 2:
        warnings.append(
            "validation spans a single service date; a score from it reflects that day"
        )
    overlap = set(train_days) & set(validation_days)
    if len(validation_days) and set(validation_days) <= overlap:
        warnings.append(
            "every validation day also appears in training — the boundary falls "
            "mid-day, so day-level effects are shared across both sides"
        )

    report = JourneySplitReport(
        boundary=boundary.to_pydatetime(),
        embargo_sec=settings.embargo_sec,
        train_rows=int(train_mask.sum()),
        validation_rows=int(validation_mask.sum()),
        embargoed_rows=int(embargoed.sum()),
        train_blocks=int(journeys.loc[train_mask, "block"].nunique()),
        validation_blocks=int(journeys.loc[validation_mask, "block"].nunique()),
        train_days=train_days,
        validation_days=validation_days,
        warnings=warnings,
    )
    if report.warnings:
        logger.warning("journey split is not yet meaningful:\n%s", report.format())
    return train_mask, validation_mask, report
