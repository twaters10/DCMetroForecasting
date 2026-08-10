"""Single source of configuration for the feature layer.

Mirrors `src/etl/config.py`: nothing here reads credentials or touches the network at
import time, and every tunable lives in one place rather than scattered through the
feature functions.

The feature layer is deliberately **pandas, not Spark**. A service day is ~30k segment
rows and the whole 90-day retention is ~3M — comfortably in-memory. More importantly the
same feature definitions must run in two places, batch over Parquet and per-request at
inference, and "write once, call from both" is trivial with pandas functions and awkward
with a Spark job. Spark stays where it earns its place, in the ETL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from zoneinfo import ZoneInfo

# Local time is what the model cares about: "the 08:00 rush" is 08:00 in Washington DC.
# The segment table stores UTC timestamps with `service_date` already resolved to this
# zone, so this is used for deriving hour-of-day, not for redoing that conversion.
SERVICE_TZ: Final[ZoneInfo] = ZoneInfo("America/New_York")

# The regression target. Modelled directly rather than as delay, log, or a ratio.
#
# Measured on 83,614 clean rows: residual spread of the segment-median baseline is far
# more stable in absolute terms than relative (CV 0.28 vs 0.50 across segment-length
# buckets). The dominant error is arrival quantization, which is ±30s regardless of how
# long the segment is, so a relative target amplifies noise exactly where the data is
# densest — the <90s bucket already carries 65.8% relative spread. `delay_sec` remains
# recoverable by subtracting `scheduled_duration_sec`, which is a feature either way.
TARGET: Final[str] = "actual_duration_sec"

# Identifies one journey. NOT `trip_id` alone: WMATA reuses a trip_id within a service
# day and 18.9% of rows belong to a repeat run. Accumulating upstream delay on `trip_id`
# would sum across two separate journeys and invent delay that never happened.
TRIP_KEY: Final[tuple[str, ...]] = ("service_date", "trip_id", "trip_run")

# Identifies one physical segment, for rolling history and categorical encoding.
SEGMENT_KEY: Final[tuple[str, ...]] = ("from_stop_id", "to_stop_id")

# Columns the model must never see. The target and anything derived from the segment's
# own completion — knowing when it arrived is knowing the answer.
LABEL_SIDE_COLUMNS: Final[tuple[str, ...]] = (
    "actual_duration_sec",
    "actual_arrival_ts",
    "delay_sec",
    "observed_at_utc",
)


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Every tunable in the feature layer."""

    # ---- row filtering -------------------------------------------------
    # Defaults keep only arrivals that were observed rather than predicted, with a
    # bracket tight enough to trust. Configurable because a sparse backfill may
    # legitimately want the weaker rows, flagged.
    require_plausible_duration: bool = True
    require_confident_arrival: bool = True
    require_observed_arrival: bool = False

    # A "segment" whose scheduled duration is 77 minutes is a layover or a
    # `stop_span > 1` artefact, not a run. Observed range reaches 4,620s. Rows beyond
    # this are flagged rather than dropped, so the rate stays visible.
    max_plausible_scheduled_sec: int = 900

    # ---- historical windows --------------------------------------------
    # How many previous *completed* traversals of the same segment feed the rolling
    # statistics. Small enough to track conditions within an hour, large enough that the
    # median is not itself noise.
    rolling_traversals: int = 5

    # Ignore a prior traversal older than this — a segment's state two hours ago says
    # nothing about now, and including it makes the feature look informative in
    # backtests while being useless at serving time.
    rolling_max_age_sec: int = 3600

    # Hour-of-week norms need several weeks before a cell holds enough observations to
    # mean anything. Below this the feature emits null rather than a number derived from
    # one or two samples.
    min_observations_for_norm: int = 5

    # ---- split ----------------------------------------------------------
    # Gap between train and validation. Rolling features look back
    # `rolling_max_age_sec`, so without an embargo at least that wide a validation row
    # near the boundary can see training data through its own history.
    embargo_sec: int = 3600
    validation_fraction: float = 0.2

    # ---- feature group toggles ------------------------------------------
    # Named so a group can be disabled wholesale when isolating a leakage suspicion.
    enable_temporal: bool = True
    enable_schedule: bool = True
    enable_identity: bool = True
    enable_historical: bool = True

    # ---- categorical -----------------------------------------------------
    # Native LightGBM categorical handling, not target encoding. 659 segments is small
    # enough that native splits are sufficient, and target encoding would have to be fit
    # inside a CV fold or a strictly-prior window to be safe — leakage risk for modest
    # lift. Integer codes plus a persisted mapping so serving encodes identically.
    categorical_columns: tuple[str, ...] = (
        "route_id",
        "direction_id",
        "segment_id",
        "from_station",
        "to_station",
    )

    # ---- output -----------------------------------------------------------
    output_path: str = "data/processed/features"

    def filter_mask_columns(self) -> list[str]:
        """Which quality columns the row filter consults, given the toggles."""
        columns: list[str] = []
        if self.require_plausible_duration:
            columns.append("duration_plausible")
        if self.require_confident_arrival:
            columns.append("arrival_confident")
        return columns


# WMATA fare periods, verified against the current published policy rather than
# remembered. **Metro eliminated peak fares.** The structure is now:
#
#   weekday before 21:30  -> one distance-based fare, no AM/PM rush distinction
#   weekday after 21:30   -> $2 flat
#   all weekend           -> $2 flat
#
# This matters for feature design: a "peak fare" flag no longer encodes rush hour,
# because the fare no longer distinguishes it. All the flag separates is
# weekday-daytime / weekday-evening / weekend, which `hour` and `is_weekend` already
# carry. It is emitted as a low-value categorical for completeness, not as the peak
# signal.
#
# The real service-intensity signal is SCHEDULED HEADWAY from static GTFS — the
# operator's own plan for how much service runs at a given time. It is known in advance
# (so trivially safe), it is continuous rather than a bucket, and it encodes what "rush
# hour" actually means for a duration model: more trains, denser platforms, more
# dwell-time pressure. See `schedule.scheduled_headway_sec`.
FARE_EVENING_START_HOUR: Final[float] = 21.5

FARE_PERIODS: Final[tuple[str, ...]] = (
    "weekday_daytime",
    "weekday_evening",
    "weekend",
)

# Federal holidays run a weekend/Sunday-style service pattern, so they are flagged and
# should not be treated as ordinary weekdays.
HOLIDAY_COUNTRY: Final[str] = "US"
HOLIDAY_SUBDIVISION: Final[str] = "DC"


DEFAULT_CONFIG: Final[FeatureConfig] = FeatureConfig()
