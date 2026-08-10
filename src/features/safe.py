"""Temporal and schedule-derived features — the group with no leakage risk.

Everything here is computable from the calendar or from the published timetable, both
of which are known long before the trip runs. Nothing reads an observed outcome. They
live in one module precisely so the leakage-critical work sits somewhere else and the
distinction shows up in the import list rather than buried in a docstring.

Prediction time throughout is **T = the segment's `actual_departure_ts`**: a rider is
standing at stop A at T asking how long the trip to B will take.
"""

from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd

from .config import FARE_EVENING_START_HOUR, SERVICE_TZ, FeatureConfig

logger = logging.getLogger("features.safe")

# `PF_A08_C` -> platform prefix, station code, track suffix. The station is the middle
# part: several platform ids map to one physical station, so any station-level feature
# has to work on the extracted code, not the raw stop_id.
_STATION_PATTERN = r"^PF_([A-Z]\d{2})"


def local_departure(segments: pd.DataFrame) -> pd.Series:
    """Departure time in America/New_York.

    The segment table stores UTC and has `service_date` already resolved to local, so
    this converts rather than re-deriving. Doing it once here means no downstream
    feature has to remember — and a UTC hour is four or five hours off depending on the
    season, which produces a rush-hour feature that peaks at lunchtime.
    """
    return pd.to_datetime(segments["actual_departure_ts"], utc=True).dt.tz_convert(
        SERVICE_TZ
    )


@functools.lru_cache(maxsize=8)
def _holiday_dates(year: int) -> frozenset:
    """US federal holidays for a year, cached.

    Imported lazily so the package still imports if `holidays` is absent — the feature
    degrades to all-False rather than breaking every other feature with it.
    """
    try:
        import holidays
    except ImportError:  # pragma: no cover - exercised only on a bare install
        logger.warning("holidays package not installed; is_holiday will be all False")
        return frozenset()
    return frozenset(holidays.country_holidays("US", years=year).keys())


def temporal_features(segments: pd.DataFrame) -> pd.DataFrame:
    """Calendar features. Trivially safe: known years in advance."""
    local = local_departure(segments)
    hour = local.dt.hour + local.dt.minute / 60.0
    dow = local.dt.dayofweek

    out = pd.DataFrame(index=segments.index)
    out["local_hour"] = local.dt.hour.astype("int16")
    out["local_hour_frac"] = hour.astype("float32")
    out["day_of_week"] = dow.astype("int8")
    out["month"] = local.dt.month.astype("int8")
    out["is_weekend"] = (dow >= 5).astype("int8")

    # Both cyclical and raw are provided. Trees split on the raw integer perfectly well
    # and often prefer it; linear and distance-based models need the cyclical form so
    # that 23:00 and 00:00 are adjacent rather than maximally far apart. Cheap to carry
    # both and let training decide rather than guessing here.
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype("float32")
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype("float32")

    dates = local.dt.date
    years = {d.year for d in dates.dropna().unique()}
    holiday_set: set = set()
    for year in years:
        holiday_set |= set(_holiday_dates(year))
    out["is_holiday"] = dates.isin(holiday_set).astype("int8")

    # Federal holidays run a weekend-style service pattern, so "weekday" for service
    # purposes means a weekday that is not a holiday.
    out["is_service_weekday"] = ((dow < 5) & (out["is_holiday"] == 0)).astype("int8")

    # WMATA's current fare periods, which no longer encode rush hour: peak fares were
    # eliminated, so the only distinction left is weekday-daytime / weekday-evening /
    # weekend. Kept as a low-value categorical for completeness. The real
    # service-intensity signal is scheduled headway — see `schedule_features`.
    evening = hour >= FARE_EVENING_START_HOUR
    out["fare_period"] = np.where(
        out["is_service_weekday"] == 0,
        "weekend",
        np.where(evening, "weekday_evening", "weekday_daytime"),
    )
    return out


def station_code(stop_ids: pd.Series) -> pd.Series:
    """`PF_A08_C` -> `A08`. Falls back to the raw id when the pattern does not match."""
    extracted = stop_ids.astype(str).str.extract(_STATION_PATTERN, expand=False)
    return extracted.fillna(stop_ids.astype(str))


def identity_features(segments: pd.DataFrame) -> pd.DataFrame:
    """Route and segment identity, as categoricals.

    High cardinality but not extreme — ~659 segments and 6 routes — which is why these
    stay as native categoricals rather than being target-encoded. Target encoding would
    have to be fit inside a CV fold or a strictly-prior window to avoid leaking the mean
    of the label into a feature, and that is a lot of machinery for a modest lift on a
    cardinality LightGBM handles natively.
    """
    out = pd.DataFrame(index=segments.index)
    out["route_id"] = segments["route_id"].astype("category")
    out["direction_id"] = segments["direction_id"].astype("category")
    out["segment_id"] = (
        segments["from_stop_id"].astype(str) + ">" + segments["to_stop_id"].astype(str)
    ).astype("category")
    out["from_station"] = station_code(segments["from_stop_id"]).astype("category")
    out["to_station"] = station_code(segments["to_stop_id"]).astype("category")
    return out


def schedule_features(
    segments: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Timetable-derived features. Safe: the schedule is published in advance.

    Scheduled headway is the honest replacement for a "peak" flag — the operator's own
    plan for how much service runs at this time and place, continuous rather than a
    bucket, and it captures what rush hour actually means for a duration model: more
    trains, denser platforms, more dwell pressure.
    """
    settings = config or FeatureConfig()
    out = pd.DataFrame(index=segments.index)

    out["scheduled_duration_sec"] = segments["scheduled_duration_sec"].astype("float32")
    out["stop_sequence"] = segments["stop_sequence"].astype("int16")
    out["from_stop_sequence"] = segments["from_stop_sequence"].astype("int16")
    out["stop_span"] = segments["stop_span"].astype("int8")

    # Position within the trip. A segment near the end of a run behaves differently from
    # one near the start — accumulated delay, crew changes, terminal effects.
    #
    # CAVEAT, and it is the sharpest one in this module. The *value* is safe: the
    # timetable says where a trip terminates, so "stops remaining" is knowable at T. The
    # *computation* below is not equivalent to that — it takes the max `stop_sequence`
    # over the trip's observed rows, which includes rows after T.
    #
    # Two consequences. When a trip is fully observed the two agree exactly. When it is
    # truncated — the archive starts mid-trip, or the vehicle drops out of the feed —
    # the observed max understates the terminus, and the error correlates with data
    # availability rather than with anything about the journey. That is a real, if
    # small, leak of observability into a feature.
    #
    # Kept because the signal is genuinely useful and the distortion is bounded, but
    # **serving must resolve the terminus from static GTFS**, not from a window of
    # observed rows it will not have. Flagged in the parity test for that reason.
    trip_max = segments.groupby(["service_date", "trip_id", "trip_run"], observed=True)[
        "stop_sequence"
    ].transform("max")
    out["stops_remaining"] = (trip_max - segments["stop_sequence"]).astype("int16")
    out["trip_progress"] = (
        segments["stop_sequence"] / trip_max.replace(0, np.nan)
    ).astype("float32")

    # Trip start time-of-day: a trip that began in the morning peak may run differently
    # mid-route than one that began off-peak.
    trip_start = segments.groupby(
        ["service_date", "trip_id", "trip_run"], observed=True
    )["actual_departure_ts"].transform("min")
    start_local = pd.to_datetime(trip_start, utc=True).dt.tz_convert(SERVICE_TZ)
    out["trip_start_hour"] = (
        start_local.dt.hour + start_local.dt.minute / 60.0
    ).astype("float32")
    out["minutes_into_trip"] = (
        (
            pd.to_datetime(segments["actual_departure_ts"], utc=True)
            - pd.to_datetime(trip_start, utc=True)
        ).dt.total_seconds()
        / 60.0
    ).astype("float32")

    # A "segment" whose scheduled duration is 77 minutes is a layover or a multi-stop
    # span, not a run. Flagged rather than dropped so the rate stays visible and the
    # model can learn to treat them differently.
    out["scheduled_duration_implausible"] = (
        out["scheduled_duration_sec"] > settings.max_plausible_scheduled_sec
    ).astype("int8")

    return out


def build_safe_features(
    segments: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """All leakage-free features in one frame."""
    settings = config or FeatureConfig()
    parts = []
    if settings.enable_temporal:
        parts.append(temporal_features(segments))
    if settings.enable_schedule:
        parts.append(schedule_features(segments, settings))
    if settings.enable_identity:
        parts.append(identity_features(segments))
    if not parts:
        return pd.DataFrame(index=segments.index)
    return pd.concat(parts, axis=1)
