"""Historical-condition features — the leakage-critical group.

Prediction time is **T = the segment's `actual_departure_ts`**. Everything here must be
computable from what a dispatcher standing at stop A at time T could know, which imposes
two rules that are easy to state and easy to get subtly wrong:

1. **Strictly before T.** Not `<=`. A traversal that completed at exactly T is not
   information you had *at* T, and an off-by-one here produces a model that validates
   beautifully and fails in production.
2. **Completed, not merely started.** A train that departed this segment two minutes ago
   but has not yet arrived tells you nothing about how long it took. The as-of joins key
   on `actual_arrival_ts`, the completion time — using departure time instead would leak
   the outcome of an in-flight traversal.

Measured usefulness on three service days, against the residual left after a
segment×hour-of-day median:

| feature | correlation with residual |
| --- | --- |
| previous traversal's residual on this segment | **+0.255** |
| upstream cumulative delay for this trip | **-0.010** |

Upstream delay is the one the brief expected to dominate. On this data it is worthless
linearly. It is still computed — a tree may find interactions a correlation misses, and
its absence would be conspicuous — but it should not be assumed to carry the model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import SEGMENT_KEY, TRIP_KEY, FeatureConfig

logger = logging.getLogger("features.historical")


def upstream_delay(segments: pd.DataFrame) -> pd.DataFrame:
    """Delay this trip accumulated on strictly prior segments.

    Keyed on `(service_date, trip_id, trip_run)`, **not** `trip_id`. WMATA reuses a
    trip_id within a service day and 18.9% of rows are a repeat run; accumulating on
    trip_id alone would carry the first journey's delay into the second and invent delay
    that never happened.

    `shift(1)` before `cumsum` is what makes it strictly prior — without it the row's
    own delay is included, which is the label leaking straight into a feature.
    """
    frame = segments.sort_values([*TRIP_KEY, "stop_sequence"])
    keys = [frame[key] for key in TRIP_KEY]

    # shift(1) first, then cumsum within the same groups. Order matters:
    # cumsum-then-shift also excludes the current row, but shift-then-cumsum keeps the
    # total anchored to "everything before this stop", which is what is known at T.
    prior_delay = frame.groupby(list(TRIP_KEY), observed=True, sort=False)[
        "delay_sec"
    ].shift(1)

    out = pd.DataFrame(index=frame.index)
    out["upstream_delay_sec"] = prior_delay.groupby(keys).cumsum().astype("float32")
    out["upstream_delay_last_sec"] = prior_delay.astype("float32")
    out["segments_completed"] = (
        frame.groupby(list(TRIP_KEY), observed=True, sort=False)
        .cumcount()
        .astype("int16")
    )
    return out.reindex(segments.index)


def rolling_segment_conditions(
    segments: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """How the last N *completed* traversals of this segment went.

    The strongest signal available (+0.255 with the residual) and the one that most
    rewards getting the as-of logic right.

    Implemented as a two-step rather than a per-row scan, because the naive version is
    both slow and easy to get wrong:

    1. Build a completions timeline per segment, ordered by `actual_arrival_ts`. Rolling
       statistics over that ordering give, at each completion, the stats of the last N
       completions up to and including it.
    2. `merge_asof` each query row's departure time T backwards onto that timeline with
       `allow_exact_matches=False`. That resolves to the newest completion **strictly**
       before T, and carries its rolling statistic.

    Step 2's `allow_exact_matches=False` is the strict `<`. With the default `True`, a
    traversal completing exactly at T would be visible to a prediction made at T.
    """
    settings = config or FeatureConfig()
    window = settings.rolling_traversals

    completions = (
        segments.loc[
            :, [*SEGMENT_KEY, "actual_arrival_ts", "actual_duration_sec", "delay_sec"]
        ]
        .dropna(subset=["actual_arrival_ts"])
        .sort_values("actual_arrival_ts")
        .reset_index(drop=True)
    )
    grouped = completions.groupby(list(SEGMENT_KEY), observed=True, sort=False)
    completions["roll_median"] = grouped["actual_duration_sec"].transform(
        lambda s: s.rolling(window, min_periods=1).median()
    )
    completions["roll_mean"] = grouped["actual_duration_sec"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    completions["roll_delay_mean"] = grouped["delay_sec"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    # How many traversals actually back the statistics above — 1 means `roll_median` is
    # a single observation, `window` means it is the full window. Clipped, because the
    # raw cumcount is a running total that grows across the archive: as a feature it
    # would encode "how late in the dataset is this row", which is a trend the model can
    # fit in training and which no serving request can reproduce.
    completions["roll_n"] = (grouped.cumcount() + 1).clip(upper=window)

    # The signal is in the DEVIATION, not the level. `roll_median` is dominated by how
    # long this segment normally takes, which the model already knows from `segment_id`
    # and `scheduled_duration_sec` — correlating it against the residual gives only
    # +0.086. What actually predicts is whether recent trains ran slow *for this
    # segment*, which measured +0.255.
    #
    # So: each completion's deviation from that segment's own strictly-prior expanding
    # median, averaged over the last N. `shift(1)` on the expanding median keeps it
    # prior — an expanding median including the current row would subtract part of the
    # value from itself.
    prior_norm = grouped["actual_duration_sec"].transform(
        lambda s: s.expanding().median().shift(1)
    )
    completions["deviation"] = completions["actual_duration_sec"] - prior_norm
    completions["roll_deviation"] = completions.groupby(
        list(SEGMENT_KEY), observed=True, sort=False
    )["deviation"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    completions = completions.rename(columns={"actual_arrival_ts": "completed_at"})

    queries = (
        segments.loc[:, [*SEGMENT_KEY, "actual_departure_ts"]]
        .assign(_row=np.arange(len(segments)))
        .sort_values("actual_departure_ts")
    )

    merged = pd.merge_asof(
        queries,
        completions[
            [
                *SEGMENT_KEY,
                "completed_at",
                "roll_median",
                "roll_mean",
                "roll_delay_mean",
                "roll_n",
                "roll_deviation",
            ]
        ],
        left_on="actual_departure_ts",
        right_on="completed_at",
        by=list(SEGMENT_KEY),
        direction="backward",
        allow_exact_matches=False,  # strict `<`: this is the whole leakage guard
    )

    # A traversal from two hours ago says nothing about now. Without this the feature
    # looks informative in a backtest — there is always *some* prior traversal — while
    # being useless at serving time, when the endpoint has no such history to hand.
    age = (merged["actual_departure_ts"] - merged["completed_at"]).dt.total_seconds()
    too_old = age > settings.rolling_max_age_sec
    for column in (
        "roll_median",
        "roll_mean",
        "roll_delay_mean",
        "roll_n",
        "roll_deviation",
    ):
        merged.loc[too_old, column] = np.nan

    merged = merged.sort_values("_row")
    out = pd.DataFrame(index=segments.index)
    out["recent_duration_median"] = merged["roll_median"].to_numpy(dtype="float32")
    out["recent_duration_mean"] = merged["roll_mean"].to_numpy(dtype="float32")
    out["recent_delay_mean"] = merged["roll_delay_mean"].to_numpy(dtype="float32")
    out["recent_traversals"] = merged["roll_n"].to_numpy(dtype="float32")
    out["recent_age_sec"] = np.where(too_old, np.nan, age).astype("float32")
    # The headline feature: how far recent traversals ran from this segment's own norm.
    out["recent_deviation"] = merged["roll_deviation"].to_numpy(dtype="float32")

    # Deviation of the schedule from recent reality: if the last few trains took much
    # longer than the timetable says, this one probably will too.
    out["recent_vs_scheduled"] = (
        out["recent_duration_median"] - segments["scheduled_duration_sec"].to_numpy()
    ).astype("float32")
    return out


def headway(segments: pd.DataFrame) -> pd.DataFrame:
    """Time since the previous vehicle departed this stop, same route and direction.

    Available at T by construction — it is about a departure that already happened, not
    an arrival that has not. This is the feature-side use of headway; measuring headway
    at the segment's *arrival* would be label-side, and the same arithmetic is safe or
    fatal depending on which timestamp it is anchored to.
    """
    keys = ["route_id", "direction_id", "from_stop_id"]
    frame = segments.sort_values([*keys, "actual_departure_ts"])
    gap = (
        frame.groupby(keys, observed=True, sort=False)["actual_departure_ts"]
        .diff()
        .dt.total_seconds()
    )

    out = pd.DataFrame(index=frame.index)
    out["headway_sec"] = gap.astype("float32")
    # A headway of six hours is the first train of the day, not a service gap. Capped so
    # the feature does not carry an outlier that swamps every split on it.
    out.loc[out["headway_sec"] > 3600, "headway_sec"] = np.nan
    return out.reindex(segments.index)


def build_historical_features(
    segments: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Every leakage-critical feature, in one frame."""
    settings = config or FeatureConfig()
    if not settings.enable_historical:
        return pd.DataFrame(index=segments.index)

    frame = segments.copy()
    frame["actual_departure_ts"] = pd.to_datetime(
        frame["actual_departure_ts"], utc=True
    )
    frame["actual_arrival_ts"] = pd.to_datetime(frame["actual_arrival_ts"], utc=True)

    parts = [
        upstream_delay(frame),
        rolling_segment_conditions(frame, settings),
        headway(frame),
    ]
    out = pd.concat(parts, axis=1)
    logger.info(
        "historical features: %d rows, recent-conditions fill rate %.1f%%",
        len(out),
        100 * out["recent_duration_median"].notna().mean(),
    )
    return out
