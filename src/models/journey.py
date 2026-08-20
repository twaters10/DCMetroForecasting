"""Journey-level windows over the per-segment table.

The model predicts one segment at a time; a rider asks about a journey. This module is
the bridge, and it exists as importable code rather than notebook cells because the
naive version is wrong in a way that does not announce itself.

**Segments telescope, but only where they are contiguous.** `actual_departure_ts` is the
upstream *arrival*, so segment `i`'s arrival is exactly segment `i+1`'s departure and a
journey's duration is a difference of two cumulative sums — verified on the raw segment
table at 100.00% of consecutive pairs, 0.0s discrepancy.

**The feature table is not the raw segment table.** `features.io.load_segments` applies
the quality filter, dropping ~2.5% of segments for implausible duration or unconfident
arrival, which punches holes into the middle of otherwise intact trips. Measured on 12
service days: 767 broken pairs, the largest spanning 6,420s. Summing across one of those
invents a journey that never ran and quietly inflates every error metric.

So every window here is confined to a **block** — a maximal run of genuinely consecutive
segments — and never crosses a hole.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.config import TARGET, TRIP_KEY

TRIP_COLUMNS: list[str] = list(TRIP_KEY)

# Segments are ordered within a trip by the stop they depart from. `stop_sequence` is
# the downstream stop and would order identically; `from_stop_sequence` is used because
# it is the one the contiguity check compares against.
ORDER_COLUMN = "from_stop_sequence"


def sort_for_journeys(frame: pd.DataFrame) -> pd.DataFrame:
    """Trip order. Every function here depends on it, so it is done in one place."""
    return frame.sort_values(TRIP_COLUMNS + [ORDER_COLUMN])


def contiguous_blocks(frame: pd.DataFrame) -> pd.Series:
    """Label each maximal run of consecutive segments, aligned to `frame.index`.

    A run breaks on any of three conditions:

    - a new trip begins
    - the previous segment's arrival is not this one's departure (a filtered-out segment
      left a hole in time)
    - `from_stop_sequence` does not continue the previous `stop_sequence` (a hole in
      stop order, which the time check usually also catches but not always)

    A segment with `stop_span > 1` is **not** a break. It spans a stop passed between
    polls, and the ETL records that honestly rather than leaving a gap — the timeline is
    still continuous.
    """
    ordered = sort_for_journeys(frame)

    work = ordered[TRIP_COLUMNS].copy()
    work["departure"] = ordered["actual_departure_ts"]
    work["arrival"] = ordered["actual_departure_ts"] + pd.to_timedelta(
        ordered[TARGET], unit="s"
    )
    work["from_seq"] = ordered[ORDER_COLUMN]
    work["seq"] = ordered["stop_sequence"]

    grouped = work.groupby(TRIP_COLUMNS, sort=False)
    previous_arrival = grouped["arrival"].shift(1)
    previous_seq = grouped["seq"].shift(1)

    new_trip = grouped.cumcount() == 0
    time_break = (work["departure"] != previous_arrival) & previous_arrival.notna()
    seq_break = (work["from_seq"] != previous_seq) & previous_seq.notna()

    blocks = (new_trip | time_break | seq_break).cumsum()
    return blocks.reindex(frame.index)


def block_diagnostics(frame: pd.DataFrame, blocks: pd.Series | None = None) -> dict:
    """How far the quality filter fragmented trips. Report it, do not assume it."""
    ordered = sort_for_journeys(frame)
    labels = (contiguous_blocks(frame) if blocks is None else blocks).loc[ordered.index]

    grouped = ordered.groupby(TRIP_COLUMNS, sort=False)
    arrival = ordered["actual_departure_ts"] + pd.to_timedelta(
        ordered[TARGET], unit="s"
    )
    previous_arrival = arrival.groupby(
        [ordered[c] for c in TRIP_COLUMNS], sort=False
    ).shift(1)
    joined = previous_arrival.notna()
    held = ordered.loc[joined, "actual_departure_ts"] == previous_arrival[joined]

    sizes = labels.value_counts()
    return {
        "rows": int(len(ordered)),
        "trips": int(grouped.ngroups),
        "blocks": int(len(sizes)),
        "contiguous_pair_pct": float(100 * held.mean()) if joined.any() else 100.0,
        "broken_pairs": int((~held).sum()) if joined.any() else 0,
        "median_block_len": float(sizes.median()),
    }


def journey_windows(
    frame: pd.DataFrame,
    prediction_column: str,
    lengths: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10, 12, 15, 17),
    blocks: pd.Series | None = None,
) -> pd.DataFrame:
    """Score journeys of each length, summing predictions within blocks.

    Returns one row per length: how many journeys of that length exist, their median
    actual duration, and the error of the summed prediction against the summed actual.

    Implemented as differences of cumulative sums rather than by materialising every
    (origin, destination) pair. The pair table would be ~3M rows and carries no
    information the segment rows lack — it is the same numbers, re-expressed.

    `residual = actual - predicted`, matching `features.baselines._score`, so a positive
    bias means the model runs short.
    """
    ordered = sort_for_journeys(frame)
    labels = (contiguous_blocks(frame) if blocks is None else blocks).loc[ordered.index]

    residual = ordered[TARGET] - ordered[prediction_column]
    grouped_res = residual.groupby(labels, sort=False)
    grouped_act = ordered[TARGET].groupby(labels, sort=False)

    cumulative_residual = grouped_res.cumsum()
    cumulative_actual = grouped_act.cumsum()
    position = ordered.groupby(labels, sort=False).cumcount()

    rows = []
    for n in lengths:
        # At position n-1 the window starts at the block's first row, so there is
        # nothing to subtract; `where` supplies the 0 that `shift` cannot.
        prior_residual = (
            cumulative_residual.groupby(labels, sort=False)
            .shift(n)
            .where(position >= n, 0.0)
        )
        prior_actual = (
            cumulative_actual.groupby(labels, sort=False)
            .shift(n)
            .where(position >= n, 0.0)
        )
        usable = position >= (n - 1)
        if not usable.any():
            continue

        window_residual = (cumulative_residual - prior_residual)[usable]
        window_actual = (cumulative_actual - prior_actual)[usable]
        median_actual = float(window_actual.median())

        rows.append(
            {
                "segments": n,
                "journeys": int(usable.sum()),
                "median_duration_sec": median_actual,
                "mae_sec": float(window_residual.abs().mean()),
                "rmse_sec": float(np.sqrt((window_residual**2).mean())),
                "bias_sec": float(window_residual.mean()),
                "sd_sec": float(window_residual.std()),
                "mae_pct_of_duration": (
                    float(100 * window_residual.abs().mean() / median_actual)
                    if median_actual
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)
