"""Segments in, journey-level training table out.

    python -m src.journeys.build

## What a row is

One row per *(trip run, origin stop A, destination stop B)*, where the target is

    journey_duration_sec = arrival(B) - arrival(A) = sum of the segments between them

Because `actual_departure_ts` is the upstream **arrival**, consecutive segments
telescope exactly — verified at 100.00% of pairs, 0.0s discrepancy — so a journey
label needs only two timestamps regardless of distance. That is why this table is
worth building: measurement error stays at ~±24s whether the journey is 2 stops or
20, falling from 20% of a single segment to ~1% of a 40-minute trip.

## The rule every feature obeys

**Everything must be knowable when the train arrives at A.** That gives three groups and
one hard exclusion:

1. **Origin-segment features, taken as-is.** `historical.py` computes them as of that
   segment's own departure, which for the journey's *first* segment IS time A. So the
   origin row's `recent_deviation`, `upstream_delay_sec` and `headway_sec` are
   legitimate without modification.
2. **Calendar at A** — likewise the origin row's, unmodified.
3. **Timetable structure of the whole journey** — `n_segments`, summed
   `scheduled_duration_sec`, stops spanned, destination station. All published ahead of
   time, so all fair.

**Excluded: every downstream segment's observed conditions.** The `recent_*` values on
segments 2..n are computed as of moments that have not happened yet when the rider
is standing at A. Aggregating them would be the leak that makes a model validate
beautifully and fail in production — and it is the precise failure the summed-segment
approach suffers from, which this pipeline exists to avoid.

## Why this exists alongside the segment model

The segment model wins per segment (+14.2% over the median baseline) and loses per
journey (-9.9% at 17 segments), because its errors are positively correlated along a
trip and summing amplifies them: error scaling n^0.663 against the baseline's n^0.558.
Bias calibration narrowed that to n^0.602 but could not close it — a constant offset
cannot fix correlated error. Training directly on the journey target optimises the
quantity actually being evaluated.

This pipeline **never writes to the per-segment feature table**, so both approaches stay
trainable and comparable from the same source.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..features.config import TARGET
from ..models.journey import ORDER_COLUMN, contiguous_blocks, sort_for_journeys
from .config import JOURNEY_TARGET, JourneyConfig

logger = logging.getLogger("journeys.build")

# Carried from the ORIGIN segment unchanged. Each is either calendar (known long in
# advance) or computed by historical.py as of the origin's own departure, which is
# exactly time A for the journey's first segment.
ORIGIN_FEATURES: tuple[str, ...] = (
    "local_hour",
    "local_hour_frac",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "is_service_weekday",
    "fare_period",
    "route_id",
    "direction_id",
    "from_station",
    "trip_start_hour",
    "minutes_into_trip",
    "trip_progress",
    "stops_remaining",
    "upstream_delay_sec",
    "upstream_delay_last_sec",
    "segments_completed",
    "recent_duration_median",
    "recent_duration_mean",
    "recent_delay_mean",
    "recent_traversals",
    "recent_age_sec",
    "recent_deviation",
    "recent_vs_scheduled",
    "headway_sec",
)

# Summed across the journey's segments. Both come from the published timetable, so a
# rider at A could look them up; neither depends on an outcome.
SUMMED_FEATURES: tuple[str, ...] = ("scheduled_duration_sec", "stop_span")


def window_sums(
    frame: pd.DataFrame, blocks: pd.Series, columns: list[str], length: int
) -> pd.DataFrame | None:
    """Sum `columns` over the `length` segments starting at each row.

    Start-indexed on purpose: the window beginning at row `s` is the journey departing
    from row `s`'s origin, so every origin-side feature is just that row, no shifting.

    A window is valid only if all `length` rows lie in one block. Shifting back by
    `length - 1` within the block returns null past its end — the invalid case.
    """
    grouped = frame.groupby(blocks, sort=False)
    out = {}
    for column in columns:
        cumulative = grouped[column].cumsum()
        end = cumulative.groupby(blocks, sort=False).shift(-(length - 1))
        # total over [s, s+L-1] == cum[s+L-1] - cum[s] + value[s]
        out[column] = end - cumulative + frame[column]
    return pd.DataFrame(out, index=frame.index)


def build_journeys(
    segments: pd.DataFrame, config: JourneyConfig | None = None
) -> pd.DataFrame:
    """Expand the per-segment table into one row per (origin, destination) journey."""
    settings = config or JourneyConfig()

    ordered = sort_for_journeys(segments).reset_index(drop=True)
    blocks = contiguous_blocks(ordered).reset_index(drop=True)
    ordered["block"] = blocks.to_numpy()

    grouped = ordered.groupby("block", sort=False)
    sum_columns = [TARGET, *SUMMED_FEATURES]

    pieces: list[pd.DataFrame] = []
    for length in settings.lengths:
        sums = window_sums(ordered, blocks, sum_columns, length)
        # Destination is the LAST segment's downstream stop, so it shifts too.
        destination = grouped["to_stop_id"].shift(-(length - 1))
        destination_stop_sequence = grouped["stop_sequence"].shift(-(length - 1))

        usable = sums[TARGET].notna() & destination.notna()
        if not usable.any():
            logger.info("length %d: no complete windows", length)
            continue

        rows = ordered.loc[
            usable, ["service_date", "trip_id", "trip_run", "block"]
        ].copy()
        rows["origin_stop_id"] = ordered.loc[usable, "from_stop_id"]
        rows["destination_stop_id"] = destination[usable]
        # The origin SEGMENT's downstream stop, which is not the journey's destination
        # once the journey is longer than one segment. Carried because the `recent_*`
        # features above are that segment's, and both serving and monitoring key the
        # live conditions table on (origin_stop_id, first_leg_to) — see
        # serving/inference.py. Without it, monitoring can only match single-segment
        # journeys and reports the model's strongest feature as null on the rest.
        rows["first_leg_to"] = ordered.loc[usable, "to_stop_id"]
        rows["origin_departure_ts"] = ordered.loc[usable, "actual_departure_ts"]
        rows["origin_stop_sequence"] = ordered.loc[usable, ORDER_COLUMN]
        rows["destination_stop_sequence"] = destination_stop_sequence[usable]
        rows["n_segments"] = np.int16(length)

        for column in ORIGIN_FEATURES:
            if column in ordered.columns:
                rows[column] = ordered.loc[usable, column]

        rows["scheduled_total_sec"] = sums.loc[usable, "scheduled_duration_sec"]
        rows["stops_spanned"] = sums.loc[usable, "stop_span"]
        # `to_station` is the DESTINATION's station, not the origin segment's. The
        # rider knows where they are going, so this is fair — and it is the strongest
        # identity feature a journey row has.
        rows["to_station"] = _station_of(destination[usable])
        rows[JOURNEY_TARGET] = sums.loc[usable, TARGET]

        pieces.append(rows)
        logger.info("length %2d: %8d journeys", length, len(rows))

    journeys = pd.concat(pieces, ignore_index=True)
    logger.info("built %d journey row(s) over %d length(s)", len(journeys), len(pieces))
    return journeys


def _station_of(stop_ids: pd.Series) -> pd.Series:
    """`PF_A08_C` -> `A08`, mirroring safe._STATION_PATTERN.

    Several platform ids map to one physical station, so station-level identity has to
    work on the extracted code rather than the raw stop_id.
    """
    return stop_ids.astype(str).str.extract(r"^PF_([A-Z]\d{2})", expand=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=None, help="per-segment feature table")
    parser.add_argument("--output", default=None)
    parser.add_argument("--start", help="first service_date, inclusive")
    parser.add_argument("--end", help="last service_date, inclusive")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    config = JourneyConfig()

    segments = pd.read_parquet(args.features or config.features_path)
    segments["service_date"] = segments["service_date"].astype(str)
    if args.start:
        segments = segments[segments["service_date"] >= args.start]
    if args.end:
        segments = segments[segments["service_date"] <= args.end]
    logger.info(
        "read %d segment(s) over %d service date(s)",
        len(segments),
        segments["service_date"].nunique(),
    )

    journeys = build_journeys(segments, config)

    output = Path(args.output or config.output_path)
    output.mkdir(parents=True, exist_ok=True)
    pq.write_to_dataset(
        pa.Table.from_pandas(journeys, preserve_index=False),
        output / "table",
        partition_cols=["service_date"],
        existing_data_behavior="delete_matching",
    )
    logger.info("wrote %d rows to %s", len(journeys), output / "table")

    print(summarise(journeys))
    return 0


def summarise(journeys: pd.DataFrame) -> str:
    """Rows and target spread per journey length — the shape of the training set."""
    by_length = journeys.groupby("n_segments").agg(
        rows=(JOURNEY_TARGET, "size"),
        median_sec=(JOURNEY_TARGET, "median"),
        p05=(JOURNEY_TARGET, lambda s: s.quantile(0.05)),
        p95=(JOURNEY_TARGET, lambda s: s.quantile(0.95)),
    )
    return "\n".join(
        [
            "",
            "=" * 70,
            "JOURNEY TABLE",
            "=" * 70,
            by_length.round(0).to_string(),
            "=" * 70,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
