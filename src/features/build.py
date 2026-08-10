"""Orchestration: segments in, model-ready feature table out.

    python -m src.features.build --start 2026-08-07 --end 2026-08-09

Also the home of the **single feature entry point** both paths call:

    compute_features(segments, history=None)

Batch passes no history and the recent-conditions features are derived from the segment
frame itself. Serving passes a `RecentConditionsLookup` loaded from S3. Writing the two
as one function is the whole defence against train/serve skew — a Spark implementation
and a separate Python reimplementation drift silently and you find out from production
predictions, not from a test.

## Why a lookup exists at all

Every feature except one is computable from a single request: the calendar, the
timetable, the segment's identity. `recent_deviation` is not — it needs the last N
*completed* traversals of this segment within the past hour, and a serverless endpoint
handling one request has no such history.

It is also the only feature carrying real signal (+0.244 against the residual, where
upstream delay manages -0.010), so dropping it at serving would leave the model a
segment-by-hour lookup with calendar trimmings.

So the batch job publishes a small table — one row per segment, the recent-conditions
values as of its latest completed traversal — and serving reads that. It is a few
hundred rows and refreshes whenever the pipeline runs.

**The staleness rule has to match on both sides.** Batch nulls these features when the
last traversal is older than `rolling_max_age_sec`; serving applies the same rule to
the lookup's `completed_at`. Without that, a lookup published three hours ago would feed
the endpoint a confident number that batch would have refused to produce — which is
exactly the skew the parity test exists to catch, and it would not show up in any
offline metric.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import SEGMENT_KEY, TARGET, FeatureConfig
from .historical import build_historical_features, rolling_segment_conditions
from .safe import build_safe_features
from .split import temporal_split

logger = logging.getLogger("features.build")

# What serving actually reads out of the lookup and hands the model.
LOOKUP_FEATURE_COLUMNS = (
    "recent_duration_median",
    "recent_duration_mean",
    "recent_delay_mean",
    "recent_traversals",
    "recent_deviation",
)

# The published table: the features above, plus the segment they belong to and the
# completion time they were computed as of. `completed_at` is metadata, not a feature —
# it exists so serving can apply the same staleness cutoff batch does.
LOOKUP_COLUMNS = ("from_stop_id", "to_stop_id", "completed_at", *LOOKUP_FEATURE_COLUMNS)

# Output layout under the configured root. The feature table lives in its own directory
# so the sidecars can sit beside it without breaking a dataset read.
TABLE_DIRNAME = "table"
LOOKUP_FILENAME = "recent_conditions_lookup.parquet"


@dataclass(frozen=True, slots=True)
class RecentConditionsLookup:
    """Per-segment recent conditions, as of the last completed traversal.

    Small by construction — one row per segment, a few hundred rows — so it fits in a
    single Parquet object and can be loaded into endpoint memory on cold start.
    """

    table: pd.DataFrame
    published_at: datetime

    def for_segment(
        self, from_stop_id: str, to_stop_id: str, now: datetime, config: FeatureConfig
    ) -> dict[str, float | None]:
        """Recent conditions for one segment, or nulls if the entry is stale.

        The staleness check is not an optimisation — it is what keeps serving
        numerically identical to batch. Batch discards a prior traversal older than
        `rolling_max_age_sec`; if serving did not, the endpoint would emit a feature
        value the training data never contained.
        """
        match = self.table[
            (self.table["from_stop_id"] == from_stop_id)
            & (self.table["to_stop_id"] == to_stop_id)
        ]
        empty: dict[str, float | None] = dict.fromkeys(LOOKUP_FEATURE_COLUMNS)
        if match.empty:
            return empty

        row = match.iloc[0]
        age = (now - pd.Timestamp(row["completed_at"]).to_pydatetime()).total_seconds()
        if age > config.rolling_max_age_sec:
            return empty
        return {
            column: (None if pd.isna(row[column]) else float(row[column]))
            for column in empty
        }


def build_recent_conditions_lookup(
    segments: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """What a request arriving now would get, per segment.

    Note what this is *not*: the recent-conditions values already attached to the last
    feature row of each segment. Those were computed as of that row's own departure,
    from completions strictly before it — so by the time it finished, they are one
    traversal out of date. Publishing them would hand serving a value batch would never
    produce for a fresh query, which is precisely the skew this table exists to avoid.

    Instead a **probe row** is appended per segment, departing one second after that
    segment's last completion, and the ordinary batch function is run over the result.
    The probe carries no `actual_arrival_ts`, so it is dropped from the completions
    timeline and cannot contribute to its own answer; the value it resolves to is by
    construction the one `merge_asof` gives any query after that completion. Reusing
    the batch function rather than reimplementing the aggregation is what makes the
    parity test a real check instead of a comparison of one function against itself.
    """
    settings = config or FeatureConfig()
    frame = segments.copy()
    frame["actual_departure_ts"] = pd.to_datetime(
        frame["actual_departure_ts"], utc=True
    )
    frame["actual_arrival_ts"] = pd.to_datetime(frame["actual_arrival_ts"], utc=True)

    last_completion = (
        frame.dropna(subset=["actual_arrival_ts"])
        .groupby(list(SEGMENT_KEY), observed=True, as_index=False)["actual_arrival_ts"]
        .max()
        .rename(columns={"actual_arrival_ts": "completed_at"})
    )
    probes = last_completion.assign(
        actual_departure_ts=last_completion["completed_at"] + pd.Timedelta(seconds=1),
        actual_arrival_ts=pd.NaT,
        actual_duration_sec=float("nan"),
        delay_sec=float("nan"),
        scheduled_duration_sec=float("nan"),
    )

    combined = pd.concat(
        [frame, probes.drop(columns=["completed_at"])], ignore_index=True
    )
    # A scalar NaT broadcast into `assign` lands as object dtype, and concatenating that
    # with a real datetime column poisons the whole column — `merge_asof` then refuses
    # the join outright rather than silently misbehaving, which is the good case.
    combined["actual_arrival_ts"] = pd.to_datetime(
        combined["actual_arrival_ts"], utc=True
    )
    combined["actual_departure_ts"] = pd.to_datetime(
        combined["actual_departure_ts"], utc=True
    )
    conditions = rolling_segment_conditions(combined, settings)
    published = conditions.iloc[len(frame) :].reset_index(drop=True)

    out = pd.concat([last_completion.reset_index(drop=True), published], axis=1)
    return out.loc[:, list(LOOKUP_COLUMNS)]


def compute_features(
    segments: pd.DataFrame,
    config: FeatureConfig | None = None,
    lookup: RecentConditionsLookup | None = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    """The one feature definition, called by batch and by serving.

    `lookup` is the serving path: with it, recent-conditions features come from the
    published table instead of being derived from the frame. Without it they are
    derived — which requires the frame to contain the segment's history, true in batch
    and never true at request time.
    """
    settings = config or FeatureConfig()
    safe = build_safe_features(segments, settings)

    if lookup is None:
        historical = build_historical_features(segments, settings)
    else:
        moment = now or datetime.now(UTC)
        rows = [
            lookup.for_segment(f, t, moment, settings)
            for f, t in zip(
                segments["from_stop_id"], segments["to_stop_id"], strict=True
            )
        ]
        historical = pd.DataFrame(rows, index=segments.index)
        # Serving cannot know a trip's upstream delay or the headway behind it from a
        # single request either. Emitted as nulls so the column set matches batch
        # exactly — a model fed a column set it was not trained on fails obscurely.
        for column in (
            "upstream_delay_sec",
            "upstream_delay_last_sec",
            "segments_completed",
            "headway_sec",
            "recent_age_sec",
            "recent_vs_scheduled",
        ):
            if column not in historical:
                historical[column] = pd.NA

    return pd.concat([safe, historical], axis=1)


def feature_summary(features: pd.DataFrame, target: pd.Series) -> str:
    """Null rates, cardinality and spread — so a behaviour change is visible early.

    Run every time. A feature whose fill rate quietly drops from 98% to 40% because an
    upstream window changed is invisible in a training metric and obvious here.
    """
    lines = ["", "=" * 78, "FEATURE SUMMARY", "=" * 78, f"\nrows: {len(features):,}\n"]
    lines.append(f"  {'feature':<32} {'fill%':>7} {'unique':>8}  {'mean':>10}")
    lines.append(f"  {'-' * 32} {'-' * 7} {'-' * 8}  {'-' * 10}")
    for column in features.columns:
        series = features[column]
        fill = 100 * series.notna().mean()
        unique = series.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(series):
            mean = f"{series.mean():10.2f}"
        else:
            mean = " " * 10
        flag = "  <-- sparse" if fill < 50 else ""
        lines.append(f"  {column:<32} {fill:7.1f} {unique:8,}  {mean}{flag}")
    lines.append(
        f"\n  target {TARGET}: mean {target.mean():.1f}s median {target.median():.0f}s"
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="first service_date, YYYY-MM-DD")
    parser.add_argument("--end", help="last service_date, inclusive")
    parser.add_argument("--output", default=None, help="override the output path")
    parser.add_argument(
        "--no-lookup", action="store_true", help="skip publishing the lookup"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    from .io import load_segments

    config = FeatureConfig()
    segments = load_segments(args.start, args.end, config)
    if segments.empty:
        logger.error("no segments in range — nothing to build")
        return 1

    features = compute_features(segments, config)

    # Join keys and the target travel with the features. A feature table that cannot be
    # traced back to its source rows is not debuggable.
    #
    # Some of these are also features in their own right (`stop_sequence` is both a join
    # key and a schedule feature). Taking them from the feature frame rather than
    # emitting both copies keeps the table single-valued per name — a duplicate column
    # is rejected outright by Arrow, and would be worse if it were not.
    key_columns = [
        "service_date",
        "trip_id",
        "trip_run",
        "from_stop_id",
        "to_stop_id",
        "stop_sequence",
        "actual_departure_ts",
        "arrival_bracket_sec",
        "arrival_source",
    ]
    keys = segments[[c for c in key_columns if c not in features.columns]].reset_index(
        drop=True
    )
    out = pd.concat([keys, features.reset_index(drop=True)], axis=1)
    out[TARGET] = segments[TARGET].to_numpy()

    print(feature_summary(features, segments[TARGET]))

    # `temporal_split` already logs the report when it has warnings, so printing it here
    # too would double every caveat and train the reader to skim past them.
    _, _, report = temporal_split(out, config=config)
    if report.is_trustworthy:
        print("\n" + report.format() + "\n")

    # The partitioned dataset gets a directory of its own, with the sidecars beside it
    # rather than inside it. A stray .json under the dataset root makes
    # `read_parquet(root)` fail outright — Arrow tries to parse it as Parquet — so the
    # obvious way to load the output would be broken by its own metadata.
    output = Path(args.output or config.output_path)
    table_path = output / TABLE_DIRNAME
    table_path.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_to_dataset(
        table,
        table_path,
        partition_cols=["service_date"],
        existing_data_behavior="delete_matching",
    )
    logger.info("wrote %d rows to %s", len(out), table_path)

    (output / "split_metadata.json").write_text(
        json.dumps(report.as_metadata(), indent=2)
    )

    if not args.no_lookup:
        lookup = build_recent_conditions_lookup(segments, config)
        lookup_path = output / LOOKUP_FILENAME
        lookup.to_parquet(lookup_path, index=False)
        logger.info(
            "published recent-conditions lookup: %d segment(s) -> %s",
            len(lookup),
            lookup_path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
