"""Head-to-head: journey model vs the segment model summed over the same journeys.

    python -m src.journeys.compare

Both approaches are scored on **identical validation journey rows**, so the comparison
answers one question and not two. The segment model's predictions are summed over the
same windows the journey rows describe, reusing the window arithmetic that built the
journey table — so no discrepancy can creep in between how a journey was labelled and
how it was predicted.

Three predictions per journey:

- **journey model** — trained directly on `journey_duration_sec`, nothing summed
- **segment model, summed** — the existing per-segment model, predictions added up
- **baseline, summed** — segment x hour-of-day median fitted on train, added up

The last is the bar both must clear. It is unbiased and its errors behave close to
independent (n^0.558), which is exactly why it is hard to beat over long journeys.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from ..features.config import SEGMENT_KEY, TARGET
from ..models import plots
from ..models.artifacts import resolve_run
from ..models.journey import contiguous_blocks, sort_for_journeys
from ..models.train import build_matrix as build_segment_matrix
from .build import window_sums
from .config import JOURNEY_TARGET, JourneyConfig
from .train import by_length

logger = logging.getLogger("journeys.compare")

SEGMENT_MODEL = "data/models/segment_duration"
JOURNEY_MODEL = "data/models/journey_duration"


def segment_predictions(features: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    """Score every segment with the saved segment model, plus the fitted baseline."""
    import lightgbm as lgb

    from ..models.calibrate import BiasCalibration
    from ..models.encode import CategoricalEncoder

    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    encoder = CategoricalEncoder.load(model_dir / "encoder.json")
    columns = json.loads((model_dir / "feature_columns.json").read_text())

    frame = features.copy()
    frame["segment_prediction"] = booster.predict(
        build_segment_matrix(frame, encoder, columns)
    )

    calibration_path = model_dir / "calibration.json"
    if calibration_path.exists():
        calibration = BiasCalibration.load(calibration_path)
        frame["segment_prediction_cal"] = calibration.apply(frame["segment_prediction"])
    else:
        frame["segment_prediction_cal"] = frame["segment_prediction"]
    return frame


def sum_over_journeys(
    features: pd.DataFrame, lengths: tuple[int, ...], columns: list[str]
) -> pd.DataFrame:
    """Sum per-segment predictions over the same windows the journey table used."""
    ordered = sort_for_journeys(features).reset_index(drop=True)
    blocks = contiguous_blocks(ordered).reset_index(drop=True)

    pieces = []
    for length in lengths:
        sums = window_sums(ordered, blocks, columns, length)
        usable = sums[columns[0]].notna()
        if not usable.any():
            continue
        rows = ordered.loc[
            usable, ["service_date", "trip_id", "trip_run", "from_stop_id"]
        ].copy()
        rows = rows.rename(columns={"from_stop_id": "origin_stop_id"})
        rows["n_segments"] = length
        for column in columns:
            rows[f"summed_{column}"] = sums.loc[usable, column]
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True)


def fitted_segment_hour_median(
    train_frame: pd.DataFrame, target_frame: pd.DataFrame
) -> pd.Series:
    """Out-of-sample segment x hour median — medians from train, applied to target."""
    keys = [*SEGMENT_KEY, "local_hour"]
    by_segment_hour = train_frame.groupby(keys, observed=True)[TARGET].median()
    by_segment = train_frame.groupby(list(SEGMENT_KEY), observed=True)[TARGET].median()

    prediction = pd.Series(
        target_frame.set_index(keys).index.map(by_segment_hour),
        index=target_frame.index,
        dtype="float64",
    )
    fallback = pd.Series(
        target_frame.set_index(list(SEGMENT_KEY)).index.map(by_segment),
        index=target_frame.index,
        dtype="float64",
    )
    return prediction.fillna(fallback).fillna(float(train_frame[TARGET].median()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-model", default=SEGMENT_MODEL)
    parser.add_argument("--journey-model", default=JOURNEY_MODEL)
    parser.add_argument("--segment-run", default=None, help="run id, else latest")
    parser.add_argument("--journey-run", default=None, help="run id, else latest")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    config = JourneyConfig()

    journey_run = resolve_run(args.journey_model, args.journey_run)
    segment_run = resolve_run(args.segment_model, args.segment_run)
    logger.info("journey run %s | segment run %s", journey_run.name, segment_run.name)
    validation = pd.read_parquet(journey_run / "validation_predictions.parquet")
    validation["service_date"] = validation["service_date"].astype(str)
    logger.info("journey validation rows: %d", len(validation))

    features = pd.read_parquet(config.features_path)
    features["service_date"] = features["service_date"].astype(str)

    # The segment model's own train side, for an honest out-of-sample baseline. Journey
    # validation blocks are excluded so the baseline never sees what it is graded on.
    validation_blocks = set(validation["block"].unique())
    ordered = sort_for_journeys(features).reset_index(drop=True)
    ordered["block"] = contiguous_blocks(ordered).to_numpy()
    train_side = ordered[~ordered["block"].isin(validation_blocks)]
    logger.info("segment rows outside journey validation: %d", len(train_side))

    scored = segment_predictions(ordered, segment_run)
    scored["baseline_prediction"] = fitted_segment_hour_median(train_side, scored)

    summed = sum_over_journeys(
        scored,
        config.lengths,
        ["segment_prediction", "segment_prediction_cal", "baseline_prediction"],
    )

    merged = validation.merge(
        summed,
        on=["service_date", "trip_id", "trip_run", "origin_stop_id", "n_segments"],
        how="inner",
    )
    logger.info("matched %d of %d validation journeys", len(merged), len(validation))

    table = by_length(
        merged,
        {
            "journey_model": "prediction",
            "segment_summed": "summed_segment_prediction",
            "segment_summed_cal": "summed_segment_prediction_cal",
            "baseline_summed": "summed_baseline_prediction",
        },
    )
    print(format_comparison(table))

    # The multi-series figure belongs here, not in training — this is the only place the
    # competing predictions exist on the same rows.
    plots.error_by_group(
        merged,
        "n_segments",
        {
            "journey model": "prediction",
            "segment model, summed": "summed_segment_prediction",
            "segment model, calibrated": "summed_segment_prediction_cal",
            "segment x hour median, summed": "summed_baseline_prediction",
        },
        journey_run / "plots",
        "comparison_by_length",
        "Journey model vs summing per-segment predictions",
        "journey length (segments)",
        JOURNEY_TARGET,
    )
    (journey_run / "comparison.json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=1, default=str)
    )
    return 0


def format_comparison(table: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 96,
        "HEAD TO HEAD — same validation journeys, MAE in seconds",
        "=" * 96,
    ]
    lines.append(
        f"{'seg':>4} {'n':>8} {'median':>8} {'journey':>9}"
        f" {'seg sum':>9} {'segsum cal':>10} {'baseline':>9} {'best':>16}"
    )
    lines.append("-" * 96)
    labels = {
        "mae_journey_model": "journey",
        "mae_segment_summed": "segment sum",
        "mae_segment_summed_cal": "segsum cal",
        "mae_baseline_summed": "baseline",
    }
    for _, r in table.iterrows():
        best = min(labels, key=lambda c: r[c])
        lines.append(
            f"{r['segments']:>4.0f} {r['journeys']:>8.0f}"
            f" {r['median_duration_sec']:>7.0f}s"
            f" {r['mae_journey_model']:>8.1f}s"
            f" {r['mae_segment_summed']:>8.1f}s {r['mae_segment_summed_cal']:>9.1f}s"
            f" {r['mae_baseline_summed']:>8.1f}s {labels[best]:>16}"
        )
    lines.append("=" * 96)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
