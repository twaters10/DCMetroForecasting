"""Scoring, at both the level the model works and the level a rider asks about.

    python -m src.models.evaluate

Per-segment MAE is reported because it is what the model optimises, but it is **not**
the headline. The 60s grid puts ~±24s of measurement error on a 120s median segment,
and the segment x hour baseline already sits at 24.9s — leaving ~0.8pp of room between
the baseline and the noise floor. A model cannot demonstrate much there, and a
per-segment number invites a claim the data cannot support.

Journey-level MAE is the headline. A journey label is a difference of two timestamps
whatever its length, so the measurement floor falls from 20.0% of duration at one
segment to 1.1% at seventeen, while genuine difficulty remains: 4.8pp of headroom.

**Baselines are fitted on train and applied to validation.** Scoring
`segment_hour_median_baseline` directly on validation rows computes its medians from the
rows it is scoring, which flatters it — the model would be competing against a baseline
that has seen the answers. The in-sample figure is still reported, and flagged, because
it is what `python -m src.features.baselines` prints and the two should reconcile.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..features.baselines import (
    BaselineResult,
    _score,
    schedule_baseline,
    segment_hour_median_baseline,
)
from ..features.config import SEGMENT_KEY, TARGET
from ..features.split import SplitReport
from .journey import block_diagnostics, journey_windows

logger = logging.getLogger("models.evaluate")

# Mirrors journeys.config.DEFAULT_LENGTHS so the summed segment model and the journey
# model are scored on identical horizons. Imported rather than restated would be better,
# but models/ must not depend on journeys/ — the segment pipeline predates it and stays
# independently runnable.
JOURNEY_LENGTHS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10, 12, 15, 17, 20, 24, 28, 32)

# Written onto the frames by train.py once a BiasCalibration has been fitted.
CALIBRATED_COLUMN = "prediction_calibrated"

# The label spans two grid-snapped endpoints however many segments it covers, and the
# lag-1 residual autocorrelation is negative (-0.106), so adjacent errors partly cancel
# rather than accumulate. Measured, not assumed — notebooks/07_journey_framing.ipynb.
MEASUREMENT_FLOOR_SEC = 24.0


def fitted_segment_hour_median(
    train_frame: pd.DataFrame, target_frame: pd.DataFrame
) -> pd.Series:
    """Out-of-sample segment x hour median: medians from train, applied to target.

    Falls back to the segment median, then the global median, where an hour or a segment
    was never seen in training. Without the fallbacks this would return NaN for exactly
    the cold-start rows that matter, and `_score` would quietly drop them from coverage.
    """
    keys = [*SEGMENT_KEY, "local_hour"]
    by_segment_hour = train_frame.groupby(keys, observed=True)[TARGET].median()
    by_segment = train_frame.groupby(list(SEGMENT_KEY), observed=True)[TARGET].median()

    prediction = target_frame.set_index(keys).index.map(by_segment_hour)
    prediction = pd.Series(prediction, index=target_frame.index, dtype="float64")

    fallback = pd.Series(
        target_frame.set_index(list(SEGMENT_KEY)).index.map(by_segment),
        index=target_frame.index,
        dtype="float64",
    )
    return prediction.fillna(fallback).fillna(float(train_frame[TARGET].median()))


def segment_level(
    train_frame: pd.DataFrame,
    frame: pd.DataFrame,
    prediction_column: str = "prediction",
) -> list[BaselineResult]:
    """Model against the baselines, all on the same rows.

    Both raw and calibrated predictions are scored when the calibrated column exists.
    Calibration is expected to make this number slightly WORSE — it moves the prediction
    off the conditional median, which is exactly what MAE rewards — while removing the
    bias that ruins summed journeys. Showing both keeps that trade visible instead of
    letting it look like a free win.
    """
    results = [
        _score("model", frame[TARGET], frame[prediction_column], in_sample=False),
        _score(
            "segment x hour median (fitted on train)",
            frame[TARGET],
            fitted_segment_hour_median(train_frame, frame),
            in_sample=False,
        ),
        schedule_baseline(frame),
    ]
    # Reported for reconciliation with the baselines CLI, and flagged as optimistic.
    results.append(segment_hour_median_baseline(frame))

    if CALIBRATED_COLUMN in frame.columns:
        results.insert(
            1,
            _score(
                "model (bias-calibrated)",
                frame[TARGET],
                frame[CALIBRATED_COLUMN],
                in_sample=False,
            ),
        )
    return results


def journey_level(
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    prediction_column: str = "prediction",
    lengths: tuple[int, ...] = JOURNEY_LENGTHS,
) -> pd.DataFrame:
    """Journey error for the model and the fitted baseline, side by side.

    Both are summed the same way, over the same contiguous blocks, so the comparison is
    like for like.
    """
    scored = frame.copy()
    scored["baseline_prediction"] = fitted_segment_hour_median(train_frame, scored)

    model = journey_windows(scored, prediction_column, lengths=lengths)
    baseline = journey_windows(scored, "baseline_prediction", lengths=lengths)

    merged = model.merge(
        baseline[["segments", "mae_sec", "mae_pct_of_duration"]],
        on="segments",
        suffixes=("", "_baseline"),
    )

    # The calibrated model is the one that matters here: a per-segment bias accumulates
    # linearly with journey length, so this is the level where removing it pays.
    if CALIBRATED_COLUMN in scored.columns:
        calibrated = journey_windows(scored, CALIBRATED_COLUMN, lengths=lengths)
        merged = merged.merge(
            calibrated[["segments", "mae_sec", "bias_sec", "mae_pct_of_duration"]],
            on="segments",
            suffixes=("", "_calibrated"),
        )
        merged["calibrated_beats_baseline_by_pct"] = 100 * (
            1 - merged["mae_sec_calibrated"] / merged["mae_sec_baseline"]
        )
    merged["floor_pct"] = 100 * MEASUREMENT_FLOOR_SEC / merged["median_duration_sec"]
    merged["headroom_pct"] = merged["mae_pct_of_duration"] - merged["floor_pct"]
    merged["beats_baseline_by_pct"] = 100 * (
        1 - merged["mae_sec"] / merged["mae_sec_baseline"]
    )
    return merged


def report(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    split_report: SplitReport,
    encoder=None,
    booster=None,
    calibration=None,
    prediction_column: str = "prediction",
) -> dict:
    """Assemble every number worth persisting, with the trustworthiness flag."""
    segments = segment_level(train_frame, validation_frame, prediction_column)
    journeys = journey_level(validation_frame, train_frame, prediction_column)

    metrics: dict = {
        # Stamped first so a metrics file can never be read as unconditionally valid.
        "trustworthy": bool(split_report.is_trustworthy),
        "split_warnings": list(split_report.warnings),
        "split": split_report.as_metadata(),
        "rows": {"train": len(train_frame), "validation": len(validation_frame)},
        "segment_level": [asdict(r) for r in segments],
        "journey_level": journeys.to_dict(orient="records"),
        "blocks": block_diagnostics(validation_frame),
    }
    if calibration is not None:
        metrics["calibration_offset_sec"] = float(calibration.offset_sec)
    if encoder is not None:
        metrics["unseen_category_pct"] = encoder.unseen_rate(validation_frame)
    if booster is not None:
        metrics["best_iteration"] = int(booster.best_iteration)
        metrics["feature_importance"] = dict(
            sorted(
                zip(
                    booster.feature_name(),
                    (int(v) for v in booster.feature_importance("gain")),
                    strict=True,
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )
    return metrics


def format_report(metrics: dict) -> str:
    """The human-readable version. Caveats first, deliberately."""
    lines = ["", "=" * 78, "MODEL EVALUATION", "=" * 78]

    if not metrics["trustworthy"]:
        lines.append(
            "\n  !! THE SPLIT IS NOT TRUSTWORTHY — these numbers are provisional"
        )
        for warning in metrics["split_warnings"]:
            lines.append(f"     - {warning}")

    lines.append(
        f"\nrows: {metrics['rows']['train']:,} train / "
        f"{metrics['rows']['validation']:,} validation\n"
    )
    lines.append("PER SEGMENT — saturated by design, not the headline")
    for result in metrics["segment_level"]:
        flag = "  (in-sample, optimistic)" if result["in_sample"] else ""
        lines.append(
            f"  {result['name']:<42} MAE {result['mae']:6.2f}s"
            f"   RMSE {result['rmse']:6.2f}s{flag}"
        )

    calibrated = "mae_sec_calibrated" in (metrics["journey_level"] or [{}])[0]
    if "calibration_offset_sec" in metrics:
        lines.append(
            f"\nbias calibration: {metrics['calibration_offset_sec']:+.3f}s per segment"
        )

    lines.append("\nPER JOURNEY — where the headroom is")
    header = (
        f"  {'segments':>8} {'median':>8} {'raw':>9} {'calib':>9} {'baseline':>9}"
        f" {'bias':>9} {'calib vs base':>14}"
        if calibrated
        else f"  {'segments':>8} {'median':>8} {'model':>9} {'baseline':>9}"
        f" {'model %':>8} {'floor %':>8} {'vs base':>8}"
    )
    lines.append(header)
    lines.append(f"  {'-' * (len(header) - 2)}")
    for row in metrics["journey_level"]:
        if calibrated:
            lines.append(
                f"  {row['segments']:>8.0f} {row['median_duration_sec']:>7.0f}s"
                f" {row['mae_sec']:>8.1f}s {row['mae_sec_calibrated']:>8.1f}s"
                f" {row['mae_sec_baseline']:>8.1f}s"
                f" {row['bias_sec_calibrated']:>+8.1f}s"
                f" {row['calibrated_beats_baseline_by_pct']:>+13.1f}%"
            )
        else:
            lines.append(
                f"  {row['segments']:>8.0f} {row['median_duration_sec']:>7.0f}s"
                f" {row['mae_sec']:>8.1f}s {row['mae_sec_baseline']:>8.1f}s"
                f" {row['mae_pct_of_duration']:>7.1f}% {row['floor_pct']:>7.1f}%"
                f" {row['beats_baseline_by_pct']:>+7.1f}%"
            )

    blocks = metrics["blocks"]
    lines.append(
        f"\ncontiguity: {blocks['contiguous_pair_pct']:.2f}% of pairs, "
        f"{blocks['broken_pairs']:,} broken, {blocks['blocks']:,} blocks "
        f"from {blocks['trips']:,} trips"
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/segment_duration")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    path = Path(args.model) / "metrics.json"
    if not path.exists():
        logger.error("no metrics at %s — run `python -m src.models.train` first", path)
        return 1

    print(format_report(json.loads(path.read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
