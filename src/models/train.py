"""Stage 3: fit the per-segment duration model.

    python -m src.models.train

## Why the model stays per-segment

Measured in `notebooks/07_journey_framing.ipynb`, on 12 service days:

- Summed per-segment errors scale as `n^0.58` — near the 0.5 of independent errors, so
  summing predictions across a journey does **not** compound them.
- Within-trip residual autocorrelation is `-0.106` at lag 1 and ~0 beyond. There is no
  trip-level state to model. The persistence that matters is per segment *across
  trains*, which `recent_deviation` already captures (+0.255 against the residual).
- Journey error is spread-dominated; bias is under 4% of MAE out to 12 segments.

So a journey is predicted by summing its segments, and journey rows are never
materialised — they carry no information the segment rows lack, being a linear
re-expression of the same numbers.

## What is deliberately kept out of the feature matrix

`arrival_bracket_sec` and `arrival_source` describe how the *label* was measured. They
are only knowable after the arrival they describe, so feeding them in as inputs leaks.
`docs/polling-cadence.md` recommends the bracket as a **sample weight** across a cadence
change — legitimate, and a different thing from a feature.

Nulls are passed through untouched. `recent_*` is null when nothing completed a
segment inside `rolling_max_age_sec`; LightGBM splits on missing natively, and a mean
would both destroy that meaning and break parity with the serving staleness rule.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from ..features.config import LABEL_SIDE_COLUMNS, TARGET, FeatureConfig
from ..features.split import temporal_split
from . import plots
from .artifacts import mark_latest, new_run_dir, write_manifest
from .calibrate import BiasCalibration
from .encode import CategoricalEncoder

logger = logging.getLogger("models.train")

DEFAULT_FEATURES_PATH = "data/processed/features/table"
DEFAULT_OUTPUT_PATH = "data/models/segment_duration"

# Join keys and provenance. Identity that the model *should* see arrives as
# `segment_id`, `from_station` and `to_station`, which are encoded categoricals; the raw
# stop ids would be the same information at higher cardinality.
KEY_COLUMNS: tuple[str, ...] = (
    "service_date",
    "trip_id",
    "trip_run",
    "from_stop_id",
    "to_stop_id",
    "actual_departure_ts",
)

# Measurement metadata: known only once the arrival has been observed.
MEASUREMENT_COLUMNS: tuple[str, ...] = ("arrival_bracket_sec", "arrival_source")

# MAE is the objective because MAE is the reported metric. Training on squared error and
# reporting absolute error optimises a different thing than the one being judged, and on
# a right-skewed target the gap is not small.
PARAMS: dict = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 17,
    "verbosity": -1,
}
# The first run stopped at best_iteration=1988 against a cap of 2000 — it never
# early-stopped, it ran out of ceiling while still improving. Early stopping is the real
# guard; this is only a backstop. If a run lands within ~100 of this, raise it again
# rather than assuming the model converged.
NUM_BOOST_ROUND = 6000
EARLY_STOPPING_ROUNDS = 100


def feature_columns(
    frame: pd.DataFrame, config: FeatureConfig | None = None
) -> list[str]:
    """Every column the model is allowed to see, in a stable order.

    Order matters: LightGBM records column order at fit time and serving must rebuild
    the matrix identically, so it is persisted with the model rather than re-derived.
    """
    del config  # accepted for symmetry; the exclusion set is not tunable
    excluded = set(KEY_COLUMNS) | set(MEASUREMENT_COLUMNS) | set(LABEL_SIDE_COLUMNS)
    return [c for c in frame.columns if c not in excluded]


def categorical_columns(
    frame: pd.DataFrame, columns: list[str], config: FeatureConfig | None = None
) -> list[str]:
    """Declared categoricals, plus any feature column that is simply not numeric.

    The declared list in `FeatureConfig` is the feature layer's intent. It omits
    `fare_period`, which is a string — and a string column handed to LightGBM raises
    rather than training. Detecting the rest from dtype means the failure mode is a log
    line about an undeclared categorical, not a crash mid-fit or a silently dropped
    feature.
    """
    settings = config or FeatureConfig()
    declared = [c for c in settings.categorical_columns if c in columns]
    undeclared = [
        c for c in columns if c not in declared and frame[c].dtype.kind not in "ifb"
    ]
    if undeclared:
        logger.info("encoding undeclared non-numeric feature(s): %s", undeclared)
    return declared + undeclared


def build_matrix(
    frame: pd.DataFrame, encoder: CategoricalEncoder, columns: list[str]
) -> pd.DataFrame:
    """Encode categoricals and select the model's columns, in the persisted order."""
    encoded = encoder.transform(frame)
    missing = [c for c in columns if c not in encoded.columns]
    if missing:
        raise ValueError(f"feature columns absent from the frame: {missing}")
    return encoded[columns]


def fit(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: FeatureConfig | None = None,
):
    """Fit on train, early-stop on validation.

    Returns (booster, encoder, columns, history) — history is the per-iteration loss for
    both splits, which `plots.learning_curve` needs to show whether the round cap bound.
    """
    import lightgbm as lgb

    settings = config or FeatureConfig()

    # Fit the encoder on training rows ONLY. Fitting on the full frame would let a
    # category that appears only in validation influence the codes.
    columns = feature_columns(train_frame, settings)
    categoricals = categorical_columns(train_frame, columns, settings)
    encoder = CategoricalEncoder.fit(train_frame, settings, extra_columns=categoricals)

    history: dict = {}
    x_train = build_matrix(train_frame, encoder, columns)
    x_validation = build_matrix(validation_frame, encoder, columns)

    train_set = lgb.Dataset(
        x_train, train_frame[TARGET], categorical_feature=categoricals
    )
    validation_set = lgb.Dataset(
        x_validation,
        validation_frame[TARGET],
        reference=train_set,
        categorical_feature=categoricals,
    )

    logger.info(
        "fitting on %d rows x %d features (%d categorical)",
        len(x_train),
        len(columns),
        len(categoricals),
    )
    booster = lgb.train(
        PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        # Train is scored as well as validation purely so the learning curve has both
        # lines. Without the train curve a rising gap — overfitting — is invisible.
        valid_sets=[train_set, validation_set],
        valid_names=["train", "validation"],
        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING_ROUNDS, verbose=False, first_metric_only=True
            ),
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(history),
        ],
    )
    logger.info("stopped at iteration %d", booster.best_iteration)
    return booster, encoder, columns, history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    from .evaluate import format_report, report

    frame = pd.read_parquet(args.features)
    frame["service_date"] = frame["service_date"].astype(str)

    train_mask, validation_mask, split_report = temporal_split(frame)
    train_frame = frame[train_mask].copy()
    validation_frame = frame[validation_mask].copy()

    if validation_frame.empty or train_frame.empty:
        logger.error(
            "split produced an empty side (train=%d, validation=%d) — nothing to fit",
            len(train_frame),
            len(validation_frame),
        )
        return 1

    # Loud, and before the numbers, so a provisional score is never read as a result.
    if not split_report.is_trustworthy:
        logger.warning(
            "THE SPLIT IS NOT TRUSTWORTHY — every metric below is provisional:\n%s",
            "\n".join(f"  - {w}" for w in split_report.warnings),
        )

    booster, encoder, columns, history = fit(train_frame, validation_frame)

    for name, part in (("train", train_frame), ("validation", validation_frame)):
        part["prediction"] = booster.predict(
            build_matrix(part, encoder, columns), num_iteration=booster.best_iteration
        )
        logger.info(
            "%s MAE %.2fs", name, (part[TARGET] - part["prediction"]).abs().mean()
        )

    # Fitted on TRAIN residuals only — fitting on validation would use the answers to
    # correct the predictions being graded. See calibrate.py for why the L1 objective
    # leaves a bias that is harmless per segment and fatal once summed.
    calibration = BiasCalibration.fit(train_frame[TARGET], train_frame["prediction"])
    for part in (train_frame, validation_frame):
        part["prediction_calibrated"] = calibration.apply(part["prediction"])

    metrics = report(
        train_frame, validation_frame, split_report, encoder, booster, calibration
    )

    # Every run gets its own immutable directory; `latest` moves, history stays.
    root = Path(args.output)
    run = new_run_dir(root)

    booster.save_model(str(run / "model.txt"), num_iteration=booster.best_iteration)
    encoder.save(run / "encoder.json")
    calibration.save(run / "calibration.json")
    (run / "feature_columns.json").write_text(json.dumps(columns, indent=1))
    (run / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    validation_frame.to_parquet(run / "validation_predictions.parquet", index=False)

    plots.write_all(
        run / "plots",
        history=history,
        best_iteration=booster.best_iteration,
        importance=metrics.get("feature_importance", {}),
        validation=validation_frame,
        target=TARGET,
    )

    segment = {r["name"]: r["mae"] for r in metrics["segment_level"]}
    write_manifest(
        run,
        model_name="segment_duration",
        target=TARGET,
        trustworthy=split_report.is_trustworthy,
        training_data={
            "features_path": args.features,
            "service_dates": sorted(frame["service_date"].unique().tolist()),
            "train_rows": len(train_frame),
            "validation_rows": len(validation_frame),
            "split": split_report.as_metadata(),
        },
        params=PARAMS,
        best_iteration=booster.best_iteration,
        feature_columns=columns,
        categorical_columns=categorical_columns(train_frame, columns),
        headline_metrics={
            "segment_mae": segment,
            "calibration_offset_sec": metrics.get("calibration_offset_sec"),
            "journey_level": metrics["journey_level"],
        },
        notes="Per-segment model. Journey predictions are the sum of its segments.",
    )
    mark_latest(root, run)

    print(format_report(metrics))
    logger.info("run %s complete — latest now points here", run.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
