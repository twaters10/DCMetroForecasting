"""Train a model directly on journey duration.

    python -m src.journeys.train

## Why this exists

The segment model wins per segment and loses per journey:

| segments | segment model | median baseline | verdict |
| --- | --- | --- | --- |
| 1 | 22.6s | 26.4s | +14.2% |
| 17 | 150.1s | 125.8s | **-9.9%** |

Its errors are positively correlated along a trip (error scaling n^0.663 against the
baseline's n^0.558), so summing amplifies them. Bias calibration narrowed that to
n^0.602 and no further: a constant offset cannot fix correlated error.

This model optimises the journey target directly, so nothing is summed and there is no
error to accumulate. Whether that is actually better is the point of `compare`, which
scores both on the **same validation journeys**.

## No bias calibration here, deliberately

The segment model needs one: its per-segment bias accumulates linearly across a summed
journey, reaching +95s over 17 segments. **This model sums nothing**, so there is
nothing to accumulate — and calibration measured as actively harmful:

| n | MAE raw | MAE if bias zeroed | median residual |
| --- | --- | --- | --- |
| 1 | 23.71s | 25.35s | +0.01s |
| 4 | 49.33s | 54.40s | +0.39s |
| 17 | 115.45s | 116.13s | +31.51s |

`regression_l1` already places the prediction at the conditional **median**, which is
exactly what MAE rewards. "Bias" is the *mean* residual, and on a right-skewed target
the two differ, so correcting the mean moves it off the MAE optimum. The median
residual is ~0 at short lengths, confirming the model is already where it should be.

Bias is still **reported** per length, because it is a real diagnostic. It is simply not
acted on. If unbiased predictions are ever needed downstream, the correct lever is the
objective (`regression_l2` or Huber targets the mean), not a post-hoc offset.

## The comparison is like for like, deliberately

Both models are scored on identical journey rows, with the segment model's predictions
summed over exactly the windows the journey rows describe. Anything less would be
comparing different questions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from ..features.config import FeatureConfig
from ..models import plots
from ..models.artifacts import mark_latest, new_run_dir, write_manifest
from ..models.encode import CategoricalEncoder
from .config import JOURNEY_TARGET, JourneyConfig
from .split import split_journeys

logger = logging.getLogger("journeys.train")

DEFAULT_OUTPUT = "data/models/journey_duration"

# Keys and provenance — never features.
KEY_COLUMNS: tuple[str, ...] = (
    "service_date",
    "trip_id",
    "trip_run",
    "block",
    "origin_stop_id",
    "destination_stop_id",
    "origin_departure_ts",
    "origin_stop_sequence",
    "destination_stop_sequence",
)

PARAMS: dict = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 17,
    "verbosity": -1,
}
NUM_BOOST_ROUND = 6000
EARLY_STOPPING_ROUNDS = 100


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every column the journey model may see, in a stable persisted order."""
    excluded = set(KEY_COLUMNS) | {JOURNEY_TARGET}
    return [c for c in frame.columns if c not in excluded]


def categorical_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Declared categoricals plus anything simply not numeric (e.g. `fare_period`)."""
    settings = FeatureConfig()
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
    encoded = encoder.transform(frame)
    missing = [c for c in columns if c not in encoded.columns]
    if missing:
        raise ValueError(f"feature columns absent from the frame: {missing}")
    return encoded[columns]


def fit(train_frame: pd.DataFrame, validation_frame: pd.DataFrame):
    import lightgbm as lgb

    columns = feature_columns(train_frame)
    categoricals = categorical_columns(train_frame, columns)
    encoder = CategoricalEncoder.fit(train_frame, extra_columns=categoricals)

    history: dict = {}
    x_train = build_matrix(train_frame, encoder, columns)
    x_validation = build_matrix(validation_frame, encoder, columns)

    train_set = lgb.Dataset(
        x_train, train_frame[JOURNEY_TARGET], categorical_feature=categoricals
    )
    validation_set = lgb.Dataset(
        x_validation,
        validation_frame[JOURNEY_TARGET],
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
        # Train is scored too, purely so the learning curve has both lines — without
        # the train curve a widening gap (overfitting) is invisible.
        valid_sets=[train_set, validation_set],
        valid_names=["train", "validation"],
        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING_ROUNDS, verbose=False, first_metric_only=True
            ),
            lgb.log_evaluation(period=200),
            lgb.record_evaluation(history),
        ],
    )
    logger.info("stopped at iteration %d", booster.best_iteration)
    return booster, encoder, columns, history


def by_length(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """MAE, bias and relative error per journey length, for each named prediction."""
    rows = []
    for length, part in frame.groupby("n_segments"):
        row: dict[str, float] = {
            "segments": int(length),
            "journeys": len(part),
            "median_duration_sec": float(part[JOURNEY_TARGET].median()),
        }
        for label, column in columns.items():
            residual = part[JOURNEY_TARGET] - part[column]
            row[f"mae_{label}"] = float(residual.abs().mean())
            row[f"bias_{label}"] = float(residual.mean())
            row[f"pct_{label}"] = float(
                100 * residual.abs().mean() / row["median_duration_sec"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("segments").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journeys", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    config = JourneyConfig()

    journeys = pd.read_parquet(Path(args.journeys or config.output_path) / "table")
    journeys["service_date"] = journeys["service_date"].astype(str)
    logger.info("read %d journey row(s)", len(journeys))

    train_mask, validation_mask, report = split_journeys(journeys)
    train_frame = journeys[train_mask].copy()
    validation_frame = journeys[validation_mask].copy()
    if train_frame.empty or validation_frame.empty:
        logger.error("split produced an empty side — nothing to fit")
        return 1
    if not report.is_trustworthy:
        logger.warning("SPLIT NOT TRUSTWORTHY — every metric below is provisional")

    booster, encoder, columns, history = fit(train_frame, validation_frame)

    for part in (train_frame, validation_frame):
        part["prediction"] = booster.predict(
            build_matrix(part, encoder, columns), num_iteration=booster.best_iteration
        )

    # `by_length` reports bias alongside MAE. Reported, not corrected — see the module
    # docstring for the measurements showing correction makes MAE worse at every length.
    table = by_length(validation_frame, {"journey_model": "prediction"})

    root = Path(args.output)
    run = new_run_dir(root)
    booster.save_model(str(run / "model.txt"), num_iteration=booster.best_iteration)
    encoder.save(run / "encoder.json")
    (run / "feature_columns.json").write_text(json.dumps(columns, indent=1))
    validation_frame.to_parquet(run / "validation_predictions.parquet", index=False)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "trustworthy": report.is_trustworthy,
                "split": report.as_metadata(),
                "best_iteration": int(booster.best_iteration),
                "by_length": table.to_dict(orient="records"),
                "feature_importance": dict(
                    sorted(
                        zip(
                            booster.feature_name(),
                            (int(v) for v in booster.feature_importance("gain")),
                            strict=True,
                        ),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                ),
            },
            indent=1,
            default=str,
        )
    )

    importance = json.loads((run / "metrics.json").read_text())["feature_importance"]
    plots.write_all(
        run / "plots",
        history=history,
        best_iteration=booster.best_iteration,
        importance=importance,
        validation=validation_frame,
        target=JOURNEY_TARGET,
        group_column="n_segments",
        comparison_series={"journey model": "prediction"},
        group_label="journey length (segments)",
    )
    write_manifest(
        run,
        model_name="journey_duration",
        target=JOURNEY_TARGET,
        trustworthy=report.is_trustworthy,
        training_data={
            "journeys_path": str(args.journeys or config.output_path),
            "service_dates": sorted(journeys["service_date"].unique().tolist()),
            "train_rows": len(train_frame),
            "validation_rows": len(validation_frame),
            "split": report.as_metadata(),
        },
        params=PARAMS,
        best_iteration=booster.best_iteration,
        feature_columns=columns,
        categorical_columns=categorical_columns(train_frame, columns),
        headline_metrics={
            "by_length": table.to_dict(orient="records"),
            # How many TRAINING journeys existed at each length. Serving warns when a
            # requested length is thinly supported — a length can be "trained on" and
            # still have too few examples to answer from, which a max-length cutoff
            # cannot express.
            "training_support": {
                str(int(k)): int(v)
                for k, v in train_frame["n_segments"]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        notes="Journey model. Predicts arrival(B)-arrival(A) directly; nothing summed.",
    )
    mark_latest(root, run)

    print("\n" + report.format())
    print("\nJOURNEY MODEL, by length")
    print(table.round(2).to_string(index=False))
    print(f"\nrun {run.name} — plots in {run / 'plots'}")
    print("Run `python -m src.journeys.compare` for the head-to-head.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
