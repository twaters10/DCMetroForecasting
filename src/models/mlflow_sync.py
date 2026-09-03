"""Project every run into MLflow, so the comparison has a UI.

    python -m src.models.mlflow_sync              # sync every model root
    python -m src.models.mlflow_sync --model data/models/journey_duration
    python -m src.models.mlflow_sync --registry   # tag runs with their registry version
    make mlflow                                   # then browse it

## The tracking store is derived, not authoritative

Every run directory is immutable and its `manifest.json` already records params,
metrics, git commit, service dates and the split — which is, almost exactly, MLflow's
data model. So nothing here is a new source of truth: the store is a **view**, and
deleting `data/mlflow` and re-running this rebuilds it completely.

That is the same bargain `journeys/config.py` strikes for the journey table — derived,
reproducible, deliberately not authoritative — and it is what makes a local tracking
server acceptable. There is no durability requirement on a file that can be regenerated
from artifacts that are themselves immutable.

The artifact root is S3 all the same, because plots and metrics JSON are worth keeping
next to everything else the project stores there.

## Why sync rather than log during training

Logging from inside `train.py` would put an MLflow dependency in the training path and
mean a failed log could fail a run that had otherwise succeeded. Syncing afterwards from
the manifest cannot: the run is already on disk and already complete, and a broken sync
costs a re-run of this and nothing else. It also means the whole history — every run
trained before MLflow existed here — backfills on the first invocation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("models.mlflow_sync")

MODEL_ROOTS = (
    "data/models/journey_duration",
    "data/models/journey_duration_p80",
    "data/models/segment_duration",
)

# Local, and gitignored via `data/`. Rebuildable, so it is not backed up.
TRACKING_URI = "sqlite:///data/mlflow/tracking.db"
ARTIFACT_PREFIX = "mlflow"

# Recorded as a tag so a run in the UI can be traced back to the exact directory that
# produced it, which is the only thing that is authoritative.
RUN_ID_TAG = "metro_pulse.run_id"


def _flatten_metrics(manifest: dict[str, Any]) -> dict[str, float]:
    """Headline metrics onto flat `name -> float` pairs, from either manifest shape."""
    headline = manifest.get("headline_metrics", {})
    metrics: dict[str, float] = {}

    for row in headline.get("by_length", []) or []:
        n = int(row["segments"])
        for key, name in (
            ("mae_journey_model", "mae"),
            ("bias_journey_model", "bias"),
            ("pct_journey_model", "mae_pct"),
        ):
            if key in row:
                metrics[f"{name}_segments_{n:02d}"] = float(row[key])
        metrics[f"journeys_segments_{n:02d}"] = float(row["journeys"])

    # The segment model records the same thing under different keys.
    for row in headline.get("journey_level", []) or []:
        n = int(row["segments"])
        for key, name in (
            ("mae_sec", "mae"),
            ("bias_sec", "bias"),
            ("rmse_sec", "rmse"),
            ("mae_pct_of_duration", "mae_pct"),
        ):
            if key in row:
                metrics[f"{name}_segments_{n:02d}"] = float(row[key])

    for name, value in (headline.get("segment_mae") or {}).items():
        # Metric names allow a restricted character set; the baseline labels are prose.
        safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in name)
        metrics[f"segment_mae.{safe}"] = float(value)

    for n, pct in (headline.get("coverage_pct") or {}).items():
        metrics[f"coverage_pct_segments_{int(n):02d}"] = float(pct)

    if headline.get("calibration_offset_sec") is not None:
        metrics["calibration_offset_sec"] = float(headline["calibration_offset_sec"])

    # One number for sorting a run list. Journey-count weighted, because journeys are
    # very unevenly distributed across lengths — see `register.model_quality`.
    rows = headline.get("by_length") or []
    total = sum(r.get("journeys", 0) for r in rows)
    if total:
        metrics["mae_weighted"] = (
            sum(r["mae_journey_model"] * r["journeys"] for r in rows) / total
        )

    split = manifest.get("training_data", {}).get("split", {})
    for key in ("train_rows", "validation_rows", "embargoed_rows"):
        if split.get(key) is not None:
            metrics[key] = float(split[key])
    if manifest.get("best_iteration") is not None:
        metrics["best_iteration"] = float(manifest["best_iteration"])
    return metrics


def _tags(manifest: dict[str, Any]) -> dict[str, str]:
    split = manifest.get("training_data", {}).get("split", {})
    dates = manifest.get("training_data", {}).get("service_dates", []) or []
    git = manifest.get("git", {}) or {}
    validation_days = split.get("validation_days", []) or []
    return {
        RUN_ID_TAG: manifest["run_id"],
        "metro_pulse.model": manifest.get("model_name", "?"),
        "metro_pulse.target": manifest.get("target", "?"),
        # First-class, because a provisional score must never read as a validated one.
        "metro_pulse.trustworthy": str(manifest.get("trustworthy")),
        "metro_pulse.service_dates": f"{dates[0]}..{dates[-1]}" if dates else "?",
        "metro_pulse.n_service_dates": str(len(dates)),
        # The validation window, so the UI can show *why* two runs' numbers differ.
        "metro_pulse.split_boundary": str(split.get("boundary_utc", "?")),
        "metro_pulse.validation_days": (
            f"{validation_days[0]}..{validation_days[-1]}" if validation_days else "?"
        ),
        "mlflow.source.git.commit": str(git.get("commit", "?")),
        "metro_pulse.git_dirty": str(git.get("dirty")),
        "mlflow.note.content": manifest.get("notes", ""),
    }


def sync_root(root: Path | str, client, s3_bucket: str | None) -> int:
    """Log every run under one model root. Idempotent: existing runs are skipped."""
    root = Path(root)
    run_dirs = sorted((root / "runs").glob("*")) if (root / "runs").is_dir() else []
    if not run_dirs:
        logger.warning("no runs under %s", root)
        return 0

    experiment = root.name
    artifact_location = (
        f"s3://{s3_bucket}/{ARTIFACT_PREFIX}/{experiment}" if s3_bucket else None
    )
    existing = client.get_experiment_by_name(experiment)
    if existing:
        experiment_id = existing.experiment_id
    else:
        experiment_id = client.create_experiment(
            experiment, artifact_location=artifact_location
        )
        logger.info("created experiment %s -> %s", experiment, artifact_location)

    # One query, not one per run: `run_id` is unique per directory, so whatever is
    # already logged is exactly what should be skipped.
    already = {
        run.data.tags.get(RUN_ID_TAG)
        for run in client.search_runs([experiment_id], max_results=50_000)
    }

    synced = 0
    for path in run_dirs:
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            logger.warning("skipping %s — no manifest (interrupted run?)", path.name)
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest["run_id"] in already:
            continue

        run = client.create_run(
            experiment_id, tags=_tags(manifest), run_name=manifest["run_id"]
        )
        for key, value in (manifest.get("params") or {}).items():
            client.log_param(run.info.run_id, key, value)
        client.log_param(
            run.info.run_id, "n_features", manifest["feature_schema"]["n_features"]
        )
        for key, value in _flatten_metrics(manifest).items():
            client.log_metric(run.info.run_id, key, value)

        # Evidence, not serving inputs: the plots and the full metrics blob. The model
        # binary is deliberately not uploaded — it already lives in S3 under
        # `models/`, put there by `register.py`, and a second copy would be a second
        # thing to keep consistent.
        for artifact in ("metrics.json", "manifest.json", "comparison.json"):
            if (path / artifact).exists():
                client.log_artifact(run.info.run_id, str(path / artifact))
        if (path / "plots").is_dir():
            client.log_artifacts(run.info.run_id, str(path / "plots"), "plots")

        client.set_terminated(run.info.run_id, "FINISHED")
        synced += 1
        logger.info("logged %s/%s", experiment, manifest["run_id"])

    logger.info(
        "%s: %d new, %d already present", experiment, synced, len(run_dirs) - synced
    )
    return synced


def tag_registry(client, group: str, region: str = "us-east-1") -> int:
    """Stamp each MLflow run with the registry version(s) built from it.

    Without this the UI shows fifteen equally-plausible runs and no indication which one
    is actually serving traffic. Approval status lives in the registry and nowhere else,
    so it has to be fetched rather than inferred.
    """
    import boto3

    sagemaker = boto3.client("sagemaker", region_name=region)
    packages = sagemaker.list_model_packages(
        ModelPackageGroupName=group, SortBy="CreationTime", SortOrder="Ascending"
    )["ModelPackageSummaryList"]

    # A run can be registered more than once — versions 1 to 4 here are all the same
    # run — so the tag collects every version rather than the last one seen.
    versions: dict[str, list[str]] = {}
    approved: dict[str, str] = {}
    for summary in packages:
        detail = sagemaker.describe_model_package(
            ModelPackageName=summary["ModelPackageArn"]
        )
        run_id = detail.get("CustomerMetadataProperties", {}).get("run_id")
        if not run_id:
            continue
        version = str(summary["ModelPackageVersion"])
        versions.setdefault(run_id, []).append(version)
        if summary.get("ModelApprovalStatus") == "Approved":
            approved[run_id] = version

    tagged = 0
    for experiment in client.search_experiments():
        for run in client.search_runs([experiment.experiment_id], max_results=50_000):
            run_id = run.data.tags.get(RUN_ID_TAG)
            if run_id not in versions:
                continue
            client.set_tag(
                run.info.run_id,
                "metro_pulse.registry_versions",
                ",".join(versions[run_id]),
            )
            client.set_tag(
                run.info.run_id,
                "metro_pulse.approved_version",
                approved.get(run_id, ""),
            )
            tagged += 1
    logger.info("tagged %d run(s) with registry state", tagged)
    return tagged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="*", default=None, help="model roots to sync")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument(
        "--registry",
        action="store_true",
        help="also tag runs with their SageMaker registry version (needs AWS)",
    )
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="keep artifacts local instead of writing them to S3",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    import mlflow
    from mlflow.tracking import MlflowClient

    Path("data/mlflow").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    bucket = None
    if not args.no_s3:
        from ..etl.config import EtlConfig

        bucket = EtlConfig.from_env().s3_bucket

    total = 0
    for root in args.model or MODEL_ROOTS:
        if not Path(root).is_dir():
            logger.warning("no such model root: %s", root)
            continue
        total += sync_root(root, client, bucket)

    if args.registry:
        from ..serving.register import MODEL_PACKAGE_GROUP

        tag_registry(client, MODEL_PACKAGE_GROUP)

    print(f"\nsynced {total} new run(s) to {args.tracking_uri}")
    print("browse with:  make mlflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
