"""Package a run, upload it, and register it in the SageMaker Model Registry.

    python -m src.serving.register                    # latest run
    python -m src.serving.register --run 2026-08-20T19-34-21Z

Three steps, in order, each verifiable on its own:

1. **Package** — `model.tar.gz` in the layout a framework container expects: model
   files at the archive root, serving inputs beside them, inference code under `code/`.
2. **Upload** — to `s3://<bucket>/models/journey_duration/<run_id>/model.tar.gz`. Keyed
   by run id, so an upload can never overwrite a previous one.
3. **Register** — a new version in the `metro-pulse-journey-duration` package group.

## Approval status is not a formality here

Versions register as **PendingManualApproval**, and the manifest's `trustworthy` flag is
copied into the registry entry's metadata. The current model was trained on 13 service
days against the 14 `split.py` requires, so its scores are provisional. A registry that
presents a provisional model as validated is worse than no registry at all — the whole
point of the thing is to be the place where "is this safe to deploy?" has an answer.

Approving a version is a deliberate, separate act:

    aws sagemaker update-model-package --model-package-arn <arn> \\
        --model-approval-status Approved

## Where the metrics actually live

Two copies, deliberately, because they answer different questions.

`ModelMetrics.ModelQuality` points at a `model_quality.json` in S3, in the
`regression_metrics` schema SageMaker understands. That is the **record**: complete,
untruncated, one entry per journey length, and the field any AWS-native tooling reads.

`CustomerMetadataProperties` keeps a squashed `mae_sec_by_segments` string as well. That
one is a **convenience**: `list_model_packages` returns it inline, so `compare_runs
--registry` can tabulate every version without an S3 fetch each.

It is not the record, for two reasons. It carries MAE only — no bias, no counts, no
split — and it is sliced at 256 characters. At the current 14 lengths it runs to 112, so
the cap has never actually fired; it is a silent failure waiting rather than one already
paid for, and the slice now warns before it happens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

from ..etl.config import EtlConfig
from ..models.artifacts import _git_state, package, resolve_run

logger = logging.getLogger("serving.register")

MODEL_PACKAGE_GROUP = "metro-pulse-journey-duration"
MODEL_ROOT = "data/models/journey_duration"

# The "arrive by" companion, shipped in the same tarball. Two models answering two
# questions from one artifact: the median for "how long will this take", the upper
# quantile for "when should I budget to arrive". Optional — if never trained, the
# endpoint simply omits `arrive_by_sec`.
QUANTILE_MODEL_ROOT = "data/models/journey_duration_p80"
SERVING_INPUTS = "data/processed/serving"
LOOKUP = "data/processed/features/recent_conditions_lookup.parquet"
REGION = "us-east-1"

# The SKLearn framework container hosts LightGBM: it already ships numpy, pandas and
# scikit-learn, and installs the rest from code/requirements.txt at cold start. A
# dedicated LightGBM image would skip that install but means maintaining a container.
FRAMEWORK = "sklearn"
FRAMEWORK_VERSION = "1.2-1"

# Training and serving deliberately run DIFFERENT LightGBM versions: the sklearn 1.2-1
# container is Python 3.9 and 4.7.0 requires >=3.10, so serving pins 4.6.0. The skew is
# verified rather than assumed: 4.6.0 reading the 4.7.0-written model.txt reproduced
# its predictions to a maximum absolute difference of 0.0.
#
# Recorded on the registry entry because a version difference between train and serve is
# exactly what an incident review goes looking for, and it should not require unpacking
# the artifact to discover.
LIGHTGBM_TRAIN_VERSION = "4.7.0"
LIGHTGBM_SERVE_VERSION = "4.6.0"


def _date_range(manifest: dict) -> str:
    """`2026-08-07 to 2026-08-22`. Commas are illegal in registry metadata values."""
    dates = manifest["training_data"]["service_dates"]
    return f"{dates[0]} to {dates[-1]}" if dates else "unknown"


def model_quality(manifest: dict) -> dict:
    """The manifest's headline metrics in SageMaker's `regression_metrics` schema.

    `mae` is the journey-count-weighted mean across lengths, which is the only honest
    way to collapse this model to one number: journeys are wildly unevenly distributed
    across lengths (107k at one segment, 31 at thirty-two), so an unweighted mean would
    let the 32-segment bucket — 118 training examples — drag the headline around.

    Per-length figures are kept beside it under custom keys rather than folded away.
    Aggregate MAE hides a regression at one horizon, which is the same reason
    `monitoring/config.py` refuses to publish a single accuracy number.
    """
    rows = manifest.get("headline_metrics", {}).get("by_length", []) or []
    metrics: dict[str, dict] = {}

    total = sum(r.get("journeys", 0) for r in rows)
    if total:
        weighted = sum(r["mae_journey_model"] * r["journeys"] for r in rows) / total
        metrics["mae"] = {"value": round(weighted, 4), "standard_deviation": "NaN"}

    for row in rows:
        segments = int(row["segments"])
        metrics[f"mae_segments_{segments:02d}"] = {
            "value": round(float(row["mae_journey_model"]), 4),
            "standard_deviation": "NaN",
        }
        # Bias is signed and matters on its own: an L1 objective leaves a per-segment
        # offset that is harmless alone and accumulates once summed over a journey.
        if "bias_journey_model" in row:
            metrics[f"bias_segments_{segments:02d}"] = {
                "value": round(float(row["bias_journey_model"]), 4),
                "standard_deviation": "NaN",
            }

    return {
        "regression_metrics": metrics,
        # Not part of the AWS schema, and deliberately so: a consumer reading these
        # numbers needs to know they may be provisional and which rows produced them.
        # Anything that only understands `regression_metrics` ignores this key.
        "metro_pulse": {
            "run_id": manifest["run_id"],
            "trustworthy": manifest["trustworthy"],
            "target": manifest["target"],
            "validation_rows": manifest["training_data"]["validation_rows"],
            "validation_journeys": total,
            "service_dates": manifest["training_data"]["service_dates"],
            "split": manifest["training_data"]["split"],
            "git": manifest["git"],
            "best_iteration": manifest["best_iteration"],
        },
    }


def staged_serving_dir(destination: Path) -> Path:
    """Collect the three serving inputs the handler loads from the model directory.

    The recent-conditions lookup is converted from Parquet to CSV on the way in. Batch
    keeps Parquet because batch already has pyarrow; the container does not, and adding
    a 126 MB dependency to read a 922-row table would dominate every cold start.
    """
    import pandas as pd

    # Cleared, not merged. The directory persists between runs, so a file that was
    # shipped once and then renamed or dropped would keep travelling in every future
    # archive — which is how the stale Parquet copies survived the switch to CSV.
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for source in (
        Path(SERVING_INPUTS) / "station_index.json",
        Path(SERVING_INPUTS) / "journey_schedule.csv",
        # Routing: without these the container loads, but refuses every journey that
        # needs a train change — which is 64% of the network.
        Path(SERVING_INPUTS) / "walk_edges.csv",
        Path(SERVING_INPUTS) / "departures.csv",
        Path(SERVING_INPUTS) / "service_calendar.csv",
    ):
        if not source.exists():
            raise FileNotFoundError(
                f"{source} missing — run `python -m src.serving.stations` first"
            )
        shutil.copy(source, destination / source.name)

    lookup = Path(LOOKUP)
    if not lookup.exists():
        raise FileNotFoundError(
            f"{lookup} missing — run `python -m src.features.build` first"
        )
    pd.read_parquet(lookup).to_csv(
        destination / "recent_conditions_lookup.csv", index=False
    )
    return destination


def image_uri(region: str = REGION) -> str:
    """Resolve the container image rather than hardcoding an ECR URI.

    Account ids differ per region and the tags move; a literal URI is wrong somewhere
    and goes stale everywhere.
    """
    # SDK v3 moved this to `sagemaker.core`; v2 had it at the top level. Both are
    # tried so a version bump does not silently break registration.
    try:
        from sagemaker.core import image_uris
    except ImportError:  # sagemaker < 3
        from sagemaker import image_uris

    return image_uris.retrieve(
        framework=FRAMEWORK,
        region=region,
        version=FRAMEWORK_VERSION,
        image_scope="inference",
        instance_type="ml.m5.large",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="run id; default is `latest`")
    parser.add_argument(
        "--dry-run", action="store_true", help="package only, no upload"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    import boto3

    run = resolve_run(MODEL_ROOT, args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    logger.info("run %s | trustworthy=%s", manifest["run_id"], manifest["trustworthy"])

    if not manifest["trustworthy"]:
        logger.warning(
            "THIS MODEL'S SPLIT IS NOT TRUSTWORTHY. Registering anyway, as "
            "PendingManualApproval, with the flag recorded on the registry entry."
        )

    serving = staged_serving_dir(run / "serving_inputs")

    # The quantile model rides along as `model_p80.txt` plus its coverage table. It uses
    # the SAME encoder and feature columns as the median model — both are trained on the
    # same journey table — so only the booster and the coverage need shipping.
    quantile_root = Path(QUANTILE_MODEL_ROOT)
    if (quantile_root / "latest").exists():
        quantile_run = resolve_run(quantile_root)
        quantile_manifest = json.loads((quantile_run / "manifest.json").read_text())
        shutil.copy(quantile_run / "model.txt", serving / "model_p80.txt")
        (serving / "coverage_p80.json").write_text(
            json.dumps(
                quantile_manifest["headline_metrics"].get("coverage_pct", {}), indent=1
            )
        )
        logger.info("bundled quantile model from run %s", quantile_run.name)
    else:
        logger.warning(
            "no quantile model at %s — the endpoint will omit arrive_by_sec",
            quantile_root,
        )
    code_dir = Path(__file__).parent
    archive = package(
        run,
        serving_dir=serving,
        code_files={
            "inference.py": code_dir / "inference.py",
            "stations.py": code_dir / "stations.py",
            "routing.py": code_dir / "routing.py",
            "requirements.txt": code_dir / "requirements.txt",
        },
    )
    if args.dry_run:
        logger.info("dry run — packaged only, nothing uploaded")
        return 0

    config = EtlConfig.from_env()
    # Keyed on run id AND a hash of the archive. Keying on run id alone meant
    # re-packaging the same run overwrote the previous version's artifact — versions 1
    # through 4 all resolved to the same file, each describing a packaging its bytes no
    # longer matched. A registry version that can change underneath you is worthless.
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()[:8]
    prefix = f"models/journey_duration/{manifest['run_id']}/{digest}"
    key = f"{prefix}/model.tar.gz"
    s3 = boto3.client("s3", region_name=config.aws_region)
    s3.upload_file(str(archive), config.s3_bucket, key)
    model_data_url = f"s3://{config.s3_bucket}/{key}"
    logger.info("uploaded %s", model_data_url)

    # Beside the archive, under the same run+digest prefix, so the metrics can never
    # drift from the artifact they describe.
    quality = json.dumps(model_quality(manifest), indent=1, default=str)
    quality_key = f"{prefix}/model_quality.json"
    s3.put_object(
        Bucket=config.s3_bucket,
        Key=quality_key,
        Body=quality.encode(),
        ContentType="application/json",
    )
    quality_url = f"s3://{config.s3_bucket}/{quality_key}"
    logger.info("uploaded %s", quality_url)

    headline = manifest.get("headline_metrics", {})
    # CustomerMetadataProperties values are restricted to
    # [\p{L}\p{Z}\p{N}_.:/=+\-@] — no braces, quotes or commas, so JSON is rejected.
    # Space-separated `segments=mae` pairs carry the same information legally.
    #
    # A CONVENIENCE COPY ONLY. `list_model_packages` returns this inline, which is what
    # lets `compare_runs --registry` tabulate every version without an S3 fetch each.
    # The metrics of record are in model_quality.json above.
    by_length = " ".join(
        f"{int(r['segments'])}={r['mae_journey_model']:.1f}"
        for r in headline.get("by_length", [])
    )
    if len(by_length) > 256:
        # Said out loud rather than sliced silently. The tail is the longest journeys,
        # which are exactly the ones with the weakest support and the most to say.
        logger.warning(
            "mae_sec_by_segments is %d chars and will be truncated to 256 in the "
            "registry metadata — the full table is in %s",
            len(by_length),
            quality_url,
        )
    sagemaker = boto3.client("sagemaker", region_name=REGION)
    response = sagemaker.create_model_package(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelPackageDescription=(
            f"Journey duration, run {manifest['run_id']}. "
            f"Trained on {len(manifest['training_data']['service_dates'])} dates. "
            f"trustworthy={manifest['trustworthy']}"
        ),
        InferenceSpecification={
            "Containers": [{"Image": image_uri(), "ModelDataUrl": model_data_url}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
            "SupportedTransformInstanceTypes": ["ml.m5.large"],
        },
        # The metrics of record: complete and untruncated, unlike the metadata copy
        # below. This is the field AWS-native tooling reads.
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": quality_url,
                }
            }
        },
        # Never Approved automatically. See the module docstring.
        ModelApprovalStatus="PendingManualApproval",
        # Values must match [\p{L}\p{Z}\p{N}_.:/=+\-@] — no braces, quotes or commas,
        # which is why date_range uses "to" and nothing here is JSON-encoded.
        CustomerMetadataProperties={
            # --- what this model IS ---
            "algorithm": "lightgbm",
            "objective": str(manifest["params"].get("objective", "unknown")),
            "framework": f"sagemaker-scikit-learn:{FRAMEWORK_VERSION}-cpu-py3",
            "target": manifest["target"],
            "target_description": "arrival at B minus arrival at A in seconds",
            "n_features": str(manifest["feature_schema"]["n_features"]),
            "n_categorical": str(len(manifest["feature_schema"]["categorical"])),
            "lightgbm_train_version": LIGHTGBM_TRAIN_VERSION,
            "lightgbm_serve_version": LIGHTGBM_SERVE_VERSION,
            "date_range": _date_range(manifest),
            "validation_rows": str(manifest["training_data"]["validation_rows"]),
            # --- where it came from and how it scored ---
            "run_id": manifest["run_id"],
            "artifact_sha8": digest,
            "trustworthy": str(manifest["trustworthy"]),
            "git_commit": str(manifest["git"]["commit"]),
            "git_dirty": str(manifest["git"]["dirty"]),
            # The manifest's commit describes the code that TRAINED the model. The
            # inference handler, station index and routing tables are packaged now and
            # can move without retraining — as they did when transfer journeys were
            # added — so the serving commit is recorded separately. Reading one for the
            # other would misdescribe the archive's contents.
            "serving_commit": str(_git_state()["commit"]),
            "serving_dirty": str(_git_state()["dirty"]),
            "service_dates": str(len(manifest["training_data"]["service_dates"])),
            "train_rows": str(manifest["training_data"]["train_rows"]),
            "best_iteration": str(manifest["best_iteration"]),
            "mae_sec_by_segments": by_length[:256],
        },
    )
    arn = response["ModelPackageArn"]
    logger.info("registered %s", arn)
    (run / "model_package_arn.txt").write_text(arn)

    print(f"\nregistered  {arn}")
    print(f"artifact    {model_data_url}")
    print(f"status      PendingManualApproval (trustworthy={manifest['trustworthy']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
