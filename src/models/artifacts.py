"""Immutable, versioned model runs — and the manifest a registry needs.

## Why runs are immutable

Retraining used to overwrite `data/models/{name}/model.txt` in place, so the previous
model and the metrics that justified it were gone the moment a new run started. That
makes "did this change help?" unanswerable and makes a rollback impossible.

Every run now writes to its own directory, named for its UTC start:

    data/models/segment_duration/
      runs/
        2026-08-20T18-42-07Z/
          model.txt  encoder.json  feature_columns.json
          metrics.json  manifest.json
          validation_predictions.parquet
          plots/*.png
      latest        -> symlink to the newest run
      latest.json   -> the same pointer, for tools that will not follow symlinks

Nothing is ever rewritten. `latest` moves; history stays.

## What the manifest is for

The next stage pushes an artifact to S3 and registers it in the SageMaker Model
Registry. A registry entry is only useful if it answers "what is this, and could I
rebuild it?", so the manifest records provenance the model file itself does not carry:
the git commit, the service dates the training data covered, row counts, library
versions, hyperparameters, headline metrics, **and the trustworthiness flag** — a
registry must never present a provisional score as a validated one.

SHA-256 of every artifact is recorded so a file that changed after the fact is
detectable. `package()` tars a run into the `model.tar.gz` layout SageMaker expects,
which is the only step the registry work should need from here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("models.artifacts")

RUN_STAMP = "%Y-%m-%dT%H-%M-%SZ"


def new_run_dir(root: Path | str, when: datetime | None = None) -> Path:
    """Create and return a fresh run directory. Never reuses an existing one."""
    root = Path(root)
    stamp = (when or datetime.now(UTC)).strftime(RUN_STAMP)
    run = root / "runs" / stamp

    # A second run inside the same clock second would otherwise land on the identical
    # path and silently merge two runs' artifacts.
    suffix = 1
    while run.exists():
        suffix += 1
        run = root / "runs" / f"{stamp}-{suffix}"

    (run / "plots").mkdir(parents=True)
    logger.info("run directory %s", run)
    return run


def resolve_run(root: Path | str, run: str | None = None) -> Path:
    """Locate a run: an explicit id, else `latest`, else the newest on disk."""
    root = Path(root)
    if run:
        path = root / "runs" / run
        if not path.is_dir():
            raise FileNotFoundError(f"no such run: {path}")
        return path

    pointer = root / "latest"
    if pointer.exists():
        return pointer.resolve()

    runs = sorted((root / "runs").glob("*")) if (root / "runs").is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no runs under {root}/runs — train a model first")
    return runs[-1]


def mark_latest(root: Path | str, run: Path) -> None:
    """Point `latest` at this run. Symlink plus a plain-JSON fallback."""
    root = Path(root)
    link = root / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(run.resolve(), target_is_directory=True)
    (root / "latest.json").write_text(
        json.dumps({"run": run.name, "path": str(run)}, indent=1)
    )


def _git_state() -> dict[str, Any]:
    """Commit and dirty flag.

    A model built from an uncommitted tree is not reproducible from its commit, so the
    dirty flag is recorded rather than inferred.
    """

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        # Explicitly surfaced: a dirty tree means the commit does not describe the code
        # that produced this model.
        "dirty": bool(status) if status is not None else None,
    }


def _versions() -> dict[str, str]:
    import lightgbm
    import numpy
    import pandas
    import pyarrow

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lightgbm": lightgbm.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
    }


def _checksums(run: Path) -> dict[str, str]:
    sums = {}
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sums[str(path.relative_to(run))] = digest
    return sums


def write_manifest(
    run: Path,
    *,
    model_name: str,
    target: str,
    trustworthy: bool,
    training_data: dict[str, Any],
    params: dict[str, Any],
    best_iteration: int,
    feature_columns: list[str],
    categorical_columns: list[str],
    headline_metrics: dict[str, Any],
    notes: str = "",
) -> Path:
    """Write manifest.json. Called last, so checksums cover every other artifact."""
    manifest = {
        "run_id": run.name,
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "target": target,
        # First-class, not buried: a registry must not present a provisional score as
        # validated. Everything downstream should gate on this.
        "trustworthy": bool(trustworthy),
        "training_data": training_data,
        "params": params,
        "best_iteration": int(best_iteration),
        "feature_schema": {
            "columns": feature_columns,
            "categorical": categorical_columns,
            "n_features": len(feature_columns),
        },
        "headline_metrics": headline_metrics,
        "git": _git_state(),
        "environment": _versions(),
        "notes": notes,
        "artifacts": _checksums(run),
    }
    path = run / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1, default=str))
    logger.info("wrote manifest for run %s", run.name)
    return path


def package(
    run: Path,
    output: Path | None = None,
    serving_dir: Path | None = None,
    code_files: dict[str, Path] | None = None,
) -> Path:
    """Tar a run into the `model.tar.gz` layout a SageMaker framework container expects.

    Model files sit at the archive root, because SageMaker extracts straight into
    `/opt/ml/model` and a nested folder would put them a level deeper than the handler
    looks. **Inference code is the exception**: framework containers look for it under
    `code/`, and `code/requirements.txt` is what installs LightGBM at cold start.

    Serving inputs (station index, journey schedule, recent-conditions lookup) are
    copied in too — the handler loads them from the model directory, so they must travel
    with the model, or the endpoint starts and then fails on its first request.

    Plots and validation predictions are excluded: evaluation evidence, not serving
    inputs, and shipping them bloats every container pull.
    """
    output = output or run / "model.tar.gz"
    root_files = {"model.txt", "encoder.json", "feature_columns.json", "manifest.json"}

    with tarfile.open(output, "w:gz") as tar:
        for name in sorted(root_files):
            path = run / name
            if path.exists():
                tar.add(path, arcname=name)

        if serving_dir:
            for path in sorted(Path(serving_dir).glob("*")):
                if path.is_file():
                    tar.add(path, arcname=path.name)

        for arcname, path in sorted((code_files or {}).items()):
            tar.add(path, arcname=f"code/{arcname}")

    logger.info("packaged %s (%.1f KB)", output, output.stat().st_size / 1024)
    return output
