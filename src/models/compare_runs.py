"""Run-over-run comparison: did this retrain actually help?

    python -m src.models.compare_runs                       # last 5 journey runs
    python -m src.models.compare_runs --model segment_duration
    python -m src.models.compare_runs --runs A B            # two specific runs
    python -m src.models.compare_runs --rescore             # honest, common holdout
    python -m src.models.compare_runs --registry            # registered versions

`artifacts.py` keeps every run immutably, and `journeys/compare.py` scores the competing
*approaches* against each other inside one run. Neither answers the question a retrain
actually raises, which is whether the new model is better than the one it replaces.

## Why the headline numbers are not comparable by default

`temporal_split` puts its boundary at a fraction of whatever data exists, so **every
retrain grades against a different validation window**. A run that ingests four new
service days validates on four days the previous run never saw — different journeys,
different weather, different incidents. Comparing the two `metrics.json` files directly
compares two measurements, not two models.

This is not hypothetical. A previous retrain here moved every absolute number in the
wrong direction while the ranking between approaches held, purely because the window
moved; the Makefile still carries the warning that came out of it.

So the default table prints the split boundary and validation days beside every metric,
and says plainly when the windows differ. Believe the numbers only when it says they are
comparable.

## What `--rescore` does instead

It loads one common holdout — the newest run's saved `validation_predictions.parquet`,
which carries the full feature frame, not just predictions — and scores every selected
run's booster on those identical rows with that run's own encoder and column order.

That is honest in one direction only. The holdout must be the **newest** run's
validation set: those rows sit after every older run's split boundary, so no older model
was trained on them. Scoring a newer model on an older run's window would grade it on
rows it learned from. `_leakage_free` enforces exactly that using the recorded
boundaries, and drops any run that fails rather than printing a flattering number.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("models.compare_runs")

DEFAULT_MODEL_ROOT = "data/models/journey_duration"
MODEL_PACKAGE_GROUP = "metro-pulse-journey-duration"
REGION = "us-east-1"

# Enough to see a trend without the table scrolling off a terminal.
DEFAULT_LIMIT = 5


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def load_runs(
    root: Path | str, runs: list[str] | None = None, limit: int = DEFAULT_LIMIT
) -> list[tuple[Path, dict[str, Any]]]:
    """Load (path, manifest) pairs, oldest first.

    Run directories are named for their UTC start, so a lexical sort is chronological.
    A run without a manifest is skipped loudly: training writes the manifest last, so a
    manifest-less directory is an interrupted run, not a comparable result.
    """
    root = Path(root)
    run_root = root / "runs"
    if not run_root.is_dir():
        raise FileNotFoundError(f"no runs under {run_root} — train a model first")

    if runs:
        paths = []
        for name in runs:
            path = run_root / name
            if not path.is_dir():
                raise FileNotFoundError(f"no such run: {path}")
            paths.append(path)
        paths.sort(key=lambda p: p.name)
    else:
        paths = sorted(p for p in run_root.glob("*") if p.is_dir())[-limit:]

    loaded = []
    for path in paths:
        manifest = path / "manifest.json"
        if not manifest.exists():
            logger.warning("skipping %s — no manifest (interrupted run?)", path.name)
            continue
        loaded.append((path, json.loads(manifest.read_text())))

    if not loaded:
        raise FileNotFoundError(f"no complete runs under {run_root}")
    return loaded


def _split(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest.get("training_data", {}).get("split", {})


def mae_by_length(manifest: dict[str, Any]) -> dict[int, float]:
    """Journey-length MAE from either manifest shape.

    The two model families record the same measurement under different keys — the
    journey model as `by_length`/`mae_journey_model`, the segment model as
    `journey_level`/`mae_sec`. `evaluate.py` names journey-level MAE the headline for
    both, so both are read onto that one axis and stay comparable.
    """
    headline = manifest.get("headline_metrics", {})
    for key, metric in (
        ("by_length", "mae_journey_model"),
        ("journey_level", "mae_sec"),
    ):
        rows = headline.get(key)
        if rows:
            return {int(r["segments"]): float(r[metric]) for r in rows if metric in r}
    return {}


# --------------------------------------------------------------------------------------
# comparability
# --------------------------------------------------------------------------------------


def windows(manifests: list[dict[str, Any]]) -> list[tuple[str, tuple[str, ...]]]:
    """The (boundary, validation days) each run was graded on."""
    return [
        (
            str(_split(m).get("boundary_utc", "?")),
            tuple(_split(m).get("validation_days", []) or []),
        )
        for m in manifests
    ]


def comparable(manifests: list[dict[str, Any]]) -> bool:
    """True only when every run was graded on the identical validation window."""
    return len(set(windows(manifests))) <= 1


def _boundary(manifest: dict[str, Any]) -> datetime | None:
    raw = _split(manifest).get("boundary_utc")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def _leakage_free(
    candidate: dict[str, Any], holdout: dict[str, Any]
) -> tuple[bool, str]:
    """May `candidate` be scored on `holdout`'s validation rows?

    Only if the candidate's split boundary is at or before the holdout's. The holdout
    rows all sit after the holdout boundary, and a candidate trained up to an earlier
    boundary therefore never saw them. A candidate with a *later* boundary may well have
    trained on those very rows, and its score would be a memory test.
    """
    mine, theirs = _boundary(candidate), _boundary(holdout)
    if mine is None or theirs is None:
        return False, "split boundary not recorded — cannot prove the rows are unseen"
    if mine > theirs:
        return False, (
            f"boundary {mine:%Y-%m-%d %H:%M} is after the holdout's "
            f"{theirs:%Y-%m-%d %H:%M} — it may have trained on these rows"
        )
    return True, ""


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


def provenance(loaded: list[tuple[Path, dict[str, Any]]]) -> pd.DataFrame:
    """One row per run: what it is, what it was built from, whether to believe it."""
    rows = []
    for _, manifest in loaded:
        split = _split(manifest)
        dates = manifest.get("training_data", {}).get("service_dates", []) or []
        git = manifest.get("git", {}) or {}
        commit = git.get("commit") or "?"
        validation_days = split.get("validation_days", []) or []
        rows.append(
            {
                "run_id": manifest["run_id"],
                "trust": "yes" if manifest.get("trustworthy") else "NO",
                "days": len(dates),
                "range": f"{dates[0]}..{dates[-1]}" if dates else "?",
                "train_rows": split.get("train_rows"),
                "val_rows": split.get("validation_rows"),
                "boundary": str(split.get("boundary_utc", "?"))[:16],
                "val_days": (
                    f"{validation_days[0]}..{validation_days[-1]}"
                    if validation_days
                    else "?"
                ),
                "iters": manifest.get("best_iteration"),
                # A dirty tree means the commit does not describe the code that trained
                # this model, so the two are shown together or not at all.
                "commit": f"{commit[:8]}{'+dirty' if git.get('dirty') else ''}",
            }
        )
    return pd.DataFrame(rows)


def headline(loaded: list[tuple[Path, dict[str, Any]]]) -> pd.DataFrame:
    """MAE by journey length, one column per run, plus the delta across the extremes."""
    series = {m["run_id"]: mae_by_length(m) for _, m in loaded}
    lengths = sorted({n for s in series.values() for n in s})
    if not lengths:
        return pd.DataFrame()

    table = pd.DataFrame({"segments": lengths})
    for run_id, values in series.items():
        table[run_id] = [values.get(n) for n in lengths]

    ids = list(series)
    if len(ids) >= 2:
        first, last = table[ids[0]], table[ids[-1]]
        table["delta"] = last - first
        # Percent of the older number, so a 2s move on a 24s segment and on a 300s
        # journey are not read as the same size of change.
        table["delta_pct"] = 100 * (last - first) / first
    return table


def rescore(
    loaded: list[tuple[Path, dict[str, Any]]], holdout: Path | None = None
) -> tuple[pd.DataFrame, Path]:
    """Score every run's booster on one common set of rows.

    The holdout defaults to the newest selected run's validation frame — the only choice
    that is leakage-free for all the others. See the module docstring.
    """
    import lightgbm as lgb

    from .encode import CategoricalEncoder
    from .train import build_matrix

    holdout_path = holdout or loaded[-1][0]
    holdout_manifest = next((m for p, m in loaded if p == holdout_path), loaded[-1][1])
    frame = pd.read_parquet(holdout_path / "validation_predictions.parquet")
    target = holdout_manifest["target"]
    logger.info(
        "holdout %s — %d rows, target %s", holdout_path.name, len(frame), target
    )

    # Grouped by journey length where the frame has one. The segment model's validation
    # frame is per-segment rows, so it pools into a single row instead: reconstructing
    # journey-level error from it needs the window arithmetic in `journeys/build`, and
    # `models/` deliberately does not depend on `journeys/`. Pooled per-segment MAE is
    # the honest number available here — `journeys/compare` is where the summed view
    # lives.
    group = "n_segments" if "n_segments" in frame.columns else None

    rows = []
    for path, manifest in loaded:
        run_id = manifest["run_id"]
        ok, reason = _leakage_free(manifest, holdout_manifest)
        if not ok:
            logger.warning("skipping %s — %s", run_id, reason)
            continue
        if manifest["target"] != target:
            logger.warning(
                "skipping %s — target %s does not match the holdout's %s",
                run_id,
                manifest["target"],
                target,
            )
            continue

        columns = json.loads((path / "feature_columns.json").read_text())
        absent = [c for c in columns if c not in frame.columns]
        if absent:
            # Schema drift between runs. Filling these would score a different model
            # than the one that was trained, so the run is reported unscoreable.
            logger.warning(
                "skipping %s — %d feature(s) absent from the holdout frame: %s",
                run_id,
                len(absent),
                ", ".join(absent[:5]) + (" ..." if len(absent) > 5 else ""),
            )
            continue

        booster = lgb.Booster(model_file=str(path / "model.txt"))
        encoder = CategoricalEncoder.load(path / "encoder.json")
        prediction = booster.predict(build_matrix(frame, encoder, columns))
        residual = frame[target] - prediction

        if group:
            for length, part in residual.groupby(frame[group]):
                rows.append(
                    {
                        "run_id": run_id,
                        "segments": int(length),
                        "n": len(part),
                        "mae": float(part.abs().mean()),
                        "bias": float(part.mean()),
                    }
                )
        else:
            rows.append(
                {
                    "run_id": run_id,
                    # A label, not a length — "0 segments" would read as a real bucket.
                    "segments": "all",
                    "n": len(residual),
                    "mae": float(residual.abs().mean()),
                    "bias": float(residual.mean()),
                }
            )

    if not rows:
        raise RuntimeError(
            "no run could be scored on this holdout — see the warnings above"
        )

    scored = pd.DataFrame(rows)
    wide = scored.pivot(index="segments", columns="run_id", values="mae").reset_index()
    wide.columns.name = None
    counts = scored.groupby("segments")["n"].first()
    wide.insert(1, "n", wide["segments"].map(counts))

    ids = [c for c in wide.columns if c not in ("segments", "n")]
    if len(ids) >= 2:
        wide["delta"] = wide[ids[-1]] - wide[ids[0]]
        wide["delta_pct"] = 100 * (wide[ids[-1]] - wide[ids[0]]) / wide[ids[0]]
    return wide, holdout_path


# --------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------


def registry(
    group: str = MODEL_PACKAGE_GROUP, region: str = REGION, limit: int = 20
) -> pd.DataFrame:
    """Registered versions and the metadata `register.py` stamped onto each.

    Read straight from the Model Registry rather than from local runs: these are the
    artifacts that can actually be deployed, and a local run that was never registered
    is not among them.
    """
    import boto3

    client = boto3.client("sagemaker", region_name=region)
    packages = client.list_model_packages(
        ModelPackageGroupName=group, SortBy="CreationTime", SortOrder="Descending"
    )["ModelPackageSummaryList"][:limit]

    rows = []
    for summary in packages:
        detail = client.describe_model_package(
            ModelPackageName=summary["ModelPackageArn"]
        )
        meta = detail.get("CustomerMetadataProperties", {})
        rows.append(
            {
                "version": summary.get("ModelPackageVersion"),
                "status": summary.get("ModelApprovalStatus", "?"),
                "created": summary["CreationTime"].strftime("%Y-%m-%d %H:%M"),
                "run_id": meta.get("run_id", "?"),
                "trust": meta.get("trustworthy", "?"),
                "dates": meta.get("date_range", "?"),
                "val_rows": meta.get("validation_rows", "?"),
                "commit": (meta.get("git_commit") or "?")[:8],
                "sha8": meta.get("artifact_sha8", "?"),
                "mae_by_segments": meta.get("mae_sec_by_segments", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------


def _rule(title: str, width: int = 96) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def format_report(
    loaded: list[tuple[Path, dict[str, Any]]],
    scored: pd.DataFrame | None = None,
    holdout: Path | None = None,
) -> str:
    manifests = [m for _, m in loaded]
    model = manifests[0].get("model_name", "?")
    lines = [_rule(f"RUNS — {model}, oldest first")]
    lines.append(provenance(loaded).to_string(index=False))

    same_window = comparable(manifests)
    table = headline(loaded)
    if not table.empty:
        lines.append(_rule("HEADLINE — journey MAE (seconds) as each run recorded it"))
        if same_window:
            lines.append("Every run was graded on the same validation window.")
        else:
            # The whole reason this tool exists. Stated before the numbers, so a
            # window artefact is never read as a model improvement.
            lines.append(
                "WARNING: these runs were graded on DIFFERENT validation windows.\n"
                "Each number below is honest about its own run and says nothing about\n"
                "which model is better. Re-run with --rescore for a like-for-like\n"
                "comparison on identical rows."
            )
        lines.append("")
        lines.append(table.round(2).to_string(index=False))

    if scored is not None and not scored.empty:
        lines.append(
            _rule(f"RESCORED — every model on {holdout.name if holdout else '?'} rows")
        )
        lines.append(
            "Identical rows, identical target. Negative delta = the newer run is\n"
            "better. These are the numbers to make a promotion decision on."
        )
        lines.append("")
        lines.append(scored.round(2).to_string(index=False))

    untrusted = [m["run_id"] for m in manifests if not m.get("trustworthy")]
    if untrusted:
        lines.append(
            f"\nNOT TRUSTWORTHY: {', '.join(untrusted)} — provisional scores, "
            "do not promote on them alone."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ROOT, help="model root dir")
    parser.add_argument("--runs", nargs="*", default=None, help="run ids, else newest")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="score every run on one common holdout — the only fair comparison",
    )
    parser.add_argument("--holdout", default=None, help="run id supplying holdout rows")
    parser.add_argument(
        "--registry", action="store_true", help="registered versions instead of runs"
    )
    parser.add_argument("--group", default=MODEL_PACKAGE_GROUP)
    parser.add_argument("--json", default=None, help="also write the tables here")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    if args.registry:
        table = registry(args.group)
        print(_rule(f"MODEL REGISTRY — {args.group}"))
        print(table.to_string(index=False) if not table.empty else "no versions yet")
        if args.json:
            Path(args.json).write_text(
                json.dumps(table.to_dict(orient="records"), indent=1, default=str)
            )
        return 0

    loaded = load_runs(args.model, args.runs, args.limit)
    scored = holdout_path = None
    if args.rescore:
        holdout = (Path(args.model) / "runs" / args.holdout) if args.holdout else None
        scored, holdout_path = rescore(loaded, holdout)

    print(format_report(loaded, scored, holdout_path))

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "provenance": provenance(loaded).to_dict(orient="records"),
                    "headline": headline(loaded).to_dict(orient="records"),
                    "comparable": comparable([m for _, m in loaded]),
                    "rescored": (
                        scored.to_dict(orient="records") if scored is not None else []
                    ),
                    "holdout": holdout_path.name if holdout_path else None,
                },
                indent=1,
                default=str,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
