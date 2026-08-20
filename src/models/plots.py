"""Evaluation figures, written as part of every training run.

Numbers in a JSON file answer questions you already knew to ask. These figures are for
the ones you did not — a learning curve that never flattens, residuals with a second
mode, error concentrated in one hour of the day. All are invisible in a summary metric.

**Palette and axes match `notebooks/05_segment_eda` and `06_feature_eda`** so a chart in
a run directory and a chart in a notebook read as the same system. The five chromatic
slots pass a CVD validator as an adjacent set; `grey` is a neutral for de-emphasis and
reference marks, never a sixth identity slot.

Every figure follows the same rules: one y-axis and never a second scale, a legend
whenever two or more series share a plot, recessive grid and axes, and values in ink
rather than in the series colour, so identity is never carried by colour alone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

# Written to files from a CLI, so never try to open a window. Must precede pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger("models.plots")

C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "grey": "#8a8a85",
}
INK = "#0b0b0b"
MUTED = "#52514e"

RC = {
    "figure.figsize": (9, 4.5),
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c9c9c4",
    "axes.labelcolor": MUTED,
    "axes.titlesize": 12,
    "axes.titleweight": "medium",
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": "#e8e8e4",
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
}


def _save(fig, directory: Path, name: str) -> Path:
    path = Path(directory) / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  wrote %s", path.name)
    return path


def learning_curve(
    history: dict, directory: Path, best_iteration: int | None = None
) -> Path | None:
    """Train and validation loss by boosting iteration, focused on the tail.

    The most diagnostic figure here — but only if it is scaled to the part that matters.
    Plotted full-range the first ~50 iterations fall from ~430s to ~55s and squash the
    entire converged region into an unreadable band. The y-axis is therefore fitted to
    the last three quarters of the run, where "still falling" and "flat" are actually
    distinguishable.

    A validation curve still descending at the right edge means the round cap is binding
    rather than converged — the failure that left the first segment model stopped at
    1988 of 2000 while still improving. A widening train/validation gap is overfitting.
    """
    if not history:
        return None
    colours = {"train": C["grey"], "validation": C["blue"]}
    curves = {
        f"{split} ({metric})": (values, colours.get(split, C["orange"]))
        for split, metrics in history.items()
        for metric, values in metrics.items()
    }
    longest = max(len(v) for v, _ in curves.values())
    tail_from = int(longest * 0.25)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for label, (values, colour) in curves.items():
            ax.plot(
                range(1, len(values) + 1), values, label=label, color=colour, lw=1.8
            )

        tails = [v[tail_from:] for v, _ in curves.values() if len(v) > tail_from]
        if tails:
            low = min(min(v) for v in tails)
            high = max(max(v) for v in tails)
            pad = max((high - low) * 0.35, low * 0.02)
            ax.set_ylim(low - pad, high + pad)

        if best_iteration:
            ax.axvline(best_iteration, color=C["orange"], lw=1.4, ls="--")
            # Anchored to the BOTTOM of the axis: the legend sits upper-right, and a
            # top-anchored label collides with it on any converged run.
            ax.annotate(
                f"best iteration {best_iteration:,}",
                (best_iteration, ax.get_ylim()[0]),
                xytext=(-6, 8),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=9,
                color=C["orange"],
            )

        ax.set_xlabel("boosting iteration")
        ax.set_ylabel("L1 loss (seconds)")
        ax.set_title("Learning curve — scaled to the converged region, not the drop")
        ax.legend(loc="upper right")
        return _save(fig, directory, "learning_curve")


def feature_importance(
    importance: dict[str, float], directory: Path, top: int = 20
) -> Path:
    """Gain share, highest first.

    Gain rather than split count: split count rewards high-cardinality features merely
    for being splittable, which says nothing about whether the splits helped.
    """
    series = pd.Series(importance).sort_values(ascending=False).head(top)[::-1]
    share = 100 * series / max(sum(importance.values()), 1)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 0.34 * len(series) + 1.5))
        ax.barh(range(len(series)), share.values, color=C["blue"], height=0.68)
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels(series.index, fontsize=9)
        ax.set_xlabel("share of total gain (%)")
        ax.set_title(f"Feature importance — top {len(series)} by gain")
        ax.grid(axis="y", visible=False)
        for i, value in enumerate(share.values):
            ax.annotate(
                f"{value:.1f}%",
                (value, i),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=MUTED,
            )
        return _save(fig, directory, "feature_importance")


def residual_distribution(residual: pd.Series, directory: Path) -> Path:
    """Where the error actually sits, versus what a single MAE implies.

    The mean and median markers are the whole point: on a right-skewed target they
    separate, and that separation is what makes bias correction help a summed prediction
    and hurt an unsummed one.
    """
    clipped = residual.clip(residual.quantile(0.005), residual.quantile(0.995))
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(clipped, bins=80, color=C["blue"], edgecolor="white", linewidth=0.3)
        for value, colour, label in (
            (float(residual.median()), C["orange"], "median"),
            (float(residual.mean()), C["aqua"], "mean"),
        ):
            ax.axvline(value, color=colour, lw=1.8, label=f"{label} {value:+.2f}s")
        ax.set_xlabel("residual (actual - predicted, seconds)")
        ax.set_ylabel("rows")
        ax.set_title("Residual distribution — mean and median separate under skew")
        ax.legend(loc="upper right")
        return _save(fig, directory, "residual_distribution")


def predicted_vs_actual(
    actual: pd.Series, predicted: pd.Series, directory: Path, bins: int = 40
) -> Path:
    """Binned median with an interquartile band, against the y=x reference.

    A scatter of hundreds of thousands of points is a solid block. Binning shows where
    the model is systematically high or low; the band shows the spread a median hides.
    """
    frame = pd.DataFrame(
        {"actual": actual.to_numpy(), "predicted": np.asarray(predicted)}
    )
    edges = np.quantile(frame["predicted"], np.linspace(0, 1, bins + 1))
    frame["bin"] = pd.cut(frame["predicted"], np.unique(edges), duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        centre=("predicted", "median"),
        median=("actual", "median"),
        q25=("actual", lambda s: s.quantile(0.25)),
        q75=("actual", lambda s: s.quantile(0.75)),
        n=("actual", "size"),
    )
    grouped = grouped[grouped["n"] >= 30]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4.4))
        limit = [0, float(grouped["centre"].max()) * 1.05]
        ax.plot(limit, limit, color=C["grey"], lw=1.4, ls="--", zorder=1)
        ax.annotate(
            "perfect prediction",
            (limit[1] * 0.55, limit[1] * 0.55),
            fontsize=9,
            color=MUTED,
            rotation=30,
            ha="left",
            va="bottom",
        )
        ax.fill_between(
            grouped["centre"],
            grouped["q25"],
            grouped["q75"],
            color=C["blue"],
            alpha=0.16,
            lw=0,
            zorder=2,
        )
        ax.plot(grouped["centre"], grouped["median"], color=C["blue"], lw=2, zorder=3)
        ax.set_xlabel("predicted (seconds)")
        ax.set_ylabel("actual (seconds)")
        ax.set_title("Predicted vs actual — binned median with interquartile band")
        return _save(fig, directory, "predicted_vs_actual")


def error_by_group(
    frame: pd.DataFrame,
    group_column: str,
    series: dict[str, str],
    directory: Path,
    name: str,
    title: str,
    xlabel: str,
    target: str,
) -> Path:
    """MAE per group, one line per named prediction. Two or more series get a legend."""
    order = (C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"])
    rows = []
    for value, part in frame.groupby(group_column):
        row: dict = {group_column: value}
        for label, column in series.items():
            row[label] = float((part[target] - part[column]).abs().mean())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(group_column)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for i, label in enumerate(series):
            ax.plot(
                table[group_column],
                table[label],
                label=label,
                color=order[i % len(order)],
                lw=2,
                marker="o",
                ms=4.5,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("MAE (seconds)")
        ax.set_title(title)
        if len(series) > 1:
            ax.legend(loc="upper left")
        return _save(fig, directory, name)


def write_all(
    directory: Path,
    *,
    history: dict,
    importance: dict[str, float],
    best_iteration: int | None = None,
    validation: pd.DataFrame,
    target: str,
    prediction_column: str = "prediction",
    group_column: str | None = None,
    comparison_series: dict[str, str] | None = None,
    group_label: str = "",
) -> list[Path]:
    """Produce the full figure set for one run. Returns the paths written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    residual = validation[target] - validation[prediction_column]

    written = [
        p
        for p in (
            learning_curve(history, directory, best_iteration),
            feature_importance(importance, directory),
            residual_distribution(residual, directory),
            predicted_vs_actual(
                validation[target], validation[prediction_column], directory
            ),
        )
        if p is not None
    ]

    if "local_hour" in validation.columns:
        written.append(
            error_by_group(
                validation,
                "local_hour",
                {"model": prediction_column},
                directory,
                "error_by_hour",
                "Error by hour of day — is it concentrated in the peaks?",
                "local hour (America/New_York)",
                target,
            )
        )
    if group_column and comparison_series:
        written.append(
            error_by_group(
                validation,
                group_column,
                comparison_series,
                directory,
                "error_by_length",
                (
                    "Error by journey length — how each approach scales"
                    if len(comparison_series) > 1
                    else "Error by journey length"
                ),
                group_label or group_column,
                target,
            )
        )
    logger.info("wrote %d figure(s) to %s", len(written), directory)
    return written
