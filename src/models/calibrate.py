"""Additive bias calibration, fitted on training residuals.

## Why a model that wins per segment loses per journey

Measured on the first real run, 13 service days:

| segments | model | baseline | vs baseline |
| --- | --- | --- | --- |
| 1 | 22.7s | 26.4s | **+14.2%** |
| 8 | 95.2s | 87.8s | -8.5% |
| 17 | 149.8s | 125.8s | **-19.1%** |

The model beats the median baseline on a single segment and loses badly once the
predictions are summed. The cause is **bias, not noise**: the model runs short by
+6.04s per segment, and a constant per-segment bias accumulates *linearly* with journey
length while random error only grows as the square root. Bias went from 27% of MAE at
one segment to 63% at seventeen.

The source is the objective. `regression_l1` optimises the conditional **median**, and
segment duration is right-skewed — a handful of very long traversals pull the mean above
the median. Predicting the median is exactly right for per-segment MAE and exactly wrong
for anything summed, because sums care about means.

## The trade this makes, stated plainly

Adding `mean(actual - predicted)` makes the predictions unbiased, which drives journey
bias to ~0 at every length. It will **slightly worsen per-segment MAE**, because MAE is
minimised by the median and this deliberately moves the prediction off it.

That is the correct trade for this project — journey-level accuracy is the headline and
per-segment MAE is saturated by measurement noise — but it is a real trade, so
`evaluate` reports calibrated and uncalibrated side by side rather than quietly
replacing one with the other.

## Why per segment and not per journey

The bias accumulates linearly, so correcting it once at the segment level corrects it at
*every* journey length simultaneously. A calibration fitted at journey level would have
to pick a length to optimise for and would be wrong at the others.

**Fitted on training residuals only.** Fitting on validation would use the answers to
correct the predictions being graded.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("models.calibrate")


@dataclass(frozen=True, slots=True)
class BiasCalibration:
    """A single additive offset in seconds, applied to every prediction."""

    offset_sec: float

    @classmethod
    def fit(
        cls, actual: pd.Series, predicted: pd.Series | np.ndarray
    ) -> BiasCalibration:
        """Offset that makes the mean residual zero on the frame it is fitted to."""
        residual = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
        offset = float(np.mean(residual))
        logger.info(
            "bias calibration %+.3fs per segment (%+.1fs over a 17-segment journey)",
            offset,
            offset * 17,
        )
        return cls(offset_sec=offset)

    def apply(self, predicted: pd.Series | np.ndarray) -> np.ndarray:
        return np.asarray(predicted, dtype=float) + self.offset_sec

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"offset_sec": self.offset_sec}, indent=1))

    @classmethod
    def load(cls, path: Path) -> BiasCalibration:
        return cls(offset_sec=float(json.loads(Path(path).read_text())["offset_sec"]))
