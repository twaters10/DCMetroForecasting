"""Configuration for the monitoring layer.

Two signals with different latencies, deliberately kept apart:

**Drift** is available immediately. It compares today's inputs and predictions to the
training distribution and answers "has something changed?".

**Realised accuracy** lags one to two days. A service day closes at 04:00 UTC the
following day and the ETL is run manually, so actuals arrive on D+1 at the earliest. It
answers "did the change matter?".

Conflating them produces an alarm that is either too slow to be actionable or too noisy
to trust, so they publish as separate metrics with separate thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

CLOUDWATCH_NAMESPACE: Final[str] = "MetroPulse/Model"

# Where the collector archives each live recent-conditions write. Not backfillable.
HISTORY_PREFIX: Final[str] = "models/serving/history/recent_conditions/"

# Journey lengths to report accuracy at. Mirrors journeys.config.DEFAULT_LENGTHS; an
# aggregate MAE would hide a regression at one horizon, because error is strongly
# length-dependent (23.6s at n=1 against 202.8s at n=28).
REPORT_LENGTHS: Final[tuple[int, ...]] = (1, 2, 4, 8, 12, 17, 24, 28)


@dataclass(frozen=True)
class MonitoringConfig:
    """Every tunable in the monitoring layer."""

    namespace: str = CLOUDWATCH_NAMESPACE

    # `recent_deviation` is the strongest feature (+0.255 against the residual) and goes
    # null whenever the live table is stale. That has already failed silently once — the
    # endpoint served schedule-only predictions for days without erroring. This is the
    # single most valuable alarm here.
    max_null_rate_pct: float = 40.0

    # Realised MAE must stay below the published schedule on the same rows. "Beats the
    # timetable" is the project's actual claim; if it stops being true, the model has no
    # reason to exist. Expressed as a ratio so a hard week does not trip it — the
    # baseline degrades on a hard week too.
    max_mae_vs_schedule_ratio: float = 1.0

    # Below this many scored journeys, a daily metric is too noisy to alarm on.
    min_scored_journeys: int = 200

    model_root: str = "data/models/journey_duration"
    features_path: str = "data/processed/features/table"
