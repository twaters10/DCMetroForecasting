"""SageMaker inference handler: station names in, predicted journey time out.

    POST {"origin": "Foggy Bottom-GWU", "destination": "Union Station",
          "departure_ts": "2026-08-21T13:05:00Z"}

    ->   {"predicted_sec": 1284, "scheduled_sec": 1200, "n_segments": 11, ...}

SageMaker's framework containers call four hooks — `model_fn`, `input_fn`, `predict_fn`,
`output_fn`. Everything expensive happens once in `model_fn` at cold start.

## The one rule this file exists to honour

**Features must be assembled exactly as training assembled them, or the model is being
asked a different question than it was taught.** Two specific commitments:

1. **Column order comes from the persisted `feature_columns.json`**, never re-derived.
   LightGBM records column order at fit time; a re-derived list that happens to differ
   silently scores against a shifted matrix.
2. **Nulls stay null.** `recent_*` is null when nothing completed the origin segment
   inside `rolling_max_age_sec`, and the trip-state features are null here always (see
   below). Training preserved those nulls precisely so this path is not a distribution
   shift, and LightGBM splits on missing natively.

## Why trip-state features are always null

`upstream_delay_sec`, `trip_progress`, `minutes_into_trip`, `segments_completed`,
`trip_start_hour` and `headway_sec` describe a train already partway through its run. A
rider standing on a platform has not boarded anything, so there is no train state to
report. Nulling them is the honest encoding of "unknown", not a shortcut — and it is why
the training pipeline was careful never to impute them.

## Staleness parity

The recent-conditions lookup carries `completed_at`. If the most recent traversal of the
origin segment is older than `rolling_max_age_sec`, the `recent_*` features are nulled —
the identical rule `features/build.py` applies in batch. Without this the endpoint would
serve a confident number that batch would have refused to produce, which is exactly the
skew no offline metric would reveal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# SageMaker loads `code/inference.py` as a TOP-LEVEL module, so a relative import fails
# inside the container even though it works everywhere else. Both forms are attempted so
# the same file runs under pytest, locally, and in the container without editing.
try:  # package context: tests and local use
    from .stations import StationError, StationIndex, resolve_journey
except ImportError:  # flat context: code/ inside model.tar.gz
    from stations import StationError, StationIndex, resolve_journey  # type: ignore

logger = logging.getLogger("serving.inference")

ROLLING_MAX_AGE_SEC = 3600

# The journey table is built for lengths up to 17 segments (journeys.config.
# DEFAULT_LENGTHS), but the network runs longer: Shady Grove to Glenmont is 26. Asked to
# extrapolate, the model predicted 2,471s against a 3,720s schedule — a 21-minute
# underestimate claiming the train beats the timetable by a third.
#
# Extrapolated answers are still returned, because refusing a legitimate journey is its
# own failure, but they are flagged on the response so a caller cannot mistake one for a
# supported prediction. The real fix is to train on longer windows.
MAX_TRAINED_SEGMENTS = 17
SERVICE_TZ = "America/New_York"
FARE_EVENING_START_HOUR = 21.5

# Known-unknowable at prediction time — see the module docstring.
TRIP_STATE_FEATURES = (
    "upstream_delay_sec",
    "upstream_delay_last_sec",
    "segments_completed",
    "trip_progress",
    "minutes_into_trip",
    "trip_start_hour",
    "stops_remaining",
    "headway_sec",
    "recent_vs_scheduled",
)


class Artifacts:
    """Everything loaded once at cold start."""

    def __init__(self, directory: Path):
        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=str(directory / "model.txt"))
        self.columns: list[str] = json.loads(
            (directory / "feature_columns.json").read_text()
        )
        self.encoder_mapping: dict = json.loads(
            (directory / "encoder.json").read_text()
        )
        self.index = StationIndex.from_json(
            (directory / "station_index.json").read_text()
        )
        # CSV keeps pyarrow out of the container — see stations.build_journey_schedule.
        self.schedule = pd.read_csv(directory / "journey_schedule.csv")
        # `parse_dates` is load-bearing: CSV carries no dtypes, and the staleness rule
        # needs a real timestamp rather than the string it would otherwise get back.
        self.lookup = pd.read_csv(
            directory / "recent_conditions_lookup.csv", parse_dates=["completed_at"]
        )
        self.lookup = self.lookup.set_index(["from_stop_id", "to_stop_id"])
        self.run_id = json.loads((directory / "manifest.json").read_text())["run_id"]
        self.trustworthy = json.loads((directory / "manifest.json").read_text())[
            "trustworthy"
        ]
        logger.info(
            "loaded run %s: %d features, %d OD pairs, %d lookup rows",
            self.run_id,
            len(self.columns),
            len(self.schedule),
            len(self.lookup),
        )


def model_fn(model_dir: str) -> Artifacts:
    return Artifacts(Path(model_dir))


def input_fn(body: str | bytes, content_type: str = "application/json") -> dict:
    """Parse only. Validation happens in `predict_fn`, where it can be caught.

    An exception raised here escapes as a bare HTTP 500 with gunicorn's HTML error page,
    which tells a caller nothing. Field checks therefore live downstream.
    """
    if "json" not in content_type:
        raise ValueError(
            f"unsupported content type {content_type!r}; send application/json"
        )
    return json.loads(body)


def _calendar_features(when: pd.Timestamp) -> dict:
    """Calendar features, derived the same way `features/safe.py` derives them."""
    local = when.tz_convert(SERVICE_TZ)
    hour_frac = local.hour + local.minute / 60
    weekday = local.dayofweek
    is_weekend = int(weekday >= 5)
    if is_weekend:
        fare = "weekend"
    else:
        fare = "weekday_evening" if hour_frac >= FARE_EVENING_START_HOUR else "weekday"
    return {
        "local_hour": local.hour,
        "local_hour_frac": hour_frac,
        "day_of_week": weekday,
        "month": local.month,
        "is_weekend": is_weekend,
        "hour_sin": np.sin(2 * np.pi * local.hour / 24),
        "hour_cos": np.cos(2 * np.pi * local.hour / 24),
        "dow_sin": np.sin(2 * np.pi * weekday / 7),
        "dow_cos": np.cos(2 * np.pi * weekday / 7),
        "is_holiday": 0,
        "is_service_weekday": int(not is_weekend),
        "fare_period": fare,
    }


def _recent_conditions(
    artifacts: Artifacts, segment: tuple[str, str], when: pd.Timestamp
) -> dict:
    """Recent conditions for the journey's FIRST leg, with the batch staleness rule."""
    blank = {
        "recent_duration_median": np.nan,
        "recent_duration_mean": np.nan,
        "recent_delay_mean": np.nan,
        "recent_traversals": np.nan,
        "recent_deviation": np.nan,
        "recent_age_sec": np.nan,
    }
    if segment not in artifacts.lookup.index:
        return blank

    row = artifacts.lookup.loc[segment]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    # `completed_at` arrives tz-aware from some parquet writers and naive from others;
    # assuming either one crashes on the other.
    completed = pd.Timestamp(row["completed_at"])
    completed = (
        completed.tz_localize("UTC")
        if completed.tzinfo is None
        else completed.tz_convert("UTC")
    )
    age = (when - completed).total_seconds()
    # Same rule as batch: older than the rolling window means "no recent information",
    # not "the last thing we saw, whenever that was".
    if not np.isfinite(age) or age < 0 or age > ROLLING_MAX_AGE_SEC:
        return blank

    values = {k: float(row[k]) for k in blank if k in row}
    values["recent_age_sec"] = float(age)
    return {**blank, **values}


def predict_fn(payload: dict, artifacts: Artifacts) -> dict:
    """Predict, or return a structured error.

    **Client errors are returned in the body, not raised.** An unhandled `StationError`
    reaches the caller as `500 Internal Server Error` with an HTML page — no hint
    whether they mistyped a station or asked for a journey needing a transfer, and a
    500 wrongly invites a retry that cannot succeed. This container has no clean way
    to set a 4xx status, so the error travels in a 200 body instead and callers check
    for an `error` key. Losing the status code is a real cost, paid to keep the
    resolver's messages readable — refusing clearly was the point of building it.
    """
    try:
        return _predict(payload, artifacts)
    except StationError as error:
        logger.info("client error: %s", error)
        return {
            "error": str(error),
            "error_type": "station_resolution",
            "request": payload,
        }
    except (KeyError, ValueError) as error:
        logger.info("bad request: %s", error)
        return {"error": str(error), "error_type": "bad_request", "request": payload}


def _predict(payload: dict, artifacts: Artifacts) -> dict:
    for field in ("origin", "destination"):
        if not payload.get(field):
            raise ValueError(f"missing required field {field!r}")

    when = pd.Timestamp(payload.get("departure_ts") or pd.Timestamp.utcnow())
    when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
    local_hour = when.tz_convert(SERVICE_TZ).hour

    journey = resolve_journey(
        payload["origin"],
        payload["destination"],
        artifacts.index,
        artifacts.schedule,
        local_hour,
    )

    row: dict = {
        "n_segments": journey["n_segments"],
        "scheduled_total_sec": journey["scheduled_total_sec"],
        "stops_spanned": journey["stops_spanned"],
        "route_id": journey["route_id"],
        "direction_id": journey["direction_id"],
        "from_station": journey["origin_stop_id"][3:6],
        "to_station": journey["destination_stop_id"][3:6],
        **_calendar_features(when),
        **_recent_conditions(
            artifacts, (journey["origin_stop_id"], journey["first_leg_to"]), when
        ),
    }
    for feature in TRIP_STATE_FEATURES:
        row.setdefault(feature, np.nan)

    frame = pd.DataFrame([row])
    for column, codes in artifacts.encoder_mapping.items():
        if column in frame.columns:
            frame[column] = (
                frame[column].astype(str).map(codes).fillna(-1).astype("int32")
            )

    missing = [c for c in artifacts.columns if c not in frame.columns]
    if missing:
        raise ValueError(f"handler failed to produce required feature(s): {missing}")

    predicted = float(artifacts.booster.predict(frame[artifacts.columns])[0])

    warnings: list[str] = []
    if journey["n_segments"] > MAX_TRAINED_SEGMENTS:
        warnings.append(
            f"journey spans {journey['n_segments']} segments but the model was trained "
            f"to {MAX_TRAINED_SEGMENTS}; this prediction is extrapolated and unreliable"
        )
    if not np.isfinite(row.get("recent_deviation", np.nan)):
        warnings.append(
            "no traversal of the origin segment completed within the last hour; "
            "recent-conditions features are null, so this leans on the schedule"
        )

    return {
        "predicted_sec": round(predicted, 1),
        "predicted_min": round(predicted / 60, 2),
        "scheduled_sec": journey["scheduled_total_sec"],
        "n_segments": journey["n_segments"],
        "origin": journey["origin_station"],
        "destination": journey["destination_station"],
        "origin_stop_id": journey["origin_stop_id"],
        "destination_stop_id": journey["destination_stop_id"],
        "departure_ts": when.isoformat(),
        "model_run": artifacts.run_id,
        # Surfaced on every response, not buried in a manifest: a caller must be able to
        # see that the split behind this number is provisional.
        "trustworthy": artifacts.trustworthy,
        "warnings": warnings,
    }


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    return json.dumps(prediction)
