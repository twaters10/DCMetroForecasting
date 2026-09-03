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

import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# SageMaker loads `code/inference.py` as a TOP-LEVEL module, so a relative import fails
# inside the container even though it works everywhere else. Both forms are attempted so
# the same file runs under pytest, locally, and in the container without editing.
try:  # package context: tests and local use
    from .routing import NoItineraryError, TransferGraph, service_position
    from .stations import StationError, StationIndex, resolve_journey
except ImportError:  # flat context: code/ inside model.tar.gz
    from routing import (  # type: ignore
        NoItineraryError,
        TransferGraph,
        service_position,
    )
    from stations import StationError, StationIndex, resolve_journey  # type: ignore

logger = logging.getLogger("serving.inference")

ROLLING_MAX_AGE_SEC = 3600

# Below this many training examples at a given journey length, a prediction is answered
# from too little evidence to be trusted. Read against the per-length counts the model
# records in its own manifest, rather than a hardcoded length cutoff.
#
# A cutoff cannot express what the data actually looks like. Training covers lengths up
# to 32, but coverage is wildly uneven: n=20 has 51,807 journeys, n=24 has 21,773,
# n=28 has 7,664 — and n=32 has 118. "Trained on" and "supported" are not the same
# claim, and a single MAX_TRAINED_SEGMENTS would let the model answer at 32 as
# confidently as at 4.
MIN_TRAINING_SUPPORT = 1000

# Below this, a connection is worth flagging: the rider has under a minute on the
# platform, and a first leg running even slightly late turns the short wait into a
# full headway.
TIGHT_CONNECTION_SEC = 60

# How much faster a transfer must be before it is worth mentioning next to a one-train
# answer. Two minutes: below that the saving is inside the model's own error at these
# lengths, so "change trains to save 40 seconds" is advice the numbers cannot support.
ALTERNATIVE_MIN_SAVING_SEC = 120

# Fallback only, for a manifest written before per-length support was recorded.
FALLBACK_MAX_SEGMENTS = 17

# Where the collector Lambda writes the live recent-conditions table, refreshed every
# few minutes. Read per request rather than baked into model.tar.gz: the bundled copy is
# batch-built from COMPLETED service days and is hours old before it ships, so every
# entry falls outside the 3600s window and every `recent_*` feature is null.
LIVE_LOOKUP_KEY = "models/serving/recent_conditions_live.csv"

# A warm container should not re-download per request, but must never serve an hour-old
# view of "recent". 60s is well inside the staleness window and cheap at this traffic.
LIVE_LOOKUP_TTL_SEC = 60
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
        # The bundled lookup is a FALLBACK only. It ships with the model so the endpoint
        # can answer at all if S3 is unreachable, but it is batch-built and therefore
        # always stale — serving from it means null recent_* features.
        self.fallback_lookup = _indexed(
            pd.read_csv(
                directory / "recent_conditions_lookup.csv", parse_dates=["completed_at"]
            )
        )
        # Routing artifacts are optional so an older model.tar.gz still loads: without
        # them the endpoint simply refuses transfer journeys exactly as it did before.
        routing_files = ("walk_edges.csv", "departures.csv", "service_calendar.csv")
        if all((directory / name).exists() for name in routing_files):
            self.graph: TransferGraph | None = TransferGraph.from_directory(
                directory, self.index, self.schedule
            )
        else:
            logger.warning(
                "routing artifacts absent (%s) — transfer journeys will be refused",
                ", ".join(routing_files),
            )
            self.graph = None
        self._live_lookup: pd.DataFrame | None = None
        self._live_fetched_at: float = 0.0
        self.bucket = os.environ.get("S3_BUCKET", "")
        self.run_id = json.loads((directory / "manifest.json").read_text())["run_id"]

        # The "arrive by" companion. Same encoder and feature columns — both are trained
        # on the same journey table — so only the booster differs. Absent when no
        # quantile model exists, in which case the response omits it rather than
        # inventing one.
        quantile_path = directory / "model_p80.txt"
        self.quantile_booster = (
            lgb.Booster(model_file=str(quantile_path))
            if quantile_path.exists()
            else None
        )
        coverage_path = directory / "coverage_p80.json"
        self.quantile_coverage: dict[str, float] = (
            json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
        )
        manifest = json.loads((directory / "manifest.json").read_text())
        self.trustworthy = manifest["trustworthy"]
        # {"4": 78894, "17": 93870, "32": 118} — how many training journeys existed at
        # each length. Absent on manifests predating this, hence the fallback.
        self.training_support: dict[str, int] = (
            manifest.get("headline_metrics", {}).get("training_support") or {}
        )
        logger.info(
            "loaded run %s: %d features, %d OD pairs, %d lookup rows",
            self.run_id,
            len(self.columns),
            len(self.schedule),
            len(self.fallback_lookup),
        )


def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index(["from_stop_id", "to_stop_id"])


def live_lookup(artifacts: Artifacts) -> pd.DataFrame:
    """The freshest recent-conditions table available, cached for `LIVE_LOOKUP_TTL_SEC`.

    Falls back to the bundled copy on any failure — missing object, permissions, bad
    CSV. **Degrading to stale data means null `recent_*` features, which is exactly the
    endpoint's behaviour today and is safe.** Raising instead would turn a refresh
    problem into an outage, and the model handles missing features natively.
    """
    now = time.monotonic()
    if (
        artifacts._live_lookup is not None
        and now - artifacts._live_fetched_at < LIVE_LOOKUP_TTL_SEC
    ):
        return artifacts._live_lookup
    if not artifacts.bucket:
        return artifacts.fallback_lookup

    try:
        import boto3

        body = (
            boto3.client("s3")
            .get_object(Bucket=artifacts.bucket, Key=LIVE_LOOKUP_KEY)["Body"]
            .read()
        )
        frame = _indexed(pd.read_csv(io.BytesIO(body), parse_dates=["completed_at"]))
        artifacts._live_lookup = frame
        artifacts._live_fetched_at = now
        logger.info("refreshed live lookup: %d segment(s)", len(frame))
        return frame
    except Exception as error:  # noqa: BLE001 - degrade, never fail a prediction
        logger.warning("live lookup unavailable (%s); using the bundled copy", error)
        artifacts._live_lookup = artifacts.fallback_lookup
        artifacts._live_fetched_at = now
        return artifacts.fallback_lookup


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


def _bracketed(segments: int, table: dict[str, float]) -> float | None:
    """Look a per-length value up, interpolating between the lengths that were trained.

    Training covers a **discrete set** of lengths (1, 2, 3, 4, 6, 8, ... 32) while a
    request can be any length. An exact-key lookup therefore missed most journeys: it
    reported "no training journeys" for 26 segments, which sits between n=24 and n=28
    and predicts well precisely because it is bracketed.

    Returns the **weaker** of the two neighbours, which is the conservative reading for
    both things this is used for: training support (less evidence) and quantile coverage
    (less of the distribution captured).

    `None` means the length falls outside the trained range entirely. That is genuine
    extrapolation and a different claim from "interpolated between two known points".
    """
    if not table:
        return None
    lengths = sorted(int(k) for k in table)
    if segments > lengths[-1] or segments < lengths[0]:
        return None
    below = max(n for n in lengths if n <= segments)
    above = min(n for n in lengths if n >= segments)
    return min(float(table[str(below)]), float(table[str(above)]))


def _training_support(segments: int, support: dict[str, int]) -> int | None:
    """How much training evidence backs a journey of this length."""
    value = _bracketed(segments, support)
    return None if value is None else int(value)


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
    table = live_lookup(artifacts)
    if segment not in table.index:
        return blank

    row = table.loc[segment]
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


def _station_name(artifacts: Artifacts, stop_id: str) -> str:
    return artifacts.index.code_to_name[stop_id[3:6]]


def _feature_row(artifacts: Artifacts, journey: dict, when: pd.Timestamp) -> dict:
    """The one-row feature frame a single-train ride is predicted from.

    Shared by direct journeys and by each leg of a transfer, so a leg is fed to the
    model exactly as a whole journey would be — which is the only reason legs can be
    predicted at all without retraining.
    """
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
    return row


def _score(artifacts: Artifacts, row: dict) -> tuple[float, float | None]:
    """(median prediction, p80 prediction) for one feature row."""
    frame = pd.DataFrame([row])
    for column, codes in artifacts.encoder_mapping.items():
        if column in frame.columns:
            frame[column] = (
                frame[column].astype(str).map(codes).fillna(-1).astype("int32")
            )

    missing = [c for c in artifacts.columns if c not in frame.columns]
    if missing:
        raise ValueError(f"handler failed to produce required feature(s): {missing}")

    matrix = frame[artifacts.columns]
    predicted = float(artifacts.booster.predict(matrix)[0])
    upper = (
        float(artifacts.quantile_booster.predict(matrix)[0])
        if artifacts.quantile_booster is not None
        else None
    )

    # The two boosters are INDEPENDENT models — different objectives, and nothing makes
    # them agree. An "arrive by" earlier than the typical duration is incoherent on its
    # face, and it is what a user would see: the app renders them side by side as
    # "Budget for" and the trip time.
    #
    # Dropped rather than clamped. Clamping to the median produces a budget with no
    # slack in it, presented as though it had some, and its coverage would be around 50%
    # rather than the figure quoted beside it. Omitting `arrive_by_sec` is a path the
    # response already supports — it is what happens when no quantile model shipped at
    # all — so the median still answers, minus a number that cannot be characterised.
    #
    # Most likely where the p80 is already weakest: long journeys, where measured
    # coverage falls to 62%. Logged because it means the two models are out of step,
    # which is a packaging problem, not a per-request one.
    if upper is not None and upper < predicted:
        logger.warning(
            "quantile prediction %.1fs is below the median %.1fs — dropping the "
            "arrive-by estimate. The p80 and median models are out of step.",
            upper,
            predicted,
        )
        upper = None

    return predicted, upper


def _support_warnings(
    artifacts: Artifacts, segments: int, prefix: str = ""
) -> list[str]:
    """Warn when a journey length sits outside or thinly inside the trained range."""
    warnings: list[str] = []
    support = _training_support(segments, artifacts.training_support)
    if support is None:
        warnings.append(
            f"{prefix}journey spans {segments} segments, beyond anything the model was "
            "trained on; this prediction is extrapolated and unreliable"
        )
    elif support < MIN_TRAINING_SUPPORT:
        warnings.append(
            f"{prefix}only {support:,} comparable training journeys near {segments} "
            f"segments (under {MIN_TRAINING_SUPPORT:,}); treat this as weakly supported"
        )
    return warnings


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

    # Dispatch on whether one train connects the two stations, rather than on catching
    # the resolver's failure: `resolve_journey` also refuses genuinely ambiguous
    # platform pairs, and that refusal must survive rather than be rerouted around.
    if artifacts.graph is not None and not artifacts.graph.is_direct(
        payload["origin"], payload["destination"]
    ):
        return _predict_transfer(payload, artifacts, when, local_hour)

    return _predict_direct(payload, artifacts, when, local_hour)


def _predict_direct(
    payload: dict, artifacts: Artifacts, when: pd.Timestamp, local_hour: int
) -> dict:
    journey = resolve_journey(
        payload["origin"],
        payload["destination"],
        artifacts.index,
        artifacts.schedule,
        local_hour,
    )

    row = _feature_row(artifacts, journey, when)
    predicted, upper = _score(artifacts, row)

    segments = journey["n_segments"]
    warnings = _support_warnings(artifacts, segments)

    if not np.isfinite(row.get("recent_deviation", np.nan)):
        warnings.append(
            "no traversal of the origin segment completed within the last hour; "
            "recent-conditions features are null, so this leans on the schedule"
        )

    arrive_by: dict = {}
    if upper is not None:
        # Report the coverage this model ACHIEVED at this length, not the nominal 80%.
        # Measured: 80.2% at one segment falling to 62.3% at 28, because a single
        # quantile fitted across pooled lengths under-covers the long tail. Quoting the
        # nominal figure would overstate the guarantee exactly where it is weakest.
        achieved = _bracketed(segments, artifacts.quantile_coverage)
        arrive_by = {
            "arrive_by_sec": round(upper, 1),
            "arrive_by_min": round(upper / 60, 2),
            "arrive_by_coverage_pct": achieved,
        }
        if achieved is not None and achieved < 70:
            warnings.append(
                f"the arrive-by estimate covers only {achieved:.0f}% of journeys at "
                f"{segments} segments, not the nominal 80%"
            )

    result = {
        "predicted_sec": round(predicted, 1),
        **arrive_by,
        "predicted_min": round(predicted / 60, 2),
        # A direct journey is entirely riding — no platform change, no connection to
        # wait for. The zeros are reported rather than omitted so that every response
        # carries the same breakdown and a caller never has to special-case which kind
        # of journey came back.
        "ride_sec": round(predicted, 1),
        "ride_min": round(predicted / 60, 2),
        "walk_sec": 0,
        "walk_min": 0.0,
        "wait_sec": 0,
        "wait_min": 0.0,
        "scheduled_sec": journey["scheduled_total_sec"],
        "n_segments": journey["n_segments"],
        # Which train to actually board. Absent until now because a direct journey has
        # exactly one, so it felt implied — but the map has to draw it, and a rider
        # standing on a platform has the same question.
        "line": journey["route_id"],
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

    # Attached only when a transfer genuinely beats this ride. The key is ABSENT rather
    # than null otherwise, so a direct answer with no better option is unchanged from
    # before this existed.
    alternative = _faster_alternative(payload, artifacts, when, local_hour, result)
    if alternative is not None:
        result["alternative"] = alternative
        result["warnings"] = [
            *warnings,
            f"a change at {alternative['transfer_station']} is about "
            f"{alternative['saving_min']:.0f} min faster than staying on this train",
        ]
    return result


def _leg_journey(origin_stop_id: str, destination_stop_id: str, leg: dict) -> dict:
    """Adapt a scheduled leg into the shape `_feature_row` expects."""
    return {
        "origin_stop_id": origin_stop_id,
        "destination_stop_id": destination_stop_id,
        "n_segments": int(leg["n_segments"]),
        "scheduled_total_sec": int(leg["sched_sec"]),
        "stops_spanned": int(leg["stop_span"]),
        "route_id": leg["route_id"],
        "direction_id": int(leg["direction_id"]),
        "first_leg_to": leg["first_leg_to"],
    }


def _score_itinerary(
    artifacts: Artifacts,
    candidate,
    when: pd.Timestamp,
    depart_sec: int,
    services: list[str],
) -> dict | None:
    """Predict both legs and time the connection between them.

    The connection is timed against the **predicted** arrival, not the scheduled one.
    That is the point of doing it this way round: if leg 1 is running late the model
    says so, and the rider is shown the connection they will actually make rather than
    the one the timetable promises.
    """
    graph = artifacts.graph
    leg1 = _leg_journey(candidate.origin_stop_id, candidate.transfer_in, candidate.leg1)
    row1 = _feature_row(artifacts, leg1, when)
    ride1, upper1 = _score(artifacts, row1)

    arrival_sec = depart_sec + int(round(ride1))
    timing = graph.connection(candidate, arrival_sec, services)
    if timing is None:
        return None

    # Leg 2 is predicted at the moment it is actually boarded. A long first leg can
    # cross a fare period or a rush-hour boundary, and scoring it at the request's
    # departure time would use the wrong hour-of-day features.
    board = when + pd.Timedelta(int(timing["departure_sec"] - depart_sec), unit="s")
    board_hour = board.tz_convert(SERVICE_TZ).hour
    leg2_schedule = (
        graph.leg(candidate.transfer_out, candidate.destination_stop_id, board_hour)
        or candidate.leg2
    )
    leg2 = _leg_journey(
        candidate.transfer_out, candidate.destination_stop_id, leg2_schedule
    )
    row2 = _feature_row(artifacts, leg2, board)
    ride2, upper2 = _score(artifacts, row2)

    walk = candidate.walk_sec
    wait = timing["wait_sec"]
    total = ride1 + walk + wait + ride2
    upper = None if upper1 is None or upper2 is None else upper1 + walk + wait + upper2
    return {
        "candidate": candidate,
        "timing": timing,
        "ride_sec": ride1 + ride2,
        "walk_sec": walk,
        "wait_sec": wait,
        "row1": row1,
        "row2": row2,
        "ride1": ride1,
        "ride2": ride2,
        "leg1": leg1,
        "leg2": leg2,
        "board": board,
        "total_sec": total,
        "arrive_by_sec": upper,
        "scheduled_sec": int(
            leg1["scheduled_total_sec"] + walk + wait + leg2["scheduled_total_sec"]
        ),
    }


def _itinerary_legs(artifacts: Artifacts, best: dict) -> list[dict]:
    """Ride / transfer / ride, in the order they are travelled."""
    candidate = best["candidate"]
    timing = best["timing"]
    return [
        {
            "type": "ride",
            "from": _station_name(artifacts, candidate.origin_stop_id),
            "to": _station_name(artifacts, candidate.transfer_in),
            "from_stop_id": candidate.origin_stop_id,
            "to_stop_id": candidate.transfer_in,
            "line": best["leg1"]["route_id"],
            "predicted_sec": round(best["ride1"], 1),
            "scheduled_sec": best["leg1"]["scheduled_total_sec"],
            "n_segments": best["leg1"]["n_segments"],
        },
        {
            "type": "transfer",
            "at": candidate.transfer_station,
            "from_stop_id": candidate.transfer_in,
            "to_stop_id": candidate.transfer_out,
            "walk_sec": candidate.walk_sec,
            "wait_sec": timing["wait_sec"],
            "if_missed_sec": timing["if_missed_sec"],
        },
        {
            "type": "ride",
            "from": _station_name(artifacts, candidate.transfer_out),
            "to": _station_name(artifacts, candidate.destination_stop_id),
            "from_stop_id": candidate.transfer_out,
            "to_stop_id": candidate.destination_stop_id,
            "line": best["leg2"]["route_id"],
            "predicted_sec": round(best["ride2"], 1),
            "scheduled_sec": best["leg2"]["scheduled_total_sec"],
            "n_segments": best["leg2"]["n_segments"],
            "boards_at": best["board"].isoformat(),
        },
    ]


def _faster_alternative(
    payload: dict,
    artifacts: Artifacts,
    when: pd.Timestamp,
    local_hour: int,
    direct: dict,
) -> dict | None:
    """A transfer worth mentioning beside a one-train answer, or None.

    **A direct ride is not automatically the quickest one.** The Blue line reaches
    Eastern Market from Potomac Yard with no change at all, but it travels via
    Rosslyn: 16 segments and 30 minutes, against 8 segments and 23.5 for Yellow to
    L'Enfant Plaza and a change there. Answering "there is a direct train" is true and
    unhelpful.

    Measured over the whole network, a transfer beats the direct ride on 165 of 3,220
    connected pairs — 110 of them by more than five minutes — concentrated in the
    Virginia Blue-line stations where that Rosslyn detour costs the most.

    The direct route stays the answer. This only adds the alternative next to it, so a
    rider who would rather stay seated is not overruled by two minutes of arithmetic.
    """
    graph = artifacts.graph
    if graph is None:
        return None

    local = when.tz_convert(SERVICE_TZ)
    service_date, depart_sec = service_position(local)
    services = graph.services_on(service_date)
    if not services:
        return None

    try:
        candidates = graph.candidates(
            payload["origin"], payload["destination"], local_hour
        )
    except StationError:
        # An ambiguous or unknown name is the direct answer's problem to report, not a
        # reason to fail the whole response over an optional extra.
        return None
    if not candidates:
        return None

    # Prune on the TIMETABLE before spending any model calls. Scoring an itinerary
    # costs two predictions, and on 95% of pairs no transfer is close to competitive.
    shortlist = [
        (candidate, timing)
        for candidate, timing in graph.rank(candidates, depart_sec, services)
        if (
            int(candidate.leg1["sched_sec"])
            + candidate.walk_sec
            + timing["wait_sec"]
            + int(candidate.leg2["sched_sec"])
        )
        <= direct["scheduled_sec"] - ALTERNATIVE_MIN_SAVING_SEC
    ]
    if not shortlist:
        return None

    scored = [
        result
        for result in (
            _score_itinerary(artifacts, candidate, when, depart_sec, services)
            for candidate, _ in shortlist
        )
        if result is not None
    ]
    if not scored:
        return None
    best = min(scored, key=lambda result: result["total_sec"])

    # Re-checked against the PREDICTION, not the timetable that got it shortlisted. A
    # route the schedule likes can lose once the model has seen how the lines actually
    # run, and recommending it then would be worse than saying nothing.
    saving = direct["predicted_sec"] - best["total_sec"]
    if saving < ALTERNATIVE_MIN_SAVING_SEC:
        return None

    candidate = best["candidate"]
    return {
        "reason": "faster with one train change",
        "saving_sec": round(saving, 1),
        "saving_min": round(saving / 60, 2),
        "predicted_sec": round(best["total_sec"], 1),
        "predicted_min": round(best["total_sec"] / 60, 2),
        "ride_sec": round(best["ride_sec"], 1),
        "ride_min": round(best["ride_sec"] / 60, 2),
        "walk_sec": best["walk_sec"],
        "wait_sec": best["wait_sec"],
        "scheduled_sec": best["scheduled_sec"],
        "n_segments": best["leg1"]["n_segments"] + best["leg2"]["n_segments"],
        "transfers": 1,
        "transfer_station": candidate.transfer_station,
        "legs": _itinerary_legs(artifacts, best),
    }


def _predict_transfer(
    payload: dict, artifacts: Artifacts, when: pd.Timestamp, local_hour: int
) -> dict:
    """Compose a two-leg answer for a journey no single train covers."""
    graph = artifacts.graph
    origin_name = payload["origin"]
    destination_name = payload["destination"]

    local = when.tz_convert(SERVICE_TZ)
    service_date, depart_sec = service_position(local)
    services = graph.services_on(service_date)
    if not services:
        raise NoItineraryError(f"no scheduled service on {service_date}")

    candidates = graph.candidates(origin_name, destination_name, local_hour)
    if not candidates:
        raise NoItineraryError(
            f"no route from {origin_name!r} to {destination_name!r}, with or without "
            "a train change"
        )

    shortlist = graph.rank(candidates, depart_sec, services)
    if not shortlist:
        raise NoItineraryError(
            f"no connection from {origin_name!r} to {destination_name!r} is scheduled "
            f"late enough on {service_date}; service has ended for the night"
        )

    scored = [
        result
        for result in (
            _score_itinerary(artifacts, candidate, when, depart_sec, services)
            for candidate, _ in shortlist
        )
        if result is not None
    ]
    if not scored:
        raise NoItineraryError(
            f"no connection could be timed from {origin_name!r} to "
            f"{destination_name!r} at {when.isoformat()}"
        )
    best = min(scored, key=lambda result: result["total_sec"])

    candidate = best["candidate"]
    timing = best["timing"]
    warnings: list[str] = []
    for label, row, leg in (
        ("first leg: ", best["row1"], best["leg1"]),
        ("second leg: ", best["row2"], best["leg2"]),
    ):
        warnings.extend(_support_warnings(artifacts, leg["n_segments"], label))
        if not np.isfinite(row.get("recent_deviation", np.nan)):
            warnings.append(
                f"{label}no traversal of the origin segment completed within the last "
                "hour; recent-conditions features are null, so this leans on the "
                "schedule"
            )

    if (
        timing["if_missed_sec"] is not None
        and timing["wait_sec"] < TIGHT_CONNECTION_SEC
    ):
        warnings.append(
            f"tight connection: {timing['wait_sec']}s to change at "
            f"{candidate.transfer_station}. Missing it costs "
            f"{timing['if_missed_sec'] / 60:.0f} min waiting for the next train"
        )

    warnings.append(
        "the connection is timed from the published timetable, so it assumes the "
        "onward train departs on schedule"
    )

    arrive_by: dict = {}
    if best["arrive_by_sec"] is not None:
        arrive_by = {
            "arrive_by_sec": round(best["arrive_by_sec"], 1),
            "arrive_by_min": round(best["arrive_by_sec"] / 60, 2),
            # Deliberately no coverage figure. Adding two p80 legs does NOT give the
            # p80 of the total — it is more conservative than that, by an amount that
            # depends on how correlated the legs are. Quoting 80% here would be a
            # number this arithmetic does not support.
            "arrive_by_basis": "sum of per-leg p80 estimates; conservative, not a "
            "calibrated 80th percentile of the total",
        }

    total = best["total_sec"]
    ride = best["ride_sec"]
    walk = best["walk_sec"]
    wait = best["wait_sec"]
    legs = _itinerary_legs(artifacts, best)

    return {
        "predicted_sec": round(total, 1),
        **arrive_by,
        "predicted_min": round(total / 60, 2),
        # The four numbers a rider actually reasons about. `ride_sec` is time on
        # trains only: it is the part the model predicts, and the part that does not
        # change if they miss the connection.
        "ride_sec": round(ride, 1),
        "ride_min": round(ride / 60, 2),
        "walk_sec": walk,
        "walk_min": round(walk / 60, 2),
        "wait_sec": wait,
        "wait_min": round(wait / 60, 2),
        "scheduled_sec": best["scheduled_sec"],
        "n_segments": best["leg1"]["n_segments"] + best["leg2"]["n_segments"],
        "transfers": 1,
        "transfer_station": candidate.transfer_station,
        "legs": legs,
        "origin": _station_name(artifacts, candidate.origin_stop_id),
        "destination": _station_name(artifacts, candidate.destination_stop_id),
        "origin_stop_id": candidate.origin_stop_id,
        "destination_stop_id": candidate.destination_stop_id,
        "departure_ts": when.isoformat(),
        "service_date": service_date,
        "model_run": artifacts.run_id,
        "trustworthy": artifacts.trustworthy,
        "warnings": warnings,
    }


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    return json.dumps(prediction)
