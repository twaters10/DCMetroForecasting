"""Score composed two-leg predictions against transfer journeys that really happened.

Serving answers a transfer journey by adding three estimates together:

    total = ride(A -> C1) + walk + wait for the onward train + ride(C2 -> B)

Adding predictions together is exactly where this project has been burned before.
Summing **per-segment** predictions into a journey compounded error — it grew as
n^0.663 against the baseline's n^0.558 — because consecutive segments on one train
share a delay, so their errors point the same way and never cancel. That is the reason
the journey model exists at all.

Two legs of a transfer are a different proposition: different trains, usually different
lines, boarded at different times. Their errors *should* be closer to independent. But
"should" is the same word that preceded the last surprise, so this module measures it
rather than assuming it.

## What it measures

Real transfer journeys are reconstructed from the journey table, which already contains
both legs as ordinary single-train rows: leg 1 arrives at the transfer station, and the
rider takes the first onward train departing after the walk. Realised total is the
second leg's arrival minus the first leg's departure — nothing modelled, purely
observed.

Against that, three errors are separated:

| error | question it answers |
| --- | --- |
| ride | is the model good at each leg on its own? |
| wait | does the timetable predict the connection the rider actually catches? |
| total | does the composition hold together? |

Whether the ride errors compound is read off `corr(leg1 error, leg2 error)` directly,
and by comparing the combined ride error against the two limits: `sqrt(e1^2 + e2^2)` if
the legs are independent, `|e1| + |e2|` if they move together.

Scoring runs on the **validation** side of the journey model's split, so no leg here was
trained on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.encode import CategoricalEncoder
from ..serving.routing import MIN_CONNECTION_BUFFER_SEC, TransferGraph, service_position
from ..serving.stations import StationIndex
from .config import JOURNEY_TARGET
from .train import build_matrix

logger = logging.getLogger("journeys.transfers")

DEFAULT_JOURNEYS = "data/processed/journeys/table"
DEFAULT_SERVING = "data/processed/serving"
DEFAULT_MODEL = "data/models/journey_duration/latest"

# Itineraries are chosen once per OD pair at a representative midday hour rather than
# re-routed per journey. The connecting station barely moves across the day; the wait
# does, and that is timed per journey against the real timetable.
ROUTING_HOUR = 12

# A reconstructed connection is only believable if the onward leg was actually observed
# soon after the rider was ready. The journey table holds journeys of a fixed set of
# lengths inside contiguous quality-filtered blocks, so the next row for a given OD pair
# is not always the next TRAIN — sometimes it is hours later, because the intervening
# runs did not survive into the table. Measured: 23% of naive reconstructions implied a
# wait over 30 minutes and 17% over two hours, which no Metro rider has ever
# experienced. Those are coverage holes, not connections, and counting them would make
# the wait error meaningless. Real late-night headways top out near 20 minutes.
MAX_OBSERVED_WAIT_SEC = 2700


@dataclass(frozen=True)
class Itinerary:
    origin: str
    transfer_in: str
    transfer_out: str
    destination: str
    station: str
    walk_sec: int
    route2: str
    direction2: int


def choose_itineraries(
    graph: TransferGraph,
    pairs: list[tuple[str, str]],
    depart_sec: int,
    services: list[str],
) -> list[Itinerary]:
    """Route each station-name pair the same way serving would."""
    chosen: list[Itinerary] = []
    for origin_name, destination_name in pairs:
        candidates = graph.candidates(origin_name, destination_name, ROUTING_HOUR)
        ranked = graph.rank(candidates, depart_sec, services)
        if not ranked:
            continue
        best = ranked[0][0]
        chosen.append(
            Itinerary(
                origin=best.origin_stop_id,
                transfer_in=best.transfer_in,
                transfer_out=best.transfer_out,
                destination=best.destination_stop_id,
                station=best.transfer_station,
                walk_sec=best.walk_sec,
                route2=str(best.leg2["route_id"]),
                direction2=int(best.leg2["direction_id"]),
            )
        )
    logger.info("routed %d of %d requested pair(s)", len(chosen), len(pairs))
    return chosen


def reconstruct(
    pool: dict[tuple[str, str], pd.DataFrame], itinerary: Itinerary
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Pair each first leg with the first onward train the rider could actually catch.

    The connection is made on **observed** times, so this is a record of what happened,
    not a simulation of it.
    """
    first = pool.get((itinerary.origin, itinerary.transfer_in))
    second = pool.get((itinerary.transfer_out, itinerary.destination))
    if first is None or second is None or first.empty or second.empty:
        return None

    first = first.sort_values("origin_departure_ts")
    second = second.sort_values("origin_departure_ts")

    arrival = first["origin_departure_ts"] + pd.to_timedelta(
        first[JOURNEY_TARGET], unit="s"
    )
    ready = arrival + pd.Timedelta(
        seconds=itinerary.walk_sec + MIN_CONNECTION_BUFFER_SEC
    )

    onward = second["origin_departure_ts"].to_numpy()
    position = np.searchsorted(onward, ready.to_numpy())
    caught = position < len(onward)
    if not caught.any():
        return None

    first_rows = first.loc[caught].reset_index(drop=True)
    second_rows = second.iloc[position[caught]].reset_index(drop=True)

    # A rider cannot "change" onto the train they are already on.
    different = ~(
        (first_rows["trip_id"].to_numpy() == second_rows["trip_id"].to_numpy())
        & (first_rows["trip_run"].to_numpy() == second_rows["trip_run"].to_numpy())
    )
    first_rows = first_rows.loc[different].reset_index(drop=True)
    second_rows = second_rows.loc[different].reset_index(drop=True)
    if first_rows.empty:
        return None

    depart = first_rows["origin_departure_ts"]
    arrive_transfer = depart + pd.to_timedelta(first_rows[JOURNEY_TARGET], unit="s")
    board = second_rows["origin_departure_ts"]
    arrive_final = board + pd.to_timedelta(second_rows[JOURNEY_TARGET], unit="s")

    return (
        pd.DataFrame(
            {
                "origin_stop_id": itinerary.origin,
                "transfer_in": itinerary.transfer_in,
                "transfer_out": itinerary.transfer_out,
                "destination_stop_id": itinerary.destination,
                "transfer_station": itinerary.station,
                "walk_sec": itinerary.walk_sec,
                "route2": itinerary.route2,
                "direction2": itinerary.direction2,
                "service_date": first_rows["service_date"].astype(str),
                "departure_ts": depart,
                "leg1_actual_sec": first_rows[JOURNEY_TARGET].to_numpy(),
                "leg2_actual_sec": second_rows[JOURNEY_TARGET].to_numpy(),
                "actual_wait_sec": (board - arrive_transfer)
                .dt.total_seconds()
                .to_numpy(),
                "actual_total_sec": (arrive_final - depart)
                .dt.total_seconds()
                .to_numpy(),
                "n_segments": (
                    first_rows["n_segments"].to_numpy()
                    + second_rows["n_segments"].to_numpy()
                ),
                "leg1_index": first_rows.index.to_numpy(),
                "leg2_index": second_rows.index.to_numpy(),
            }
        ),
        first_rows,
        second_rows,
    )


def predicted_wait(
    graph: TransferGraph,
    frame: pd.DataFrame,
    predicted_arrival: np.ndarray,
) -> np.ndarray:
    """The wait serving would have quoted, timed off the PREDICTED arrival."""
    waits = np.full(len(frame), np.nan)
    local = frame["departure_ts"].dt.tz_convert("America/New_York")
    for i, (timestamp, walk, stop, route, direction, arrival) in enumerate(
        zip(
            local,
            frame["walk_sec"],
            frame["transfer_out"],
            frame["route2"],
            frame["direction2"],
            predicted_arrival,
            strict=False,
        )
    ):
        date, depart_sec = service_position(timestamp)
        services = graph.services_on(date)
        if not services:
            continue
        ready = int(depart_sec + arrival + walk + MIN_CONNECTION_BUFFER_SEC)
        departure = graph.next_departure(stop, route, int(direction), ready, services)
        if departure is not None:
            waits[i] = departure - ready + MIN_CONNECTION_BUFFER_SEC
    return waits


def summarise(scored: pd.DataFrame, direct: pd.DataFrame | None) -> dict:
    """Decompose the composed error and test the independence assumption."""
    error1 = scored["leg1_pred_sec"] - scored["leg1_actual_sec"]
    error2 = scored["leg2_pred_sec"] - scored["leg2_actual_sec"]
    ride_error = error1 + error2
    wait_error = scored["pred_wait_sec"] - scored["actual_wait_sec"]
    total_error = scored["pred_total_sec"] - scored["actual_total_sec"]

    independent = float(np.sqrt(error1**2 + error2**2).mean())
    compounding = float((error1.abs() + error2.abs()).mean())
    observed = float(ride_error.abs().mean())

    report = {
        "journeys": int(len(scored)),
        "od_pairs": int(
            scored.groupby(["origin_stop_id", "destination_stop_id"]).ngroups
        ),
        "service_dates": sorted(scored["service_date"].unique().tolist()),
        "median_segments": float(scored["n_segments"].median()),
        "mae_total_sec": float(total_error.abs().mean()),
        "mae_ride_sec": observed,
        "mae_wait_sec": float(wait_error.abs().mean()),
        "median_actual_wait_sec": float(scored["actual_wait_sec"].median()),
        "median_pred_wait_sec": float(scored["pred_wait_sec"].median()),
        "mae_leg1_sec": float(error1.abs().mean()),
        "mae_leg2_sec": float(error2.abs().mean()),
        "leg_error_correlation": float(error1.corr(error2)),
        "ride_error_if_independent_sec": independent,
        "ride_error_if_compounding_sec": compounding,
        "bias_total_sec": float(total_error.mean()),
        "scheduled_mae_sec": float(
            (scored["scheduled_total_sec"] - scored["actual_total_sec"]).abs().mean()
        ),
    }
    # Where the observed combined ride error sits between the two limits. 0 means the
    # legs behaved as independent draws, 1 means their errors moved together and the
    # composition compounds exactly as summing per-segment predictions did.
    span = compounding - independent
    report["compounding_index"] = (
        float((observed - independent) / span) if span > 1e-9 else None
    )

    if direct is not None and len(direct):
        matched = []
        for segments, group in scored.groupby("n_segments"):
            near = direct[(direct["n_segments"] - segments).abs() <= 2]
            if len(near) < 200:
                continue
            matched.append(
                {
                    "n_segments": int(segments),
                    "transfer_journeys": int(len(group)),
                    "transfer_mae_sec": float(
                        (group["pred_total_sec"] - group["actual_total_sec"])
                        .abs()
                        .mean()
                    ),
                    "direct_mae_sec": float(near["abs_error"].mean()),
                    "direct_journeys": int(len(near)),
                }
            )
        report["length_matched"] = matched

    # Per station, conditioned on the onward line and direction. The 4-minute figure
    # measured early on counted ANY departure from the far platform, which flatters
    # every station served by several lines; a rider needs one specific line going one
    # specific way, and this is that number.
    by_station = (
        scored.groupby(["transfer_station", "route2", "direction2"])
        .agg(
            journeys=("actual_wait_sec", "size"),
            median_actual_wait_sec=("actual_wait_sec", "median"),
            p90_actual_wait_sec=("actual_wait_sec", lambda s: float(s.quantile(0.9))),
            median_pred_wait_sec=("pred_wait_sec", "median"),
            median_walk_sec=("walk_sec", "median"),
        )
        .reset_index()
        .sort_values("journeys", ascending=False)
    )
    report["wait_by_station"] = by_station.head(20).to_dict("records")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journeys", default=DEFAULT_JOURNEYS)
    parser.add_argument("--serving", default=DEFAULT_SERVING)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--pairs", type=int, default=400, help="station pairs to sample (0 = all)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-wait-sec",
        type=int,
        default=MAX_OBSERVED_WAIT_SEC,
        help="discard reconstructions whose onward leg was not observed within this",
    )
    parser.add_argument("--output", default=None, help="write the report as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    model_dir = Path(args.model)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    split = manifest["training_data"]["split"]
    boundary = pd.Timestamp(split["boundary_utc"]) + pd.Timedelta(
        seconds=int(split["embargo_sec"])
    )
    logger.info("model run %s | validation starts %s", manifest["run_id"], boundary)

    serving = Path(args.serving)
    index = StationIndex.from_json((serving / "station_index.json").read_text())
    schedule = pd.read_csv(serving / "journey_schedule.csv")
    graph = TransferGraph.from_directory(serving, index, schedule)

    journeys = pd.read_parquet(args.journeys)
    # The journey table stores departures tz-naive in UTC; the split boundary and the
    # service-day arithmetic below are both tz-aware. Reconciling here rather than at
    # each use keeps a naive/aware mix from silently shifting a connection by hours.
    departures = journeys["origin_departure_ts"]
    journeys["origin_departure_ts"] = (
        departures.dt.tz_localize("UTC")
        if departures.dt.tz is None
        else departures.dt.tz_convert("UTC")
    )
    journeys = journeys[journeys["origin_departure_ts"] >= boundary]
    logger.info("validation journeys: %d row(s)", len(journeys))

    names = sorted(set(index.code_to_name.values()))
    transfer_pairs = [
        (a, b) for a in names for b in names if a != b and not graph.is_direct(a, b)
    ]
    rng = np.random.default_rng(args.seed)
    if args.pairs and args.pairs < len(transfer_pairs):
        picked = rng.choice(len(transfer_pairs), size=args.pairs, replace=False)
        transfer_pairs = [transfer_pairs[i] for i in sorted(picked)]
    logger.info("sampled %d transfer pair(s)", len(transfer_pairs))

    reference = pd.Timestamp("2026-08-25 12:00:00", tz="America/New_York")
    date, depart_sec = service_position(reference)
    itineraries = choose_itineraries(
        graph, transfer_pairs, depart_sec, graph.services_on(date)
    )

    needed = {(i.origin, i.transfer_in) for i in itineraries} | {
        (i.transfer_out, i.destination) for i in itineraries
    }
    keys = list(
        zip(journeys["origin_stop_id"], journeys["destination_stop_id"], strict=False)
    )
    journeys = journeys.loc[[k in needed for k in keys]]
    logger.info("journeys on the legs of interest: %d row(s)", len(journeys))
    pool = {
        key: group
        for key, group in journeys.groupby(
            ["origin_stop_id", "destination_stop_id"], sort=False
        )
    }

    frames, leg1_frames, leg2_frames = [], [], []
    for itinerary in itineraries:
        built = reconstruct(pool, itinerary)
        if built is None:
            continue
        frame, first_rows, second_rows = built
        frames.append(frame)
        leg1_frames.append(first_rows)
        leg2_frames.append(second_rows)
    if not frames:
        logger.error("no transfer journeys could be reconstructed")
        return 1

    scored = pd.concat(frames, ignore_index=True)
    leg1 = pd.concat(leg1_frames, ignore_index=True)
    leg2 = pd.concat(leg2_frames, ignore_index=True)
    logger.info("reconstructed %d candidate transfer journey(s)", len(scored))

    believable = scored["actual_wait_sec"] <= args.max_wait_sec
    dropped = int((~believable).sum())
    coverage = {
        "candidates": int(len(scored)),
        "dropped_unobserved_connection": dropped,
        "dropped_pct": round(100 * dropped / max(len(scored), 1), 1),
        "max_wait_sec": args.max_wait_sec,
    }
    logger.info(
        "kept %d journey(s); dropped %d (%.1f%%) whose onward leg was not observed "
        "within %ds",
        int(believable.sum()),
        dropped,
        coverage["dropped_pct"],
        args.max_wait_sec,
    )
    scored = scored.loc[believable].reset_index(drop=True)
    leg1 = leg1.loc[believable.to_numpy()].reset_index(drop=True)
    leg2 = leg2.loc[believable.to_numpy()].reset_index(drop=True)
    if scored.empty:
        logger.error("no believable connections survived filtering")
        return 1

    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    encoder = CategoricalEncoder.load(model_dir / "encoder.json")
    columns = json.loads((model_dir / "feature_columns.json").read_text())
    scored["leg1_pred_sec"] = booster.predict(build_matrix(leg1, encoder, columns))
    scored["leg2_pred_sec"] = booster.predict(build_matrix(leg2, encoder, columns))
    scored["scheduled_total_sec"] = (
        leg1["scheduled_total_sec"].to_numpy()
        + leg2["scheduled_total_sec"].to_numpy()
        + scored["walk_sec"].to_numpy()
        + scored["actual_wait_sec"].to_numpy()
    )

    scored["pred_wait_sec"] = predicted_wait(
        graph, scored, scored["leg1_pred_sec"].to_numpy()
    )
    usable = scored["pred_wait_sec"].notna()
    if not usable.all():
        logger.warning(
            "%d journey(s) had no timetabled connection and are excluded",
            int((~usable).sum()),
        )
        scored = scored.loc[usable].reset_index(drop=True)

    scored["pred_total_sec"] = (
        scored["leg1_pred_sec"]
        + scored["walk_sec"]
        + scored["pred_wait_sec"]
        + scored["leg2_pred_sec"]
    )

    direct = None
    predictions_path = model_dir / "validation_predictions.parquet"
    if predictions_path.exists():
        direct = pd.read_parquet(predictions_path)
        if "abs_error" not in direct.columns:
            candidates = [c for c in direct.columns if "pred" in c.lower()]
            if candidates and JOURNEY_TARGET in direct.columns:
                direct["abs_error"] = (
                    direct[candidates[0]] - direct[JOURNEY_TARGET]
                ).abs()
            else:
                direct = None

    report = summarise(scored, direct)
    report["coverage"] = coverage
    print("\n" + json.dumps(report, indent=1))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=1))
        scored.to_parquet(output.with_suffix(".parquet"), index=False)
        logger.info("wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
