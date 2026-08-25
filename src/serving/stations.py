"""Station names in, platform codes and a scheduled duration out.

The model works in platform ids (`PF_A08_C`); a user interface works in names
("Foggy Bottom"). This module is the adapter, and it is built now rather than later for
one reason: feature internals can change freely afterwards, but the endpoint's public
request shape cannot once anything integrates with it.

## Names come from `parent_station`, not `stop_name`

`stop_name` on a rail platform is platform-level — `"Wiehle-Reston East, Silver Line
Center Platform"` — so parsing it into a station name means stripping a line-and-track
suffix that varies. GTFS already models this properly: every rail platform carries a
`parent_station` (100% populated, verified), and the parent row is the station with a
clean name. Measured on the archived bundle: 125 platforms -> 102 station codes -> 98
clean names, with zero disagreement inside a code.

## Four names are genuinely ambiguous

Fort Totten, Gallery Place, L'Enfant Plaza and Metro Center are transfer stations where
two lines meet at separate platforms, so each maps to **two** station codes. They are
resolved by asking which pairing a scheduled trip actually connects — if only one
origin/destination platform pairing is served by a real trip, that is the answer.

Where that is still ambiguous, or where no trip connects the two, the caller gets an
explicit error naming the candidates. **A wrong platform is a wrong answer delivered
confidently**, which is worse than a refusal.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import zipfile
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

logger = logging.getLogger("serving.stations")

RAIL_PLATFORM_PREFIX = "PF_"
STATION_PATTERN = re.compile(r"^PF_([A-Z]\d{2})")

# Sentinel hour meaning "no hour-specific schedule known, use the all-day median".
ANY_HOUR = -1


class StationError(ValueError):
    """Raised for a name that cannot be resolved to exactly one platform."""


def normalise(name: str) -> str:
    """Casefold and strip punctuation so 'L'Enfant Plaza' matches "l enfant plaza"."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).casefold()).strip()


# NOTE: no `slots=True`. It requires Python 3.10 and the SageMaker sklearn 1.2-1
# container runs 3.9, where it raises at import — the container starts, the endpoint
# reaches InService, and every request fails. This class ships inside model.tar.gz, so
# it must stay 3.9-compatible even though the rest of the project targets 3.12.
@dataclass(frozen=True)
class StationIndex:
    """name -> station codes, and station code -> platforms."""

    names_to_codes: dict[str, list[str]]
    code_to_name: dict[str, str]
    code_to_platforms: dict[str, list[str]]

    def candidates(self, name: str) -> list[str]:
        """Station codes a name could mean. Raises with suggestions if unknown."""
        key = normalise(name)
        if key in self.names_to_codes:
            return self.names_to_codes[key]
        near = get_close_matches(key, self.names_to_codes, n=3, cutoff=0.6)
        suggestions = sorted(
            {self.code_to_name[self.names_to_codes[n][0]] for n in near}
        )
        raise StationError(
            f"unknown station {name!r}"
            + (f" — did you mean {', '.join(suggestions)}?" if suggestions else "")
        )

    def platforms(self, name: str) -> list[str]:
        return [
            p for code in self.candidates(name) for p in self.code_to_platforms[code]
        ]

    def to_json(self) -> str:
        return json.dumps(
            {
                "names_to_codes": self.names_to_codes,
                "code_to_name": self.code_to_name,
                "code_to_platforms": self.code_to_platforms,
            },
            indent=1,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> StationIndex:
        raw = json.loads(text)
        return cls(
            names_to_codes=raw["names_to_codes"],
            code_to_name=raw["code_to_name"],
            code_to_platforms=raw["code_to_platforms"],
        )


def build_station_index(stops: pd.DataFrame) -> StationIndex:
    """Build the name index from `stops.txt`, resolving names via `parent_station`."""
    rail = stops[
        stops["stop_id"].astype(str).str.startswith(RAIL_PLATFORM_PREFIX)
    ].copy()
    rail["station"] = rail["stop_id"].str.extract(STATION_PATTERN)
    rail = rail[rail["station"].notna()]

    parents = stops.set_index("stop_id")["stop_name"]
    rail["station_name"] = rail["parent_station"].map(parents)
    # Fall back to the platform name only if a parent is missing; on the verified
    # bundle this never fires, but a silent NaN here would poison the whole index.
    missing = rail["station_name"].isna()
    if missing.any():
        logger.warning(
            "%d platform(s) without a parent station name", int(missing.sum())
        )
        rail.loc[missing, "station_name"] = rail.loc[missing, "stop_name"]

    code_to_name = (
        rail.groupby("station")["station_name"].agg(lambda s: s.mode().iat[0]).to_dict()
    )
    code_to_platforms = (
        rail.groupby("station")["stop_id"].agg(lambda s: sorted(s.unique())).to_dict()
    )

    names_to_codes: dict[str, list[str]] = {}
    for code, name in code_to_name.items():
        names_to_codes.setdefault(normalise(name), []).append(code)
    names_to_codes = {k: sorted(v) for k, v in names_to_codes.items()}

    ambiguous = {k: v for k, v in names_to_codes.items() if len(v) > 1}
    logger.info(
        "%d platforms -> %d station codes -> %d names (%d ambiguous: %s)",
        len(rail),
        len(code_to_name),
        len(names_to_codes),
        len(ambiguous),
        ", ".join(sorted(code_to_name[v[0]] for v in ambiguous.values())),
    )
    return StationIndex(names_to_codes, code_to_name, code_to_platforms)


def build_journey_schedule(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Every (origin, destination) pair a scheduled trip connects, and its planned time.

    Keyed by departure hour as well as the pair. Measured on the archived bundle, only
    43% of pairs have an identical scheduled duration across every trip and the p95
    spread is 420s, so an all-day median alone would misstate the pairs that vary most.
    An `ANY_HOUR` row is emitted per pair as the fallback.
    """
    ordered = stop_times.sort_values(["static_trip_id", "stop_sequence"])
    rows: list[tuple] = []
    for _, group in ordered.groupby("static_trip_id", sort=False):
        ids = group["stop_id"].to_numpy()
        offsets = group["scheduled_arrival_offset_sec"].to_numpy()
        sequences = group["stop_sequence"].to_numpy()
        route = group["route_id"].iat[0]
        direction = group["direction_id"].iat[0]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                rows.append(
                    (
                        ids[i],
                        ids[j],
                        (offsets[i] // 3600) % 24,
                        offsets[j] - offsets[i],
                        j - i,
                        sequences[j] - sequences[i],
                        route,
                        direction,
                        # The stop immediately after the origin. The recent-conditions
                        # lookup is keyed on a SEGMENT, so serving needs the journey's
                        # first leg, not its endpoints.
                        ids[i + 1],
                    )
                )
    pairs = pd.DataFrame(
        rows,
        columns=[
            "origin",
            "destination",
            "hour",
            "sched_sec",
            "n_segments",
            "stop_span",
            "route_id",
            "direction_id",
            "first_leg_to",
        ],
    )

    def summarise(
        frame: pd.DataFrame, keys: list[str], hour: int | None
    ) -> pd.DataFrame:
        out = frame.groupby(keys, as_index=False).agg(
            sched_sec=("sched_sec", "median"),
            n_segments=("n_segments", "median"),
            stop_span=("stop_span", "median"),
            trips=("sched_sec", "size"),
            # Modal rather than first: a platform pair served by more than one route
            # should report the one that actually runs it most.
            route_id=("route_id", lambda s: s.mode().iat[0]),
            direction_id=("direction_id", lambda s: s.mode().iat[0]),
            first_leg_to=("first_leg_to", lambda s: s.mode().iat[0]),
        )
        if hour is not None:
            out["hour"] = hour
        return out

    by_hour = summarise(pairs, ["origin", "destination", "hour"], None)
    fallback = summarise(pairs, ["origin", "destination"], ANY_HOUR)
    schedule = pd.concat([by_hour, fallback], ignore_index=True)
    for column in ("sched_sec", "n_segments", "stop_span"):
        schedule[column] = schedule[column].round().astype("int32")

    logger.info(
        "%d OD pair(s), %d rows including hourly variants",
        fallback.shape[0],
        len(schedule),
    )
    return schedule


def resolve_journey(
    origin_name: str,
    destination_name: str,
    index: StationIndex,
    schedule: pd.DataFrame,
    hour: int | None = None,
) -> dict:
    """Resolve two names to the one platform pair a scheduled trip actually connects."""
    origins = index.platforms(origin_name)
    destinations = index.platforms(destination_name)

    connected = schedule[
        schedule["origin"].isin(origins) & schedule["destination"].isin(destinations)
    ]
    if connected.empty:
        raise StationError(
            f"no scheduled trip runs from {origin_name!r} to {destination_name!r} — "
            "they may be on different lines, or in the opposite direction"
        )

    pairs = connected[["origin", "destination"]].drop_duplicates()
    if len(pairs) > 1:
        # Both platforms of a transfer station are served, and the request is genuinely
        # ambiguous. Refuse rather than guess.
        options = ", ".join(f"{o} -> {d}" for o, d in pairs.itertuples(index=False))
        raise StationError(
            f"{origin_name!r} to {destination_name!r} is ambiguous across platforms "
            f"({options}); specify platform ids directly"
        )

    origin, destination = pairs.iloc[0]
    exact = connected[
        (connected["origin"] == origin)
        & (connected["destination"] == destination)
        & (connected["hour"] == (ANY_HOUR if hour is None else hour))
    ]
    if exact.empty:
        exact = connected[connected["hour"] == ANY_HOUR]
    row = exact.iloc[0]

    return {
        "origin_stop_id": origin,
        "destination_stop_id": destination,
        "origin_station": index.code_to_name[STATION_PATTERN.match(origin).group(1)],
        "destination_station": index.code_to_name[
            STATION_PATTERN.match(destination).group(1)
        ],
        "scheduled_total_sec": int(row["sched_sec"]),
        "n_segments": int(row["n_segments"]),
        "stops_spanned": int(row["stop_span"]),
        "route_id": row["route_id"],
        "direction_id": int(row["direction_id"]),
        "first_leg_to": row["first_leg_to"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", default=None, help="static GTFS zip; default newest"
    )
    parser.add_argument("--stop-times", default=None, help="cached stop_times.parquet")
    parser.add_argument("--output", default="data/processed/serving")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    bundle = (
        Path(args.bundle)
        if args.bundle
        else sorted(Path("data/cache/static").rglob("*.zip"))[-1]
    )
    with zipfile.ZipFile(bundle) as archive:
        stops = pd.read_csv(io.BytesIO(archive.read("stops.txt")))
    logger.info("bundle %s", bundle.name)

    stop_times_path = (
        Path(args.stop_times)
        if args.stop_times
        else sorted(Path("data/cache/schedule/rail").glob("*/stop_times.parquet"))[-1]
    )
    stop_times = pd.read_parquet(stop_times_path)
    logger.info("stop_times %s (%d rows)", stop_times_path.parent.name, len(stop_times))

    index = build_station_index(stops)
    schedule = build_journey_schedule(stop_times)

    # Imported here, not at module scope: `routing` imports this module, so a top-level
    # import would be circular.
    from .routing import (
        build_departures,
        build_service_calendar,
        build_walk_edges,
        read_bundle_tables,
    )

    tables = read_bundle_tables(bundle)
    walk_edges = build_walk_edges(tables["pathways.txt"], index)
    departures = build_departures(stop_times, tables["trips.txt"], index)
    calendar = build_service_calendar(tables["calendar_dates.txt"])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "station_index.json").write_text(index.to_json())
    # CSV, not Parquet, deliberately. The only reason the serving container would need
    # pyarrow is to read this file — and pyarrow is 126 MB installed against LightGBM's
    # 5.5 MB, all of it downloaded and installed at every serverless cold start. A ~5 MB
    # CSV that pandas reads natively removes the dependency entirely.
    schedule.to_csv(output / "journey_schedule.csv", index=False)
    walk_edges.to_csv(output / "walk_edges.csv", index=False)
    departures.to_csv(output / "departures.csv", index=False)
    calendar.to_csv(output / "service_calendar.csv", index=False)
    logger.info(
        "wrote station_index.json, journey_schedule.csv and 3 routing "
        "artifact(s) to %s",
        output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
