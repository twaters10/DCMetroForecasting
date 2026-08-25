"""Journeys that need a train change: routing, walk cost and connection wait.

The journey model predicts a **single-train** ride, so the endpoint could only answer
35.7% of the network — the other 64.3% of station pairs got an honest refusal from
`stations.resolve_journey`. This module supplies the missing pieces so those pairs can
be answered by composing two model calls:

    total = ride(A -> C1) + walk(C1 -> C2) + wait for the next train + ride(C2 -> B)

Nothing here retrains anything. Each leg is an ordinary single-train journey the model
already handles well; the new work is picking C, costing the walk, and timing the
connection.

## One transfer is always enough

Measured over all 9,506 ordered station-name pairs: 3,396 direct, 6,110 with exactly one
transfer, **zero** needing two. So the search is a single intermediate station, not a
general shortest-path problem, and `MAX_TRANSFERS` is a fact about this network rather
than a tuning knob.

## Walk cost comes from `pathways.txt`

Every one of the 60 same-station platform pairs is reachable through the GTFS pathway
graph, so the walk is summed along real corridors rather than assumed. Measured: median
152s, max 217s (Reagan National). A single hardcoded constant would be wrong at both
ends, and the wrong direction of wrong at the stations where changing trains is slowest.

Note that C1 and C2 are frequently the **same platform** — Blue, Orange and Silver share
the island at Metro Center (`PF_C01_C`), so changing between them costs no walk at all.
That is a zero-cost walk edge, not a missing one.

## Service patterns are not optional

`stop_times` covers 14 `service_id`s but only **two run on any given date**, and they
partition by route: one covers Red, the other covers the remaining five lines. Pooling
all 14 would show ~5.7x more trains than actually run and make every connection look far
easier to catch than it is. Departures are therefore filtered to the services active on
the requested date, via `calendar_dates.txt`.
"""

from __future__ import annotations

import bisect
import io
import logging
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:  # inside the SageMaker container these are siblings, not a package
    from .stations import ANY_HOUR, STATION_PATTERN, StationError, StationIndex
except ImportError:  # pragma: no cover - exercised only in the container
    from stations import ANY_HOUR, STATION_PATTERN, StationError, StationIndex

logger = logging.getLogger("serving.routing")

# Measured, not chosen: every ordered station pair on this network is reachable within
# one transfer. See the module docstring.
MAX_TRANSFERS = 1

# How many candidate itineraries get a model call. Candidates are ranked by their fully
# scheduled total (including the scheduled connection wait) and the cheapest few are
# scored for real; the rest cannot plausibly win.
MAX_CANDIDATES_SCORED = 3

SECONDS_PER_DAY = 86_400

# GTFS expresses times as an offset from noon-minus-12h, so a trip running past midnight
# carries an offset above 86400 (this feed reaches 27.5h). A request just after midnight
# therefore belongs to the PREVIOUS service date.
SERVICE_DAY_ROLLOVER_HOUR = 3

# A connection needs a moment on the platform beyond the walk itself. Without it a train
# departing the same second the rider arrives counts as catchable, which it is not.
MIN_CONNECTION_BUFFER_SEC = 30


class NoItineraryError(StationError):
    """Raised when no route exists between two stations, even with a transfer."""


# ---------------------------------------------------------------------------
# Artifact builders (run offline, from the archived GTFS bundle)
# ---------------------------------------------------------------------------


def build_walk_edges(pathways: pd.DataFrame, index: StationIndex) -> pd.DataFrame:
    """Shortest walking time between every pair of platforms at the same station.

    Dijkstra over the GTFS pathway graph, whose nodes are platforms, mezzanines,
    entrances and generic corridor nodes. Only platform-to-platform pairs within one
    station survive into the output — an entrance is not a place anyone changes trains.
    """
    graph: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in pathways.itertuples(index=False):
        seconds = int(row.traversal_time)
        graph[str(row.from_stop_id)].append((str(row.to_stop_id), seconds))
        if int(row.is_bidirectional) == 1:
            graph[str(row.to_stop_id)].append((str(row.from_stop_id), seconds))

    # Platforms grouped by station NAME rather than code. Fort Totten is two codes
    # (B06 Red, E06 Green/Yellow) and is exactly the case that matters.
    by_name: dict[str, list[str]] = defaultdict(list)
    for code, name in index.code_to_name.items():
        by_name[name].extend(index.code_to_platforms[code])

    import heapq

    rows: list[tuple[str, str, int]] = []
    unreachable = 0
    for platforms in by_name.values():
        if len(platforms) < 2:
            continue
        targets = set(platforms)
        for source in platforms:
            distance = {source: 0}
            queue = [(0, source)]
            while queue:
                cost, node = heapq.heappop(queue)
                if cost > distance.get(node, 1 << 30):
                    continue
                for neighbour, weight in graph.get(node, ()):
                    step = cost + weight
                    if step < distance.get(neighbour, 1 << 30):
                        distance[neighbour] = step
                        heapq.heappush(queue, (step, neighbour))
            for target in targets:
                if target == source:
                    continue
                if target in distance:
                    rows.append((source, target, distance[target]))
                else:
                    unreachable += 1

    edges = pd.DataFrame(rows, columns=["from_stop_id", "to_stop_id", "walk_sec"])
    if unreachable:
        # Not fatal — the pair simply cannot be used as a transfer — but it means the
        # pathway graph is incomplete, which is worth saying out loud.
        logger.warning("%d same-station platform pair(s) unreachable", unreachable)
    logger.info(
        "walk edges: %d pair(s), median %.0fs, max %.0fs",
        len(edges),
        edges["walk_sec"].median() if len(edges) else 0,
        edges["walk_sec"].max() if len(edges) else 0,
    )
    return edges.sort_values(["from_stop_id", "to_stop_id"]).reset_index(drop=True)


def build_departures(
    stop_times: pd.DataFrame, trips: pd.DataFrame, index: StationIndex
) -> pd.DataFrame:
    """Scheduled departures at every platform of a multi-route station.

    Restricted to stations where a transfer is actually possible — a station served by
    one route is somewhere you stay on the train. Carries `service_id`, without which
    the 14 pooled patterns would show trains that do not run on the requested date.
    """
    times = stop_times.copy()
    times["scheduled_trip_id"] = times["scheduled_trip_id"].astype(str)
    lookup = trips[["scheduled_trip_id", "service_id"]].copy()
    lookup["scheduled_trip_id"] = lookup["scheduled_trip_id"].astype(str)
    times = times.merge(lookup, on="scheduled_trip_id", how="left")

    unmatched = int(times["service_id"].isna().sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} stop_times row(s) have no service_id in trips.txt — the "
            "departure calendar would silently omit them"
        )

    times["station"] = times["stop_id"].str.extract(STATION_PATTERN)
    times["name"] = times["station"].map(index.code_to_name)
    routes = times.groupby("name")["route_id"].nunique()
    transfer_names = set(routes[routes > 1].index)

    departures = (
        times[times["name"].isin(transfer_names)][
            [
                "service_id",
                "stop_id",
                "route_id",
                "direction_id",
                "scheduled_departure_offset_sec",
            ]
        ]
        .rename(columns={"scheduled_departure_offset_sec": "departure_sec"})
        .drop_duplicates()
        .sort_values(
            ["service_id", "stop_id", "route_id", "direction_id", "departure_sec"]
        )
        .reset_index(drop=True)
    )
    departures["departure_sec"] = departures["departure_sec"].astype("int32")
    logger.info(
        "departures: %d row(s) across %d transfer station(s), %d service pattern(s)",
        len(departures),
        len(transfer_names),
        departures["service_id"].nunique(),
    )
    return departures


def build_service_calendar(calendar_dates: pd.DataFrame) -> pd.DataFrame:
    """date -> service_ids running that date, from `calendar_dates.txt`."""
    added = calendar_dates[calendar_dates["exception_type"] == 1].copy()
    added["date"] = pd.to_datetime(added["date"], format="%Y%m%d").dt.strftime(
        "%Y-%m-%d"
    )
    calendar = (
        added[["date", "service_id"]]
        .drop_duplicates()
        .sort_values(["date", "service_id"])
        .reset_index(drop=True)
    )
    logger.info(
        "service calendar: %d date(s) from %s to %s",
        calendar["date"].nunique(),
        calendar["date"].min(),
        calendar["date"].max(),
    )
    return calendar


def read_bundle_tables(bundle: Path) -> dict[str, pd.DataFrame]:
    """The three bundle tables routing needs, read in one pass."""
    wanted = ("pathways.txt", "trips.txt", "calendar_dates.txt")
    with zipfile.ZipFile(bundle) as archive:
        return {name: pd.read_csv(io.BytesIO(archive.read(name))) for name in wanted}


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One possible way through: ride to C1, walk to C2, ride on to B."""

    origin_stop_id: str
    transfer_in: str
    transfer_out: str
    destination_stop_id: str
    transfer_station: str
    walk_sec: int
    leg1: dict
    leg2: dict

    @property
    def scheduled_ride_sec(self) -> int:
        return int(self.leg1["sched_sec"]) + int(self.leg2["sched_sec"])

    @property
    def n_segments(self) -> int:
        return int(self.leg1["n_segments"]) + int(self.leg2["n_segments"])


def service_position(local: pd.Timestamp) -> tuple[str, int]:
    """(service date, seconds into the service day) for a local timestamp.

    A train at 01:10 belongs to the previous service day, where the timetable calls it
    25:10. Getting this wrong would look up the wrong day's services and, past midnight,
    find no connection at all.
    """
    seconds = local.hour * 3600 + local.minute * 60 + local.second
    if local.hour < SERVICE_DAY_ROLLOVER_HOUR:
        return (local - pd.Timedelta(1, unit="D")).strftime(
            "%Y-%m-%d"
        ), seconds + SECONDS_PER_DAY
    return local.strftime("%Y-%m-%d"), seconds


class TransferGraph:
    """Ride edges, walk edges and the departure calendar, indexed for lookup."""

    def __init__(
        self,
        schedule: pd.DataFrame,
        walk_edges: pd.DataFrame,
        departures: pd.DataFrame,
        calendar: pd.DataFrame,
        index: StationIndex,
    ) -> None:
        self.index = index

        self.legs: dict[tuple[str, str, int], dict] = {}
        self.ride_out: dict[str, set] = defaultdict(set)
        for row in schedule.itertuples(index=False):
            self.legs[(row.origin, row.destination, int(row.hour))] = {
                "sched_sec": int(row.sched_sec),
                "n_segments": int(row.n_segments),
                "stop_span": int(row.stop_span),
                "route_id": row.route_id,
                "direction_id": int(row.direction_id),
                "first_leg_to": row.first_leg_to,
            }
            self.ride_out[row.origin].add(row.destination)

        # Every platform can reach itself at zero cost: Blue, Orange and Silver share
        # one island at Metro Center, so changing between them involves no walking.
        self.walk: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for stop in self.ride_out:
            self.walk[stop].append((stop, 0))
        for row in walk_edges.itertuples(index=False):
            self.walk[row.from_stop_id].append((row.to_stop_id, int(row.walk_sec)))

        self.departures: dict[tuple[str, str, str, int], list[int]] = {}
        for key, group in departures.groupby(
            ["service_id", "stop_id", "route_id", "direction_id"], sort=False
        ):
            service, stop, route, direction = key
            self.departures[(str(service), str(stop), str(route), int(direction))] = (
                sorted(int(v) for v in group["departure_sec"])
            )

        self.calendar: dict[str, list[str]] = (
            calendar.groupby("date")["service_id"].apply(list).to_dict()
        )
        self._calendar_dow: dict[int, list[str]] = {}
        for date, services in self.calendar.items():
            weekday = pd.Timestamp(date).dayofweek
            self._calendar_dow.setdefault(weekday, list(services))

        logger.info(
            "transfer graph: %d ride origin(s), %d walk edge(s), %d departure group(s)",
            len(self.ride_out),
            len(walk_edges),
            len(self.departures),
        )

    @classmethod
    def from_directory(
        cls, directory: Path, index: StationIndex, schedule: pd.DataFrame
    ) -> TransferGraph:
        """Load the three routing artifacts that ship alongside the model."""
        return cls(
            schedule=schedule,
            walk_edges=pd.read_csv(directory / "walk_edges.csv"),
            departures=pd.read_csv(directory / "departures.csv"),
            calendar=pd.read_csv(directory / "service_calendar.csv", dtype=str),
            index=index,
        )

    # -- lookups ------------------------------------------------------------

    def services_on(self, date: str) -> list[str]:
        """Service ids running on a date, falling back to the same weekday.

        The published calendar ends at a fixed date. Beyond it, the same weekday's
        pattern is the best available answer and is flagged by the caller rather than
        passed off as the real timetable.
        """
        if date in self.calendar:
            return self.calendar[date]
        return self._calendar_dow.get(pd.Timestamp(date).dayofweek, [])

    def leg(self, origin: str, destination: str, hour: int | None) -> dict | None:
        """The scheduled leg for a platform pair, preferring the hour-specific row."""
        if hour is not None:
            found = self.legs.get((origin, destination, hour))
            if found is not None:
                return found
        return self.legs.get((origin, destination, ANY_HOUR))

    def next_departure(
        self,
        stop_id: str,
        route_id: str,
        direction_id: int,
        ready_sec: int,
        services: Sequence[str],
    ) -> int | None:
        """First scheduled departure at or after `ready_sec`, across active services."""
        best: int | None = None
        for service in services:
            times = self.departures.get(
                (service, stop_id, str(route_id), int(direction_id))
            )
            if not times:
                continue
            position = bisect.bisect_left(times, ready_sec)
            if position < len(times):
                candidate = times[position]
                if best is None or candidate < best:
                    best = candidate
        return best

    def is_direct(self, origin_name: str, destination_name: str) -> bool:
        origins = self.index.platforms(origin_name)
        destinations = set(self.index.platforms(destination_name))
        return any(
            destinations & self.ride_out.get(origin, set()) for origin in origins
        )

    # -- planning -----------------------------------------------------------

    def candidates(
        self, origin_name: str, destination_name: str, hour: int | None = None
    ) -> list[Candidate]:
        """Every one-transfer route between two stations, unranked."""
        origins = self.index.platforms(origin_name)
        destinations = self.index.platforms(destination_name)
        origin_codes = {STATION_PATTERN.match(p).group(1) for p in origins}
        destination_codes = {STATION_PATTERN.match(p).group(1) for p in destinations}
        blocked = origin_codes | destination_codes

        found: dict[tuple[str, str, str, str], Candidate] = {}
        for origin in origins:
            for transfer_in in self.ride_out.get(origin, ()):  # noqa: B007
                match = STATION_PATTERN.match(transfer_in)
                if match is None or match.group(1) in blocked:
                    # Changing at the origin or destination station is not a transfer.
                    continue
                leg1 = self.leg(origin, transfer_in, hour)
                if leg1 is None:
                    continue
                for transfer_out, walk_sec in self.walk.get(transfer_in, ()):
                    onward = self.ride_out.get(transfer_out, set())
                    for destination in destinations:
                        if destination not in onward:
                            continue
                        leg2 = self.leg(transfer_out, destination, hour)
                        if leg2 is None:
                            continue
                        if (
                            transfer_in == transfer_out
                            and leg1["route_id"] == leg2["route_id"]
                            and leg1["direction_id"] == leg2["direction_id"]
                        ):
                            # Same platform, same route, same direction: that is one
                            # ride, not a connection.
                            continue
                        key = (origin, transfer_in, transfer_out, destination)
                        if key in found:
                            continue
                        station = self.index.code_to_name[
                            STATION_PATTERN.match(transfer_in).group(1)
                        ]
                        found[key] = Candidate(
                            origin_stop_id=origin,
                            transfer_in=transfer_in,
                            transfer_out=transfer_out,
                            destination_stop_id=destination,
                            transfer_station=station,
                            walk_sec=int(walk_sec),
                            leg1=leg1,
                            leg2=leg2,
                        )
        # Sorted, not set-iteration order. `ride_out` is a set, so the natural order
        # varies between processes with hash randomisation — which makes any caller
        # that takes "the first candidate" quietly non-reproducible.
        return [found[key] for key in sorted(found)]

    def connection(
        self,
        candidate: Candidate,
        arrival_sec: int,
        services: Sequence[str],
    ) -> dict | None:
        """Time the connection given when the rider actually reaches the transfer.

        Returns `None` when no train runs late enough to catch — a real answer at the
        end of service, and better than wrapping round to the morning and reporting a
        six-hour journey.
        """
        ready = arrival_sec + candidate.walk_sec + MIN_CONNECTION_BUFFER_SEC
        departure = self.next_departure(
            candidate.transfer_out,
            candidate.leg2["route_id"],
            candidate.leg2["direction_id"],
            ready,
            services,
        )
        if departure is None:
            return None
        # The train after this one. A connection is a gamble — if leg 1 runs a little
        # late the rider misses it and waits a whole headway — and the size of that
        # penalty is the single most useful thing to tell them about a tight change.
        following = self.next_departure(
            candidate.transfer_out,
            candidate.leg2["route_id"],
            candidate.leg2["direction_id"],
            departure + 1,
            services,
        )
        return {
            "ready_sec": ready,
            "departure_sec": departure,
            "wait_sec": int(departure - ready),
            "following_departure_sec": following,
            "if_missed_sec": None if following is None else int(following - ready),
        }

    def rank(
        self,
        candidates: Sequence[Candidate],
        depart_sec: int,
        services: Sequence[str],
        limit: int = MAX_CANDIDATES_SCORED,
    ) -> list[tuple[Candidate, dict]]:
        """Rank by fully scheduled total, including the connection wait.

        The wait is part of the ranking rather than a detail added afterwards: a route
        with a shorter ride but a fifteen-minute connection genuinely loses to a longer
        ride that walks straight onto a train, and ranking on ride time alone would
        pick the wrong one.
        """
        scored: list[tuple[int, Candidate, dict]] = []
        for candidate in candidates:
            arrival = depart_sec + int(candidate.leg1["sched_sec"])
            timing = self.connection(candidate, arrival, services)
            if timing is None:
                continue
            total = (
                int(candidate.leg1["sched_sec"])
                + candidate.walk_sec
                + timing["wait_sec"]
                + int(candidate.leg2["sched_sec"])
            )
            scored.append((total, candidate, timing))
        scored.sort(key=lambda item: (item[0], item[1].transfer_station))
        return [(candidate, timing) for _, candidate, timing in scored[:limit]]
