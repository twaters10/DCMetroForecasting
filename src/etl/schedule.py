"""Step 2: the scheduled side — static GTFS joined to realtime, with match rates.

Two things in here are easy to get wrong and expensive to discover later.

**The join is two hops.** Realtime `trip_id` is `{scheduled_trip_id}_{version}` and
so is static `trips.trip_id`, but the versions differ whenever WMATA has published a
new timetable. Joining them directly matches 0%. The route is:

    realtime trip_id -> strip suffix -> trips.scheduled_trip_id
                     -> trips.trip_id (versioned) -> stop_times

`stop_times` is keyed on the *versioned* id, so `trips.txt` is a required hop, not a
convenience. Measured on real data: 0% direct, 100% via `scheduled_trip_id`.

**GTFS times run past 24:00:00.** A trip leaving at 23:50 and arriving at 00:15 has
`arrival_time = "24:15:00"`. Parsing that as a clock time raises or silently wraps.
The spec's model is an offset from the start of the service day, and the service day
starts at *noon minus 12 hours* — not midnight — precisely so that a DST transition
does not add or drop an hour mid-trip. `service_day_start` implements that rule.

See docs/static-gtfs.md for the join evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Final

import pandas as pd

from .archive import StaticBundle
from .config import NON_REVENUE_ROUTE_ID, SERVICE_TZ

# `12345678_20670` -> `12345678`. Anchored so a stray underscore inside an id cannot
# silently truncate it, and tolerant of ids with no suffix at all (non-revenue moves
# arrive as bare `NR170`).
_VERSION_SUFFIX: Final[re.Pattern[str]] = re.compile(r"^(.+)_([^_]+)$")

# GTFS `HH:MM:SS`, where HH may exceed 23.
_GTFS_TIME: Final[re.Pattern[str]] = re.compile(r"^(\d+):([0-5]\d):([0-5]\d)$")


def base_trip_id(trip_id: str) -> str:
    """Strip the schedule-version suffix: `12345678_20670` -> `12345678`."""
    match = _VERSION_SUFFIX.match(trip_id)
    return match.group(1) if match else trip_id


def schedule_version(trip_id: str) -> str | None:
    """Extract the schedule-version suffix, or None if the id carries none."""
    match = _VERSION_SUFFIX.match(trip_id)
    return match.group(2) if match else None


def gtfs_time_to_seconds(value: str) -> int | None:
    """`HH:MM:SS` -> seconds from the start of the service day. Handles HH >= 24."""
    match = _GTFS_TIME.match(value.strip())
    if match is None:
        return None
    hours, minutes, seconds = (int(g) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def service_day_start(service_date: date) -> datetime:
    """The instant a GTFS service day begins, in America/New_York.

    Noon minus twelve hours, per the GTFS specification — not midnight. On a normal
    day the two are identical. On a DST boundary they are not: local midnight is
    ambiguous or nonexistent, while noon is always well defined, so anchoring at noon
    means a 25:30:00 stop time lands at a real instant on both the spring-forward and
    fall-back days. Getting this wrong shifts an hour of trips by an hour, twice a
    year, in a way that looks like a delay spike rather than a bug.

    The conversion to UTC before subtracting is the whole point and is easy to lose.
    Timedelta arithmetic on a zoneinfo-aware datetime is *wall-clock* arithmetic: the
    result's offset is recomputed for its new local time, so `noon_local - 12h` yields
    local midnight with midnight's offset — exactly the naive answer this function
    exists to avoid, differing by an hour on both transition days. Subtracting in UTC
    makes it absolute. This matches `segments.add_scheduled_timestamps`, which does the
    same arithmetic in Spark; the two must agree or the same stop time resolves to two
    different instants depending on which code path touched it.
    """
    noon_local = datetime.combine(service_date, time(12), tzinfo=SERVICE_TZ)
    return noon_local.astimezone(UTC) - timedelta(hours=12)


def scheduled_timestamp(service_date: date, gtfs_time: str) -> datetime | None:
    """Resolve a GTFS `HH:MM:SS` on a service date to an absolute instant (UTC)."""
    offset = gtfs_time_to_seconds(gtfs_time)
    if offset is None:
        return None
    return service_day_start(service_date) + timedelta(seconds=offset)


# --------------------------------------------------------------------------
# The joinable schedule
# --------------------------------------------------------------------------


def scheduled_stop_times(bundle: StaticBundle) -> pd.DataFrame:
    """Flatten the bundle into one row per (scheduled_trip_id, stop_sequence).

    This is the frame realtime observations join against. It carries
    `scheduled_trip_id` — the version-stable key — rather than requiring callers to
    know about the two-hop translation.

    Times stay as offsets in seconds rather than timestamps, because a timestamp
    needs a service date and the schedule is date-independent. The caller supplies
    the date; see `scheduled_timestamp`.
    """
    trips = bundle.trips[
        ["trip_id", "scheduled_trip_id", "route_id", "direction_id", "trip_headsign"]
    ].copy()
    stop_times = bundle.stop_times[
        ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"]
    ].copy()

    joined = stop_times.merge(trips, on="trip_id", how="inner", validate="many_to_one")

    joined["stop_sequence"] = joined["stop_sequence"].astype(int)
    joined["direction_id"] = pd.to_numeric(joined["direction_id"], errors="coerce")
    joined["scheduled_arrival_offset_sec"] = joined["arrival_time"].map(
        gtfs_time_to_seconds
    )
    joined["scheduled_departure_offset_sec"] = joined["departure_time"].map(
        gtfs_time_to_seconds
    )

    return joined.rename(columns={"trip_id": "static_trip_id"})[
        [
            "scheduled_trip_id",
            "static_trip_id",
            "route_id",
            "direction_id",
            "trip_headsign",
            "stop_id",
            "stop_sequence",
            "scheduled_arrival_offset_sec",
            "scheduled_departure_offset_sec",
        ]
    ].sort_values(["scheduled_trip_id", "stop_sequence"])


# --------------------------------------------------------------------------
# Match-rate reporting
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchRateReport:
    """How well realtime observations reached the schedule.

    A low match rate is the single most likely silent failure in this pipeline: the
    output still looks like a segment table, it just has the wrong scheduled times or
    far fewer rows than it should. So this is reported loudly on every run rather
    than logged at debug level.
    """

    total_rows: int
    non_revenue_rows: int
    revenue_rows: int
    matched_rows: int
    naive_matched_rows: int
    realtime_versions: dict[str, int]
    bundle_version: str | None
    bundle_key: str

    @property
    def match_rate(self) -> float:
        if self.revenue_rows == 0:
            return 0.0
        return self.matched_rows / self.revenue_rows

    @property
    def version_agrees(self) -> bool:
        """Whether the bundle's version is the one the realtime feed was emitting.

        Disagreement does not break the join — `scheduled_trip_id` is version-stable
        — but it means the scheduled times may come from a timetable that was not in
        force, so any delay computed from them is suspect.
        """
        return self.bundle_version is not None and set(self.realtime_versions) == {
            self.bundle_version
        }

    def format(self) -> str:
        lines = [
            "static GTFS join",
            f"  bundle                {self.bundle_key.split('/')[-1]}",
            f"  bundle version        {self.bundle_version}",
            f"  realtime versions     {self.realtime_versions}",
            f"  rows                  {self.total_rows:,}",
            f"  non-revenue excluded  {self.non_revenue_rows:,} "
            f"(route_id == '{NON_REVENUE_ROUTE_ID}')",
            f"  revenue rows          {self.revenue_rows:,}",
            f"  matched to schedule   {self.matched_rows:,} "
            f"({100 * self.match_rate:.2f}%)",
            f"  (direct trip_id join  {self.naive_matched_rows:,} — nonzero only for "
            "rows whose\n     realtime version already matches the bundle)",
        ]
        if not self.version_agrees:
            lines.append(
                "  !! VERSION MISMATCH: scheduled times may not be the ones that "
                "were in force.\n"
                "     The join is still valid; treat delay_sec on these rows as "
                "lower confidence."
            )
        if self.match_rate < 0.95:
            lines.append(
                f"  !! MATCH RATE {100 * self.match_rate:.1f}% IS LOW. Expected ~100% "
                "for revenue service.\n"
                "     Most likely cause: no archived bundle covers this service "
                "date. Check static/."
            )
        return "\n".join(lines)


def build_match_report(
    observations: pd.DataFrame, bundle: StaticBundle
) -> MatchRateReport:
    """Compute the match-rate report for a set of realtime observations.

    `observations` needs `trip_id` and `route_id`. Non-revenue trips are excluded from
    the denominator *before* the rate is computed — they have no schedule by
    definition, so counting them makes normal operations look like a regression and
    hides a real one.
    """
    frame = observations.dropna(subset=["trip_id"])
    revenue = frame[frame["route_id"] != NON_REVENUE_ROUTE_ID]

    scheduled_ids = set(bundle.trips["scheduled_trip_id"])
    static_ids = set(bundle.trips["trip_id"])

    bases = revenue["trip_id"].map(base_trip_id)
    versions = revenue["trip_id"].map(schedule_version).value_counts().to_dict()

    return MatchRateReport(
        total_rows=len(frame),
        non_revenue_rows=len(frame) - len(revenue),
        revenue_rows=len(revenue),
        matched_rows=int(bases.isin(scheduled_ids).sum()),
        naive_matched_rows=int(revenue["trip_id"].isin(static_ids).sum()),
        realtime_versions={str(k): int(v) for k, v in versions.items()},
        bundle_version=bundle.schedule_version,
        bundle_key=bundle.key,
    )
