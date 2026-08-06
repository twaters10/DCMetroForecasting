"""Step 1: what does WMATA actually put in these feeds?

GTFS-realtime marks nearly every field optional and agencies vary widely in what
they populate, so the derivation approach cannot be chosen from the spec. This
script reads a real continuous window out of the archive and reports what is
actually there.

    python -m src.etl.explore --mode rail --start 2026-08-05T11 --end 2026-08-05T14

It answers four questions, in the order they constrain the design:

1. **Coverage** — is the window continuous, or is there collector downtime that
   would make any progression analysis meaningless?
2. **Field census** — which fields are set, and how often.
3. **Static join** — does realtime `trip_id` reach `stop_times`, and is the bundle
   the one that was in force?
4. **Arrival derivation** — the decisive one. Whether the STOPPED_AT transition in
   VehiclePositions and the last-prediction-before-disappearance in TripUpdates
   each actually work on consecutive snapshots, and how far apart they land where
   both fire.

Writes nothing. Reads the archive and prints.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import boto3
import pandas as pd
from google.transit import gtfs_realtime_pb2

from .archive import (
    Snapshot,
    read_snapshots,
    resolve_static_bundle,
    snapshot_keys_by_hour,
)
from .config import (
    EXPECTED_SNAPSHOTS_PER_HOUR,
    FEEDS_BY_MODE,
    MAX_ARRIVAL_BRACKET_SEC,
    MIN_PLAUSIBLE_EPOCH_SEC,
    NON_REVENUE_ROUTE_ID,
    SERVICE_TZ,
    EtlConfig,
)

_VEHICLE_STATUS = gtfs_realtime_pb2.VehiclePosition.VehicleStopStatus
_SCHEDULE_RELATIONSHIP = (
    gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship
)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Convert a tz-aware datetime column to integer unix seconds.

    Subtracting the epoch and dividing by a Timedelta rather than casting to
    int64: pandas 3 infers `datetime64[us]` from Python datetimes where pandas 2
    inferred `datetime64[ns]`, so `astype("int64") // 10**9` silently returns
    microseconds-over-a-billion on one of them. This is unit-agnostic.
    """
    return (series - _EPOCH) // pd.Timedelta("1s")


# --------------------------------------------------------------------------
# 1. Coverage
# --------------------------------------------------------------------------


def report_coverage(keys_by_hour: dict[str, list[str]], feed: str) -> None:
    rule(f"1. SNAPSHOT COVERAGE — {feed}")
    total = sum(len(k) for k in keys_by_hour.values())
    print(f"{total} snapshots across {len(keys_by_hour)} hour partitions")
    print(f"(a healthy hour holds {EXPECTED_SNAPSHOTS_PER_HOUR}: one per minute)\n")

    for partition, keys in keys_by_hour.items():
        count = len(keys)
        short = count < EXPECTED_SNAPSHOTS_PER_HOUR
        flag = f"  <-- SHORT by {EXPECTED_SNAPSHOTS_PER_HOUR - count}" if short else ""
        print(f"  {partition:<44} {count:>3}{flag}")


# --------------------------------------------------------------------------
# 2. Field census
# --------------------------------------------------------------------------


def field_census(objects: list[Any], label: str) -> None:
    """Count how often each field is actually set.

    `ListFields()` returns only fields that are set, which is exactly the question.
    A field absent from this table is absent from the feed — not merely rare.
    """
    counts: Counter[str] = Counter()
    for obj in objects:
        for descriptor, _ in obj.ListFields():
            counts[descriptor.name] += 1

    print(f"\n{label} (n={len(objects)})")
    if not objects:
        print("  no records")
        return
    for name, seen in counts.most_common():
        print(f"  {name:<28} {seen:>7}  {pct(seen, len(objects)):>7}")


def report_vehicle_positions_census(snapshots: list[Snapshot]) -> pd.DataFrame:
    rule("2a. FIELD CENSUS — VehiclePositions")

    rows: list[dict[str, Any]] = []
    vehicles: list[Any] = []
    trips: list[Any] = []
    for snap in snapshots:
        for entity in snap.message.entity:
            v = entity.vehicle
            vehicles.append(v)
            trips.append(v.trip)
            rows.append(
                {
                    "captured_at": snap.captured_at,
                    "trip_id": v.trip.trip_id or None,
                    "route_id": v.trip.route_id or None,
                    "direction_id": (
                        v.trip.direction_id if v.trip.HasField("direction_id") else None
                    ),
                    "vehicle_id": v.vehicle.id or None,
                    "stop_id": v.stop_id or None,
                    "stop_sequence": (
                        v.current_stop_sequence
                        if v.HasField("current_stop_sequence")
                        else None
                    ),
                    "status": (
                        _VEHICLE_STATUS.Name(v.current_status)
                        if v.HasField("current_status")
                        else None
                    ),
                    "vehicle_ts": v.timestamp if v.HasField("timestamp") else None,
                    "occupancy": (
                        gtfs_realtime_pb2.VehiclePosition.OccupancyStatus.Name(
                            v.occupancy_status
                        )
                        if v.HasField("occupancy_status")
                        else None
                    ),
                }
            )

    field_census(vehicles, "vehicle.*")
    field_census(trips, "vehicle.trip.*")

    frame = pd.DataFrame(rows)
    print("\nvalue distributions")
    for column in ("status", "occupancy"):
        counts = frame[column].value_counts(dropna=False)
        print(f"  {column}:")
        for value, count in counts.items():
            note = ""
            if column == "occupancy" and len(counts) == 1:
                note = "   <-- single value: present but carries no signal"
            print(f"    {str(value):<18} {count:>7}  {pct(count, len(frame)):>7}{note}")

    # Observation age: each vehicle stamps its own report time, which is not when the
    # collector captured it. This is the floor on arrival precision, independent of
    # the polling interval.
    age = (to_epoch_seconds(frame["captured_at"]) - frame["vehicle_ts"]).dropna()
    if not age.empty:
        print(
            f"\nvehicle report age at capture (sec): "
            f"median {age.median():.0f}, p95 {age.quantile(0.95):.0f}, "
            f"max {age.max():.0f}"
        )
    return frame


def report_trip_updates_census(snapshots: list[Snapshot]) -> pd.DataFrame:
    rule("2b. FIELD CENSUS — TripUpdates")

    rows: list[dict[str, Any]] = []
    trip_updates: list[Any] = []
    stop_time_updates: list[Any] = []
    for snap in snapshots:
        for entity in snap.message.entity:
            tu = entity.trip_update
            trip_updates.append(tu)
            for stu in tu.stop_time_update:
                stop_time_updates.append(stu)
                rows.append(
                    {
                        "captured_at": snap.captured_at,
                        "trip_id": tu.trip.trip_id or None,
                        "route_id": tu.trip.route_id or None,
                        "stop_sequence": (
                            stu.stop_sequence if stu.HasField("stop_sequence") else None
                        ),
                        "stop_id": stu.stop_id or None,
                        "arrival_time": (
                            stu.arrival.time if stu.arrival.HasField("time") else None
                        ),
                        "arrival_uncertainty": (
                            stu.arrival.uncertainty
                            if stu.arrival.HasField("uncertainty")
                            else None
                        ),
                        "departure_time": (
                            stu.departure.time
                            if stu.departure.HasField("time")
                            else None
                        ),
                        "relationship": _SCHEDULE_RELATIONSHIP.Name(
                            stu.schedule_relationship
                        ),
                    }
                )

    field_census(trip_updates, "trip_update.*")
    field_census(stop_time_updates, "trip_update.stop_time_update.*")

    frame = pd.DataFrame(rows)
    n = len(frame)
    print(f"\narrival vs departure availability (n={n} stop_time_updates)")
    has_arr = frame["arrival_time"].notna()
    has_dep = frame["departure_time"].notna()
    for label, mask in (
        ("arrival.time", has_arr),
        ("departure.time", has_dep),
        ("both", has_arr & has_dep),
        ("neither", ~has_arr & ~has_dep),
    ):
        print(f"  {label:<16} {int(mask.sum()):>7}  {pct(int(mask.sum()), n):>7}")

    print("\nschedule_relationship")
    for value, count in frame["relationship"].value_counts().items():
        note = "   <-- must never become a segment row" if value == "SKIPPED" else ""
        print(f"  {str(value):<16} {count:>7}  {pct(count, n):>7}{note}")

    unc = frame["arrival_uncertainty"].dropna()
    if not unc.empty:
        print("\narrival.uncertainty distribution")
        for value, count in unc.value_counts().head(8).items():
            note = "   <-- spec: 0 means known, not predicted" if value == 0 else ""
            print(f"  {int(value):<16} {count:>7}  {pct(count, len(unc)):>7}{note}")

    # Departure is only ever set on the first stop of a trip in WMATA rail. If that
    # holds, segment departure has to be approximated by arrival at the upstream
    # stop, which folds dwell time into the segment.
    if has_dep.any():
        seqs = frame.loc[has_dep, "stop_sequence"].value_counts().head(5)
        print("\nstop_sequence values carrying departure.time:")
        for value, count in seqs.items():
            print(f"  seq {int(value):<12} {count:>7}")
    return frame


# --------------------------------------------------------------------------
# 3. Static join
# --------------------------------------------------------------------------


def report_static_join(tu_frame: pd.DataFrame, bundle: Any, mode: str) -> None:
    rule("3. STATIC GTFS JOIN")
    print(f"bundle:           s3://.../{bundle.key.split('/')[-1]}")
    print(f"feed window:      {bundle.feed_start_date} .. {bundle.feed_end_date}")
    print(f"schedule version: {bundle.schedule_version}")
    print(
        f"static trips:     {len(bundle.trips):,}   "
        f"stop_times: {len(bundle.stop_times):,}"
    )

    frame = tu_frame.dropna(subset=["trip_id"]).copy()
    revenue = frame[frame["route_id"] != NON_REVENUE_ROUTE_ID]
    non_revenue = len(frame) - len(revenue)

    print(f"\nrealtime rows:    {len(frame):,}")
    print(
        f"non-revenue:      {non_revenue:,} "
        f"(route_id == '{NON_REVENUE_ROUTE_ID}') — excluded, they have no schedule"
    )

    rt_versions = Counter(
        str(t).rsplit("_", 1)[-1] for t in revenue["trip_id"].unique()
    )
    print(f"realtime versions: {dict(rt_versions)}")
    if bundle.schedule_version not in rt_versions:
        print(
            f"  !! realtime version(s) {sorted(rt_versions)} != bundle "
            f"{bundle.schedule_version}"
        )
        print(
            "     The join below still works (scheduled_trip_id is version-stable), "
            "but scheduled\n     times come from a timetable that may not have been "
            "in force. Flag per row."
        )

    # The two-hop join: realtime trip_id -> trips.scheduled_trip_id -> the versioned
    # trips.trip_id -> stop_times.
    by_scheduled = dict(
        zip(bundle.trips["scheduled_trip_id"], bundle.trips["trip_id"], strict=True)
    )
    static_trip_ids = set(bundle.trips["trip_id"])
    reachable = set(bundle.stop_times["trip_id"])

    naive = revenue["trip_id"].isin(static_trip_ids).sum()
    bases = revenue["trip_id"].str.rsplit("_", n=1).str[0]
    matched_mask = bases.isin(by_scheduled)
    mapped = bases[matched_mask].map(by_scheduled)
    reached = mapped.isin(reachable).sum()

    total = len(revenue)
    print(f"\n{'join attempt':<44} {'matched':>10} {'rate':>8}")
    print(f"{'-' * 64}")
    print(
        f"{'realtime trip_id -> static trip_id':<44} {naive:>10,} "
        f"{pct(naive, total):>8}"
    )
    print(
        f"{'base -> trips.scheduled_trip_id':<44} "
        f"{int(matched_mask.sum()):>10,} {pct(int(matched_mask.sum()), total):>8}"
    )
    print(f"{'  -> reached stop_times':<44} {reached:>10,} {pct(reached, total):>8}")

    # stop_id needs no translation, but verify rather than assume.
    stop_ids = set(revenue["stop_id"].dropna().unique())
    static_stops = set(bundle.stops["stop_id"])
    hit = len(stop_ids & static_stops)
    print(
        f"\n{'distinct stop_id -> stops.txt':<44} "
        f"{hit:>10,} {pct(hit, len(stop_ids)):>8}"
    )
    if stop_ids - static_stops:
        print(f"  unmatched examples: {sorted(stop_ids - static_stops)[:5]}")


# --------------------------------------------------------------------------
# 4. Arrival derivation — the decisive analysis
# --------------------------------------------------------------------------


def derive_vp_arrivals(vp_frame: pd.DataFrame) -> pd.DataFrame:
    """Derive arrivals from VehiclePositions stop_sequence transitions.

    A vehicle's `current_stop_sequence` incrementing between two consecutive
    snapshots means it arrived at the new stop somewhere in that interval. The
    arrival is bracketed by the two capture times, so the midpoint is the estimate
    and the interval width is the error bar — an honest observation with a bound,
    rather than a prediction.

    That the increment marks *arrival* and not departure was verified, not assumed,
    because the two would differ by a whole segment. Measured over 6,119 transitions
    in a 3-hour rail window:

    - the status reported at the newly-incremented sequence is `STOPPED_AT` 92.3% of
      the time (`INCOMING_AT` 6.5%, `IN_TRANSIT_TO` 1.2%)
    - 98.9% of (trip, stop) pairs are already `STOPPED_AT` at their first sighting

    So WMATA advances the sequence on arrival. Note this makes `IN_TRANSIT_TO` with
    sequence N mean "departed N", not the spec's "heading to N" — the last status
    seen at a sequence is `IN_TRANSIT_TO` 54.1% of the time. Reading it the spec's
    way would shift every arrival by one segment.
    """
    frame = vp_frame.dropna(subset=["trip_id", "stop_sequence"]).copy()
    frame = frame.sort_values(["trip_id", "captured_at"])

    grouped = frame.groupby("trip_id", sort=False)
    frame["prev_seq"] = grouped["stop_sequence"].shift(1)
    frame["prev_at"] = grouped["captured_at"].shift(1)

    advanced = frame[
        frame["prev_seq"].notna() & (frame["stop_sequence"] > frame["prev_seq"])
    ]
    window = (advanced["captured_at"] - advanced["prev_at"]).dt.total_seconds()
    return advanced.assign(
        window_sec=window,
        seq_jump=advanced["stop_sequence"] - advanced["prev_seq"],
        arrival_estimate=advanced["prev_at"]
        + (advanced["captured_at"] - advanced["prev_at"]) / 2,
        # The bracket width is the error bar, so a wide bracket is a weak estimate
        # rather than a wrong one. Kept and flagged, not dropped — the caller decides.
        bracket_ok=window <= MAX_ARRIVAL_BRACKET_SEC,
    )


def derive_tu_arrivals(tu_frame: pd.DataFrame) -> pd.DataFrame:
    """Derive arrivals from TripUpdates: last prediction before a stop disappears.

    Each (trip, stop) is predicted repeatedly, and the prediction converges as the
    vehicle approaches. Once passed, the stop drops off the feed. The final observed
    prediction is therefore the best available estimate of the actual arrival.
    """
    frame = tu_frame.dropna(subset=["trip_id", "stop_sequence", "arrival_time"]).copy()
    # WMATA occasionally sets arrival.time to 0 instead of omitting it. Dropping
    # these here rather than downstream: a single 0 in a group destroys the min and
    # makes prediction drift read as 56 years.
    frame = frame[frame["arrival_time"] >= MIN_PLAUSIBLE_EPOCH_SEC]
    frame = frame.sort_values("captured_at")

    grouped = frame.groupby(["trip_id", "stop_sequence"], sort=False)
    return pd.DataFrame(
        {
            "observations": grouped.size(),
            "last_seen_at": grouped["captured_at"].last(),
            "final_estimate": grouped["arrival_time"].last(),
            "first_estimate": grouped["arrival_time"].first(),
            "estimate_range_sec": grouped["arrival_time"].max()
            - grouped["arrival_time"].min(),
            "stop_id": grouped["stop_id"].last(),
        }
    ).reset_index()


def report_arrival_derivation(
    vp_frame: pd.DataFrame, tu_frame: pd.DataFrame, window_end: datetime
) -> None:
    rule("4. ARRIVAL DERIVATION VIABILITY")

    # ---- VehiclePositions ----
    vp_arrivals = derive_vp_arrivals(vp_frame)
    tracked = vp_frame.dropna(subset=["trip_id", "stop_sequence"])
    observed_pairs = tracked.groupby(["trip_id", "stop_sequence"]).size()
    stopped_at = tracked[tracked["status"] == "STOPPED_AT"]
    stopped_pairs = stopped_at.groupby(["trip_id", "stop_sequence"]).size()

    print("VehiclePositions — stop_sequence transitions")
    print(f"  distinct trips tracked           {tracked['trip_id'].nunique():,}")
    print(f"  distinct (trip, stop) observed   {len(observed_pairs):,}")
    print(
        f"  of those, ever seen STOPPED_AT    {len(stopped_pairs):,}  "
        f"{pct(len(stopped_pairs), len(observed_pairs))}"
    )
    print(f"  arrival transitions detected     {len(vp_arrivals):,}")

    if not vp_arrivals.empty:
        tight = vp_arrivals[vp_arrivals["bracket_ok"]]
        wide = vp_arrivals[~vp_arrivals["bracket_ok"]]
        print(
            f"  usable bracket (<= {MAX_ARRIVAL_BRACKET_SEC}s)      {len(tight):,}  "
            f"{pct(len(tight), len(vp_arrivals))}"
        )
        print(
            f"  wide bracket (feed dropout)      {len(wide):,}  "
            f"{pct(len(wide), len(vp_arrivals))} — keep but flag low confidence"
        )
        print(
            f"\n  quantization window (sec), usable only: median "
            f"{tight['window_sec'].median():.0f}, "
            f"p95 {tight['window_sec'].quantile(0.95):.0f}, "
            f"max {tight['window_sec'].max():.0f}"
        )
        jumps = tight["seq_jump"].value_counts().sort_index()
        skipped = int(tight.loc[tight["seq_jump"] > 1, "seq_jump"].sub(1).sum())
        print("  sequence jump per transition:")
        for value, count in jumps.head(6).items():
            note = (
                "   <-- stop(s) passed between polls, arrival unobserved"
                if value > 1
                else ""
            )
            print(
                f"    +{int(value):<12} {count:>7}  {pct(count, len(tight)):>7}{note}"
            )
        print(
            f"  stops passed unobserved:         {skipped:,}  "
            f"{pct(skipped, len(tight) + skipped)} of traversals"
        )

    # ---- TripUpdates ----
    tu_arrivals = derive_tu_arrivals(tu_frame)
    # A stop still in the feed at the end of the window has not been passed yet, so
    # its "final" prediction is not an arrival. Excluding these is essential or the
    # estimate is contaminated by trips still in progress.
    cutoff = pd.Timestamp(window_end).tz_convert("UTC")
    settled = tu_arrivals[
        tu_arrivals["last_seen_at"] < cutoff - pd.Timedelta(minutes=2)
    ]

    print("\nTripUpdates — last prediction before the stop disappears")
    print(f"  distinct (trip, stop) predicted   {len(tu_arrivals):,}")
    print(
        f"  settled (disappeared in window)   {len(settled):,}  "
        f"{pct(len(settled), len(tu_arrivals))}"
    )
    if not settled.empty:
        print(
            f"  observations per stop: median {settled['observations'].median():.0f}, "
            f"max {settled['observations'].max():.0f}"
        )
        drift = settled["estimate_range_sec"]
        print(
            f"  prediction drift over life (sec): median {drift.median():.0f}, "
            f"p90 {drift.quantile(0.90):.0f}, max {drift.max():.0f}"
        )
        stable = int((drift <= 60).sum())
        print(
            f"  drifted <= 60s across all observations: {stable:,}  "
            f"{pct(stable, len(settled))}"
        )

    # ---- Cross-source agreement: the number that decides the design ----
    print("\nCross-source agreement (where both fire on the same trip+stop)")
    if vp_arrivals.empty or settled.empty:
        print("  insufficient overlap in this window")
        return

    vp_keyed = vp_arrivals.loc[
        vp_arrivals["bracket_ok"],
        ["trip_id", "stop_sequence", "arrival_estimate", "window_sec"],
    ]
    merged = vp_keyed.merge(settled, on=["trip_id", "stop_sequence"], how="inner")
    if merged.empty:
        print("  no overlapping (trip, stop) pairs")
        return

    vp_epoch = to_epoch_seconds(merged["arrival_estimate"])
    delta = (merged["final_estimate"] - vp_epoch).astype(float)
    print(f"  overlapping pairs                {len(merged):,}")
    print(
        f"  TripUpdates minus VehiclePositions (sec): "
        f"median {delta.median():.0f}, "
        f"p10 {delta.quantile(0.10):.0f}, p90 {delta.quantile(0.90):.0f}"
    )
    within = int((delta.abs() <= 60).sum())
    print(f"  agree within 60s                 {within:,}  {pct(within, len(merged))}")
    print(
        "\n  Interpretation: VehiclePositions is a bracketed observation, TripUpdates\n"
        "  is WMATA's prediction. This spread is TripUpdates' error against observed\n"
        "  truth — the basis for preferring VP and falling back to TU with a\n"
        "  provenance column."
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_hour(value: str) -> datetime:
    """Parse `YYYY-MM-DDTHH` (or a fuller ISO stamp) as UTC."""
    text = value if len(value) > 13 else f"{value}:00:00"
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(FEEDS_BY_MODE), default="rail")
    parser.add_argument(
        "--start", type=parse_hour, required=True, help="UTC hour, e.g. 2026-08-05T11"
    )
    parser.add_argument(
        "--end", type=parse_hour, required=True, help="UTC hour, exclusive"
    )
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args(argv)

    config = EtlConfig.from_env()
    s3 = boto3.client("s3")
    vp_feed, tu_feed = FEEDS_BY_MODE[args.mode]

    local_start = args.start.astimezone(SERVICE_TZ)
    local_end = args.end.astimezone(SERVICE_TZ)
    print(f"mode:   {args.mode}")
    print(f"window: {args.start:%Y-%m-%d %H:%M} .. {args.end:%H:%M} UTC")
    print(f"        {local_start:%Y-%m-%d %H:%M} .. {local_end:%H:%M} {local_start:%Z}")
    print(f"bucket: s3://{config.s3_bucket}/{config.raw_prefix}")

    for feed in (vp_feed, tu_feed):
        keys_by_hour = snapshot_keys_by_hour(s3, config, feed, args.start, args.end)
        report_coverage(keys_by_hour, feed)
        if feed == vp_feed:
            vp_keys = [k for keys in keys_by_hour.values() for k in keys]
        else:
            tu_keys = [k for keys in keys_by_hour.values() for k in keys]

    print(f"\nfetching {len(vp_keys) + len(tu_keys)} snapshots ...")
    vp_snaps = list(read_snapshots(s3, config, vp_keys, args.max_workers))
    tu_snaps = list(read_snapshots(s3, config, tu_keys, args.max_workers))

    vp_frame = report_vehicle_positions_census(vp_snaps)
    tu_frame = report_trip_updates_census(tu_snaps)

    service_date = local_start.date().isoformat()
    bundle = resolve_static_bundle(s3, config, args.mode, service_date)
    report_static_join(tu_frame, bundle, args.mode)

    report_arrival_derivation(vp_frame, tu_frame, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
