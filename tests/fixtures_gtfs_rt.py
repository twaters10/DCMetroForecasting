"""Build tiny synthetic GTFS-realtime messages for tests.

The real feeds are 6 KB–1 MB per snapshot and need network and S3. Everything the
derivation logic does can be exercised on a handful of hand-built entities instead,
which makes the tests fast, offline, and — more importantly — able to construct the
exact situations that broke the pipeline on real data: a stop passed between polls, a
vehicle dropping out of the feed, a trip_id reused for a second journey.

These build real `FeedMessage` protobufs, not mocks, so they go through the same
`decode.py` parsing as production data.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from google.transit import gtfs_realtime_pb2

# A fixed instant so every test is deterministic. A weekday, well inside the archive,
# on a date with no DST transition.
BASE_TIME = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)
SERVICE_DATE = "20260805"


def at(minutes: int) -> datetime:
    """Capture time `minutes` after the base instant — one snapshot per minute."""
    return BASE_TIME + timedelta(minutes=minutes)


def epoch(moment: datetime) -> int:
    return int(moment.timestamp())


def vehicle_positions(
    entities: list[dict[str, object]],
    captured_at: datetime | None = None,
) -> gtfs_realtime_pb2.FeedMessage:
    """Build a VehiclePositions FeedMessage.

    Each entity dict takes `trip_id`, `stop_sequence`, `status`, and optionally
    `route_id`, `stop_id`, `vehicle_id`, `direction_id`, `start_date`.
    """
    moment = captured_at or BASE_TIME
    message = gtfs_realtime_pb2.FeedMessage()
    message.header.gtfs_realtime_version = "2.0"
    message.header.timestamp = epoch(moment)

    for index, spec in enumerate(entities):
        entity = message.entity.add()
        entity.id = str(index)
        v = entity.vehicle
        v.trip.trip_id = str(spec["trip_id"])
        v.trip.route_id = str(spec.get("route_id", "RED"))
        v.trip.direction_id = int(spec.get("direction_id", 0))
        v.trip.start_date = str(spec.get("start_date", SERVICE_DATE))
        v.trip.start_time = "08:00:00"
        v.vehicle.id = str(spec.get("vehicle_id", "V1"))
        if spec.get("stop_sequence") is not None:
            v.current_stop_sequence = int(spec["stop_sequence"])  # type: ignore[arg-type]
            v.stop_id = str(spec.get("stop_id", f"PF_{spec['stop_sequence']}"))
            v.current_status = getattr(
                gtfs_realtime_pb2.VehiclePosition, str(spec.get("status", "STOPPED_AT"))
            )
        v.timestamp = epoch(moment)
    return message


def trip_updates(
    entities: list[dict[str, object]],
    captured_at: datetime | None = None,
) -> gtfs_realtime_pb2.FeedMessage:
    """Build a TripUpdates FeedMessage.

    Each entity dict takes `trip_id` and `stops`, a list of
    `{stop_sequence, arrival_ts, ...}`. `arrival_ts` is a datetime or None.
    """
    moment = captured_at or BASE_TIME
    message = gtfs_realtime_pb2.FeedMessage()
    message.header.gtfs_realtime_version = "2.0"
    message.header.timestamp = epoch(moment)

    for index, spec in enumerate(entities):
        entity = message.entity.add()
        entity.id = str(index)
        tu = entity.trip_update
        tu.trip.trip_id = str(spec["trip_id"])
        tu.trip.route_id = str(spec.get("route_id", "RED"))
        tu.trip.direction_id = int(spec.get("direction_id", 0))
        tu.trip.start_date = str(spec.get("start_date", SERVICE_DATE))
        tu.vehicle.id = str(spec.get("vehicle_id", "V1"))
        tu.timestamp = epoch(moment)

        for stop in spec["stops"]:  # type: ignore[union-attr]
            stu = tu.stop_time_update.add()
            stu.stop_sequence = int(stop["stop_sequence"])
            stu.stop_id = str(stop.get("stop_id", f"PF_{stop['stop_sequence']}"))
            stu.schedule_relationship = getattr(
                gtfs_realtime_pb2.TripUpdate.StopTimeUpdate,
                str(stop.get("relationship", "SCHEDULED")),
            )
            if stop.get("arrival_ts") is not None:
                stu.arrival.time = epoch(stop["arrival_ts"])
                stu.arrival.uncertainty = 0
            if stop.get("departure_ts") is not None:
                stu.departure.time = epoch(stop["departure_ts"])
    return message


def progressing_vehicle(
    trip_id: str,
    sequences: list[tuple[int, int]],
    vehicle_id: str = "V1",
) -> list[tuple[datetime, gtfs_realtime_pb2.FeedMessage]]:
    """A vehicle advancing through stops, one snapshot per entry.

    `sequences` is a list of `(minute_offset, stop_sequence)`. Skipping a minute
    produces a feed dropout; skipping a sequence produces a stop passed between polls.
    Both are situations that broke the real pipeline, so both need to be constructible.
    """
    return [
        (
            at(minute),
            vehicle_positions(
                [
                    {
                        "trip_id": trip_id,
                        "stop_sequence": sequence,
                        "status": "STOPPED_AT",
                        "vehicle_id": vehicle_id,
                    }
                ],
                captured_at=at(minute),
            ),
        )
        for minute, sequence in sequences
    ]
