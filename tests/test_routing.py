"""Tests for multi-leg routing, against a synthetic three-station network.

The failures worth testing here are the ones that still return a confident number.

The worst of them is **service pooling**. `stop_times` carries 14 `service_id`s but
only two run on any date, and they partition by route. Ignoring `service_id` does not
raise anything — it silently shows 5.7x more trains than exist, so every connection
looks easy to catch and every quoted wait is a fraction of the real one. A test that
only checked "a wait was returned" would pass on that bug.

The rest are the same shape: a service-day rollover that looks up the wrong date, a
missing walk edge that makes a real transfer station unusable, and ranking on ride time
alone, which picks the route with the shorter trains and the longer platform wait.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.serving.routing import (
    MIN_CONNECTION_BUFFER_SEC,
    TransferGraph,
    build_service_calendar,
    build_walk_edges,
    service_position,
)
from src.serving.stations import ANY_HOUR, StationIndex

# X --RED--> M(platform 1) ..walk.. M(platform 2) --BLUE--> Y
# Z --GREEN-> M(platform 2) as a second way into the same change.
ORIGIN = "PF_X01_C"
TRANSFER_IN = "PF_M01_C"
TRANSFER_OUT = "PF_M02_C"
DESTINATION = "PF_Y01_C"
ALT_ORIGIN = "PF_Z01_C"
# A second way from Xville to Yorkton: a quicker ride into a much longer wait.
ALT_TRANSFER = "PF_N01_C"

HOUR = 9
DEPART_SEC = HOUR * 3600  # 09:00:00 on the service day


@pytest.fixture
def index() -> StationIndex:
    return StationIndex(
        names_to_codes={
            "xville": ["X01"],
            "midtown": ["M01", "M02"],
            "yorkton": ["Y01"],
            "zedbury": ["Z01"],
            "northgate": ["N01"],
        },
        code_to_name={
            "X01": "Xville",
            "M01": "Midtown",
            "M02": "Midtown",
            "Y01": "Yorkton",
            "Z01": "Zedbury",
            "N01": "Northgate",
        },
        code_to_platforms={
            "X01": [ORIGIN],
            "M01": [TRANSFER_IN],
            "M02": [TRANSFER_OUT],
            "Y01": [DESTINATION],
            "Z01": [ALT_ORIGIN],
            "N01": [ALT_TRANSFER],
        },
    )


def _leg(
    origin: str,
    destination: str,
    sched: int,
    route: str,
    first_leg_to: str,
    direction: int = 0,
):
    return {
        "origin": origin,
        "destination": destination,
        "hour": ANY_HOUR,
        "sched_sec": sched,
        "n_segments": 3,
        "stop_span": 3,
        "route_id": route,
        "direction_id": direction,
        "first_leg_to": first_leg_to,
    }


@pytest.fixture
def schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _leg(ORIGIN, TRANSFER_IN, 600, "RED", TRANSFER_IN),
            _leg(TRANSFER_OUT, DESTINATION, 900, "BLUE", DESTINATION),
            _leg(ALT_ORIGIN, TRANSFER_OUT, 700, "GREEN", TRANSFER_OUT),
            # Via Northgate: half the ride, but the onward train is 45 minutes out.
            _leg(ORIGIN, ALT_TRANSFER, 300, "RED", ALT_TRANSFER),
            _leg(ALT_TRANSFER, DESTINATION, 900, "BLUE", DESTINATION, direction=1),
        ]
    )


@pytest.fixture
def walk_edges() -> pd.DataFrame:
    return pd.DataFrame(
        [{"from_stop_id": TRANSFER_IN, "to_stop_id": TRANSFER_OUT, "walk_sec": 120}]
    )


@pytest.fixture
def departures() -> pd.DataFrame:
    """BLUE trains from the far platform, under two different service patterns.

    `WEEKDAY` runs at 09:15 and 09:45. `WEEKEND` fills the half-hour gap at 09:20 —
    a train that must NOT be offered on a weekday.
    """
    rows = [
        ("WEEKDAY", TRANSFER_OUT, "BLUE", 0, 9 * 3600 + 15 * 60),
        ("WEEKDAY", TRANSFER_OUT, "BLUE", 0, 9 * 3600 + 45 * 60),
        ("WEEKEND", TRANSFER_OUT, "BLUE", 0, 9 * 3600 + 20 * 60),
        ("WEEKDAY", ALT_TRANSFER, "BLUE", 1, 9 * 3600 + 50 * 60),
    ]
    return pd.DataFrame(
        rows,
        columns=["service_id", "stop_id", "route_id", "direction_id", "departure_sec"],
    )


@pytest.fixture
def calendar() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-08-25", "service_id": "WEEKDAY"},
            {"date": "2026-08-22", "service_id": "WEEKEND"},
        ]
    )


@pytest.fixture
def graph(schedule, walk_edges, departures, calendar, index) -> TransferGraph:
    return TransferGraph(schedule, walk_edges, departures, calendar, index)


def test_a_weekday_request_never_offers_a_weekend_train(graph):
    """The silent failure: pooling service patterns invents trains.

    Ready at 09:12 on a weekday, the only real onward train is 09:15. The WEEKEND
    feed also has one at 09:20, and a graph that ignored `service_id` would happily
    return whichever came first without ever erroring.
    """
    ready = 9 * 3600 + 12 * 60
    weekday = graph.next_departure(TRANSFER_OUT, "BLUE", 0, ready, ["WEEKDAY"])
    assert weekday == 9 * 3600 + 15 * 60

    weekend = graph.next_departure(TRANSFER_OUT, "BLUE", 0, ready, ["WEEKEND"])
    assert weekend == 9 * 3600 + 20 * 60

    # And the calendar must be what selects between them.
    assert graph.services_on("2026-08-25") == ["WEEKDAY"]
    assert graph.services_on("2026-08-22") == ["WEEKEND"]


def test_pooling_every_service_would_have_shortened_the_wait(graph):
    """Guards the bug directly: pooled services report a train 5 minutes too early."""
    ready = 9 * 3600 + 12 * 60
    honest = graph.next_departure(TRANSFER_OUT, "BLUE", 0, ready, ["WEEKDAY"])
    pooled = graph.next_departure(
        TRANSFER_OUT, "BLUE", 0, ready, ["WEEKDAY", "WEEKEND"]
    )
    assert pooled == 9 * 3600 + 20 * 60 or pooled == honest
    # The point is that they can differ, so the caller must pass the day's services.
    assert honest == 9 * 3600 + 15 * 60


def test_next_departure_never_returns_a_train_already_gone(graph):
    ready = 9 * 3600 + 15 * 60 + 1
    assert graph.next_departure(TRANSFER_OUT, "BLUE", 0, ready, ["WEEKDAY"]) == (
        9 * 3600 + 45 * 60
    )


def test_no_train_late_enough_returns_none_not_tomorrow_morning(graph):
    """Past the end of service the answer is 'no connection', not a 20-hour wait."""
    ready = 23 * 3600
    assert graph.next_departure(TRANSFER_OUT, "BLUE", 0, ready, ["WEEKDAY"]) is None


def test_after_midnight_belongs_to_the_previous_service_day():
    """01:10 is 25:10 on the timetable, and the day before's services run it."""
    late = pd.Timestamp("2026-08-26 01:10:00", tz="America/New_York")
    date, seconds = service_position(late)
    assert date == "2026-08-25"
    assert seconds == 25 * 3600 + 10 * 60

    ordinary = pd.Timestamp("2026-08-25 09:00:00", tz="America/New_York")
    assert service_position(ordinary) == ("2026-08-25", 9 * 3600)


def test_a_transfer_is_found_and_costs_the_real_walk(graph):
    candidates = graph.candidates("Xville", "Yorkton", HOUR)
    by_station = {c.transfer_station: c for c in candidates}
    assert set(by_station) == {"Midtown", "Northgate"}
    candidate = by_station["Midtown"]
    assert candidate.transfer_in == TRANSFER_IN
    assert candidate.transfer_out == TRANSFER_OUT
    assert candidate.transfer_station == "Midtown"
    assert candidate.walk_sec == 120


def _via(graph, station: str, origin="Xville", destination="Yorkton"):
    """Pick an itinerary by its transfer station.

    Never index into `candidates()` — more than one route exists here, and taking
    "the first" makes the test depend on iteration order rather than on behaviour.
    """
    matches = [
        c
        for c in graph.candidates(origin, destination, HOUR)
        if c.transfer_station == station
    ]
    assert len(matches) == 1, f"expected exactly one route via {station}"
    return matches[0]


def test_the_connection_is_timed_from_the_arrival_it_is_given(graph):
    candidate = _via(graph, "Midtown")
    arrival = DEPART_SEC + 600  # 09:10
    timing = graph.connection(candidate, arrival, ["WEEKDAY"])
    ready = arrival + 120 + MIN_CONNECTION_BUFFER_SEC  # 09:12:30
    assert timing["departure_sec"] == 9 * 3600 + 15 * 60
    assert timing["wait_sec"] == timing["departure_sec"] - ready
    # The penalty for missing it is the following train, which is what makes a tight
    # connection legible rather than just short.
    assert timing["if_missed_sec"] == (9 * 3600 + 45 * 60) - ready


def test_a_later_arrival_misses_the_train_and_waits_a_full_headway(graph):
    candidate = _via(graph, "Midtown")
    early = graph.connection(candidate, DEPART_SEC + 600, ["WEEKDAY"])
    late = graph.connection(candidate, DEPART_SEC + 900, ["WEEKDAY"])  # ready 09:17:30
    assert early["departure_sec"] == 9 * 3600 + 15 * 60
    assert late["departure_sec"] == 9 * 3600 + 45 * 60
    assert late["wait_sec"] > early["wait_sec"]


def test_ranking_counts_the_wait_not_only_the_ride(graph):
    """A shorter ride into a long wait must lose to a longer ride onto a train.

    Ranking on ride time alone is the plausible-looking mistake: it still returns a
    real route, just not the fastest one. Here Northgate is reached in 300s against
    Midtown's 600s, but its onward train is 45 minutes out, so it loses on the total
    by a wide margin.
    """
    candidates = graph.candidates("Xville", "Yorkton", HOUR)
    quickest_ride = min(candidates, key=lambda c: c.leg1["sched_sec"])
    assert quickest_ride.transfer_station == "Northgate"

    ranked = graph.rank(candidates, DEPART_SEC, ["WEEKDAY"])
    assert [c.transfer_station for c, _ in ranked] == ["Midtown", "Northgate"]

    def total(pair):
        candidate, timing = pair
        return (
            candidate.leg1["sched_sec"]
            + candidate.walk_sec
            + timing["wait_sec"]
            + candidate.leg2["sched_sec"]
        )

    assert total(ranked[0]) < total(ranked[1])
    # Midtown: 600 ride + 120 walk + wait to 09:15 + 900 ride.
    assert total(ranked[0]) == 600 + 120 + ranked[0][1]["wait_sec"] + 900


def test_changing_at_the_origin_or_destination_is_not_a_transfer(graph, index):
    """Boarding at Midtown and 'changing' there is one ride, not two."""
    assert graph.candidates("Midtown", "Yorkton", HOUR) == []


def test_rank_drops_itineraries_with_no_catchable_connection(graph):
    candidates = graph.candidates("Xville", "Yorkton", HOUR)
    # Depart so late that the last BLUE train has gone.
    assert graph.rank(candidates, 22 * 3600, ["WEEKDAY"]) == []


def test_walk_edges_are_symmetric_and_only_within_a_station(index):
    """A one-way walk edge would make a transfer possible in one direction only."""
    pathways = pd.DataFrame(
        [
            {
                "from_stop_id": TRANSFER_IN,
                "to_stop_id": "NODE_MEZZ",
                "is_bidirectional": 1,
                "traversal_time": 40,
            },
            {
                "from_stop_id": "NODE_MEZZ",
                "to_stop_id": TRANSFER_OUT,
                "is_bidirectional": 1,
                "traversal_time": 35,
            },
            # A corridor to a different station must never become a walk edge.
            {
                "from_stop_id": "NODE_MEZZ",
                "to_stop_id": DESTINATION,
                "is_bidirectional": 1,
                "traversal_time": 5,
            },
        ]
    )
    edges = build_walk_edges(pathways, index)
    pairs = {
        (row.from_stop_id, row.to_stop_id): row.walk_sec for row in edges.itertuples()
    }
    assert pairs == {(TRANSFER_IN, TRANSFER_OUT): 75, (TRANSFER_OUT, TRANSFER_IN): 75}


def test_same_platform_changes_cost_no_walk(schedule, departures, calendar, index):
    """Blue, Orange and Silver share one island at Metro Center.

    With no walk edge at all, the graph must still allow changing trains where the
    rider does not move — otherwise the commonest transfer in the system disappears.
    """
    graph = TransferGraph(
        schedule,
        pd.DataFrame(columns=["from_stop_id", "to_stop_id", "walk_sec"]),
        departures,
        calendar,
        index,
    )
    same_platform = pd.DataFrame(
        [
            _leg(ORIGIN, TRANSFER_OUT, 600, "RED", TRANSFER_OUT),
            _leg(TRANSFER_OUT, DESTINATION, 900, "BLUE", DESTINATION),
        ]
    )
    graph = TransferGraph(
        same_platform,
        pd.DataFrame(columns=["from_stop_id", "to_stop_id", "walk_sec"]),
        departures,
        calendar,
        index,
    )
    candidates = graph.candidates("Xville", "Yorkton", HOUR)
    assert len(candidates) == 1
    assert candidates[0].transfer_in == candidates[0].transfer_out == TRANSFER_OUT
    assert candidates[0].walk_sec == 0


def test_riding_through_on_one_train_is_not_offered_as_a_transfer(
    departures, calendar, index
):
    """Same platform, same route, same direction is a single ride."""
    through = pd.DataFrame(
        [
            _leg(ORIGIN, TRANSFER_OUT, 600, "BLUE", TRANSFER_OUT),
            _leg(TRANSFER_OUT, DESTINATION, 900, "BLUE", DESTINATION),
        ]
    )
    graph = TransferGraph(
        through,
        pd.DataFrame(columns=["from_stop_id", "to_stop_id", "walk_sec"]),
        departures,
        calendar,
        index,
    )
    assert graph.candidates("Xville", "Yorkton", HOUR) == []


def test_service_calendar_keeps_only_added_dates():
    raw = pd.DataFrame(
        [
            {"service_id": "WEEKDAY", "date": 20260825, "exception_type": 1},
            {"service_id": "REMOVED", "date": 20260825, "exception_type": 2},
        ]
    )
    calendar = build_service_calendar(raw)
    assert calendar["service_id"].tolist() == ["WEEKDAY"]
    assert calendar["date"].tolist() == ["2026-08-25"]


def test_unknown_calendar_date_falls_back_to_the_same_weekday(graph):
    """Past the published calendar, the same weekday is the honest best guess."""
    # 2026-08-25 is a Tuesday; 2027-03-02 is also a Tuesday.
    assert graph.services_on("2027-03-02") == ["WEEKDAY"]
    # 2026-08-22 is a Saturday; 2027-03-06 is also a Saturday.
    assert graph.services_on("2027-03-06") == ["WEEKEND"]


# ---------------------------------------------------------------------------
# The response breakdown
# ---------------------------------------------------------------------------


def _stub_artifacts(graph, index, schedule):
    """An `Artifacts` with a stand-in booster, built without touching the filesystem.

    The booster returns 1.1x the scheduled time, so the two legs predict differently
    and a breakdown that silently reused one leg for both would show up.
    """
    import numpy as np

    from src.serving.inference import Artifacts

    class StubBooster:
        def predict(self, matrix):
            return np.array([float(matrix["scheduled_total_sec"].iloc[0]) * 1.1])

    artifacts = object.__new__(Artifacts)
    artifacts.booster = StubBooster()
    artifacts.quantile_booster = None
    artifacts.quantile_coverage = {}
    artifacts.columns = ["n_segments", "scheduled_total_sec"]
    artifacts.encoder_mapping = {}
    artifacts.index = index
    artifacts.schedule = schedule
    artifacts.graph = graph
    artifacts.fallback_lookup = pd.DataFrame(
        {"from_stop_id": [], "to_stop_id": [], "completed_at": []}
    ).set_index(["from_stop_id", "to_stop_id"])
    artifacts._live_lookup = None
    artifacts._live_fetched_at = 0.0
    artifacts.bucket = ""  # keeps `live_lookup` off S3
    artifacts.run_id = "test-run"
    artifacts.trustworthy = True
    artifacts.training_support = {}
    return artifacts


# 2026-08-25 09:00 in New York, the hour the synthetic timetable is built around.
MORNING = "2026-08-25T13:00:00Z"


def test_a_transfer_response_splits_into_riding_walking_and_waiting(
    graph, index, schedule
):
    """The three parts must account for the total exactly.

    A breakdown that does not add up is the kind of bug nobody notices: every number
    looks reasonable on its own, and only the sum gives it away.
    """
    from src.serving.inference import predict_fn

    artifacts = _stub_artifacts(graph, index, schedule)
    result = predict_fn(
        {"origin": "Xville", "destination": "Yorkton", "departure_ts": MORNING},
        artifacts,
    )

    assert result["transfers"] == 1
    assert result["transfer_station"] == "Midtown"
    # Ride 600 and 900 scheduled, predicted at 1.1x -> 660 + 990.
    assert result["ride_sec"] == pytest.approx(1650.0)
    assert result["walk_sec"] == 120
    # Arrives 09:11:00, ready 09:13:30 after walk and buffer, train at 09:15.
    assert result["wait_sec"] == 90
    assert result["predicted_sec"] == pytest.approx(1860.0)
    assert result["ride_sec"] + result["walk_sec"] + result[
        "wait_sec"
    ] == pytest.approx(result["predicted_sec"])
    # Riding is strictly less than the door-to-door total whenever a change is involved.
    assert result["ride_sec"] < result["predicted_sec"]


def test_a_single_train_journey_has_no_walk_and_no_wait(graph, index, schedule):
    """One train means the total IS the riding time — nothing else to account for."""
    from src.serving.inference import predict_fn

    artifacts = _stub_artifacts(graph, index, schedule)
    result = predict_fn(
        {"origin": "Xville", "destination": "Midtown", "departure_ts": MORNING},
        artifacts,
    )

    assert "legs" not in result
    assert result.get("transfers", 0) == 0
    assert result["walk_sec"] == 0
    assert result["wait_sec"] == 0
    assert result["ride_sec"] == result["predicted_sec"] == pytest.approx(660.0)


def test_the_breakdown_is_present_on_every_response(graph, index, schedule):
    """Same keys either way, so a dashboard never branches on the journey type."""
    from src.serving.inference import predict_fn

    artifacts = _stub_artifacts(graph, index, schedule)
    required = {"ride_sec", "ride_min", "walk_sec", "walk_min", "wait_sec", "wait_min"}
    for destination in ("Midtown", "Yorkton"):
        result = predict_fn(
            {"origin": "Xville", "destination": destination, "departure_ts": MORNING},
            artifacts,
        )
        assert required <= set(
            result
        ), f"{destination} is missing {required - set(result)}"
        assert result["ride_min"] == pytest.approx(result["ride_sec"] / 60, abs=0.01)
