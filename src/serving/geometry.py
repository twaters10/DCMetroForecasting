"""Network geometry for drawing the map: line shapes, station points, stop order.

Built from the archived GTFS bundle alongside the station index, and consumed only by
the local dashboard — it is deliberately NOT packaged into `model.tar.gz`. The
inference container has no use for coordinates, and the archive is already 5.5 MB.

Three things come out of here:

`lines`      one representative path per route, simplified. `shapes.txt` carries 282
             shapes and 115,029 points, most of them near-duplicate variants of the
             same track (Silver alone has 79). Drawing all of them is slow and looks
             no different, so the longest shape per route is kept and thinned.

`stations`   one point per station code, from `parent_station` coordinates.

`sequences`  the canonical stop order per route and direction, which is what turns
             "Vienna to Metro Center on Orange" into a list of stations to highlight.
             Taken from the longest trip on that route and direction, so a short
             turnback working does not truncate the line.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger("serving.geometry")

# Roughly 25 m at this latitude. Below that, thinning is invisible on any screen the
# dashboard will be viewed on, and the point count is what makes the map sluggish.
SIMPLIFY_TOLERANCE_DEG = 0.00025


def _perpendicular_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    if start == end:
        return math.dist(point, start)
    (x, y), (x1, y1), (x2, y2) = point, start, end
    numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    return numerator / math.dist(start, end)


def simplify(
    points: list[tuple[float, float]], tolerance: float
) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker, iteratively so a long line cannot blow the stack."""
    if len(points) < 3:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        worst, index = 0.0, -1
        for i in range(first + 1, last):
            distance = _perpendicular_distance(points[i], points[first], points[last])
            if distance > worst:
                worst, index = distance, i
        if worst > tolerance and index != -1:
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))
    return [p for p, k in zip(points, keep, strict=False) if k]


def build_geometry(
    stops: pd.DataFrame,
    shapes: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    stop_times: pd.DataFrame,
    station_pattern,
    code_to_name: dict[str, str],
) -> dict:
    rail = stops[stops["stop_id"].astype(str).str.startswith("PF_")].copy()
    rail["station"] = rail["stop_id"].str.extract(station_pattern)
    rail = rail[rail["station"].notna()]

    stations = {
        code: {
            "name": code_to_name.get(code, code),
            "lon": round(float(group["stop_lon"].mean()), 6),
            "lat": round(float(group["stop_lat"].mean()), 6),
        }
        for code, group in rail.groupby("station")
    }

    colours = routes.set_index("route_id")["route_color"].to_dict()
    longest = trips.groupby(["route_id", "shape_id"]).size().reset_index(name="trips")
    shape_lengths = shapes.groupby("shape_id").size()

    lines: dict[str, dict] = {}
    for route_id, group in longest.groupby("route_id"):
        # Longest by point count, not most-used: a heavily-run short turnback would
        # otherwise become "the Red line" and the map would lose half of it.
        candidates = group["shape_id"][group["shape_id"].isin(shape_lengths.index)]
        if candidates.empty:
            continue
        shape_id = shape_lengths.loc[candidates].idxmax()
        points = (
            shapes[shapes["shape_id"] == shape_id]
            .sort_values("shape_pt_sequence")[["shape_pt_lon", "shape_pt_lat"]]
            .to_numpy()
            .tolist()
        )
        thinned = simplify(
            [(float(x), float(y)) for x, y in points], SIMPLIFY_TOLERANCE_DEG
        )
        lines[route_id] = {
            "color": f"#{colours.get(route_id, '888888')}",
            "path": [[round(x, 6), round(y, 6)] for x, y in thinned],
        }
        logger.info(
            "%-7s %6d points -> %4d after simplify", route_id, len(points), len(thinned)
        )

    sequences: dict[str, list[str]] = {}
    ordered = stop_times.sort_values(["static_trip_id", "stop_sequence"])
    for (route_id, direction), group in ordered.groupby(["route_id", "direction_id"]):
        counts = group.groupby("static_trip_id").size()
        canonical = group[group["static_trip_id"] == counts.idxmax()]
        codes: list[str] = []
        for stop_id in canonical["stop_id"]:
            match = station_pattern.match(str(stop_id))
            if match and (not codes or codes[-1] != match.group(1)):
                codes.append(match.group(1))
        sequences[f"{route_id}:{int(direction)}"] = codes

    logger.info(
        "geometry: %d station(s), %d line(s), %d route-direction sequence(s)",
        len(stations),
        len(lines),
        len(sequences),
    )
    return {"stations": stations, "lines": lines, "sequences": sequences}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--stop-times", default=None)
    parser.add_argument("--output", default="data/processed/serving")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    from .stations import STATION_PATTERN, build_station_index

    bundle = (
        Path(args.bundle)
        if args.bundle
        else sorted(Path("data/cache/static").rglob("*.zip"))[-1]
    )
    with zipfile.ZipFile(bundle) as archive:
        tables = {
            name: pd.read_csv(io.BytesIO(archive.read(name)))
            for name in ("stops.txt", "shapes.txt", "trips.txt", "routes.txt")
        }
    stop_times_path = (
        Path(args.stop_times)
        if args.stop_times
        else sorted(Path("data/cache/schedule/rail").glob("*/stop_times.parquet"))[-1]
    )
    index = build_station_index(tables["stops.txt"])
    geometry = build_geometry(
        tables["stops.txt"],
        tables["shapes.txt"],
        tables["trips.txt"],
        tables["routes.txt"],
        pd.read_parquet(stop_times_path),
        STATION_PATTERN,
        index.code_to_name,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "network_geometry.json"
    path.write_text(json.dumps(geometry, separators=(",", ":"), sort_keys=True))
    logger.info("wrote %s (%.0f KB)", path, path.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
