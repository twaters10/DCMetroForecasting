"""Draw the WMATA rail network and highlight the route a prediction describes.

Matplotlib rather than pydeck or `st.map`, for one reason that matters here: it renders
identically with no network access and no basemap tiles. The dashboard is a local tool
that already depends on reaching a SageMaker endpoint; making the map fail separately
when a tile server is slow would be a second, unrelated way for the page to look broken.

The route is drawn from the STOP SEQUENCE, not from the track shape. Highlighting a
slice of the shape would need the nearest point on the polyline to each station, and
gets visibly wrong where lines share track — the Blue and Yellow both run through
Pentagon on geometry that is metres apart. Walking the canonical stop order for the
leg's route and direction is exact by construction.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

GEOMETRY_PATH = Path("data/processed/serving/network_geometry.json")

STATION_PATTERN_LENGTH = slice(3, 6)  # "PF_C11_1" -> "C11"

# Everything not on the route fades back rather than disappearing: the point of a map
# is that a rider can see where the route sits in the network they already know.
IDLE_ALPHA = 0.30
IDLE_WIDTH = 1.8
ROUTE_WIDTH = 5.5

# About 4 km. Below this the view is zoomed so far that the surrounding network
# carries no information and the map stops being a map.
MIN_EXTENT_DEG = 0.035


@lru_cache(maxsize=1)
def load_geometry(path: str = str(GEOMETRY_PATH)) -> dict | None:
    file = Path(path)
    if not file.exists():
        return None
    return json.loads(file.read_text())


def _code(stop_id: str) -> str:
    return str(stop_id)[STATION_PATTERN_LENGTH]


def route_segments(result: dict) -> list[tuple[str, str, str]]:
    """(line, origin code, destination code) for each ride, in travel order."""
    legs = result.get("legs")
    if legs:
        return [
            (leg["line"], _code(leg["from_stop_id"]), _code(leg["to_stop_id"]))
            for leg in legs
            if leg["type"] == "ride"
        ]
    if result.get("line") and result.get("origin_stop_id"):
        return [
            (
                result["line"],
                _code(result["origin_stop_id"]),
                _code(result["destination_stop_id"]),
            )
        ]
    return []


def station_path(geometry: dict, line: str, origin: str, destination: str) -> list[str]:
    """The stations a leg passes through, in order.

    Direction is INFERRED rather than taken from the response: the leg payload does not
    carry `direction_id`, and the sequence that contains the origin before the
    destination is the one being travelled. Falling back to the two endpoints keeps a
    branch the canonical trip does not cover from erasing the whole overlay.
    """
    for direction in (0, 1):
        sequence = geometry["sequences"].get(f"{line}:{direction}")
        if not sequence or origin not in sequence or destination not in sequence:
            continue
        start, end = sequence.index(origin), sequence.index(destination)
        if start < end:
            return sequence[start : end + 1]
    return [origin, destination]


def _overlaps(a: tuple, b: tuple) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _place_labels(
    fig, ax, marks: list, ink: str, halo: str, fontsize: float = 9.5
) -> None:
    """Label each marked station, choosing an offset that does not collide.

    A fixed offset is not good enough here. The stations worth labelling are the
    origin, the change and the destination, and on a transfer through central DC the
    change and the destination can be a few hundred metres apart — at which point two
    labels print on top of each other and neither can be read. Each label takes the
    first of eight positions that is clear of the ones already placed.
    """
    from matplotlib.patheffects import withStroke

    fig.canvas.draw()
    scale = fig.dpi / 72
    candidates = (
        (11, 7, "left", "bottom"),
        (11, -9, "left", "top"),
        (-11, 7, "right", "bottom"),
        (-11, -9, "right", "top"),
        (0, 16, "center", "bottom"),
        (0, -18, "center", "top"),
        (18, 0, "left", "center"),
        (-18, 0, "right", "center"),
    )
    placed: list[tuple] = []
    for (lon, lat), label in marks:
        x, y = ax.transData.transform((lon, lat))
        width = 0.58 * fontsize * len(label) * scale
        height = 1.5 * fontsize * scale
        choice, box = candidates[0], None
        for dx, dy, ha, va in candidates:
            px, py = x + dx * scale, y + dy * scale
            left = (
                px if ha == "left" else px - width if ha == "right" else px - width / 2
            )
            bottom = (
                py
                if va == "bottom"
                else py - height if va == "top" else py - height / 2
            )
            trial = (left, bottom, left + width, bottom + height)
            if not any(_overlaps(trial, other) for other in placed):
                choice, box = (dx, dy, ha, va), trial
                break
        if box is None:
            # Nothing was clear; keep the default and record it so the NEXT label at
            # least avoids this one rather than compounding the pile-up.
            dx, dy, ha, va = choice
            px, py = x + dx * scale, y + dy * scale
            box = (px, py, px + width, py + height)
        placed.append(box)
        dx, dy, ha, va = choice
        ax.annotate(
            label,
            (lon, lat),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontweight="bold",
            color=ink,
            zorder=8,
            path_effects=[withStroke(linewidth=3.0, foreground=halo)],
        )


def render(result: dict, dark: bool = False):
    """A figure of the network with this route picked out, or None without geometry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = load_geometry()
    if geometry is None:
        return None

    ink = "#e6e6e6" if dark else "#1a1a1a"
    muted = "#8a8a8a"
    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    for line in geometry["lines"].values():
        path = line["path"]
        ax.plot(
            [p[0] for p in path],
            [p[1] for p in path],
            color=line["color"],
            alpha=IDLE_ALPHA,
            linewidth=IDLE_WIDTH,
            solid_capstyle="round",
            zorder=1,
        )

    stations = geometry["stations"]
    ax.scatter(
        [s["lon"] for s in stations.values()],
        [s["lat"] for s in stations.values()],
        s=7,
        color=muted,
        alpha=0.35,
        zorder=2,
        linewidths=0,
    )

    segments = route_segments(result)
    touched: list[str] = []
    for line, origin, destination in segments:
        codes = [
            c
            for c in station_path(geometry, line, origin, destination)
            if c in stations
        ]
        if len(codes) < 2:
            continue
        colour = geometry["lines"].get(line, {}).get("color", muted)
        # Drawn twice: a background stroke in the page colour, then the line itself.
        # Downtown has five routes inside a few hundred metres, and without the halo
        # the highlighted one is just another coloured line in the knot.
        halo = "#ffffff" if not dark else "#0e1117"
        for width, shade, order in (
            (ROUTE_WIDTH + 4.0, halo, 3),
            (ROUTE_WIDTH, colour, 4),
        ):
            ax.plot(
                [stations[c]["lon"] for c in codes],
                [stations[c]["lat"] for c in codes],
                color=shade,
                linewidth=width,
                solid_capstyle="round",
                zorder=order,
                alpha=1.0 if shade == halo else 0.97,
            )
        touched.extend(codes)

    if touched:
        ax.scatter(
            [stations[c]["lon"] for c in touched],
            [stations[c]["lat"] for c in touched],
            s=26,
            facecolor="white" if not dark else "#111111",
            edgecolor=ink,
            linewidths=1.1,
            zorder=5,
        )

    # Only the stations a rider has to act at are labelled. Naming all 98 turns the map
    # into a wall of text and hides the three that matter.
    marks: list[tuple[str, str]] = []
    if segments:
        marks.append((segments[0][1], result.get("origin", "")))
        for leg in result.get("legs", []):
            if leg["type"] == "transfer":
                marks.append((_code(leg["from_stop_id"]), f"\u21c4 {leg['at']}"))
        marks.append((segments[-1][2], result.get("destination", "")))

    marked = [(code, label) for code, label in marks if code in stations]
    if marked:
        ax.scatter(
            [stations[c]["lon"] for c, _ in marked],
            [stations[c]["lat"] for c, _ in marked],
            s=120,
            facecolor="white" if not dark else "#111111",
            edgecolor=ink,
            linewidths=2.0,
            zorder=7,
        )
        _place_labels(
            fig,
            ax,
            [((stations[c]["lon"], stations[c]["lat"]), label) for c, label in marked],
            ink,
            "#ffffff" if not dark else "#0e1117",
        )

    # Frame the route, not the network. The system runs from Ashburn to Largo, and a
    # central four-stop hop drawn at that scale is a smudge — which is exactly the case
    # where a rider most needs to see which way the line goes. The generous padding
    # keeps enough of the surrounding network visible to stay recognisable, and the
    # floor stops a two-stop journey zooming to the point of meaninglessness.
    if touched:
        lons = [stations[c]["lon"] for c in touched]
        lats = [stations[c]["lat"] for c in touched]
        span = max(max(lons) - min(lons), max(lats) - min(lats), MIN_EXTENT_DEG)
        pad = span * 0.45
        cx, cy = (max(lons) + min(lons)) / 2, (max(lats) + min(lats)) / 2
        ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
        ax.set_ylim(cy - span / 2 - pad, cy + span / 2 + pad)

    # Latitude degrees are a fixed distance; longitude degrees shrink by cos(lat). Using
    # a 1:1 aspect on raw degrees would stretch the region east-west by about a quarter.
    mid = sum(s["lat"] for s in stations.values()) / len(stations)
    ax.set_aspect(1 / math.cos(math.radians(mid)))
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig
