"""Configuration for the journey layer.

A **separate pipeline**, deliberately. It consumes the per-segment feature table and
leaves it completely untouched, so the segment model keeps training on exactly the data
it trains on today and the two approaches stay comparable. Nothing here writes to
`data/processed/features/`.

Journeys are *derived*, never collected: `journeys = f(segments)`. The table is a linear
re-expression of the segment table plus origin-side features, so it is fully
reproducible, is not authoritative, and — at ~2.5M rows — is deliberately **not** synced
to S3. Only `processed/segments/` is authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# The journey target: elapsed time from the train arriving at A to arriving at B.
# Equals the sum of `actual_duration_sec` over the constituent segments exactly, because
# `actual_departure_ts` is the upstream arrival so consecutive segments telescope.
JOURNEY_TARGET: Final[str] = "journey_duration_sec"

# Identifies one journey row.
JOURNEY_KEY: Final[tuple[str, ...]] = (
    "service_date",
    "trip_id",
    "trip_run",
    "origin_stop_id",
    "destination_stop_id",
)

# Enumerating every window is b(b+1)/2 per block — 4.1M rows, most of them
# near-duplicate lengths. These ten span 1 segment (~2 min) to 17 (~36 min) and
# match the lengths `models.evaluate` already reports, so the journey model and the
# summed segment model are scored on exactly the same horizons.
DEFAULT_LENGTHS: Final[tuple[int, ...]] = (1, 2, 3, 4, 6, 8, 10, 12, 15, 17)


@dataclass(frozen=True, slots=True)
class JourneyConfig:
    """Every tunable in the journey layer."""

    lengths: tuple[int, ...] = DEFAULT_LENGTHS

    # Source is the per-segment feature table — read-only, never written back.
    features_path: str = "data/processed/features/table"
    output_path: str = "data/processed/journeys"

    # A journey is assigned to train/validation as a whole **block**, not row by row.
    # One block generates up to ~90 overlapping journeys sharing the same underlying
    # traversals; splitting them across the boundary would leak the same observations
    # into both sides and flatter the validation score.
    group_split_by_block: bool = True
