"""Single source of configuration for the ETL.

Mirrors the collector's `infra/lambda_collector/config.py`: nothing here reads
credentials or touches the network at import time, and every environment-specific
value is resolved in one place rather than scattered through the pipeline.

Paths are never hardcoded. The archive location comes from the same `.env` the
collector uses, so the ETL and the collector cannot drift onto different buckets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------
# The archive is partitioned in UTC because that is what the collector writes and
# what makes partition pruning unambiguous. Everything the *model* cares about is
# local: "the 08:00 rush" is 08:00 in Washington DC, not in UTC, and the offset
# moves twice a year. Conversion happens at exactly one point in the pipeline —
# when service_date and time-of-day features are derived — and never in storage.
ARCHIVE_TZ: Final[ZoneInfo] = ZoneInfo("UTC")
SERVICE_TZ: Final[ZoneInfo] = ZoneInfo("America/New_York")

# --------------------------------------------------------------------------
# Feeds
# --------------------------------------------------------------------------
# Feed names match the collector's `FeedSpec.name` values exactly, because they
# are the top-level partition in the archive. Renaming one here silently reads an
# empty prefix rather than failing.
RAIL_VEHICLE_POSITIONS: Final[str] = "rail_vehicle_positions"
RAIL_TRIP_UPDATES: Final[str] = "rail_trip_updates"
BUS_VEHICLE_POSITIONS: Final[str] = "bus_vehicle_positions"
BUS_TRIP_UPDATES: Final[str] = "bus_trip_updates"

FEEDS_BY_MODE: Final[dict[str, tuple[str, str]]] = {
    "rail": (RAIL_VEHICLE_POSITIONS, RAIL_TRIP_UPDATES),
    "bus": (BUS_VEHICLE_POSITIONS, BUS_TRIP_UPDATES),
}

# What the collector appends to every snapshot. The ETL keys off it, so it lives
# next to the feed names rather than being spelled inline.
SNAPSHOT_SUFFIX: Final[str] = ".pb.gz"

# Fallback only. The real value is MEASURED per window by
# `archive.modal_interval_seconds` and passed through the stage-A summary, because the
# collector's cadence is a deployment setting that has already changed once (60s -> 30s)
# and the archive therefore spans both eras. This constant is used only when a window
# holds too few snapshots to measure a gap at all.
EXPECTED_SNAPSHOTS_PER_HOUR: Final[int] = 60

# The collector's original cadence, kept as the nominal scale for
# MAX_ARRIVAL_BRACKET_SEC below. Per-row quantization is NOT read from here — every
# segment carries its own `arrival_bracket_sec`, measured from the two snapshots that
# bracketed the arrival, which is correct whatever cadence produced it.
SAMPLING_QUANTIZATION_SEC: Final[int] = 60

# Widest gap between two consecutive observations of a vehicle that still yields a
# usable arrival estimate. A stop_sequence increment brackets the arrival between the
# two capture times, so the bracket width *is* the error bar. Measured on real rail
# data at 60s: 98.7% of transitions fall inside 180s, and the rest are vehicles that
# dropped out of the feed and reappeared minutes later — a gap in observation, not a
# fast train.
#
# Deliberately an ABSOLUTE bound rather than a multiple of the current cadence. What a
# consumer cares about is "how wrong can this arrival be", and 180s of uncertainty is
# 180s regardless of how many polls it spans. Halving the interval therefore does not
# move the bar; it just means more rows clear it, which is correct because they really
# are more precise.
MAX_ARRIVAL_BRACKET_SEC: Final[int] = 3 * SAMPLING_QUANTIZATION_SEC

# How an arrival was derived. Defined here rather than in `arrivals.py` because that
# module imports pyspark at module scope, and the live derivation runs in a Lambda that
# has no Spark. One definition shared by both implementations, not two that agree today.
SOURCE_VEHICLE_POSITION: Final[str] = "vehicle_position"
SOURCE_TRIP_UPDATE: Final[str] = "trip_update"

# WMATA sometimes sets `arrival.time` to 0 rather than omitting the field (473 of
# 629,030 stop_time_updates in a sampled 3-hour rail window). Zero is a valid
# protobuf int but not a valid 2026 timestamp, and it silently destroys any min/max
# over predictions. Anything below this floor is a sentinel, not a time.
MIN_PLAUSIBLE_EPOCH_SEC: Final[int] = 1_000_000_000

# How far past local midnight a service day keeps running. A GTFS service day is not a
# calendar day — that is why the spec allows stop times beyond 24:00:00 — so any window
# that ends at midnight drops the late-night tail of every day.
#
# Measured on this archive: trips carrying `start_date = D` still run 3.5 hours past
# local midnight, and the tail thins out naturally rather than being clipped by the
# sample window. Four hours covers that with margin. Raising it is cheap; the overlap it
# creates between consecutive days' windows is handled by per-window staging writes plus
# the dedup on read in the pipeline.
SERVICE_DAY_OVERHANG_HOURS: Final[int] = 4

# Non-revenue equipment moves. These carry a trip_id but no schedule, so they will
# never join to static GTFS and must be excluded *before* computing a match rate —
# otherwise normal non-revenue traffic is indistinguishable from a real regression.
NON_REVENUE_ROUTE_ID: Final[str] = "NR"


class ConfigError(RuntimeError):
    """Raised when required environment variables are missing or malformed."""


@dataclass(frozen=True, slots=True)
class EtlConfig:
    """Runtime configuration, resolved from the environment."""

    s3_bucket: str
    raw_prefix: str
    static_prefix: str
    processed_prefix: str
    cache_dir: Path
    aws_region: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> EtlConfig:
        """Build config from environment variables, failing fast and loudly.

        Loads the repo-root `.env` first, mirroring what
        `infra/lambda_collector/local_run.py` does for the collector. Without this the
        ETL only worked when the caller had already `source`d `.env` by hand — which
        every interactive command here happened to do, masking the gap until an
        unattended run hit it and died on a missing S3_BUCKET.

        An explicit `env` argument skips the file entirely, so tests stay hermetic.
        """
        if env is None:
            _load_dotenv()
        source = os.environ if env is None else env

        bucket = _clean(source.get("S3_BUCKET"))
        if not bucket:
            raise ConfigError(
                "Missing required environment variable: S3_BUCKET. Set it in the "
                "repo-root .env file (see .env.example)."
            )

        cache = _clean(source.get("ETL_CACHE_DIR")) or "data/cache"

        return cls(
            s3_bucket=bucket,
            raw_prefix=_normalize_prefix(_clean(source.get("S3_PREFIX")) or "raw/"),
            static_prefix=_normalize_prefix(
                _clean(source.get("S3_STATIC_PREFIX")) or "static/"
            ),
            processed_prefix=_normalize_prefix(
                _clean(source.get("S3_PROCESSED_PREFIX")) or "processed/"
            ),
            cache_dir=Path(cache).expanduser(),
            # boto3 resolves its own region from the profile, but pyarrow's
            # S3FileSystem does not read ~/.aws/config — it needs one explicitly or it
            # pays a bucket-location lookup on every open. Matches deploy.sh's default.
            aws_region=(
                _clean(source.get("AWS_REGION"))
                or _clean(source.get("AWS_DEFAULT_REGION"))
                or "us-east-1"
            ),
        )


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


def _load_dotenv(env_file: Path | None = None) -> None:
    """Load the repo-root `.env` into the process environment.

    `override=False` on purpose: a variable already set in the environment wins over
    the file, so `S3_BUCKET=other python -m src.etl.pipeline ...` behaves as expected
    and CI can inject config without editing a file.

    Falls back to a minimal parser if python-dotenv is absent, so the ETL keeps working
    on a bare install — the same accommodation `local_run.py` makes.
    """
    path = env_file or REPO_ROOT / ".env"
    if not path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return

    load_dotenv(path, override=False)


def _clean(value: str | None) -> str | None:
    """Treat empty/whitespace-only environment variables as unset."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_prefix(prefix: str) -> str:
    """Normalize to `some/prefix/` — no leading slash, exactly one trailing."""
    return prefix.strip().lstrip("/").rstrip("/") + "/"
