"""Single source of configuration for the metro-pulse collector.

Everything environment- or feed-specific lives here so the handler stays a thin
orchestration layer. Nothing in this module reads AWS credentials or contacts
the network at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

# --------------------------------------------------------------------------
# WMATA API
# --------------------------------------------------------------------------
# The key travels in a request header, not a query string, so it never lands in
# access logs. Confirmed against the live API, which advertises the header name
# in its 401 response:
#   WWW-Authenticate: AzureApiManagementKey realm="https://api.wmata.com/gtfs",
#                     name="api_key", type="header"
WMATA_API_KEY_HEADER: Final[str] = "api_key"
WMATA_GTFS_BASE_URL: Final[str] = "https://api.wmata.com/gtfs"

# (connect, read) seconds. This function is invoked every 60s, so a slow feed
# must never let one invocation overlap the next. Worst case with the feeds
# below is 4 x (3.05 + 7) ~= 40s, comfortably inside the 60s Lambda timeout.
HTTP_TIMEOUT_SECONDS: Final[tuple[float, float]] = (3.05, 7.0)

# A single fetch attempt per feed per invocation. No retries by design: the next
# poll is only 60 seconds away, and retrying inside the invocation would eat the
# timeout budget of the feeds that have not been fetched yet.

# Static GTFS gets its own, much longer read timeout. It runs on a separate daily
# schedule, so it does not share the 60s budget above — and it could not fit in
# it anyway: the bus bundle is ~50 MB, against ~1 MB for the largest realtime
# feed. The next static poll is 24 hours away rather than 60 seconds, so giving
# a slow transfer room to finish is worth more here than failing fast.
STATIC_HTTP_TIMEOUT_SECONDS: Final[tuple[float, float]] = (3.05, 120.0)

# --------------------------------------------------------------------------
# Archive format
# --------------------------------------------------------------------------
# Snapshots are gzipped before being written. This is lossless — objects
# decompress byte-identical — so the archive still holds the exact bytes WMATA
# served, just smaller. Measured across all four feeds: 3.4x overall (2.6x-4.8x
# per feed), taking the 90-day archive from ~145 GB to ~43 GB. The storage saving
# is minor; the real win is the local PySpark ETL, which pulls this archive down
# from S3 on every run.
#
# Level 6 (zlib's default) costs ~48 ms per invocation for all four feeds, which
# is noise against the 60s timeout. Level 9 buys very little on protobuf for
# noticeably more CPU.
GZIP_COMPRESSION_LEVEL: Final[int] = 6

# The archive's file extension, in one place because the ETL keys off it.
SNAPSHOT_SUFFIX: Final[str] = ".pb.gz"


@dataclass(frozen=True, slots=True)
class FeedSpec:
    """One GTFS-realtime feed to archive.

    `name` is used verbatim as both the top-level S3 partition and the filename
    stem, so it must stay stable — renaming it after data has landed splits the
    archive into two prefixes the ETL would have to union.
    """

    name: str
    path: str
    description: str

    @property
    def url(self) -> str:
        return f"{WMATA_GTFS_BASE_URL}/{self.path}"


# The feeds this collector archives. Adding a feed is a one-line change here;
# the handler iterates this tuple and needs no edit.
#
# WMATA also publishes service-alert feeds at `rail-gtfsrt-alerts.pb` and
# `bus-gtfsrt-alerts.pb` (both verified to exist). They are deliberately not
# collected yet — alerts are low-cardinality and change slowly, so polling them
# once a minute would mostly archive duplicates. The Bus & Rail Crowding feed is
# also out of scope; the ETL fetches it on demand rather than snapshotting it
# every minute. Static GTFS *is* archived, but on a daily schedule — see
# STATIC_FEEDS below.
FEEDS: Final[tuple[FeedSpec, ...]] = (
    FeedSpec(
        name="rail_vehicle_positions",
        path="rail-gtfsrt-vehiclepositions.pb",
        description="Rail vehicle positions — where each train is right now.",
    ),
    FeedSpec(
        name="rail_trip_updates",
        path="rail-gtfsrt-tripupdates.pb",
        description="Rail trip updates — predicted/actual stop times per trip.",
    ),
    FeedSpec(
        name="bus_vehicle_positions",
        path="bus-gtfsrt-vehiclepositions.pb",
        description="Bus vehicle positions — where each bus is right now.",
    ),
    FeedSpec(
        name="bus_trip_updates",
        path="bus-gtfsrt-tripupdates.pb",
        description="Bus trip updates — predicted/actual stop times per trip.",
    ),
)

# --------------------------------------------------------------------------
# Static GTFS
# --------------------------------------------------------------------------
# The scheduled timetable — the "scheduled" half of every actual-vs-scheduled
# delta the ETL computes. Archived daily rather than per-minute because it
# changes at most weekly.
#
# Archiving it at all is not optional, and this is the reason: static `trip_id`
# is a composite, `{scheduled_trip_id}_{schedule_version}`, and WMATA serves
# only the *current* bundle. Rail's window is about ten days (feed_info.txt
# reports feed_start_date/feed_end_date); after that the bundle is gone for
# good. A version bump happens because the timetable changed, so joining
# archived realtime against a later bundle can hand back the new scheduled time
# against the old actual — a wrong delay that looks entirely plausible. Missing
# a day here silently and permanently corrupts that day's labels.
#
# These reuse FeedSpec rather than defining a parallel type: the fields and the
# `url` property are identical, and `name` plays the same role — top-level
# archive partition and filename stem.
STATIC_FEEDS: Final[tuple[FeedSpec, ...]] = (
    FeedSpec(
        name="rail",
        path="rail-gtfs-static.zip",
        description="Rail scheduled timetable — ~3 MB, feed window ~10 days.",
    ),
    FeedSpec(
        name="bus",
        path="bus-gtfs-static.zip",
        description="Bus scheduled timetable — ~50 MB, feed window ~3 months.",
    ),
)

# Members every bundle must contain to be considered valid. This is the
# *intersection* of what the two modes actually ship, not the GTFS spec's list:
# rail publishes no `calendar.txt` (only `calendar_dates.txt`) and no
# `feed_version`, while bus publishes both. Requiring `calendar.txt` here would
# reject every rail bundle.
STATIC_REQUIRED_MEMBERS: Final[tuple[str, ...]] = (
    "trips.txt",
    "stop_times.txt",
    "stops.txt",
    "routes.txt",
    "feed_info.txt",
)


class ConfigError(RuntimeError):
    """Raised when required environment variables are missing or malformed."""


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    """Runtime configuration, resolved from the environment."""

    wmata_api_key: str
    s3_prefix: str
    s3_static_prefix: str
    s3_bucket: str | None = None
    local_output_dir: str | None = None

    @property
    def writes_locally(self) -> bool:
        return self.local_output_dir is not None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> CollectorConfig:
        """Build config from environment variables, failing fast and loudly.

        All missing variables are reported in one message rather than one per
        run — a misconfigured deploy should be diagnosable from a single log
        line. This raises rather than returning cleanly: unlike a flaky feed, a
        missing variable will not fix itself on the next poll.
        """
        source = os.environ if env is None else env

        api_key = _clean(source.get("WMATA_API_KEY"))
        bucket = _clean(source.get("S3_BUCKET"))
        prefix = _clean(source.get("S3_PREFIX"))
        static_prefix = _clean(source.get("S3_STATIC_PREFIX"))
        local_dir = _clean(source.get("LOCAL_OUTPUT_DIR"))

        missing = ["WMATA_API_KEY"] if not api_key else []
        if local_dir is None:
            # S3 mode (this is what runs in Lambda).
            if not bucket:
                missing.append("S3_BUCKET")
            if not prefix:
                missing.append("S3_PREFIX")

        if missing:
            raise ConfigError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Set them on the Lambda function, or in a local .env file "
                "(see .env.example)."
            )

        assert api_key is not None  # narrowed by the `missing` check above
        raw = _normalize_prefix(prefix or "raw/")
        static = _normalize_prefix(static_prefix or "static/")

        # The raw prefix carries a 90-day expiry lifecycle rule. Static bundles
        # nested under it would be deleted on that clock, and because WMATA only
        # serves the current bundle they could never be re-fetched — the archive
        # would lose the ability to compute correct deltas for anything older
        # than the current feed window. Cheap to catch here; unrecoverable if
        # discovered later.
        if static.startswith(raw) or raw.startswith(static):
            raise ConfigError(
                f"S3_STATIC_PREFIX ({static!r}) and S3_PREFIX ({raw!r}) must not "
                "nest. The raw prefix expires after 90 days and static GTFS "
                "bundles cannot be re-fetched once WMATA rotates them."
            )

        return cls(
            wmata_api_key=api_key,
            s3_prefix=raw,
            s3_static_prefix=static,
            s3_bucket=bucket,
            local_output_dir=local_dir,
        )


def _clean(value: str | None) -> str | None:
    """Treat empty/whitespace-only environment variables as unset."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_prefix(prefix: str) -> str:
    """Normalize to `some/prefix/` — no leading slash, exactly one trailing."""
    return prefix.strip().lstrip("/").rstrip("/") + "/"
