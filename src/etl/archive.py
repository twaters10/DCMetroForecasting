"""Reading the archive: realtime snapshots and static GTFS bundles.

This is the IO layer the rest of the ETL sits on. It is deliberately separate from
any analysis, because the two change for different reasons — how a snapshot is
fetched is settled, how an arrival is derived from it is not.

Two things here are load-bearing for correctness:

**Partition pruning.** Snapshot keys are enumerated by walking the requested UTC
hours and listing each `hour=HH/` prefix directly. Nothing ever lists the bucket
and filters afterwards: at ~43k objects per feed per month, a full listing costs
real money and minutes, and it silently gets slower as the archive grows.

**Static bundle resolution.** The bundle is read from the collector's `static/`
archive in S3, not downloaded from WMATA. WMATA serves only the current bundle, so
fetching live would join historical realtime data against a timetable that was not
in force. See docs/static-gtfs.md.
"""

from __future__ import annotations

import gzip
import io
import re
import zipfile
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pandas as pd
from google.transit import gtfs_realtime_pb2

from .config import SNAPSHOT_SUFFIX, EtlConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any

# `{feed_name}-{unix_timestamp}.pb.gz` — the capture time is in the filename, so a
# snapshot's timestamp never has to be inferred from its S3 LastModified (which is
# write time, not capture time, and differs by seconds).
_KEY_TIMESTAMP: Final[re.Pattern[str]] = re.compile(r"-(\d+)\.pb\.gz$")

# `static/{mode}/feed_start=YYYYMMDD/feed_end=YYYYMMDD/{mode}-gtfs-static-{hash}.zip`
_BUNDLE_KEY: Final[re.Pattern[str]] = re.compile(
    r"feed_start=(\d{8})/feed_end=(\d{8}|unknown)/"
)

# Static GTFS files the ETL reads. `calendar.txt` is deliberately absent: WMATA rail
# does not publish it (only `calendar_dates.txt`), so requiring it would reject every
# rail bundle.
_STATIC_TABLES: Final[tuple[str, ...]] = (
    "trips.txt",
    "stop_times.txt",
    "stops.txt",
    "routes.txt",
    "feed_info.txt",
)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One decoded GTFS-realtime snapshot, with the moment it was captured."""

    captured_at: datetime
    key: str
    message: gtfs_realtime_pb2.FeedMessage

    @property
    def entities(self) -> list[Any]:
        return list(self.message.entity)


@dataclass(frozen=True, slots=True)
class StaticBundle:
    """A static GTFS bundle, parsed into frames, with its provenance.

    `schedule_version` is the suffix shared by every `trips.trip_id` in the bundle
    (e.g. `20670`). It is the identifier that realtime `trip_id`s carry too, and
    comparing the two is the only way to tell whether a join is using the timetable
    that was actually in force. `feed_start_date` is what the key is filed under.
    """

    key: str
    feed_start_date: str
    feed_end_date: str
    schedule_version: str | None
    trips: pd.DataFrame
    stop_times: pd.DataFrame
    stops: pd.DataFrame
    routes: pd.DataFrame

    @property
    def version_label(self) -> str:
        """What gets recorded in the output as `static_gtfs_version`."""
        return f"{self.feed_start_date}/{self.schedule_version or 'unknown'}"


# --------------------------------------------------------------------------
# Realtime snapshots
# --------------------------------------------------------------------------


def hour_partitions(start: datetime, end: datetime) -> Iterator[str]:
    """Yield `year=/month=/day=/hour=` prefixes covering [start, end).

    Half-open on purpose: an hour range is naturally expressed as "from 12:00 up to
    but not including 15:00", and a closed interval would silently read a 61st
    partition at the boundary.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")

    cursor = start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    limit = end.astimezone(UTC)
    while cursor < limit:
        yield (f"year={cursor:%Y}/month={cursor:%m}/day={cursor:%d}/hour={cursor:%H}/")
        cursor += timedelta(hours=1)


def snapshot_keys_by_hour(
    s3: S3Client, config: EtlConfig, feed: str, start: datetime, end: datetime
) -> dict[str, list[str]]:
    """Map each hour partition to its snapshot keys, listing only those prefixes.

    Returned per-hour rather than flattened so the caller can see collector gaps:
    an hour with 12 files is downtime, and an hour missing entirely is a different
    problem from an hour that is merely short.
    """
    found: dict[str, list[str]] = {}
    for partition in hour_partitions(start, end):
        prefix = f"{config.raw_prefix}{feed}/{partition}"
        keys: list[str] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=config.s3_bucket, Prefix=prefix):
            keys.extend(
                obj["Key"]
                for obj in page.get("Contents", [])
                if obj["Key"].endswith(SNAPSHOT_SUFFIX)
            )
        found[partition] = sorted(keys)
    return found


def earliest_snapshot(s3: S3Client, config: EtlConfig, feed: str) -> datetime | None:
    """The oldest hour partition present for a feed, or None if the feed is empty.

    Descends `year=/month=/day=/hour=` one delimiter at a time, taking the minimum at
    each level — four list calls rather than enumerating ~130k objects.

    The catch-up driver needs this to know where the archive begins. Without it, a
    14-day lookback proposes dates from before the collector existed, each of which
    starts a Spark session only to fail on missing input, and the scheduled job reports
    failure every night. It also correctly excludes the collector's first, partial day.
    """
    prefix = f"{config.raw_prefix}{feed}/"
    parts: list[str] = []

    for _ in range(4):
        response = s3.list_objects_v2(
            Bucket=config.s3_bucket, Prefix=prefix, Delimiter="/"
        )
        children = sorted(item["Prefix"] for item in response.get("CommonPrefixes", []))
        if not children:
            return None
        prefix = children[0]
        parts.append(prefix.rstrip("/").rsplit("/", 1)[-1])

    values = {key: int(value) for key, value in (p.split("=") for p in parts)}
    return datetime(
        values["year"], values["month"], values["day"], values["hour"], tzinfo=UTC
    )


def modal_interval_seconds(keys: list[str]) -> int | None:
    """The most common gap between consecutive snapshots, in seconds.

    The collector's cadence is a deployment setting that changes over time — it moved
    from 60s to 30s — and the archive therefore spans several. Anything that needs to
    know the cadence must **measure** it from the data rather than read a constant,
    or it silently describes the wrong era.

    The mode rather than the mean, because a collector outage inserts one enormous gap
    that would drag an average far off the real cadence. Returns None for fewer than
    two snapshots, where there is no gap to measure.
    """
    if len(keys) < 2:
        return None

    stamps = sorted(captured_at_from_key(key) for key in keys)
    gaps = Counter(
        int((later - earlier).total_seconds())
        for earlier, later in zip(stamps, stamps[1:], strict=False)
    )
    gaps.pop(0, None)  # duplicate keys would otherwise vote for a zero-second cadence
    if not gaps:
        return None
    return gaps.most_common(1)[0][0]


def captured_at_from_key(key: str) -> datetime:
    """Extract capture time from a snapshot key, in UTC."""
    match = _KEY_TIMESTAMP.search(key)
    if match is None:
        raise ValueError(f"key does not carry a unix timestamp: {key}")
    return datetime.fromtimestamp(int(match.group(1)), tz=UTC)


def decode_snapshot(payload: bytes) -> gtfs_realtime_pb2.FeedMessage:
    """Decompress and parse one archived snapshot.

    The collector stores gzipped protobuf with no `ContentEncoding`, precisely so
    that every reader gets identical bytes and decompresses explicitly. That is
    what this does.
    """
    message = gtfs_realtime_pb2.FeedMessage()
    message.ParseFromString(gzip.decompress(payload))
    return message


def read_snapshots(
    s3: S3Client,
    config: EtlConfig,
    keys: list[str],
    max_workers: int = 16,
) -> Iterator[Snapshot]:
    """Fetch and decode snapshots, yielding them in chronological key order.

    Fetches concurrently because these are many small objects and the wall time is
    almost entirely S3 round-trip latency, not bandwidth or CPU. Results are
    reordered to match `keys` before yielding: arrival derivation depends on
    consecutive snapshots being seen in order, and a thread pool does not preserve
    submission order.
    """

    def fetch(key: str) -> tuple[str, bytes]:
        body = s3.get_object(Bucket=config.s3_bucket, Key=key)["Body"].read()
        return key, body

    ordered = sorted(keys, key=captured_at_from_key)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for key, payload in pool.map(fetch, ordered):
            yield Snapshot(
                captured_at=captured_at_from_key(key),
                key=key,
                message=decode_snapshot(payload),
            )


# --------------------------------------------------------------------------
# Static GTFS
# --------------------------------------------------------------------------


def list_static_bundles(
    s3: S3Client, config: EtlConfig, mode: str
) -> list[dict[str, str]]:
    """List archived bundles for a mode, newest feed window first.

    Reads the feed window straight out of the key, so choosing a bundle costs one
    list call and no downloads — the reason the collector puts it there.
    """
    prefix = f"{config.static_prefix}{mode}/"
    bundles: list[dict[str, str]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=config.s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            match = _BUNDLE_KEY.search(key)
            if match is None or not key.endswith(".zip"):
                continue
            bundles.append(
                {
                    "key": key,
                    "feed_start_date": match.group(1),
                    "feed_end_date": match.group(2),
                    "size": str(obj["Size"]),
                }
            )
    return sorted(bundles, key=lambda b: b["feed_start_date"], reverse=True)


def fetch_static_bundle(s3: S3Client, config: EtlConfig, key: str) -> Path:
    """Download a bundle to the local cache, or reuse the cached copy.

    Content-addressed keys make caching trivially safe: the hash is in the filename,
    so a cached file with a matching name cannot be stale content under a reused
    name.
    """
    destination = config.cache_dir / key
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download to a sibling temp path and rename, so an interrupted transfer cannot
    # leave a truncated zip that the cache check would then trust forever.
    staging = destination.with_suffix(".partial")
    s3.download_file(config.s3_bucket, key, str(staging))
    staging.replace(destination)
    return destination


def load_static_bundle(path: Path, key: str = "") -> StaticBundle:
    """Parse a bundle zip into frames.

    All columns are read as strings. GTFS ids are opaque tokens that merely happen
    to look numeric — inferring dtypes turns `stop_sequence` into an int (fine) but
    also risks mangling ids with leading zeros, and a join key whose dtype differs
    between the two sides silently matches nothing.
    """
    with zipfile.ZipFile(path) as archive:
        frames: dict[str, pd.DataFrame] = {}
        for name in _STATIC_TABLES:
            with archive.open(name) as handle:
                frames[name] = pd.read_csv(
                    io.BytesIO(handle.read()), dtype=str, keep_default_na=False
                )

    feed_info = frames["feed_info.txt"]
    trips = frames["trips.txt"]

    # Every trip_id in a bundle shares one version suffix; take it from the first
    # row rather than asserting, so a future mixed bundle degrades to a label rather
    # than crashing the run.
    version: str | None = None
    if not trips.empty:
        candidate = str(trips["trip_id"].iloc[0]).rsplit("_", 1)
        version = candidate[1] if len(candidate) == 2 else None

    return StaticBundle(
        key=key or str(path),
        feed_start_date=_first(feed_info, "feed_start_date"),
        feed_end_date=_first(feed_info, "feed_end_date") or "unknown",
        schedule_version=version,
        trips=trips,
        stop_times=frames["stop_times.txt"],
        stops=frames["stops.txt"],
        routes=frames["routes.txt"],
    )


def resolve_static_bundle(
    s3: S3Client, config: EtlConfig, mode: str, service_date: str | None = None
) -> StaticBundle:
    """Pick, cache and parse the bundle to join a service date against.

    Preference order, and the reasoning:

    1. A bundle whose `[feed_start_date, feed_end_date]` window contains
       `service_date` — the timetable WMATA said was in force.
    2. Otherwise the newest bundle at or before `service_date`, because a schedule
       published earlier still described that day until it was superseded.
    3. Otherwise the newest bundle at all.

    Cases 2 and 3 mean the scheduled times may not be the ones that were in force.
    That is not detectable from the feed window alone — the realtime feed lags the
    static one — so the caller must compare `schedule_version` against the suffix on
    the realtime `trip_id`s and report any disagreement.
    """
    bundles = list_static_bundles(s3, config, mode)
    if not bundles:
        raise FileNotFoundError(
            f"no static GTFS bundles archived under "
            f"s3://{config.s3_bucket}/{config.static_prefix}{mode}/. "
            "Run the collector's static task first: "
            'aws lambda invoke --payload \'{"task":"static_gtfs"}\' ...'
        )

    chosen = bundles[0]
    if service_date is not None:
        compact = service_date.replace("-", "")
        covering = [
            b
            for b in bundles
            if b["feed_start_date"] <= compact
            and (b["feed_end_date"] == "unknown" or compact <= b["feed_end_date"])
        ]
        earlier = [b for b in bundles if b["feed_start_date"] <= compact]
        if covering:
            chosen = covering[0]
        elif earlier:
            chosen = earlier[0]

    path = fetch_static_bundle(s3, config, chosen["key"])
    return load_static_bundle(path, key=chosen["key"])


def _first(frame: pd.DataFrame, column: str) -> str:
    """First value of a column, or empty string if absent/empty."""
    if column not in frame.columns or frame.empty:
        return ""
    return str(frame[column].iloc[0]).strip()
