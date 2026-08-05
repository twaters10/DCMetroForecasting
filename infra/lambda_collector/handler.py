"""Lambda entrypoint: poll WMATA GTFS-realtime feeds, archive raw snapshots.

Invoked every 60 seconds by EventBridge Scheduler. Each invocation fetches every
feed in `config.FEEDS`, validates that the payload is a parseable GTFS-realtime
message, and writes the *original protobuf bytes* to a time-partitioned key.

Two design rules drive most of the code below:

1. **Never raise on a bad poll.** A raised exception makes Lambda retry, and a
   retry risks writing the same snapshot twice. Duplicate rows in a time series
   are harder to detect and repair than one missing minute, so transient
   failures are logged and swallowed. Missing *configuration*, by contrast, does
   raise — that will not fix itself on the next poll and should be loud.
2. **Store bytes, not interpretations.** The protobuf is decoded only to prove
   it parsed; what gets written is the untouched response body, gzipped. Gzip is
   lossless, so the archive still holds exactly what WMATA served — the ETL can
   be rewritten a year from now without having lost anything to a conversion.
"""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import requests
from google.transit import gtfs_realtime_pb2

from config import (
    FEEDS,
    GZIP_COMPRESSION_LEVEL,
    HTTP_TIMEOUT_SECONDS,
    SNAPSHOT_SUFFIX,
    WMATA_API_KEY_HEADER,
    CollectorConfig,
    FeedSpec,
)
from writers import LocalWriter, S3Writer, SnapshotWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Reused across warm invocations so TCP/TLS handshakes are not repaid every
# minute. Safe at module scope: it holds no per-invocation state.
_SESSION: Final[requests.Session] = requests.Session()

# Built on first use and cached for the life of the container.
_WRITER: SnapshotWriter | None = None


@dataclass(frozen=True, slots=True)
class FeedResult:
    """Outcome of collecting one feed, for logging and the handler's return."""

    feed: str
    status: str
    entity_count: int | None = None
    byte_size: int | None = None  # uncompressed, as served by WMATA
    stored_size: int | None = None  # gzipped, as written to the archive
    destination: str | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Collect every configured feed. Always returns; never raises on I/O."""
    config = CollectorConfig.from_env()  # fails fast if misconfigured
    writer = _get_writer(config)

    # One timestamp for the whole invocation, so all feeds from a single poll
    # share a partition and filename stem. That makes "the 12:03 snapshot" a
    # thing the ETL can join across feeds without a tolerance window.
    captured_at = datetime.now(UTC)

    results = [_collect_feed(spec, config, writer, captured_at) for spec in FEEDS]

    succeeded = sum(1 for r in results if r.ok)
    return {
        "captured_at": captured_at.isoformat(),
        "feeds_ok": succeeded,
        "feeds_failed": len(results) - succeeded,
        "results": [
            {
                "feed": r.feed,
                "status": r.status,
                "entity_count": r.entity_count,
                "byte_size": r.byte_size,
                "stored_size": r.stored_size,
                "destination": r.destination,
                "detail": r.detail,
            }
            for r in results
        ],
    }


def _collect_feed(
    spec: FeedSpec,
    config: CollectorConfig,
    writer: SnapshotWriter,
    captured_at: datetime,
) -> FeedResult:
    """Fetch, validate, and archive one feed. Logs exactly one summary line.

    One line per feed per invocation is a deliberate budget: this runs ~43k
    times a month across four feeds, and CloudWatch bills on ingested volume.
    The payload itself is never logged.
    """
    result = _try_collect_feed(spec, config, writer, captured_at)

    if result.ok:
        # `bytes` is what WMATA served, `stored` is what landed in the archive —
        # carrying both makes the compression ratio observable in CloudWatch
        # without any extra instrumentation.
        logger.info(
            "feed=%s status=ok entities=%s bytes=%s stored=%s dest=%s",
            result.feed,
            result.entity_count,
            result.byte_size,
            result.stored_size,
            result.destination,
        )
    else:
        logger.warning(
            "feed=%s status=%s detail=%s",
            result.feed,
            result.status,
            result.detail,
        )
    return result


def _try_collect_feed(
    spec: FeedSpec,
    config: CollectorConfig,
    writer: SnapshotWriter,
    captured_at: datetime,
) -> FeedResult:
    """The body of `_collect_feed`, minus logging. Returns instead of raising."""
    try:
        response = _SESSION.get(
            spec.url,
            headers={WMATA_API_KEY_HEADER: config.wmata_api_key},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        # Timeout, DNS failure, connection reset. Next poll is 60s away.
        return FeedResult(spec.name, "network_error", detail=_describe(exc))

    if response.status_code != 200:
        # Note the status code but not the body: a WMATA error body can be an
        # HTML page, and logging it every minute during an outage is expensive.
        return FeedResult(
            spec.name, "http_error", detail=f"http_status={response.status_code}"
        )

    payload = response.content
    entity_count = _validate_gtfs_realtime(payload)
    if entity_count is None:
        # A 200 that is not a GTFS-realtime message — an error page or a
        # truncated response. Writing it would poison the archive.
        return FeedResult(
            spec.name, "parse_error", byte_size=len(payload), detail="not_gtfs_rt"
        )

    # Compress only after validation: the archive should never contain anything
    # that did not parse as a FeedMessage, compressed or not.
    body = gzip.compress(payload, GZIP_COMPRESSION_LEVEL)

    key = build_key(config.s3_prefix, spec.name, captured_at)
    try:
        destination = writer.write(key, body)
    except Exception as exc:  # noqa: BLE001 - a failed write must not retry
        return FeedResult(spec.name, "write_error", detail=_describe(exc))

    return FeedResult(
        feed=spec.name,
        status="ok",
        entity_count=entity_count,
        byte_size=len(payload),
        stored_size=len(body),
        destination=destination,
    )


def _validate_gtfs_realtime(payload: bytes) -> int | None:
    """Return the entity count if `payload` is a GTFS-realtime FeedMessage.

    Returns None if it is not. Protobuf will happily "parse" some non-protobuf
    bytes into an empty message, so an empty parse is not proof of validity —
    the header's `gtfs_realtime_version` is checked as well. An otherwise valid
    feed with zero entities is legitimate (e.g. rail overnight) and is archived:
    "no vehicles were running" is itself a fact the ETL needs.
    """
    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        feed.ParseFromString(payload)
    except Exception:  # noqa: BLE001 - protobuf raises several unrelated types
        return None

    if not feed.header.gtfs_realtime_version:
        return None
    return len(feed.entity)


def build_key(prefix: str, feed_name: str, captured_at: datetime) -> str:
    """Build the time-partitioned object key for a snapshot.

    `{prefix}{feed}/year=YYYY/month=MM/day=DD/hour=HH/{feed}-{unix_ts}.pb.gz`

    Hive-style partitions in UTC. Partitioning costs nothing now and lets the
    PySpark ETL read one day without listing months of objects; retrofitting it
    after the archive is large means rewriting every key.
    """
    moment = captured_at.astimezone(UTC)
    return (
        f"{prefix}{feed_name}/"
        f"year={moment:%Y}/month={moment:%m}/day={moment:%d}/hour={moment:%H}/"
        f"{feed_name}-{int(moment.timestamp())}{SNAPSHOT_SUFFIX}"
    )


def _get_writer(config: CollectorConfig) -> SnapshotWriter:
    """Return the cached writer, building it on first use in this container."""
    global _WRITER
    if _WRITER is None:
        if config.writes_locally:
            assert config.local_output_dir is not None
            _WRITER = LocalWriter(config.local_output_dir)
        else:
            assert config.s3_bucket is not None
            _WRITER = S3Writer(config.s3_bucket)
    return _WRITER


def _describe(exc: BaseException) -> str:
    """Compact, single-line exception summary — type plus message, no traceback."""
    message = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
