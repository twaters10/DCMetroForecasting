"""Archive WMATA's static GTFS bundles — the scheduled side of the dataset.

Invoked once a day by its own EventBridge schedule, separately from the 60-second
realtime collection. `handler.lambda_handler` dispatches here on
`{"task": "static_gtfs"}`.

Why this exists at all: static `trip_id` is a composite,
`{scheduled_trip_id}_{schedule_version}`, and WMATA serves only the *current*
bundle — rail's feed window is about ten days. A version bump happens because the
timetable changed, so joining archived realtime against a later bundle can return
the new scheduled time against the old actual, producing a wrong delay that looks
entirely plausible. A day not archived is a day whose labels can never be
computed correctly. See docs/static-gtfs.md.

The two design rules from `handler.py` carry over unchanged:

1. **Never raise on a bad fetch.** A raised exception makes Lambda retry, and the
   next scheduled poll is only a day away. Failures are logged and swallowed;
   missing *configuration* still raises, because that will not fix itself.
2. **Store bytes, not interpretations.** The bundle is decoded only far enough to
   prove it is a real GTFS zip and to read its feed window; what gets written is
   the untouched response body. It is already compressed, so unlike the realtime
   snapshots it is not gzipped again.

This module is imported lazily by the handler, so nothing here — including its
imports — can affect the realtime collection path.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import requests

from config import (
    STATIC_FEEDS,
    STATIC_HTTP_TIMEOUT_SECONDS,
    STATIC_REQUIRED_MEMBERS,
    WMATA_API_KEY_HEADER,
    CollectorConfig,
    FeedSpec,
)
from writers import SnapshotWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Bundles are stored under their own extension, not SNAPSHOT_SUFFIX (`.pb.gz`).
STATIC_SUFFIX: Final[str] = ".zip"

# How much of the content hash goes in the key. 12 hex chars is 48 bits — ample
# to distinguish a few hundred bundles over the project's life, and short enough
# that a key stays readable in a console listing.
_HASH_CHARS: Final[int] = 12

# Used when a bundle omits `feed_end_date`. Both modes currently set it, but a
# missing value must not stop the archive: an unrecoverable bundle stored under
# an awkward key beats no bundle at all.
_UNKNOWN_DATE: Final[str] = "unknown"


@dataclass(frozen=True, slots=True)
class BundleInfo:
    """What `feed_info.txt` says about the bundle's validity window.

    `feed_version` is None for rail, which does not publish the field — the feed
    window is the only version identifier available across both modes, which is
    why the key is built from it rather than from `feed_version`.
    """

    feed_start_date: str
    feed_end_date: str
    feed_version: str | None


@dataclass(frozen=True, slots=True)
class StaticFeedResult:
    """Outcome of archiving one bundle, for logging and the handler's return."""

    feed: str
    status: str
    byte_size: int | None = None
    destination: str | None = None
    detail: str | None = None
    feed_start_date: str | None = None
    feed_end_date: str | None = None
    feed_version: str | None = None
    sha256: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def collect_static_feeds(
    config: CollectorConfig,
    writer: SnapshotWriter,
    captured_at: datetime,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Archive every bundle in `config.STATIC_FEEDS`. Never raises on I/O.

    Returns the same response shape as the realtime path so `deploy.sh`'s smoke
    test reads `feeds_failed` identically whichever task ran.
    """
    http = session if session is not None else requests.Session()

    results = [
        _collect_static_feed(spec, config, writer, captured_at, http)
        for spec in STATIC_FEEDS
    ]

    succeeded = sum(1 for r in results if r.ok)
    return {
        "task": "static_gtfs",
        "captured_at": captured_at.isoformat(),
        "feeds_ok": succeeded,
        "feeds_failed": len(results) - succeeded,
        "results": [
            {
                "feed": r.feed,
                "status": r.status,
                "byte_size": r.byte_size,
                "destination": r.destination,
                "detail": r.detail,
                "feed_start_date": r.feed_start_date,
                "feed_end_date": r.feed_end_date,
                "feed_version": r.feed_version,
                "sha256": r.sha256,
            }
            for r in results
        ],
    }


def _collect_static_feed(
    spec: FeedSpec,
    config: CollectorConfig,
    writer: SnapshotWriter,
    captured_at: datetime,
    session: requests.Session,
) -> StaticFeedResult:
    """Fetch, validate, and archive one bundle. Logs exactly one summary line."""
    result = _try_collect_static_feed(spec, config, writer, captured_at, session)

    if result.ok:
        logger.info(
            "static_feed=%s status=ok bytes=%s window=%s..%s version=%s dest=%s",
            result.feed,
            result.byte_size,
            result.feed_start_date,
            result.feed_end_date,
            result.feed_version,
            result.destination,
        )
    else:
        logger.warning(
            "static_feed=%s status=%s detail=%s",
            result.feed,
            result.status,
            result.detail,
        )
    return result


def _try_collect_static_feed(
    spec: FeedSpec,
    config: CollectorConfig,
    writer: SnapshotWriter,
    captured_at: datetime,
    session: requests.Session,
) -> StaticFeedResult:
    """The body of `_collect_static_feed`, minus logging. Returns, never raises."""
    try:
        response = session.get(
            spec.url,
            headers={WMATA_API_KEY_HEADER: config.wmata_api_key},
            timeout=STATIC_HTTP_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return StaticFeedResult(spec.name, "network_error", detail=_describe(exc))

    if response.status_code != 200:
        # The status code but not the body: a WMATA error body can be an HTML
        # page, and these are ~50 MB responses on the happy path.
        return StaticFeedResult(
            spec.name, "http_error", detail=f"http_status={response.status_code}"
        )

    payload = response.content
    info = read_bundle_info(payload)
    if info is None:
        # A 200 that is not a usable GTFS bundle — an error page, a truncated
        # transfer, or a zip missing files the ETL needs. Writing it would poison
        # the archive with something that looks like a real bundle.
        return StaticFeedResult(
            spec.name,
            "parse_error",
            byte_size=len(payload),
            detail="not_a_gtfs_bundle",
        )

    digest = hashlib.sha256(payload).hexdigest()
    key = build_static_key(config.s3_static_prefix, spec.name, info, digest)

    try:
        destination = writer.write(
            key,
            payload,
            content_type="application/zip",
            metadata={
                "feed-start-date": info.feed_start_date,
                "feed-end-date": info.feed_end_date,
                "feed-version": info.feed_version or "",
                "fetched-at": captured_at.isoformat(),
                "sha256": digest,
            },
        )
    except Exception as exc:  # noqa: BLE001 - a failed write must not retry
        return StaticFeedResult(spec.name, "write_error", detail=_describe(exc))

    return StaticFeedResult(
        feed=spec.name,
        status="ok",
        byte_size=len(payload),
        destination=destination,
        feed_start_date=info.feed_start_date,
        feed_end_date=info.feed_end_date,
        feed_version=info.feed_version,
        sha256=digest,
    )


def read_bundle_info(payload: bytes) -> BundleInfo | None:
    """Return the bundle's feed window, or None if `payload` is not usable.

    Three things are checked, in increasing cost order: that the bytes open as a
    zip, that every member the ETL depends on is present, and that
    `feed_info.txt` parses with a non-empty `feed_start_date`. The last matters
    most — the feed window is the archive's only version identifier, so a bundle
    without it cannot be filed correctly even though it is otherwise intact.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (zipfile.BadZipFile, OSError):
        return None

    members = set(archive.namelist())
    if not set(STATIC_REQUIRED_MEMBERS).issubset(members):
        return None

    try:
        with archive.open("feed_info.txt") as handle:
            rows = list(csv.DictReader(io.TextIOWrapper(handle, "utf-8-sig")))
    except (KeyError, OSError, UnicodeDecodeError, csv.Error):
        return None

    if not rows:
        return None

    row = rows[0]
    start = (row.get("feed_start_date") or "").strip()
    if not start:
        return None

    # `.strip() or None` rather than `.get(...)`: rail omits feed_version
    # entirely, and bus could plausibly ship it empty. Both mean "no version".
    version = (row.get("feed_version") or "").strip() or None
    end = (row.get("feed_end_date") or "").strip() or _UNKNOWN_DATE

    return BundleInfo(feed_start_date=start, feed_end_date=end, feed_version=version)


def build_static_key(prefix: str, mode: str, info: BundleInfo, sha256_hex: str) -> str:
    """Build the content-addressed object key for one bundle.

    `{prefix}{mode}/feed_start=YYYYMMDD/feed_end=YYYYMMDD/{mode}-gtfs-static-{hash}.zip`

    Two properties matter here.

    **Idempotent without reading S3.** The same bundle hashes to the same key, so
    the daily write is a byte-identical overwrite rather than a duplicate. That
    is not a stylistic choice: the execution role grants `s3:PutObject` and
    nothing else — no read, list, or delete — so an "already archived?" check
    would fail with AccessDenied. Storage therefore grows per *distinct* bundle,
    roughly one a week for rail and one a quarter for bus, not once a day.

    **Resolvable by listing alone.** The feed window sits in the key, so the ETL
    can pick the bundle covering a given service_date from one `list_objects_v2`
    without downloading anything to read `feed_info.txt`. The ETL runs locally
    under a full-access profile, so listing is available to it.
    """
    return (
        f"{prefix}{mode}/"
        f"feed_start={info.feed_start_date}/feed_end={info.feed_end_date}/"
        f"{mode}-gtfs-static-{sha256_hex[:_HASH_CHARS]}{STATIC_SUFFIX}"
    )


def _describe(exc: BaseException) -> str:
    """Compact, single-line exception summary — type plus message, no traceback."""
    message = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
