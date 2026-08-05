"""Unit tests for static GTFS archiving.

Everything here runs against synthetic in-memory zips. No network, no S3, no
`.env` — the real bundles are 3 MB and 50 MB, and a test suite that needs WMATA to
be up is a test suite that fails for reasons unrelated to the code.

The cases worth having are the ones that would otherwise fail silently in
production: a bundle filed under the wrong key, or an error page archived as if it
were a timetable.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import pytest
import requests

from config import STATIC_REQUIRED_MEMBERS, CollectorConfig, ConfigError
from static_gtfs import (
    BundleInfo,
    build_static_key,
    collect_static_feeds,
    read_bundle_info,
)

# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

# Mirrors the real rail bundle: no `calendar.txt`, no `feed_version`. Rail and bus
# differ in exactly these two ways, and it is the rail shape that a
# spec-literal implementation rejects.
RAIL_FEED_INFO = (
    "feed_publisher_name,feed_publisher_url,feed_lang,feed_start_date,feed_end_date\n"
    "WMATA,http://www.wmata.com,en,20260805,20260814\n"
)

# Mirrors the real bus bundle: has `feed_version`.
BUS_FEED_INFO = (
    "feed_publisher_name,feed_publisher_url,feed_lang,feed_start_date,"
    "feed_end_date,feed_version\n"
    "WMATA,http://www.wmata.com,en,20260621,20260912,S1000250\n"
)


def make_bundle(
    feed_info: str = RAIL_FEED_INFO,
    members: tuple[str, ...] = STATIC_REQUIRED_MEMBERS,
    extra: dict[str, str] | None = None,
    filler: str = "",
) -> bytes:
    """Build a minimal but structurally valid GTFS zip.

    `filler` changes the bytes without changing the parsed content, which is how
    the content-addressing tests produce two bundles that mean the same thing but
    must not share a key.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in members:
            archive.writestr(name, feed_info if name == "feed_info.txt" else "header\n")
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
        if filler:
            archive.writestr("filler.txt", filler)
    return buffer.getvalue()


@pytest.fixture
def config() -> CollectorConfig:
    return CollectorConfig(
        wmata_api_key="test-key",
        s3_prefix="raw/",
        s3_static_prefix="static/",
        s3_bucket="test-bucket",
    )


class RecordingWriter:
    """Captures what would have been written, instead of writing it."""

    def __init__(self) -> None:
        self.writes: list[dict[str, object]] = []

    def write(
        self,
        key: str,
        payload: bytes,
        content_type: str = "application/gzip",
        metadata: dict[str, str] | None = None,
    ) -> str:
        self.writes.append(
            {
                "key": key,
                "size": len(payload),
                "content_type": content_type,
                "metadata": metadata,
            }
        )
        return f"s3://test-bucket/{key}"


class StubResponse:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content


class StubSession:
    """Stands in for `requests.Session`, returning or raising per-call."""

    def __init__(self, outcome: object) -> None:
        self._outcome = outcome
        self.calls: list[str] = []

    def get(self, url: str, **_: object) -> StubResponse:
        self.calls.append(url)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        assert isinstance(self._outcome, StubResponse)
        return self._outcome


CAPTURED_AT = datetime(2026, 8, 5, 6, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# read_bundle_info — validation
# --------------------------------------------------------------------------


def test_accepts_rail_shaped_bundle_without_calendar_or_feed_version():
    """The rail/bus asymmetry a later refactor is most likely to break."""
    info = read_bundle_info(make_bundle(RAIL_FEED_INFO))

    assert info == BundleInfo(
        feed_start_date="20260805", feed_end_date="20260814", feed_version=None
    )


def test_accepts_bus_shaped_bundle_with_feed_version_and_calendar():
    info = read_bundle_info(
        make_bundle(BUS_FEED_INFO, extra={"calendar.txt": "header\n"})
    )

    assert info is not None
    assert info.feed_version == "S1000250"
    assert info.feed_start_date == "20260621"


def test_rejects_bytes_that_are_not_a_zip():
    """What an HTML error page served with a 200 looks like."""
    assert read_bundle_info(b"<html>rate limit exceeded</html>") is None


def test_rejects_bundle_missing_a_required_member():
    members = tuple(m for m in STATIC_REQUIRED_MEMBERS if m != "trips.txt")

    assert read_bundle_info(make_bundle(members=members)) is None


def test_rejects_bundle_with_empty_feed_start_date():
    """Intact but unfileable: the feed window is the only version identifier."""
    feed_info = "feed_publisher_name,feed_start_date,feed_end_date\nWMATA,,20260814\n"

    assert read_bundle_info(make_bundle(feed_info)) is None


def test_rejects_bundle_with_headers_but_no_rows():
    assert read_bundle_info(make_bundle("feed_start_date,feed_end_date\n")) is None


def test_missing_feed_end_date_falls_back_to_unknown():
    """Archive it anyway — an awkward key beats an unrecoverable lost bundle."""
    info = read_bundle_info(make_bundle("feed_start_date\n20260805\n"))

    assert info is not None
    assert info.feed_end_date == "unknown"


# --------------------------------------------------------------------------
# build_static_key — content addressing
# --------------------------------------------------------------------------


def test_key_layout():
    info = BundleInfo("20260805", "20260814", None)

    key = build_static_key("static/", "rail", info, "abcdef0123456789" * 4)

    assert key == (
        "static/rail/feed_start=20260805/feed_end=20260814/"
        "rail-gtfs-static-abcdef012345.zip"
    )


def test_key_uses_twelve_hash_characters():
    info = BundleInfo("20260805", "20260814", None)

    key = build_static_key("static/", "bus", info, "0" * 64)

    assert key.endswith("bus-gtfs-static-000000000000.zip")


def test_identical_bundles_produce_one_key_so_daily_writes_are_idempotent(config):
    """The property that keeps a daily schedule from creating 365 copies.

    It also has to hold without reading S3 — the execution role grants PutObject
    and nothing else, so an existence check is not available.
    """
    payload = make_bundle()
    writer = RecordingWriter()
    session = StubSession(StubResponse(200, payload))

    collect_static_feeds(config, writer, CAPTURED_AT, session)
    collect_static_feeds(config, writer, CAPTURED_AT, session)

    keys = {write["key"] for write in writer.writes}
    assert len(writer.writes) == 4  # two feeds, twice
    assert len(keys) == 2  # but only two distinct destinations


def test_changed_bundle_contents_produce_a_different_key():
    info = BundleInfo("20260805", "20260814", None)
    import hashlib

    first = hashlib.sha256(make_bundle(filler="a")).hexdigest()
    second = hashlib.sha256(make_bundle(filler="b")).hexdigest()

    assert build_static_key("static/", "rail", info, first) != build_static_key(
        "static/", "rail", info, second
    )


# --------------------------------------------------------------------------
# collect_static_feeds — failure handling
# --------------------------------------------------------------------------


def test_happy_path_writes_zip_content_type_and_metadata(config):
    writer = RecordingWriter()
    session = StubSession(StubResponse(200, make_bundle()))

    summary = collect_static_feeds(config, writer, CAPTURED_AT, session)

    assert summary["feeds_ok"] == 2
    assert summary["feeds_failed"] == 0
    write = writer.writes[0]
    assert write["content_type"] == "application/zip"
    assert write["metadata"]["feed-start-date"] == "20260805"
    assert write["metadata"]["fetched-at"] == CAPTURED_AT.isoformat()


def test_network_error_is_reported_not_raised(config):
    writer = RecordingWriter()
    session = StubSession(requests.ConnectionError("connection reset"))

    summary = collect_static_feeds(config, writer, CAPTURED_AT, session)

    assert summary["feeds_failed"] == 2
    assert all(r["status"] == "network_error" for r in summary["results"])
    assert writer.writes == []


def test_http_error_is_reported_and_nothing_written(config):
    writer = RecordingWriter()
    session = StubSession(StubResponse(429, b""))

    summary = collect_static_feeds(config, writer, CAPTURED_AT, session)

    assert summary["feeds_failed"] == 2
    assert summary["results"][0]["detail"] == "http_status=429"
    assert writer.writes == []


def test_unparseable_payload_is_never_archived(config):
    """A 200 carrying an error page must not enter the archive."""
    writer = RecordingWriter()
    session = StubSession(StubResponse(200, b"<html>error</html>"))

    summary = collect_static_feeds(config, writer, CAPTURED_AT, session)

    assert summary["feeds_failed"] == 2
    assert all(r["status"] == "parse_error" for r in summary["results"])
    assert writer.writes == []


def test_write_failure_is_reported_not_raised(config):
    class FailingWriter:
        def write(self, *_: object, **__: object) -> str:
            raise RuntimeError("AccessDenied")

    summary = collect_static_feeds(
        config,
        FailingWriter(),
        CAPTURED_AT,
        StubSession(StubResponse(200, make_bundle())),
    )

    assert summary["feeds_failed"] == 2
    assert all(r["status"] == "write_error" for r in summary["results"])


# --------------------------------------------------------------------------
# Configuration guard
# --------------------------------------------------------------------------

BASE_ENV = {
    "WMATA_API_KEY": "test-key",
    "S3_BUCKET": "test-bucket",
    "S3_PREFIX": "raw/",
}


def test_static_prefix_defaults_to_static_when_unset():
    """S3_PREFIX is required in S3 mode; S3_STATIC_PREFIX is not, so it defaults."""
    resolved = CollectorConfig.from_env(BASE_ENV)

    assert resolved.s3_prefix == "raw/"
    assert resolved.s3_static_prefix == "static/"


def test_prefixes_are_normalised_to_a_single_trailing_slash():
    resolved = CollectorConfig.from_env(
        {**BASE_ENV, "S3_PREFIX": "/raw", "S3_STATIC_PREFIX": "/gtfs-static//"}
    )

    assert resolved.s3_prefix == "raw/"
    assert resolved.s3_static_prefix == "gtfs-static/"


@pytest.mark.parametrize("static_prefix", ["raw/static/", "raw/"])
def test_static_prefix_nested_under_raw_is_refused(static_prefix):
    """raw/ expires after 90 days and WMATA cannot re-serve an old bundle."""
    with pytest.raises(ConfigError, match="must not nest"):
        CollectorConfig.from_env({**BASE_ENV, "S3_STATIC_PREFIX": static_prefix})
