"""Snapshot destinations: S3 in Lambda, a local directory for testing.

Both writers expose the same `write(key, payload) -> destination` call, so the
handler never branches on where the bytes are going. The local writer exists so
feed parsing can be verified before Lambda packaging is involved — two failure
modes worth debugging separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol

# What the realtime snapshots are: gzipped protobuf. Static GTFS bundles override
# this with `application/zip`.
DEFAULT_CONTENT_TYPE: Final[str] = "application/gzip"


class SnapshotWriter(Protocol):
    """Writes one snapshot and returns a human-readable destination.

    `content_type` and `metadata` exist for the static GTFS bundles, which are
    zips rather than gzipped protobuf and carry a feed window worth recording on
    the object. Both are optional so the realtime call sites stay unchanged.
    """

    def write(
        self,
        key: str,
        payload: bytes,
        content_type: str = DEFAULT_CONTENT_TYPE,
        metadata: dict[str, str] | None = None,
    ) -> str: ...


class S3Writer:
    """Writes snapshots to S3 with `PutObject`.

    The execution role grants `s3:PutObject` on `raw/*` and nothing else — no
    read, list, or delete. So this class must never check whether a key already
    exists or clean anything up; those calls would fail with AccessDenied.
    Overwriting an identical key is harmless anyway: keys are timestamped, so a
    collision only happens if the same invocation is somehow replayed, in which
    case the second write is byte-identical.

    boto3 is imported lazily and the client is created once per container so the
    cost is paid on cold start only.
    """

    def __init__(self, bucket: str) -> None:
        # boto3 is provided by the Lambda runtime and is deliberately NOT
        # bundled into the deployment zip (see build_package.sh).
        import boto3

        self._bucket = bucket
        self._client = boto3.client("s3")

    def write(
        self,
        key: str,
        payload: bytes,
        content_type: str = DEFAULT_CONTENT_TYPE,
        metadata: dict[str, str] | None = None,
    ) -> str:
        # A ContentType but deliberately NEVER `ContentEncoding`. ContentEncoding
        # invites transparent decompression by some readers (a presigned URL
        # fetched with `requests`, or CloudFront) while boto3's get_object does not
        # decompress at all — so the same object would hand back different bytes
        # depending on how it was read. Treating every object as an opaque
        # compressed blob keeps bytes-in == bytes-out for every reader, and the
        # ETL decompresses explicitly. This holds for the zip bundles too.
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            **({"Metadata": metadata} if metadata else {}),
        )
        return f"s3://{self._bucket}/{key}"


class LocalWriter:
    """Writes snapshots under a local directory, mirroring the S3 key layout.

    Keeping the same partitioned layout on disk means the PySpark ETL can be
    pointed at a local sample and at S3 without a code change.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    def write(
        self,
        key: str,
        payload: bytes,
        content_type: str = DEFAULT_CONTENT_TYPE,
        metadata: dict[str, str] | None = None,
    ) -> str:
        # Both are accepted and ignored: a filesystem has nowhere to put them, and
        # the local writer exists to verify fetch/parse/key logic, not S3
        # attributes. The key already carries everything the ETL needs to resolve
        # a bundle, so nothing load-bearing is lost here.
        del content_type, metadata

        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)
