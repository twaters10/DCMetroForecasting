"""Snapshot destinations: S3 in Lambda, a local directory for testing.

Both writers expose the same `write(key, payload) -> destination` call, so the
handler never branches on where the bytes are going. The local writer exists so
feed parsing can be verified before Lambda packaging is involved — two failure
modes worth debugging separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class SnapshotWriter(Protocol):
    """Writes one snapshot and returns a human-readable destination."""

    def write(self, key: str, payload: bytes) -> str: ...


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

    def write(self, key: str, payload: bytes) -> str:
        # `ContentType: application/gzip` and deliberately NOT
        # `ContentEncoding: gzip`. ContentEncoding invites transparent
        # decompression by some readers (a presigned URL fetched with `requests`,
        # or CloudFront) while boto3's get_object does not decompress at all — so
        # the same object would hand back different bytes depending on how it was
        # read. Treating it as an opaque gzip blob keeps bytes-in == bytes-out for
        # every reader, and the ETL decompresses explicitly.
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=payload,
            ContentType="application/gzip",
        )
        return f"s3://{self._bucket}/{key}"


class LocalWriter:
    """Writes snapshots under a local directory, mirroring the S3 key layout.

    Keeping the same partitioned layout on disk means the PySpark ETL can be
    pointed at a local sample and at S3 without a code change.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    def write(self, key: str, payload: bytes) -> str:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)
