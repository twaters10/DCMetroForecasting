"""The processed segment table as an S3-backed dataset.

Stage B writes Parquet to a local path because that is what Spark can do here — there
is no `hadoop-aws` jar in this PySpark install, so `s3a://` is not available. This
module makes S3 the authoritative copy anyway: the local tree is a working directory,
and everything downstream reads from S3.

That matters more than it looks. `raw/` expires after 90 days, so once a service day
ages out, the segment table is the **only** surviving record of it. A table living on a
single laptop with no backup is a data-loss event waiting for a disk failure. At
~0.7 MB per service day, replicating it costs essentially nothing.

Reads go through `pyarrow.fs.S3FileSystem` rather than Spark, for the same
jar-availability reason — pyarrow speaks S3 natively and picks up the same credentials
boto3 uses.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .config import EtlConfig

logger = logging.getLogger(__name__)

# The dataset's name under the processed prefix. Segments are one dataset among several
# this project will eventually write there (features, predictions), so they get a
# subdirectory rather than sitting at the prefix root.
SEGMENTS_DATASET: Final[str] = "segments"

_PARTITION_DIR: Final[re.Pattern[str]] = re.compile(r"service_date=(\d{4}-\d{2}-\d{2})")


def segments_prefix(config: EtlConfig) -> str:
    """S3 key prefix for the segment table, e.g. `processed/segments/`."""
    return f"{config.processed_prefix}{SEGMENTS_DATASET}/"


def segments_uri(config: EtlConfig) -> str:
    """Full `s3://` URI for the segment table — what downstream stages point at."""
    return f"s3://{config.s3_bucket}/{segments_prefix(config)}"


def _filesystem(config: EtlConfig) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(region=config.aws_region)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def sync_partitions(
    config: EtlConfig,
    local_root: Path,
    service_dates: list[str],
    s3: Any | None = None,
) -> dict[str, int]:
    """Replace the given service_date partitions in S3 with the local copies.

    **Deletes the destination prefix before uploading.** A re-run can legitimately
    produce a different number of part files — Spark chooses that from partition count —
    and any leftover file from a previous run would still be read as part of the
    dataset, silently inflating the row count. This is the same failure mode
    `decode.write_window` guards against in staging, one layer further out, and it is
    invisible unless you happen to be counting rows.

    Returns the number of objects uploaded per date.
    """
    import boto3

    client = s3 or boto3.client("s3", region_name=config.aws_region)
    uploaded: dict[str, int] = {}

    for service_date in service_dates:
        partition = local_root / f"service_date={service_date}"
        if not partition.is_dir():
            logger.warning("no local partition to sync for %s", service_date)
            continue

        prefix = f"{segments_prefix(config)}service_date={service_date}/"
        _delete_prefix(client, config.s3_bucket, prefix)

        count = 0
        for path in sorted(partition.rglob("*.parquet")):
            # `*.parquet` and the leading-dot check together skip Spark's `.crc`
            # checksum sidecars and `_SUCCESS` markers. They are Hadoop-internal, of no
            # use to any reader here, and would otherwise double the object count.
            if not path.is_file() or path.name.startswith("."):
                continue
            client.upload_file(
                str(path), config.s3_bucket, f"{prefix}{path.relative_to(partition)}"
            )
            count += 1
        uploaded[service_date] = count
        logger.info(
            "synced %s — %d object(s) to s3://%s/%s",
            service_date,
            count,
            config.s3_bucket,
            prefix,
        )

    return uploaded


def _delete_prefix(client: Any, bucket: str, prefix: str) -> int:
    """Delete every object under a prefix. Returns how many were removed."""
    removed = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            removed += len(keys)
    return removed


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def completed_service_dates(config: EtlConfig, s3: Any | None = None) -> set[str]:
    """Service dates already present in S3.

    This is what makes the schedule a trigger rather than a source of truth. The
    catch-up driver compares it against the dates complete in the archive and processes
    the difference, so a missed run costs nothing.
    """
    import boto3

    client = s3 or boto3.client("s3", region_name=config.aws_region)
    prefix = segments_prefix(config)

    dates: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=config.s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            # A partition counts as present only if it holds a real object. An empty
            # "directory" placeholder means an interrupted sync, not a finished date.
            if obj["Size"] == 0:
                continue
            match = _PARTITION_DIR.search(obj["Key"])
            if match:
                dates.add(match.group(1))
    return dates


def read_segments(
    config: EtlConfig, service_dates: list[str] | None = None
) -> pa.Table:
    """Read the segment table from S3. This is the entry point for stages 3-4.

    Reads through pyarrow rather than Spark deliberately: the dataset is small (~0.7 MB
    per service day) and downstream modelling is pandas/LightGBM, so pulling a Spark
    session up just to read Parquet would be pure overhead.
    """
    dataset = ds.dataset(
        f"{config.s3_bucket}/{segments_prefix(config).rstrip('/')}",
        filesystem=_filesystem(config),
        format="parquet",
        partitioning="hive",
    )
    if service_dates is None:
        return dataset.to_table()
    return dataset.to_table(filter=ds.field("service_date").isin(service_dates))


def read_local_segments(local_root: Path) -> pa.Table:
    """Read the local working copy — for verifying a sync round-tripped."""
    return pq.read_table(local_root)
