"""SparkSession construction, in one place.

Spark runs locally — no Glue, no EMR. That is a deliberate cost decision, and it
means the session defaults matter: Spark's out-of-the-box settings assume a cluster
and are actively wrong on a laptop.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession

# Where a Homebrew JDK lands on Apple Silicon and Intel respectively. Checked only
# if JAVA_HOME is unset, so an explicit JAVA_HOME always wins.
_JDK_CANDIDATES: Final[tuple[str, ...]] = (
    "/opt/homebrew/opt/openjdk@17",
    "/usr/local/opt/openjdk@17",
    "/opt/homebrew/opt/openjdk",
    "/usr/local/opt/openjdk",
)

# Spark defaults to 200 shuffle partitions, sized for a cluster and a dataset far
# larger than this one. A day of rail is tens of thousands of rows: 200 partitions
# means 200 near-empty tasks whose scheduling overhead dwarfs the work. Small enough
# to be efficient, large enough to use the cores.
DEFAULT_SHUFFLE_PARTITIONS: Final[int] = 8

# Spark's default driver heap is 1g. In local mode the driver *is* the executor, so
# `local[*]` on a 10-core laptop runs ten concurrent tasks inside that one 1g heap —
# about 100MB each. Deriving arrivals sorts a day of trip updates under window
# functions and does not fit, which surfaces as `OutOfMemoryError: Java heap space`
# followed by a confusing NullPointerException from `dagScheduler()` being null: the
# context is already dead by the time the next collect() reports it.
#
# 4g against a 16GB machine leaves room for the OS and the Python side while giving
# each task real headroom. Raise it for bus, which is ~10x the volume.
DEFAULT_DRIVER_MEMORY: Final[str] = "4g"


class JavaNotFoundError(RuntimeError):
    """Raised when no JVM can be located, with instructions rather than a stack."""


def ensure_java_home() -> str:
    """Resolve and export JAVA_HOME, or fail with something actionable.

    PySpark's failure mode without a JVM is a bare `JAVA_HOME is not set` or a
    FileNotFoundError from a subprocess, neither of which says what to install. This
    turns it into a sentence.
    """
    existing = os.environ.get("JAVA_HOME", "").strip()
    if existing and (Path(existing) / "bin" / "java").exists():
        return existing

    for candidate in _JDK_CANDIDATES:
        if (Path(candidate) / "bin" / "java").exists():
            os.environ["JAVA_HOME"] = candidate
            return candidate

    raise JavaNotFoundError(
        "No Java runtime found, and PySpark needs one.\n"
        "  macOS:  brew install openjdk@17\n"
        "  then:   export JAVA_HOME=/opt/homebrew/opt/openjdk@17\n"
        "Or set JAVA_HOME in the repo-root .env file."
    )


def pin_worker_interpreter() -> str:
    """Force Spark's Python workers to use this interpreter.

    Without this, Spark launches workers with whatever `python3` is first on PATH. On
    macOS that is the system Python 3.8, which lacks `importlib.resources.files` and
    dies inside PySpark's own imports — surfacing as a bare `EOFException occurred
    while reading the port number from pyspark.daemon's stdout`, which says nothing
    about interpreters at all. Pinning both ends to `sys.executable` means the venv is
    used no matter how the pipeline was invoked.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return sys.executable


def set_driver_memory(memory: str = DEFAULT_DRIVER_MEMORY) -> str:
    """Raise the driver heap, before the JVM exists to have one.

    This cannot be done with `SparkSession.builder.config("spark.driver.memory", ...)`.
    By the time a SparkConf is applied, py4j has already launched the JVM and its -Xmx
    is fixed, so the setting is accepted, reported back by `spark.conf.get`, and has no
    effect whatsoever — the worst kind of silent no-op. The launcher reads
    PYSPARK_SUBMIT_ARGS instead, which is consulted before the process starts.

    The trailing `pyspark-shell` is required: PYSPARK_SUBMIT_ARGS is parsed as a
    spark-submit command line and the launcher looks for that token as the "primary
    resource" terminating the argument list. Without it the JVM fails to start.
    """
    os.environ.setdefault(
        "PYSPARK_SUBMIT_ARGS", f"--driver-memory {memory} pyspark-shell"
    )
    return memory


def build_session(
    app_name: str = "metro-pulse-etl",
    shuffle_partitions: int = DEFAULT_SHUFFLE_PARTITIONS,
    cores: str = "*",
    driver_memory: str = DEFAULT_DRIVER_MEMORY,
) -> SparkSession:
    """Build a local SparkSession configured for this workload."""
    ensure_java_home()
    pin_worker_interpreter()
    set_driver_memory(driver_memory)

    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # Session-local time zone is pinned to UTC so that any timestamp Spark
        # formats or parses without an explicit zone is unambiguous. Every
        # conversion to America/New_York is done explicitly, in one place
        # (schedule.service_day_start), never implicitly by the session default.
        .config("spark.sql.session.timeZone", "UTC")
        # Dynamic overwrite makes a re-run replace only the service_date partitions
        # it actually produced, instead of deleting the whole output tree. This is
        # what makes "re-run one day" safe next to a month of existing output.
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.ui.enabled", "false")
        # Spark's console progress bar is built from carriage returns and is meant for
        # a TTY. Piped to `tee` it renders as thousands of `[Stage 72:===>]` fragments
        # on one enormous line, in the terminal AND in logs/etl-YYYY-MM.log. Our own
        # stage logging covers "where is it up to" in a form that survives a log file.
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    return session
