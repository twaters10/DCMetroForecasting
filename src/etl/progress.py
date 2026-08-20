"""Throttled progress logging for the long, quiet stages.

Stage A downloads ~2,880 objects per service day and, until this existed, said nothing
between "decoding feed=... snapshots=2880" and finishing several minutes later. A run
that is working and a run that is hung look identical from the outside, which is the
worst property a long job can have.

**Throttled by elapsed time, not by count.** A "every 100 items" rule logs 29 lines for
a fast local feed and 3 for a slow one — the cadence follows the work rather than the
clock. Time-based throttling gives the same rhythm either way, and the log stays the
same size whether the network is quick or crawling.

The ETA is deliberately a plain linear extrapolation of the observed rate. It is a
progress indicator, not a promise, and anything cleverer would imply a precision that a
variable-latency S3 read does not have.
"""

from __future__ import annotations

import logging
import time


class Progress:
    """Log progress through a countable job, at most every `every_seconds`.

    Deliberately not a dependency on tqdm or rich: this output goes to a log file as
    often as to a terminal, and a progress bar built from carriage returns turns into
    thousands of unreadable lines in `logs/etl-YYYY-MM.log`.
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        label: str,
        every_seconds: float = 5.0,
    ) -> None:
        self.logger = logger
        self.total = max(int(total), 0)
        self.label = label
        self.every_seconds = every_seconds
        self.count = 0
        self._started = time.monotonic()
        self._last_log = self._started

    def advance(self, step: int = 1) -> None:
        self.count += step
        now = time.monotonic()
        if now - self._last_log >= self.every_seconds:
            self._last_log = now
            self.logger.info("  %s", self._line(now))

    def done(self) -> None:
        """Always logs, so the final count appears even for a job that finished fast."""
        elapsed = time.monotonic() - self._started
        rate = self.count / elapsed if elapsed > 0 else 0.0
        self.logger.info(
            "  %s complete — %d in %s (%.0f/s)",
            self.label,
            self.count,
            format_duration(elapsed),
            rate,
        )

    def _line(self, now: float) -> str:
        elapsed = now - self._started
        rate = self.count / elapsed if elapsed > 0 else 0.0
        if not self.total:
            return f"{self.label} {self.count:,} ({rate:.0f}/s)"

        pct = 100 * self.count / self.total
        remaining = (self.total - self.count) / rate if rate > 0 else 0.0
        return (
            f"{self.label} {self.count:,}/{self.total:,} ({pct:.0f}%) "
            f"{rate:.0f}/s, ~{format_duration(remaining)} left"
        )


def format_duration(seconds: float) -> str:
    """Compact and human-readable: 45s, 3m12s, 1h04m."""
    seconds = max(int(seconds), 0)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
