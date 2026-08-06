"""Step 5: data-quality checks, run as part of the ETL.

Everything here **reports** rather than drops. A silently smaller output is the worst
failure mode this pipeline has: the segment table still looks correct, the schema is
unchanged, and the row count is only wrong if you knew what it should have been. So
suspect rows stay in the output carrying their flags (`duration_plausible`,
`arrival_confident`, `schedule_version_agrees`) and the counts are printed on every run.

Checks, in the order they can invalidate the run:

1. **Snapshot gaps** — hours holding materially fewer than 60 files. Collector
   downtime. Only stage A can see this, because by the time observations are decoded a
   missing hour is indistinguishable from an hour with no service.
2. **Static match rate** — realtime trips that never reached the schedule. The single
   most likely silent failure; see `schedule.MatchRateReport`.
3. **Duration sanity** — negative durations, implausibly long ones, and actual
   durations that are a large multiple of scheduled.
4. **Sequence gaps** — segments spanning more than one stop, meaning a stop was passed
   between polls and never observed.
5. **Provenance** — how much of the output rests on predictions rather than
   observations, on a wide bracket, or on a timetable that was not in force.

`QualityReport.blocking_issues` lists the conditions that mean the output should not be
trusted, so a caller can choose to fail the run. Nothing here decides that on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from .config import EXPECTED_SNAPSHOTS_PER_HOUR
from .schedule import MatchRateReport

# `year=2026/month=08/day=05/hour=22/`
_PARTITION = re.compile(r"year=(\d{4})/month=(\d{2})/day=(\d{2})/hour=(\d{2})")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame

# An hour missing more than this fraction of its snapshots is called out individually.
# A couple of absent minutes is ordinary jitter around the schedule boundary; a third of
# the hour gone is an outage that will distort every derived arrival in it.
SHORT_HOUR_TOLERANCE: Final[float] = 0.9

# Below this static-GTFS match rate the run is not trustworthy — revenue service should
# reach the schedule essentially always (measured: 100%).
MIN_ACCEPTABLE_MATCH_RATE: Final[float] = 0.95

# Above this share of rows failing the plausibility test, the derivation is suspect
# rather than the data.
MAX_ACCEPTABLE_IMPLAUSIBLE_RATE: Final[float] = 0.05


@dataclass(frozen=True, slots=True)
class SnapshotCoverage:
    """Whether the collector actually ran across the requested window."""

    feed: str
    hours_checked: int
    complete_hours: int
    short_hours: list[tuple[str, int]] = field(default_factory=list)
    empty_hours: list[str] = field(default_factory=list)
    pending_hours: list[str] = field(default_factory=list)

    @property
    def elapsed_hours(self) -> int:
        """Hours the collector actually had a chance to fill."""
        return self.hours_checked - len(self.pending_hours)

    @property
    def missing_snapshots(self) -> int:
        expected = self.elapsed_hours * EXPECTED_SNAPSHOTS_PER_HOUR
        observed = (
            sum(count for _, count in self.short_hours)
            + self.complete_hours * EXPECTED_SNAPSHOTS_PER_HOUR
        )
        return max(expected - observed, 0)

    def format(self) -> list[str]:
        lines = [
            f"  {self.feed}",
            f"    hours in window      {self.hours_checked}"
            + (
                f" ({len(self.pending_hours)} not yet elapsed)"
                if self.pending_hours
                else ""
            ),
            f"    complete (>= {EXPECTED_SNAPSHOTS_PER_HOUR})     "
            f"{self.complete_hours}/{self.elapsed_hours}",
            f"    missing snapshots    {self.missing_snapshots}",
        ]
        for partition, count in self.short_hours[:10]:
            lines.append(
                f"    !! SHORT {partition} — {count}/{EXPECTED_SNAPSHOTS_PER_HOUR}"
            )
        if len(self.short_hours) > 10:
            lines.append(f"    ... and {len(self.short_hours) - 10} more short hours")
        for partition in self.empty_hours[:10]:
            lines.append(f"    !! EMPTY {partition} — collector down")
        return lines


def _hour_has_elapsed(partition: str, now: datetime) -> bool:
    """Whether an hour partition is fully in the past.

    An hour cannot be judged for completeness until it has finished. Without this, every
    run over the current service day reports the remaining hours as collector downtime —
    a blocking issue raised on a perfectly healthy archive, which trains the reader to
    ignore the report.
    """
    match = _PARTITION.search(partition)
    if match is None:
        return True
    year, month, day, hour = (int(g) for g in match.groups())
    hour_end = datetime(year, month, day, hour, tzinfo=UTC) + timedelta(hours=1)
    return hour_end <= now


def check_snapshot_coverage(
    decode_summary: dict[str, Any], now: datetime | None = None
) -> SnapshotCoverage:
    """Turn a stage-A summary into a coverage verdict.

    `now` is injectable so tests can pin it; a check whose result depends on the wall
    clock is otherwise untestable.
    """
    moment = now or datetime.now(UTC)
    by_hour: dict[str, int] = decode_summary["snapshots_by_hour"]
    threshold = EXPECTED_SNAPSHOTS_PER_HOUR * SHORT_HOUR_TOLERANCE

    short: list[tuple[str, int]] = []
    empty: list[str] = []
    pending: list[str] = []
    complete = 0
    for partition, count in sorted(by_hour.items()):
        if not _hour_has_elapsed(partition, moment):
            pending.append(partition)
        elif count == 0:
            empty.append(partition)
        elif count < threshold:
            short.append((partition, count))
        else:
            complete += 1

    return SnapshotCoverage(
        feed=decode_summary["feed"],
        hours_checked=len(by_hour),
        complete_hours=complete,
        short_hours=short,
        empty_hours=empty,
        pending_hours=pending,
    )


@dataclass(frozen=True, slots=True)
class SegmentQuality:
    """Row-level checks over the finished segment table."""

    rows: int
    negative_duration: int
    implausibly_long: int
    ratio_outliers: int
    sequence_gaps: int
    missing_schedule: int
    low_confidence: int
    from_predictions: int
    stale_schedule_version: int
    repeat_run_rows: int
    delay_p05: int | None
    delay_median: int | None
    delay_p95: int | None

    @property
    def implausible_rate(self) -> float:
        if self.rows == 0:
            return 0.0
        return (
            self.negative_duration + self.implausibly_long + self.ratio_outliers
        ) / self.rows

    def format(self) -> list[str]:
        def share(count: int) -> str:
            pct = 100 * count / self.rows if self.rows else 0.0
            return f"{count:>8,}  {pct:>5.1f}%"

        return [
            f"  rows                       {self.rows:>8,}",
            f"  negative duration          {share(self.negative_duration)}",
            f"  implausibly long           {share(self.implausibly_long)}",
            f"  actual >> scheduled        {share(self.ratio_outliers)}",
            f"  stop_sequence gaps         {share(self.sequence_gaps)}",
            f"  no scheduled times         {share(self.missing_schedule)}",
            f"  wide arrival bracket       {share(self.low_confidence)}",
            f"  from predictions not obs   {share(self.from_predictions)}",
            f"  stale schedule version     {share(self.stale_schedule_version)}",
            f"  repeat run of a trip_id    {share(self.repeat_run_rows)}",
            f"  delay_sec  p05 {self.delay_p05}  median {self.delay_median}  "
            f"p95 {self.delay_p95}",
        ]


def check_segments(segments: DataFrame) -> SegmentQuality:
    """Compute every row-level check in a single pass over the segment table."""
    from pyspark.sql import functions as F

    from .segments import IMPLAUSIBLE_DURATION_RATIO, IMPLAUSIBLE_DURATION_SEC

    def count_where(condition: Any) -> Any:
        return F.sum(F.when(condition, 1).otherwise(0))

    ratio_outlier = (
        F.col("scheduled_duration_sec").isNotNull()
        & (F.col("scheduled_duration_sec") > 0)
        & (
            F.col("actual_duration_sec")
            > F.col("scheduled_duration_sec") * IMPLAUSIBLE_DURATION_RATIO
        )
    )

    row = segments.agg(
        F.count("*").alias("rows"),
        count_where(F.col("actual_duration_sec") < 0).alias("negative_duration"),
        count_where(F.col("actual_duration_sec") > IMPLAUSIBLE_DURATION_SEC).alias(
            "implausibly_long"
        ),
        count_where(ratio_outlier).alias("ratio_outliers"),
        count_where(F.col("stop_span") > 1).alias("sequence_gaps"),
        count_where(F.col("scheduled_duration_sec").isNull()).alias("missing_schedule"),
        count_where(~F.col("arrival_confident")).alias("low_confidence"),
        count_where(F.col("arrival_source") == "trip_update").alias("from_predictions"),
        count_where(~F.col("schedule_version_agrees")).alias("stale_schedule_version"),
        count_where(F.col("trip_run") > 0).alias("repeat_run_rows"),
        F.expr("percentile_approx(delay_sec, 0.05)").alias("delay_p05"),
        F.expr("percentile_approx(delay_sec, 0.5)").alias("delay_median"),
        F.expr("percentile_approx(delay_sec, 0.95)").alias("delay_p95"),
    ).collect()[0]

    return SegmentQuality(**row.asDict())


@dataclass(frozen=True, slots=True)
class QualityReport:
    """The per-run summary. One object, printed once, at the end of the run."""

    coverage: list[SnapshotCoverage]
    match_rate: MatchRateReport
    segments: SegmentQuality

    @property
    def blocking_issues(self) -> list[str]:
        """Conditions under which the output should not be trusted.

        Returned rather than raised: whether a low match rate aborts the run is an
        operational choice, and a backfill of a partially-collected day may legitimately
        want the rows anyway.
        """
        issues: list[str] = []

        if self.match_rate.match_rate < MIN_ACCEPTABLE_MATCH_RATE:
            issues.append(
                f"static GTFS match rate {100 * self.match_rate.match_rate:.1f}% is "
                f"below {100 * MIN_ACCEPTABLE_MATCH_RATE:.0f}% — most likely no "
                "archived bundle covers this service date"
            )
        if self.segments.implausible_rate > MAX_ACCEPTABLE_IMPLAUSIBLE_RATE:
            issues.append(
                f"{100 * self.segments.implausible_rate:.1f}% of segments failed a "
                "duration sanity check — suspect the derivation, not the data"
            )
        if self.segments.rows == 0:
            issues.append("no segments were produced")
        for coverage in self.coverage:
            if coverage.empty_hours:
                issues.append(
                    f"{coverage.feed}: {len(coverage.empty_hours)} hour(s) hold no "
                    "snapshots at all — collector downtime"
                )
        return issues

    def format(self) -> str:
        lines = ["", "=" * 78, "DATA QUALITY REPORT", "=" * 78, "", "snapshot coverage"]
        for coverage in self.coverage:
            lines.extend(coverage.format())
        lines.extend(["", self.match_rate.format(), "", "segment table"])
        lines.extend(self.segments.format())

        issues = self.blocking_issues
        lines.append("")
        if issues:
            lines.append("!! BLOCKING ISSUES")
            lines.extend(f"   - {issue}" for issue in issues)
        else:
            lines.append("no blocking issues")
        lines.extend(["=" * 78, ""])
        return "\n".join(lines)
