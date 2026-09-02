"""Build the CloudWatch dashboard for the journey-duration model.

    python -m src.monitoring.dashboard              # create or replace
    python -m src.monitoring.dashboard --dry-run    # print the body, change nothing

The dashboard used to exist only in the console, which made it invisible to review and
impossible to recreate. It also pinned `ModelRun` to a literal run id in every widget,
so it went blank the moment a new model was registered and stayed blank until someone
noticed and edited five widgets by hand.

Both are fixed by generating it here. Widgets select their series with `SEARCH()` rather
than a fixed dimension value, so they follow whichever run is publishing.

## What SEARCH costs us

`report.py` dimensions every metric by run id deliberately: without it a series silently
spans two models across a redeploy and the step change reads as drift. That property is
kept — the dimension is still there, and a retrain still starts a visibly separate line.
SEARCH only removes the need to retype the run id.

Two consequences worth knowing before they look like bugs:

- SEARCH matches only metrics that reported within the last two weeks. A dashboard that
  has gone quiet that long renders empty — which is a symptom of publishing having
  stopped, not of the dashboard being broken.
- Immediately after a retrain both runs are inside that window, so a widget briefly
  shows two lines. That is the handover, and it is the thing you want to look at.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from ..etl.config import EtlConfig
from .config import CLOUDWATCH_NAMESPACE, MonitoringConfig

logger = logging.getLogger("monitoring.dashboard")

DASHBOARD_NAME = "metro-pulse-model"

# The endpoint is monitored, but not from here — see the widget comment below.
ENDPOINT_NAME = "metro-pulse-journey"
VARIANT_NAME = "variant-name-1"

# Lengths to plot on the "beats the schedule" chart. A subset of REPORT_LENGTHS: all
# eight are published, but eight lines on one axis is unreadable and the short, middle
# and long horizons are what actually differ.
PLOTTED_LENGTHS = (1, 4, 8, 17, 28)

DAY = 86400


def _search(metric: str, stat: str, *, segments: int | None = None) -> str:
    """A SEARCH expression for one metric, across whichever model run is publishing."""
    dimensions = "JourneySegments,ModelRun" if segments is not None else "ModelRun"
    query = f'{{{CLOUDWATCH_NAMESPACE},{dimensions}}} MetricName="{metric}"'
    if segments is not None:
        query += f' JourneySegments="{segments}"'
    return f"SEARCH('{query}', '{stat}', {DAY})"


def build_body(region: str) -> dict:
    """The dashboard document. Pure — takes no AWS call, so it is testable."""
    return {
        "widgets": [
            {
                "type": "text",
                "x": 0,
                "y": 0,
                "width": 24,
                "height": 2,
                "properties": {
                    "markdown": (
                        "# Metro Pulse — journey duration model\n"
                        "**Model health** (this dashboard) is separate from **endpoint "
                        "health** (bottom row). Drift is visible immediately; realised "
                        "accuracy lags a day because a service day closes at 04:00 UTC "
                        "and is scored by the ETL cron the following morning. Series "
                        "follow whichever model run is publishing — a retrain shows up "
                        "as a new line, not a step."
                    )
                },
            },
            {
                "type": "metric",
                "x": 0,
                "y": 2,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": (
                        "Beats the published schedule (%) — the project's actual claim"
                    ),
                    "region": region,
                    "period": DAY,
                    "yAxis": {
                        "left": {"label": "% better than timetable", "showUnits": False}
                    },
                    "annotations": {
                        "horizontal": [
                            {
                                "label": "no better than the timetable",
                                "value": 0,
                                "color": "#d62728",
                            }
                        ]
                    },
                    "metrics": [
                        [
                            {
                                "expression": _search(
                                    "BeatsSchedulePct", "Average", segments=length
                                ),
                                "label": f"{length} segment"
                                + ("" if length == 1 else "s"),
                                "id": f"beats{length}",
                            }
                        ]
                        for length in PLOTTED_LENGTHS
                    ],
                },
            },
            {
                "type": "metric",
                "x": 12,
                "y": 2,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Realised MAE vs schedule baseline (8 segments)",
                    "region": region,
                    "period": DAY,
                    "yAxis": {"left": {"label": "seconds", "showUnits": False}},
                    # Both on the same rows: a hard week degrades the timetable too, so
                    # the gap between these lines is the honest signal, not either line.
                    "metrics": [
                        [
                            {
                                "expression": _search(
                                    "RealisedMAE", "Average", segments=8
                                ),
                                "label": "model",
                                "id": "mae",
                            }
                        ],
                        [
                            {
                                "expression": _search(
                                    "ScheduleBaselineMAE", "Average", segments=8
                                ),
                                "label": "schedule",
                                "id": "baseline",
                            }
                        ],
                    ],
                },
            },
            {
                "type": "metric",
                "x": 0,
                "y": 8,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "recent_deviation null rate — the silent failure",
                    "region": region,
                    "period": DAY,
                    "yAxis": {
                        "left": {
                            "label": "% of journeys",
                            "showUnits": False,
                            "min": 0,
                            "max": 100,
                        }
                    },
                    "annotations": {
                        "horizontal": [
                            {
                                "label": "alarm",
                                "value": MonitoringConfig().max_null_rate_pct,
                                "color": "#d62728",
                            }
                        ]
                    },
                    "metrics": [
                        [
                            {
                                "expression": _search(
                                    "RecentDeviationNullRate", "Maximum"
                                ),
                                "label": "null rate",
                                "id": "nulls",
                            }
                        ]
                    ],
                },
            },
            {
                "type": "metric",
                "x": 12,
                "y": 8,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Journeys scored per run — is monitoring itself alive?",
                    "region": region,
                    "period": DAY,
                    "metrics": [
                        [
                            {
                                "expression": _search("ScoredJourneys", "Sum"),
                                "label": "journeys scored",
                                "id": "scored",
                            }
                        ]
                    ],
                },
            },
            {
                # Left pinned, and left in place. SageMaker's own metrics, not ours,
                # and legitimately near-empty: organic endpoint traffic is a handful of
                # requests a day, which is exactly why model quality is measured by
                # shadow scoring instead. Absence here is not an outage.
                "type": "metric",
                "x": 0,
                "y": 14,
                "width": 24,
                "height": 6,
                "properties": {
                    "title": "Endpoint health (already emitted by SageMaker)",
                    "region": region,
                    "stat": "Average",
                    "period": 300,
                    "metrics": [
                        [
                            "AWS/SageMaker",
                            name,
                            "EndpointName",
                            ENDPOINT_NAME,
                            "VariantName",
                            VARIANT_NAME,
                            *([{"stat": "Sum", "yAxis": "right"}] if counted else []),
                        ]
                        for name, counted in (
                            ("ModelLatency", False),
                            ("OverheadLatency", False),
                            ("Invocations", True),
                            ("Invocation5XXErrors", True),
                        )
                    ],
                },
            },
        ]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print, change nothing")
    parser.add_argument("--name", default=DASHBOARD_NAME)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    body = build_body(EtlConfig.from_env().aws_region)

    if args.dry_run:
        print(json.dumps(body, indent=2))
        return 0

    import boto3

    response = boto3.client("cloudwatch").put_dashboard(
        DashboardName=args.name, DashboardBody=json.dumps(body)
    )
    # A malformed widget is reported here rather than raising: the dashboard is still
    # written, just with that widget broken, so this must not be discarded.
    warnings = response.get("DashboardValidationMessages", [])
    for warning in warnings:
        logger.warning("%s: %s", warning.get("DataPath"), warning.get("Message"))

    print(f"wrote dashboard {args.name} ({len(body['widgets'])} widgets)")
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
