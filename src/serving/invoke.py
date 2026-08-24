"""Call the deployed endpoint from the command line.

    python -m src.serving.invoke --origin Vienna --destination Rosslyn
    python -m src.serving.invoke --origin "Metro Center" --destination "Union Station" \
        --departure 2026-08-21T09:00:00Z
    python -m src.serving.invoke --check      # run the whole verification set

Exists because the raw `aws sagemaker-runtime invoke-endpoint` call is awkward in ways
that cause real mistakes: it needs `--cli-binary-format raw-in-base64-out`, it writes
the response body to a *file argument* rather than stdout, and pointing that at
`/dev/stdout` interleaves the body with the CLI's own metadata JSON so neither parses.
Worse, a failed call leaves the previous response file untouched, so the last successful
answer is read back as though it were the new one — which is exactly how a broken error
path first looked like it was working.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger("serving.invoke")

ENDPOINT = "metro-pulse-journey"
REGION = "us-east-1"

# Journeys that between them exercise every branch worth checking.
CHECKS: tuple[tuple[str, str, str], ...] = (
    ("Vienna", "Rosslyn", "parity — compare against the offline prediction"),
    ("Pentagon", "Rosslyn", "short journey, busy segment — live conditions expected"),
    ("Metro Center", "Union Station", "transfer station, resolvable"),
    ("L Enfant Plaza", "Fort Totten", "transfer station, punctuation stripped"),
    ("Shady Grove", "Glenmont", "26 segments — near the edge of training support"),
    ("Vienna", "Glenmont", "no single-train route — refusal expected"),
    ("Fogy Botom", "Union Station", "unknown name — suggestion expected"),
)


def invoke(origin: str, destination: str, departure: str | None = None) -> dict:
    """One request. Returns the parsed body, error or prediction alike."""
    import boto3

    payload = {"origin": origin, "destination": destination}
    if departure:
        payload["departure_ts"] = departure

    client = boto3.client("sagemaker-runtime", region_name=REGION)
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read())


def describe(result: dict) -> str:
    if "error" in result:
        return f"  REFUSED ({result.get('error_type')}): {result['error']}"
    lines = [
        f"  {result['predicted_min']:>6} min   "
        f"(scheduled {result['scheduled_sec'] / 60:.1f} min, "
        f"{result['n_segments']} segments)"
    ]
    lines += [f"      !! {w}" for w in result.get("warnings", [])]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin")
    parser.add_argument("--destination")
    parser.add_argument("--departure", default=None, help="ISO 8601, default now")
    parser.add_argument("--check", action="store_true", help="run the verification set")
    parser.add_argument("--json", action="store_true", help="print the raw response")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.check:
        departure = args.departure or datetime.now(UTC).isoformat()
        failures = 0
        for origin, destination, note in CHECKS:
            print(f"\n{origin} -> {destination}   [{note}]")
            try:
                print(describe(invoke(origin, destination, departure)))
            except Exception as error:  # noqa: BLE001 - report, do not abort the set
                failures += 1
                print(f"  CALL FAILED: {type(error).__name__}: {str(error)[:160]}")
        print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} calls returned a response")
        return 1 if failures else 0

    if not (args.origin and args.destination):
        parser.error("--origin and --destination are required unless --check is given")

    result = invoke(args.origin, args.destination, args.departure)
    print(json.dumps(result, indent=1) if args.json else describe(result))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
