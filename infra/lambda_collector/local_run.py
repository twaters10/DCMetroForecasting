"""Run the collector against the live WMATA feeds without deploying anything.

    python infra/lambda_collector/local_run.py                     # realtime feeds
    python infra/lambda_collector/local_run.py --task static-gtfs   # static GTFS

Loads `.env` from the repo root, invokes `lambda_handler` with a stub event, and
prints a per-feed summary. With `LOCAL_OUTPUT_DIR` set (the default in
`.env.example`) snapshots land on disk and no AWS credentials are needed; unset
it to exercise the real S3 write path with your own credentials.

This exists so feed parsing can be confirmed before Lambda packaging enters the
picture — they are separate failure modes and debugging them together is a trap.
The `--task` flag matters for the same reason: the static path can be exercised
end to end, against the live API and a real ~50 MB bundle, without a deploy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# The Lambda zip has every module at its root, so handler.py imports `config`
# and `writers` as top-level modules. Running this file directly puts its own
# directory on sys.path first, which reproduces that layout exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"


def _load_dotenv() -> None:
    """Load `.env` into the environment, without hard-depending on a library."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_dotenv_fallback()
        return
    load_dotenv(ENV_FILE)


def _load_dotenv_fallback() -> None:
    """Minimal `KEY=value` parser, so the test path works with no dev deps."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Hyphenated on the command line, underscored in the event — the event value
    # is what EventBridge sends and must match handler._STATIC_GTFS_TASK exactly.
    parser.add_argument(
        "--task",
        choices=("realtime", "static-gtfs"),
        default="realtime",
        help="which collection task to run (default: realtime)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if not ENV_FILE.exists():
        print(f"No .env at {ENV_FILE} — copy .env.example to .env first.")
        return 1
    _load_dotenv()

    # Resolve LOCAL_OUTPUT_DIR relative to the repo root, not the shell's cwd,
    # so output lands in the same place no matter where this is run from.
    local_dir = os.environ.get("LOCAL_OUTPUT_DIR", "").strip()
    if local_dir and not Path(local_dir).is_absolute():
        os.environ["LOCAL_OUTPUT_DIR"] = str(REPO_ROOT / local_dir)

    from handler import lambda_handler

    event: dict[str, object] = {"source": "local_run"}
    if args.task == "static-gtfs":
        event["task"] = "static_gtfs"
        prefix_var = "S3_STATIC_PREFIX"
        default_prefix = "static/"
    else:
        prefix_var = "S3_PREFIX"
        default_prefix = "raw/"

    destination = os.environ.get("LOCAL_OUTPUT_DIR") or (
        f"s3://{os.environ.get('S3_BUCKET')}/"
        f"{os.environ.get(prefix_var, default_prefix)}"
    )
    # flush=True so this lands before the handler's log lines, which go to stderr.
    print(f"Running task '{args.task}', writing to: {destination}\n", flush=True)

    summary = lambda_handler(event, None)
    print("\n" + json.dumps(summary, indent=2))

    # Non-zero exit if any feed failed, so this is usable as a smoke check.
    return 1 if summary["feeds_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
