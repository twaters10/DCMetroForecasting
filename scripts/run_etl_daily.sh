#!/usr/bin/env bash
#
# launchd entry point for the daily ETL catch-up.
#
# This wrapper exists because launchd starts jobs with an almost empty environment —
# no PATH beyond the bare minimum, no AWS_PROFILE, no JAVA_HOME, and a working
# directory of /. Every one of those is required here, and diagnosing their absence
# from a launchd failure is miserable. Setting them in one script beats scattering
# EnvironmentVariables through the plist, and means the same command works when you
# run it by hand.
#
#   ./scripts/run_etl_daily.sh              # process whatever is outstanding
#   ./scripts/run_etl_daily.sh --dry-run    # any catchup flag is passed through
#
# Installation of the schedule itself is in runbook.txt.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
export AWS_PROFILE="${AWS_PROFILE:-metro-pulse}"

# The venv interpreter is used directly rather than activating the venv: it is one
# fewer moving part, and src/etl/spark.py pins Spark's Python workers to
# sys.executable, so the whole job stays on this interpreter.
PYTHON="$REPO_ROOT/.venv/bin/python"

# PySpark needs a JVM. spark.ensure_java_home() searches the usual Homebrew paths, so
# this is only a hint for the common case; an already-set JAVA_HOME always wins.
if [[ -z "${JAVA_HOME:-}" ]]; then
  for candidate in /opt/homebrew/opt/openjdk@17 /usr/local/opt/openjdk@17; do
    if [[ -x "$candidate/bin/java" ]]; then
      export JAVA_HOME="$candidate"
      break
    fi
  done
fi

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/etl-$(date +%Y-%m).log"

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
{
  echo ""
  echo "================================================================================"
  echo "run started $(date -u '+%Y-%m-%dT%H:%M:%SZ')  (local $(date '+%Y-%m-%d %H:%M %Z'))"
  echo "================================================================================"
} | tee -a "$LOG_FILE"

# Output goes to BOTH the terminal and the log. It used to be `>>"$LOG_FILE" 2>&1`,
# which sent every byte to the file and left an interactive run looking hung for the
# ten-plus minutes a multi-day catch-up takes.
#
# `-u` is load-bearing, not decoration. Python line-buffers stdout only when it is a
# TTY; behind a pipe it switches to 4-8KB block buffering, so progress would arrive in
# bursts minutes after the work happened — which looks exactly like the hang this is
# meant to cure. `-u` forces unbuffered writes.
#
# `set -o pipefail` (set at the top) is equally load-bearing: without it the pipeline's
# status is `tee`'s, which is always 0, and a failed ETL would exit green.
#
# `set -e` would abort before the exit status could be logged, so the failure is
# captured explicitly and re-raised at the end.
status=0
"$PYTHON" -u -m src.etl.catchup "$@" 2>&1 | tee -a "$LOG_FILE" || status=$?

echo "run finished status=$status $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$LOG_FILE"
exit "$status"
