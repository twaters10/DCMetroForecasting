#!/bin/bash
#
# Cron launcher for the daily ETL. INSTALLED COPY — do not run from the repo.
#
# `scripts/install_etl_cron.sh` writes a copy of this file OUTSIDE ~/Documents and
# points cron at that copy. It cannot live in the repo, and the reason is the whole
# design of this file.
#
# ## Why this exists instead of cron calling run_etl_daily.sh
#
# macOS TCC protects ~/Documents, where this repo lives, and it grants access per
# BINARY. Probed from cron:
#
#     ls / head / bash  reading the repo   -> Operation not permitted
#     .venv/bin/python  reading the same   -> OK
#
# `.venv/bin/python` resolves to /Library/Frameworks/Python.framework, which holds
# Full Disk Access; /bin/bash does not. So cron can run the pipeline, but it cannot
# read a shell script stored in the repo to find out how — bash is denied before the
# first line executes. This launcher therefore lives outside ~/Documents, and every
# path it touches inside the repo is touched by Python, never by the shell.
#
# The log goes outside the repo for the same reason: the shell performs the redirect,
# so it must write somewhere TCC does not police.
#
# ## The failure mode to watch
#
# This depends on that framework Python keeping Full Disk Access. Rebuilding the venv
# against a different interpreter (Homebrew, pyenv) silently loses it, and the job
# would fail with a permission error that looks nothing like one. The preflight below
# checks exactly that and says so in plain terms.

set -uo pipefail

REPO="@REPO@"
LOG_DIR="@LOG_DIR@"
PYTHON="$REPO/.venv/bin/python"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/etl-$(date +%Y-%m).log"

export AWS_PROFILE="${AWS_PROFILE:-metro-pulse}"

# Only a hint: src/etl/spark.py searches the usual Homebrew paths itself.
if [[ -z "${JAVA_HOME:-}" ]]; then
  for candidate in /opt/homebrew/opt/openjdk@17 /usr/local/opt/openjdk@17; do
    [[ -x "$candidate/bin/java" ]] && export JAVA_HOME="$candidate" && break
  done
fi

{
  echo ""
  echo "================================================================================"
  echo "cron run started $(date -u '+%Y-%m-%dT%H:%M:%SZ')  (local $(date '+%F %H:%M %Z'))"
  echo "================================================================================"
} >> "$LOG_FILE"

# Preflight. Python is asked to read one repo file; if TCC has taken the grant away
# this fails here with an explanation rather than 200 lines further on with a
# traceback that blames the pipeline.
if ! "$PYTHON" -c "open('$REPO/pyproject.toml').close()" 2>/dev/null; then
  {
    echo "FATAL: $PYTHON cannot read $REPO"
    echo ""
    echo "macOS TCC has denied this interpreter access to ~/Documents. Grant Full Disk"
    echo "Access to the interpreter the venv points at:"
    echo "    $(readlink -f "$PYTHON" 2>/dev/null || echo "$PYTHON")"
    echo "System Settings -> Privacy & Security -> Full Disk Access."
    echo ""
    echo "This usually means the venv was rebuilt against a different interpreter."
  } >> "$LOG_FILE"
  exit 78  # EX_CONFIG: wrong configuration, not a transient failure
fi

# `cd` is a shell builtin and chdir is permitted even where reading is not, so this
# is safe. Python needs the repo as CWD to resolve `src` as a package.
cd "$REPO" || { echo "FATAL: cannot chdir to $REPO" >> "$LOG_FILE"; exit 78; }

# -u because output is going straight to a file: without it Python block-buffers and
# a run that dies mid-way leaves a log that stops in an arbitrary place.
status=0
"$PYTHON" -u -m src.etl.catchup "$@" >> "$LOG_FILE" 2>&1 || status=$?

# Score whatever the ETL has landed but not yet published. Driven by what is missing
# rather than by this firing, so three of the four daily runs exit in under a second and
# a day the ETL caught up late is still scored.
#
# Its status is recorded separately and deliberately does NOT feed `status`. The monitor
# exits 2 when the model breaches a threshold, which is a finding about the model, not a
# failed run — folding it into the ETL's status would make a healthy pipeline look
# broken in exactly the log used to tell a real outage from a sleeping laptop.
monitor_status=0
"$PYTHON" -u -m src.monitoring.report --catchup >> "$LOG_FILE" 2>&1 || monitor_status=$?
echo "monitor finished status=$monitor_status $(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  >> "$LOG_FILE"

echo "cron run finished status=$status $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_FILE"
exit "$status"
