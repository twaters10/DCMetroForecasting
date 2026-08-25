#!/usr/bin/env bash
#
# Install (or remove) the local cron schedule for the daily ETL.
#
#   ./scripts/install_etl_cron.sh              # install, then verify
#   ./scripts/install_etl_cron.sh --uninstall  # remove, leaving other entries alone
#   SCHEDULE="0 */4 * * *" ./scripts/install_etl_cron.sh
#
# ## Why a copy rather than pointing cron at the repo
#
# macOS TCC grants filesystem access per binary, and /bin/bash does not have it for
# ~/Documents. cron would fail to READ a launcher stored in the repo before running a
# line of it. So the launcher is copied outside ~/Documents and cron points there.
# `scripts/etl_cron.sh` is the source of truth; re-run this script after editing it.
#
# ## Why several times a day rather than once
#
# A laptop is asleep at 03:00 and cron, unlike launchd, does not catch up a missed
# firing. The pipeline is a CATCH-UP job — every run asks which service days are
# absent from S3 and processes those — so extra firings are free (a no-op run takes
# about three seconds) and each one is another chance to find the machine awake.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Outside ~/Documents by necessity. A dot-directory in $HOME is not TCC-protected,
# so /bin/sh can read the launcher there.
INSTALL_DIR="$HOME/.metro-pulse"
LAUNCHER="$INSTALL_DIR/run-etl.sh"
LOG_DIR="$HOME/Library/Logs/metro-pulse"
SCHEDULE="${SCHEDULE:-0 9,13,17,21 * * *}"

BEGIN="# >>> metro-pulse etl (managed by scripts/install_etl_cron.sh) >>>"
END="# <<< metro-pulse etl <<<"

# Everything outside our marked block is preserved verbatim: this crontab has other
# jobs in it and clobbering them would be unforgivable.
strip_block() {
  crontab -l 2>/dev/null | awk -v b="$BEGIN" -v e="$END" '
    $0 == b {skip = 1} {if (!skip) print} $0 == e {skip = 0}
  ' || true
}

if [[ "${1:-}" == "--uninstall" ]]; then
  strip_block | crontab -
  rm -f "$LAUNCHER"
  echo "removed the metro-pulse cron entry and $LAUNCHER"
  echo "remaining crontab:"
  crontab -l 2>/dev/null || echo "  (empty)"
  exit 0
fi

mkdir -p "$INSTALL_DIR" "$LOG_DIR"
sed -e "s|@REPO@|$REPO|g" -e "s|@LOG_DIR@|$LOG_DIR|g" \
  "$REPO/scripts/etl_cron.sh" > "$LAUNCHER"
chmod +x "$LAUNCHER"
echo "launcher  $LAUNCHER"
echo "logs      $LOG_DIR"
echo "schedule  $SCHEDULE"

{
  strip_block
  echo "$BEGIN"
  echo "# Daily WMATA ETL catch-up. Regenerate with scripts/install_etl_cron.sh."
  echo "$SCHEDULE $LAUNCHER"
  echo "$END"
} | crontab -

echo
echo "crontab now:"
crontab -l | sed 's/^/  /'
