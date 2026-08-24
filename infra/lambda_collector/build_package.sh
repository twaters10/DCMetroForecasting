#!/usr/bin/env bash
#
# Build the deployment zip for the metro-pulse-collector Lambda.
#
# The constraint that shapes this whole script: the function runs on arm64
# (Graviton) / Python 3.12, but this script is normally run on a developer
# machine that is neither. `protobuf` ships architecture-specific wheels, so a
# plain `pip install --target` silently installs the *host's* wheel. Everything
# looks fine locally and the function then dies in Lambda at import time with an
# error that does not mention architecture at all.
#
# So the install is explicitly cross-targeted, and the result is verified rather
# than assumed — see check_wheel_tags below.
#
# Idempotent: any previous package/ and .zip are removed first. Runnable from
# anywhere; all paths resolve relative to this script.
#
# Usage:
#   ./infra/lambda_collector/build_package.sh
#   PYTHON_BIN=python3.12 ./infra/lambda_collector/build_package.sh

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BUILD_DIR="$SCRIPT_DIR/package"
ZIP_PATH="$SCRIPT_DIR/metro-pulse-collector.zip"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# Must match the deployed function's configuration exactly.
TARGET_PLATFORM="manylinux2014_aarch64"
TARGET_PYTHON_VERSION="3.12"
TARGET_ARCH_TOKEN="aarch64" # what a correct binary wheel tag must contain

# Modules that live at the zip root alongside the dependencies. The Lambda
# handler is configured as `handler.lambda_handler`, and handler.py imports
# `config`, `writers` and `static_gtfs` as top-level modules, so all four must be
# at the root. static_gtfs is imported lazily at runtime, which means omitting it
# would break only the daily static task and leave the 60-second one healthy —
# exactly the kind of partial failure that goes unnoticed, so the verification
# step below checks for it like the rest.
HANDLER_MODULES=(handler.py config.py writers.py static_gtfs.py recent_conditions.py)

# Shared ETL code, shipped as a PACKAGE rather than at the zip root. `src/etl/config.py`
# and the collector's own `config.py` are two different modules with the same name, so a
# flat copy would shadow one with the other — the same collision tests/conftest.py
# documents. Under `metro_etl/` the import is unambiguous.
#
# `arrivals_live.py` is the pandas port of the Spark arrival derivation, kept honest by
# tests/test_arrivals_parity.py. It is imported here, never reimplemented: a third copy
# of those rules would defeat the parity test entirely.
SHARED_PACKAGE_DIR="metro_etl"
SHARED_MODULES=(../../src/etl/config.py ../../src/etl/arrivals_live.py)

# Lambda limits for a direct (non-S3) upload.
ZIP_LIMIT_MB=50
UNZIPPED_LIMIT_MB=250
ZIP_WARN_MB=45 # warn at 90% of the direct-upload limit

# Prefer the repo's virtualenv, since that is the interpreter whose pip is known
# to support the cross-targeting flags used below.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

log() { printf '  %s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }
fail() {
  printf '\nBUILD FAILED: %s\n' "$*" >&2
  exit 1
}

# --------------------------------------------------------------------------
# 1. Clean — idempotency. A stale package/ from a previous run is exactly how a
#    wrong-architecture wheel survives into a "fixed" build.
# --------------------------------------------------------------------------
step "Cleaning previous build artifacts"
rm -rf "$BUILD_DIR"
rm -f "$ZIP_PATH"
log "removed $BUILD_DIR and $ZIP_PATH"

# --------------------------------------------------------------------------
# 2. Install dependencies, cross-targeted at arm64 / Python 3.12.
#
#    This step is deliberately self-contained: it produces a directory of
#    dependencies and nothing else. To move to a Lambda Layer later, this is the
#    only step that changes — the tree gets zipped under `python/` and published
#    as a layer instead of being merged into the function zip.
#
#    --only-binary=:all: is what makes a source-distribution fallback impossible.
#    Without it, pip would quietly build an sdist against the *host* Python and
#    architecture, which is the silent failure this script exists to prevent.
#    With it, pip errors out instead — noisy, which is what we want.
# --------------------------------------------------------------------------
step "Installing dependencies for $TARGET_PLATFORM / Python $TARGET_PYTHON_VERSION"
log "pip: $PYTHON_BIN -m pip"

"$PYTHON_BIN" -m pip install \
  --target "$BUILD_DIR" \
  --platform "$TARGET_PLATFORM" \
  --implementation cp \
  --python-version "$TARGET_PYTHON_VERSION" \
  --only-binary=:all: \
  --requirement "$REQUIREMENTS" \
  --quiet ||
  fail "pip could not resolve a binary wheel for every dependency on \
$TARGET_PLATFORM. Do NOT drop --only-binary=:all: to work around this: that \
makes pip build a source distribution for the host architecture, which imports \
fine here and fails in Lambda."

# --------------------------------------------------------------------------
# 3. Verify the wheels are actually the ones we asked for.
#
#    Each installed distribution records its wheel tag in `*.dist-info/WHEEL`.
#    A pure-Python wheel is tagged `py3-none-any` and is fine on any platform; a
#    compiled wheel must carry an aarch64 tag. Anything else (a macosx_* or
#    x86_64 tag) means platform targeting silently fell back, and this is the
#    only place that is cheap to detect.
# --------------------------------------------------------------------------
step "Verifying wheel architecture tags"
arch_violations=0
for wheel_metadata in "$BUILD_DIR"/*.dist-info/WHEEL; do
  [[ -e "$wheel_metadata" ]] || fail "no distributions were installed into $BUILD_DIR"

  dist_name="$(basename "$(dirname "$wheel_metadata")")"
  tags="$(awk '/^Tag:/ {print $2}' "$wheel_metadata" | tr '\n' ' ')"

  if [[ "$tags" == *"none-any"* ]]; then
    log "ok        $dist_name (pure python)"
  elif [[ "$tags" == *"$TARGET_ARCH_TOKEN"* ]]; then
    log "ok        $dist_name ($tags)"
  else
    log "WRONG     $dist_name ($tags)"
    arch_violations=$((arch_violations + 1))
  fi
done

if ((arch_violations > 0)); then
  fail "$arch_violations distribution(s) were installed for the wrong \
architecture. The function is arm64; these wheels are not. This would fail at \
import time in Lambda with a message that does not mention architecture."
fi

# --------------------------------------------------------------------------
# 4. Verify boto3 was not pulled in.
#
#    The Lambda runtime already provides boto3/botocore (~20 MB unzipped).
#    Bundling them bloats the artifact and, worse, shadows the runtime's version
#    with whatever was pinned at build time.
# --------------------------------------------------------------------------
step "Verifying boto3 is not bundled"
if [[ -d "$BUILD_DIR/boto3" || -d "$BUILD_DIR/botocore" ]]; then
  fail "boto3/botocore were installed into the package. Remove them from \
requirements.txt — the Lambda runtime provides them."
fi
log "ok        boto3/botocore absent (provided by the runtime)"

# --------------------------------------------------------------------------
# 5. Trim what Lambda will never execute.
# --------------------------------------------------------------------------
step "Trimming build directory"
find "$BUILD_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$BUILD_DIR/bin" # console-script shims; nothing invokes them in Lambda
log "removed __pycache__ directories and console-script shims"

# --------------------------------------------------------------------------
# 6. Assemble the zip.
#
#    Dependencies go in at the *root* of the archive, not nested under
#    `package/`. Lambda puts the archive root on sys.path, so a nested layout
#    produces `ModuleNotFoundError` for every dependency.
# --------------------------------------------------------------------------
step "Assembling $ZIP_PATH"

# `cd` into the build dir so paths are stored relative to it, i.e. at the root.
(cd "$BUILD_DIR" && zip -q -r -X "$ZIP_PATH" .)
log "added dependencies at archive root"

for module in "${HANDLER_MODULES[@]}"; do
  [[ -f "$SCRIPT_DIR/$module" ]] || fail "expected handler module not found: $module"
  (cd "$SCRIPT_DIR" && zip -q -X "$ZIP_PATH" "$module")
  log "added $module"
done

# Shared ETL package. Staged into the build dir first so the archive paths come out as
# metro_etl/*.py rather than carrying the source tree's layout.
SHARED_STAGE="$BUILD_DIR/$SHARED_PACKAGE_DIR"
mkdir -p "$SHARED_STAGE"
: >"$SHARED_STAGE/__init__.py"
for module in "${SHARED_MODULES[@]}"; do
  [[ -f "$SCRIPT_DIR/$module" ]] || fail "expected shared module not found: $module"
  cp "$SCRIPT_DIR/$module" "$SHARED_STAGE/"
done
(cd "$BUILD_DIR" && zip -q -r -X "$ZIP_PATH" "$SHARED_PACKAGE_DIR")
log "added $SHARED_PACKAGE_DIR/ (${#SHARED_MODULES[@]} shared module(s))"

# --------------------------------------------------------------------------
# 7. Verify the assembled layout, then report size.
# --------------------------------------------------------------------------
step "Verifying archive layout"
# Capture the listing once rather than piping `unzip -l` into each grep. Under
# `set -o pipefail`, `grep -q` exits on its first match and closes the pipe,
# which can kill unzip with SIGPIPE (exit 141) and fail the build for an archive
# that is perfectly fine. Whether it happens is a race, so it shows up as an
# intermittent build failure — the worst kind to debug.
zip_listing="$(unzip -l "$ZIP_PATH")"

for module in "${HANDLER_MODULES[@]}"; do
  grep -qE "[[:space:]]$module\$" <<<"$zip_listing" ||
    fail "$module is missing from the archive root"
done
grep -q "google/transit/gtfs_realtime_pb2.py" <<<"$zip_listing" ||
  fail "gtfs_realtime_pb2 is missing from the archive — the handler would not import"
log "ok        handler modules and gtfs_realtime_pb2 present at expected paths"

step "Build complete"
zip_bytes="$(wc -c <"$ZIP_PATH" | tr -d ' ')"
zip_mb="$(awk -v b="$zip_bytes" 'BEGIN { printf "%.1f", b / 1048576 }')"
unzipped_mb="$(awk -v k="$(du -sk "$BUILD_DIR" | cut -f1)" 'BEGIN { printf "%.1f", k / 1024 }')"

log "artifact:  $ZIP_PATH"
log "zipped:    ${zip_mb} MB (direct-upload limit ${ZIP_LIMIT_MB} MB)"
log "unzipped:  ${unzipped_mb} MB (limit ${UNZIPPED_LIMIT_MB} MB)"

if awk -v m="$zip_mb" -v lim="$ZIP_LIMIT_MB" 'BEGIN { exit !(m >= lim) }'; then
  fail "the zip exceeds Lambda's ${ZIP_LIMIT_MB} MB direct-upload limit. Upload \
via S3 (aws lambda update-function-code --s3-bucket/--s3-key) or move the \
dependencies into a Lambda Layer."
elif awk -v m="$zip_mb" -v warn="$ZIP_WARN_MB" 'BEGIN { exit !(m >= warn) }'; then
  log ""
  log "WARNING: ${zip_mb} MB is close to the ${ZIP_LIMIT_MB} MB direct-upload"
  log "limit. Consider moving dependencies to a Lambda Layer before adding more."
fi

if awk -v m="$unzipped_mb" -v lim="$UNZIPPED_LIMIT_MB" 'BEGIN { exit !(m >= lim) }'; then
  fail "unpacked size exceeds Lambda's ${UNZIPPED_LIMIT_MB} MB limit."
fi
