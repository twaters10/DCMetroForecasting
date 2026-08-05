#!/usr/bin/env bash
#
# Deploy the metro-pulse-collector Lambda: build the zip, then push it with
# `aws lambda update-function-code`.
#
# This script deploys CODE ONLY. The function, its execution role, its
# environment variables, and the EventBridge schedule are all provisioned
# outside this repo and are never created or modified here — the only mutating
# call is update-function-code. Everything else is a read used to verify that
# the artifact we built matches the function we are pushing it to.
#
# That verification is the point. build_package.sh cross-compiles for
# arm64/Python 3.12; if the deployed function is not actually arm64/Python 3.12,
# the upload succeeds and the function then fails at import time, once a minute,
# silently losing history that cannot be backfilled. Cheaper to check here.
#
# Usage:
#   ./infra/lambda_collector/deploy.sh
#   FUNCTION_NAME=my-fn AWS_REGION=us-west-2 ./infra/lambda_collector/deploy.sh
#   AWS_PROFILE=personal ./infra/lambda_collector/deploy.sh
#   SMOKE_TEST=1 ./infra/lambda_collector/deploy.sh   # invoke once after deploy

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration — override any of these from the environment.
# --------------------------------------------------------------------------
FUNCTION_NAME="${FUNCTION_NAME:-metro-pulse-collector}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# What build_package.sh targets. The function must match, or the artifact will
# not import in Lambda.
EXPECTED_ARCHITECTURE="arm64"
EXPECTED_RUNTIME="python3.12"
EXPECTED_HANDLER="handler.lambda_handler"

# Invoke once after deploying to confirm the new code imports and the feeds
# still collect. Off by default so deploy does one thing; worth turning on for
# the first deploy of a change.
SMOKE_TEST="${SMOKE_TEST:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SCRIPT="$SCRIPT_DIR/build_package.sh"
ZIP_PATH="$SCRIPT_DIR/metro-pulse-collector.zip"

log() { printf '  %s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }
fail() {
  printf '\nDEPLOY FAILED: %s\n' "$*" >&2
  exit 1
}

aws_lambda() { aws lambda --region "$AWS_REGION" "$@"; }

# --------------------------------------------------------------------------
# 1. Preflight — fail before building rather than after.
# --------------------------------------------------------------------------
step "Preflight"
command -v aws >/dev/null || fail "the AWS CLI is not installed or not on PATH."

# No default region is assumed anywhere; --region is passed on every call. This
# also means the script behaves identically regardless of the caller's profile.
log "function:  $FUNCTION_NAME"
log "region:    $AWS_REGION"
# An `[[ ... ]] && log ...` one-liner here would exit the script under `set -e`
# whenever AWS_PROFILE is unset, since the whole compound returns 1.
if [[ -n "${AWS_PROFILE:-}" ]]; then
  log "profile:   $AWS_PROFILE"
fi

caller_arn="$(aws sts get-caller-identity --region "$AWS_REGION" --query Arn --output text 2>/dev/null)" ||
  fail "could not authenticate to AWS. Check your credentials (aws configure) \
or AWS_PROFILE."
log "identity:  $caller_arn"

# --------------------------------------------------------------------------
# 2. Confirm the function exists and matches what we build for.
#
#    Read-only. This script never creates or reconfigures the function — if any
#    of these do not match, that is a console change for you to make knowingly,
#    not something a deploy script should silently "fix".
# --------------------------------------------------------------------------
step "Verifying target function"
# A list query, not a dict: `--output text` on a dict emits values ordered by key
# name, so adding a field later would silently reshuffle the columns below. A
# list preserves the order written here.
function_config="$(
  aws_lambda get-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --query '[Architectures[0],Runtime,Handler,Timeout,MemorySize]' \
    --output text 2>/dev/null
)" || fail "function '$FUNCTION_NAME' was not found in $AWS_REGION. This script \
deploys code to an existing function; it does not create one."

read -r actual_arch actual_runtime actual_handler actual_timeout actual_memory <<<"$function_config"

log "architecture: $actual_arch"
log "runtime:      $actual_runtime"
log "handler:      $actual_handler"
log "timeout:      ${actual_timeout}s, memory: ${actual_memory}MB"

[[ "$actual_arch" == "$EXPECTED_ARCHITECTURE" ]] ||
  fail "function architecture is '$actual_arch' but the package is built for \
'$EXPECTED_ARCHITECTURE'. The upload would succeed and then fail at import time \
in Lambda. Fix EXPECTED_ARCHITECTURE here and TARGET_PLATFORM in \
build_package.sh, or change the function's architecture."

[[ "$actual_runtime" == "$EXPECTED_RUNTIME" ]] ||
  fail "function runtime is '$actual_runtime' but the package is built for \
'$EXPECTED_RUNTIME'. Wheels compiled for a different Python minor version will \
not import."

# The handler string is a configuration concern, not a packaging one, so this is
# a warning rather than a hard failure — but a wrong value means every scheduled
# invocation fails immediately.
if [[ "$actual_handler" != "$EXPECTED_HANDLER" ]]; then
  log ""
  log "WARNING: handler is '$actual_handler', expected '$EXPECTED_HANDLER'."
  log "The zip puts handler.py at its root, so the function will not start"
  log "unless the handler is set to '$EXPECTED_HANDLER' in the console."
fi

# --------------------------------------------------------------------------
# 3. Build.
# --------------------------------------------------------------------------
step "Building deployment package"
[[ -x "$BUILD_SCRIPT" ]] || fail "$BUILD_SCRIPT is missing or not executable."
"$BUILD_SCRIPT"
[[ -f "$ZIP_PATH" ]] || fail "the build did not produce $ZIP_PATH."

# Lambda reports the SHA256 of what it stored, base64-encoded. Computing the
# same locally lets us prove the deployed artifact is byte-for-byte the one just
# built, rather than trusting that the upload did what it said.
local_sha256="$(openssl dgst -sha256 -binary "$ZIP_PATH" | openssl base64)"

# --------------------------------------------------------------------------
# 4. Upload — the only mutating call in this script.
# --------------------------------------------------------------------------
step "Uploading to $FUNCTION_NAME"
update_result="$(
  aws_lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --zip-file "fileb://$ZIP_PATH" \
    --query '[LastModified,CodeSize,Version]' \
    --output text
)" || fail "update-function-code failed. Check that your credentials grant \
lambda:UpdateFunctionCode on $FUNCTION_NAME."

read -r modified size version <<<"$update_result"
log "modified:  $modified"
log "code size: $size bytes"
log "version:   $version"

# The update is asynchronous. Without waiting, a smoke test (or a fast-following
# deploy) can race against a function still in the Pending state.
step "Waiting for the update to become active"
aws_lambda wait function-updated-v2 --function-name "$FUNCTION_NAME" ||
  fail "the function did not reach an updated state. Check the console for a \
LastUpdateStatusReason."
log "ok"

# --------------------------------------------------------------------------
# 5. Verify the deployed artifact is the one we built.
# --------------------------------------------------------------------------
step "Verifying deployed artifact"
remote_sha256="$(
  aws_lambda get-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --query CodeSha256 --output text
)"

if [[ "$local_sha256" == "$remote_sha256" ]]; then
  log "ok        deployed CodeSha256 matches the local zip"
else
  fail "deployed CodeSha256 ($remote_sha256) does not match the local zip \
($local_sha256). Something else may have deployed to this function concurrently."
fi

# --------------------------------------------------------------------------
# 6. Optional smoke test.
#
#    A real invocation, which writes a real snapshot to S3 — harmless, since the
#    schedule is doing exactly that every minute anyway. Catches an import error
#    now instead of via a gap in the archive discovered days later.
# --------------------------------------------------------------------------
if [[ "$SMOKE_TEST" == "1" ]]; then
  step "Smoke test — invoking once"
  response_file="$(mktemp)"
  trap 'rm -f "$response_file"' EXIT

  invoke_status="$(
    aws_lambda invoke \
      --function-name "$FUNCTION_NAME" \
      --payload '{"source":"deploy.sh smoke test"}' \
      --cli-binary-format raw-in-base64-out \
      --query 'FunctionError' --output text \
      "$response_file"
  )" || fail "the invocation call itself failed (missing lambda:InvokeFunction?)."

  if [[ "$invoke_status" != "None" ]]; then
    log "response: $(cat "$response_file")"
    fail "the function returned an error ($invoke_status). The new code is \
deployed but not working — check CloudWatch Logs."
  fi

  log "response: $(cat "$response_file")"
  log ""
  log "Check feeds_failed in the response above: 0 means every feed collected."
fi

step "Deploy complete"
log "Logs: aws logs tail /aws/lambda/$FUNCTION_NAME --follow --region $AWS_REGION"
