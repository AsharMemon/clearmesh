#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set. Export it before running Thunder commands." >&2
  exit 1
fi
if [ ! -x "$TNR_BIN" ]; then
  echo "tnr binary not found or not executable: $TNR_BIN" >&2
  exit 1
fi
if [ $# -gt 0 ]; then
  shift
fi
if [ $# -eq 0 ]; then
  if [ -t 0 ]; then
    echo "Usage: scripts/thunder/run_remote.sh [instance_id] <remote command...>" >&2
    echo "   or: scripts/thunder/run_remote.sh [instance_id] < script.sh" >&2
    exit 1
  fi
  {
    printf "bash -s <<'CLEARMESH_REMOTE_SCRIPT'\n"
    cat -
    printf "\nCLEARMESH_REMOTE_SCRIPT\nexit\n"
  } | "$TNR_BIN" connect "$INSTANCE_ID"
  exit 0
fi

remote_command="$*"
printf '%s\nexit\n' "$remote_command" | "$TNR_BIN" connect "$INSTANCE_ID"
