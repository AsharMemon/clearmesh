#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_ARCHIVE="${2:-}"
DOWNLOAD_DIR="${3:-}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
EXTRACT="${EXTRACT:-1}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set. Export it before fetching Thunder artifacts." >&2
  exit 1
fi
if [ -z "$REMOTE_ARCHIVE" ] || [ -z "$DOWNLOAD_DIR" ]; then
  echo "Usage: scripts/thunder/fetch_remote_artifact.sh [instance_id] <remote_archive.tar.gz> <download_dir>" >&2
  exit 1
fi
if [ ! -x "$TNR_BIN" ]; then
  echo "tnr binary not found or not executable: $TNR_BIN" >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_DIR"
local_archive="$DOWNLOAD_DIR/$(basename "$REMOTE_ARCHIVE")"
"$TNR_BIN" scp "$INSTANCE_ID:$REMOTE_ARCHIVE" "$local_archive"

if [ "$EXTRACT" = "1" ]; then
  tar -xzf "$local_archive" -C "$DOWNLOAD_DIR"
fi

echo "downloaded=$local_archive"
if [ "$EXTRACT" = "1" ]; then
  echo "extracted=$DOWNLOAD_DIR"
fi
