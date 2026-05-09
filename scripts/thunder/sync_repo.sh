#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE="$(mktemp -t clearmesh-thunder.XXXXXX.tar.gz)"
trap 'rm -f "$ARCHIVE"' EXIT

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set. Export it before syncing." >&2
  exit 1
fi

cd "$REPO_ROOT"
COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.codex_outputs' \
  --exclude='.clearmesh_state' \
  --exclude='artifacts' \
  --exclude='eval_results' \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  --exclude='__MACOSX' \
  -czf "$ARCHIVE" .

"$TNR_BIN" scp "$ARCHIVE" "$INSTANCE_ID:/home/ubuntu/clearmesh_src.tar.gz"
printf 'rm -rf %q && mkdir -p %q && tar --exclude="._*" --exclude="__MACOSX" -xzf /home/ubuntu/clearmesh_src.tar.gz -C %q && rm /home/ubuntu/clearmesh_src.tar.gz && cd %q && find . -name "._*" -delete && pwd && find . -maxdepth 2 -type f | wc -l\nexit\n' \
  "$REMOTE_DIR" "$REMOTE_DIR" "$REMOTE_DIR" "$REMOTE_DIR" | "$TNR_BIN" connect "$INSTANCE_ID"
