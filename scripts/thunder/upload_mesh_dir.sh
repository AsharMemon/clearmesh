#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
LOCAL_MESH_DIR="${2:-}"
REMOTE_MESH_DIR="${3:-/tmp/clearmesh_input_meshes}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set. Export it before uploading to Thunder." >&2
  exit 1
fi
if [ -z "$LOCAL_MESH_DIR" ] || [ ! -d "$LOCAL_MESH_DIR" ]; then
  echo "Usage: $0 [instance_id] /path/to/local_mesh_dir [/remote/mesh_dir]" >&2
  exit 2
fi

LOCAL_MESH_DIR="$(cd "$LOCAL_MESH_DIR" && pwd)"
ARCHIVE="$(mktemp -t clearmesh-meshes.XXXXXX.tar.gz)"
trap 'rm -f "$ARCHIVE"' EXIT

parent="$(dirname "$LOCAL_MESH_DIR")"
base="$(basename "$LOCAL_MESH_DIR")"
COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --exclude='.DS_Store' \
  --exclude='._*' \
  --exclude='__MACOSX' \
  -czf "$ARCHIVE" \
  -C "$parent" \
  "$base"

"$TNR_BIN" scp "$ARCHIVE" "$INSTANCE_ID:/tmp/clearmesh_mesh_upload.tar.gz"
printf 'set -euo pipefail\nrm -rf %q\nmkdir -p %q\nmkdir -p /tmp/clearmesh_mesh_upload\nrm -rf /tmp/clearmesh_mesh_upload/*\ntar --exclude="._*" --exclude="__MACOSX" -xzf /tmp/clearmesh_mesh_upload.tar.gz -C /tmp/clearmesh_mesh_upload\nsrc="$(find /tmp/clearmesh_mesh_upload -mindepth 1 -maxdepth 1 -type d | head -n 1)"\ncp -R "$src"/. %q/\nfind %q -name "._*" -delete\nfind %q -maxdepth 1 -type f | sort\nrm /tmp/clearmesh_mesh_upload.tar.gz\nexit\n' \
  "$REMOTE_MESH_DIR" "$REMOTE_MESH_DIR" "$REMOTE_MESH_DIR" "$REMOTE_MESH_DIR" "$REMOTE_MESH_DIR" | "$TNR_BIN" connect "$INSTANCE_ID"
