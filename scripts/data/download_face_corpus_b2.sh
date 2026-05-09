#!/usr/bin/env bash
# Download a packaged FACE corpus archive from Backblaze B2 via rclone.
set -euo pipefail

SOURCE="${1:-${B2_CORPUS_URI:-}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-/tmp/clearmesh_face_corpus_download}}"
B2_BUCKET_FROM_ENV="${B2_BUCKET:-}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_PREFIX="${B2_PREFIX:-face-corpora}"
RCLONE_REMOTE="${RCLONE_REMOTE:-b2}"
RCLONE_CONFIG_DIR="${RCLONE_CONFIG_DIR:-$HOME/.config/rclone}"
RCLONE_CONFIG="$RCLONE_CONFIG_DIR/rclone.conf"
EXTRACT="${EXTRACT:-1}"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/data/download_face_corpus_b2.sh b2://<bucket>/<prefix>/<archive.tar.gz> <output_dir>
  scripts/data/download_face_corpus_b2.sh <archive.tar.gz> <output_dir>

If a bare archive name is provided, B2_BUCKET and B2_PREFIX are used.
Requires either an existing rclone remote, or B2_KEY_ID and B2_APP_KEY.
EOF
}

if [ -z "$SOURCE" ]; then
  usage
  exit 1
fi
if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is required. Install with: curl -sSL https://rclone.org/install.sh | bash" >&2
  exit 2
fi

load_b2_token() {
  if [ -z "${B2_TOKEN:-}" ] || { [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_APP_KEY:-}" ]; }; then
    return 0
  fi
  local parsed key_id app_key bucket
  parsed="$(python3 - <<'PY'
import json
import os
token = os.environ.get("B2_TOKEN", "").strip()
key_id = app_key = bucket = ""
if token.startswith("{"):
    try:
        obj = json.loads(token)
    except json.JSONDecodeError:
        obj = {}
    def pick(*names):
        for name in names:
            value = obj.get(name)
            if value:
                return str(value)
        return ""
    key_id = pick("keyId", "keyID", "applicationKeyId", "applicationKeyID", "accountId", "accountID", "key_id", "application_key_id")
    app_key = pick("applicationKey", "appKey", "key", "application_key", "app_key")
    bucket = pick("bucketName", "bucket", "bucket_name")
elif ":" in token:
    key_id, app_key = token.split(":", 1)
if key_id and app_key:
    print("\t".join([key_id, app_key, bucket]))
PY
)"
  if [ -n "$parsed" ]; then
    IFS=$'\t' read -r key_id app_key bucket <<<"$parsed"
    export B2_KEY_ID="${B2_KEY_ID:-$key_id}"
    export B2_APP_KEY="${B2_APP_KEY:-$app_key}"
    if [ -z "$B2_BUCKET_FROM_ENV" ] && [ -n "${bucket:-}" ]; then
      B2_BUCKET="$bucket"
    fi
  fi
}

if ! rclone listremotes 2>/dev/null | grep -qx "${RCLONE_REMOTE}:"; then
  load_b2_token
  if [ -z "${B2_KEY_ID:-}" ] || [ -z "${B2_APP_KEY:-}" ]; then
    echo "No rclone remote '${RCLONE_REMOTE}:' and B2_KEY_ID/B2_APP_KEY are not set." >&2
    echo "B2_TOKEN is supported only as JSON with keyId/applicationKey or as key_id:application_key." >&2
    exit 3
  fi
  mkdir -p "$RCLONE_CONFIG_DIR"
  if [ -f "$RCLONE_CONFIG" ] && grep -q "^\[${RCLONE_REMOTE}\]$" "$RCLONE_CONFIG"; then
    :
  else
    {
      echo "[${RCLONE_REMOTE}]"
      echo "type = b2"
      echo "account = ${B2_KEY_ID}"
      echo "key = ${B2_APP_KEY}"
    } >> "$RCLONE_CONFIG"
  fi
fi

case "$SOURCE" in
  b2://*)
    stripped="${SOURCE#b2://}"
    bucket="${stripped%%/*}"
    key="${stripped#*/}"
    remote_path="${RCLONE_REMOTE}:${bucket}/${key}"
    archive_name="$(basename "$key")"
    ;;
  ${RCLONE_REMOTE}:*)
    remote_path="$SOURCE"
    archive_name="$(basename "$SOURCE")"
    ;;
  */*)
    remote_path="${RCLONE_REMOTE}:${B2_BUCKET}/${SOURCE}"
    archive_name="$(basename "$SOURCE")"
    ;;
  *)
    remote_path="${RCLONE_REMOTE}:${B2_BUCKET}/${B2_PREFIX}/${SOURCE}"
    archive_name="$(basename "$SOURCE")"
    ;;
esac

mkdir -p "$OUTPUT_DIR"
archive_path="$OUTPUT_DIR/$archive_name"

echo "Downloading $remote_path -> $archive_path"
rclone copyto "$remote_path" "$archive_path" --progress

if rclone lsf "${remote_path}.json" >/dev/null 2>&1; then
  echo "Downloading ${remote_path}.json -> ${archive_path}.json"
  rclone copyto "${remote_path}.json" "${archive_path}.json" --progress
fi

if [ "$EXTRACT" = "1" ]; then
  extract_dir="$OUTPUT_DIR/extracted"
  rm -rf "$extract_dir"
  mkdir -p "$extract_dir"
  tar -xzf "$archive_path" -C "$extract_dir"
  echo "extracted=$extract_dir"
fi

echo "archive=$archive_path"
