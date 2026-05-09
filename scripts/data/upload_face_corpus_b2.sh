#!/usr/bin/env bash
# Upload a packaged FACE corpus archive to Backblaze B2 via rclone.
set -euo pipefail

ARCHIVE="${1:-}"
B2_BUCKET_FROM_ENV="${B2_BUCKET:-}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_PREFIX="${B2_PREFIX:-face-corpora}"
RCLONE_REMOTE="${RCLONE_REMOTE:-b2}"
RCLONE_CONFIG_DIR="${RCLONE_CONFIG_DIR:-$HOME/.config/rclone}"
RCLONE_CONFIG="$RCLONE_CONFIG_DIR/rclone.conf"

if [ -z "$ARCHIVE" ] || [ ! -f "$ARCHIVE" ]; then
  echo "Usage: B2_BUCKET=<bucket> scripts/data/upload_face_corpus_b2.sh <corpus.tar.gz>" >&2
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
  if [ ! -f "$RCLONE_CONFIG" ] || ! grep -q "^\[${RCLONE_REMOTE}\]$" "$RCLONE_CONFIG"; then
    {
      echo "[${RCLONE_REMOTE}]"
      echo "type = b2"
      echo "account = ${B2_KEY_ID}"
      echo "key = ${B2_APP_KEY}"
    } >> "$RCLONE_CONFIG"
  fi
fi

archive_name="$(basename "$ARCHIVE")"
manifest=""
if [ -f "${ARCHIVE}.json" ]; then
  manifest="${ARCHIVE}.json"
elif [ -f "${ARCHIVE%.tar.gz}.tar.gz.json" ]; then
  manifest="${ARCHIVE%.tar.gz}.tar.gz.json"
fi

remote_path="${RCLONE_REMOTE}:${B2_BUCKET}/${B2_PREFIX}/${archive_name}"
echo "Uploading $ARCHIVE -> $remote_path"
rclone copyto "$ARCHIVE" "$remote_path" --progress
if [ -n "$manifest" ]; then
  echo "Uploading $manifest -> ${remote_path}.json"
  rclone copyto "$manifest" "${remote_path}.json" --progress
fi
echo "b2://${B2_BUCKET}/${B2_PREFIX}/${archive_name}"
