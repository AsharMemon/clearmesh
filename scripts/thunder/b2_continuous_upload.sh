#!/usr/bin/env bash
# Continuously uploads FACE run/corpus artifacts to Backblaze B2 via rclone.
# Credentials are read from env only; this script intentionally contains no keys.
set -euo pipefail

MODE="${MODE:-face_run}"              # face_run | face_shard
LOCAL_ROOT="${LOCAL_ROOT:?LOCAL_ROOT is required}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_PREFIX="${B2_PREFIX:?B2_PREFIX is required}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
STABILITY_SECONDS="${STABILITY_SECONDS:-120}"
STATE_DIR="${STATE_DIR:-/tmp/clearmesh_b2_upload_state}"
RCLONE_REMOTE="${RCLONE_REMOTE:-b2env}"
LOG_TIME() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
mkdir -p "$STATE_DIR"

if [[ -z "${B2_KEY_ID:-}" || -z "${B2_APP_KEY:-}" ]] && [[ -n "${B2_TOKEN:-}" ]]; then
  parsed_b2="$(
    python3 - <<'PY'
import json
import os
import sys

token = os.environ.get("B2_TOKEN", "").strip()
key_id = app_key = ""
if token:
    if token.startswith("{"):
        payload = json.loads(token)
        key_id = (
            payload.get("keyId")
            or payload.get("keyID")
            or payload.get("applicationKeyId")
            or payload.get("applicationKeyID")
            or payload.get("key_id")
            or payload.get("application_key_id")
            or payload.get("accountId")
            or payload.get("accountID")
            or ""
        )
        app_key = (
            payload.get("applicationKey")
            or payload.get("application_key")
            or payload.get("appKey")
            or payload.get("app_key")
            or payload.get("key")
            or ""
        )
    elif ":" in token:
        key_id, app_key = token.split(":", 1)
if key_id and app_key:
    sys.stdout.write(key_id + "\n" + app_key)
PY
  )"
  if [[ -n "$parsed_b2" ]]; then
    B2_KEY_ID="${B2_KEY_ID:-$(printf '%s\n' "$parsed_b2" | sed -n '1p')}"
    B2_APP_KEY="${B2_APP_KEY:-$(printf '%s\n' "$parsed_b2" | sed -n '2p')}"
  fi
fi
if [[ -n "${B2_KEY_ID:-}" && -z "${B2_APP_KEY:-}" && -n "${B2_TOKEN:-}" ]]; then
  case "$B2_TOKEN" in
    \{*|*:*) ;;
    *) B2_APP_KEY="$B2_TOKEN" ;;
  esac
fi

if [[ -n "${B2_KEY_ID:-}" && -n "${B2_APP_KEY:-}" ]]; then
  export RCLONE_CONFIG_${RCLONE_REMOTE^^}_TYPE=b2
  export RCLONE_CONFIG_${RCLONE_REMOTE^^}_ACCOUNT="$B2_KEY_ID"
  export RCLONE_CONFIG_${RCLONE_REMOTE^^}_KEY="$B2_APP_KEY"
fi

# Also support pre-populated RCLONE_CONFIG_B2ENV_* env vars.
if ! rclone lsd "${RCLONE_REMOTE}:" >/dev/null 2>&1; then
  echo "[$(LOG_TIME)] ERROR: cannot access rclone remote ${RCLONE_REMOTE}:" >&2
  exit 2
fi

remote_base="${RCLONE_REMOTE}:${B2_BUCKET}/${B2_PREFIX}"
state_file="$STATE_DIR/$(echo "${B2_BUCKET}_${B2_PREFIX}" | tr '/:' '__').state"
touch "$state_file"

is_stable() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  local mtime now age
  mtime=$(stat -c '%Y' "$path" 2>/dev/null || echo 0)
  now=$(date +%s)
  age=$(( now - mtime ))
  (( age >= STABILITY_SECONDS ))
}

upload_changed_file() {
  local path="$1" rel="$2"
  [[ -f "$path" ]] || return 0
  is_stable "$path" || return 0
  local size mtime key
  size=$(stat -c '%s' "$path")
  mtime=$(stat -c '%Y' "$path")
  key="$rel|$size|$mtime"
  if grep -Fqx "$key" "$state_file"; then
    return 0
  fi
  echo "[$(LOG_TIME)] upload file $rel ($size bytes)"
  rclone copyto "$path" "$remote_base/$rel" --stats 30s
  echo "$key" >> "$state_file"
}

upload_manifest() {
  local manifest="$STATE_DIR/manifest.$(date -u +%Y%m%dT%H%M%SZ).txt"
  {
    echo "mode=$MODE"
    echo "local_root=$LOCAL_ROOT"
    echo "remote_base=$remote_base"
    echo "time=$(LOG_TIME)"
    find "$LOCAL_ROOT" -maxdepth 5 -type f -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TSZ\n' 2>/dev/null | sort | head -n 20000
  } > "$manifest"
  rclone copyto "$manifest" "$remote_base/upload_manifests/$(basename "$manifest")" --stats 30s >/dev/null 2>&1 || true
  rm -f "$manifest"
}

sync_face_run_metadata() {
  echo "[$(LOG_TIME)] sync face_run metadata -> $remote_base"
  local filter_file="$STATE_DIR/face_run_filters.txt"
  cat > "$filter_file" <<'EOF'
+ /status.jsonl
+ /logs/**
+ /runs/**/logs/**
+ /runs/**/summary*.json
+ /runs/**/scale_readiness.json
+ /runs/**/face_gate_inspection.json
+ /runs/**/*.json
+ /runs/**/*.jsonl
+ /runs/**/*.png
+ /runs/**/*.jpg
+ /runs/**/*.jpeg
+ /runs/**/*.glb
- /corpus/**
- *.pt
- *
EOF
  rclone copy "$LOCAL_ROOT" "$remote_base" \
    --filter-from "$filter_file" \
    --transfers 8 --checkers 16 --stats 30s || true
  while IFS= read -r -d '' ckpt; do
    local rel
    rel="${ckpt#"$LOCAL_ROOT"/}"
    upload_changed_file "$ckpt" "$rel"
    # Snapshot mutable checkpoint names as immutable resume points too.
    case "$(basename "$ckpt")" in
      checkpoint.latest.pt|checkpoint.current.pt|checkpoint.pt)
        if is_stable "$ckpt"; then
          local size mtime snap_rel key
          size=$(stat -c '%s' "$ckpt")
          mtime=$(stat -c '%Y' "$ckpt")
          snap_rel="checkpoint_snapshots/${rel//\//__}.${mtime}.pt"
          key="$snap_rel|$size|$mtime"
          if ! grep -Fqx "$key" "$state_file"; then
            echo "[$(LOG_TIME)] upload checkpoint snapshot $snap_rel ($size bytes)"
            rclone copyto "$ckpt" "$remote_base/$snap_rel" --stats 30s
            echo "$key" >> "$state_file"
          fi
        fi
        ;;
    esac
  done < <(find "$LOCAL_ROOT" -type f \( -name 'checkpoint*.pt' -o -name '*.ckpt' \) -print0 2>/dev/null)
}

sync_face_shard() {
  echo "[$(LOG_TIME)] sync face_shard progress -> $remote_base"
  local filter_file="$STATE_DIR/face_shard_filters.txt"
  cat > "$filter_file" <<'EOF'
+ /source_annotations.jsonl
+ /status.jsonl
+ /logs/**
+ /corpus/raw/summary.json
+ /corpus/raw/selected_annotations.jsonl
+ /corpus/raw/download_manifest.json
+ /corpus/curated_candidates.jsonl
+ /corpus/strict_targets/**
+ /corpus/tokens/**
+ /corpus/split*/**
+ /pilot_summary*.json
+ /strict_gate*.json
+ /lean_face_corpus.tar.gz
+ /lean_face_corpus.tar.gz.json
- *
EOF
  rclone copy "$LOCAL_ROOT" "$remote_base" \
    --filter-from "$filter_file" \
    --transfers 16 --checkers 32 --stats 30s || true
  upload_changed_file "$LOCAL_ROOT/lean_face_corpus.tar.gz" "lean_face_corpus.tar.gz"
  upload_changed_file "$LOCAL_ROOT/lean_face_corpus.tar.gz.json" "lean_face_corpus.tar.gz.json"
}

run_once() {
  if [[ ! -d "$LOCAL_ROOT" ]]; then
    echo "[$(LOG_TIME)] waiting for LOCAL_ROOT=$LOCAL_ROOT"
    return 0
  fi
  case "$MODE" in
    face_run) sync_face_run_metadata ;;
    face_shard) sync_face_shard ;;
    *) echo "[$(LOG_TIME)] ERROR: unknown MODE=$MODE" >&2; return 2 ;;
  esac
  upload_manifest
  echo "[$(LOG_TIME)] sync pass complete"
}

echo "[$(LOG_TIME)] starting clearmesh B2 continuous uploader mode=$MODE root=$LOCAL_ROOT remote=$remote_base interval=${INTERVAL_SECONDS}s"
while true; do
  run_once || true
  sleep "$INTERVAL_SECONDS"
done
