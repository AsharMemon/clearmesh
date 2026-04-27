#!/usr/bin/env bash
set -u
BASE="/Users/Ashar/Documents/GitHub/clearmesh/.codex_outputs/thunder_r51_j6h0jrtl"
WATCH_LOG="$BASE/watch_r51.log"
TNR="/Users/Ashar/.tnr/bin/tnr"
mkdir -p "$BASE"
echo "[watch] start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
while true; do
  echo "[watch] tick $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
  STATUS="$($TNR status --json 2>&1 || true)"
  printf '%s\n' "$STATUS" > "$BASE/tnr_status_latest.json"
  if ! printf '%s\n' "$STATUS" | grep -q 'j6h0jrtl'; then
    echo "[watch] instance not found; exiting" >> "$WATCH_LOG"
    exit 0
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=12 tnr-1 'test -f /workspace/dualprim_compress_r51.log' >/dev/null 2>&1 && \
    rsync -az -e ssh tnr-1:/workspace/dualprim_compress_r51.log "$BASE/" >> "$WATCH_LOG" 2>&1 || true
  REMOTE_STATE="$(ssh -o BatchMode=yes -o ConnectTimeout=12 tnr-1 'if test -f /workspace/r51.done; then echo done; elif test -f /workspace/r51.failed; then echo failed; elif test -f /workspace/r51.pid && ps -p $(cat /workspace/r51.pid) >/dev/null 2>&1; then echo running; else echo stopped; fi' 2>>"$WATCH_LOG" || echo ssh_failed)"
  echo "[watch] state=$REMOTE_STATE" >> "$WATCH_LOG"
  case "$REMOTE_STATE" in
    done|failed|stopped)
      echo "[watch] final copy $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
      rsync -az -e ssh tnr-1:/workspace/dualprim_compress_r51.log "$BASE/" >> "$WATCH_LOG" 2>&1 || true
      rsync -az -e ssh tnr-1:/workspace/dualprim_k086_compress_from_r50_r51/ "$BASE/dualprim_k086_compress_from_r50_r51/" >> "$WATCH_LOG" 2>&1 || true
      echo "[watch] deleting instance $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
      $TNR delete 0 --yes >> "$WATCH_LOG" 2>&1 || true
      $TNR status --json > "$BASE/tnr_status_after_delete.json" 2>>"$WATCH_LOG" || true
      echo "[watch] exit $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
      exit 0
      ;;
  esac
  sleep 180
done
