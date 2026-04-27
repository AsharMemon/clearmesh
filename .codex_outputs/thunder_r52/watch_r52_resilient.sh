#!/usr/bin/env bash
# Resilient watch script for r52. Local laptop side. Polls Thunder, rsyncs
# log + outputs, auto-deletes instance on completion. Self-restarts if it
# crashes via outer wrapper (start_watch_r52.sh).
BASE="/Users/Ashar/Documents/GitHub/clearmesh/.codex_outputs/thunder_r52"
WATCH_LOG="$BASE/watch_r52.log"
TNR="/Users/Ashar/.tnr/bin/tnr"
INSTANCE_NAME="${INSTANCE_NAME:-j6h0jrtl}"
mkdir -p "$BASE"
echo "[watch] start pid=$$ $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
trap 'echo "[watch] EXIT trap pid=$$ $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"' EXIT

while true; do
  echo "[watch] tick $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"

  # Snapshot tnr status; non-fatal if it errors
  STATUS="$("$TNR" status --json 2>>"$WATCH_LOG" || true)"
  printf '%s\n' "$STATUS" > "$BASE/tnr_status_latest.json"

  # If our instance is no longer in the list, exit cleanly
  if ! printf '%s\n' "$STATUS" | grep -q "$INSTANCE_NAME"; then
    echo "[watch] instance $INSTANCE_NAME not found; exiting" >> "$WATCH_LOG"
    exit 0
  fi

  # Always try to sync the log (best-effort; ignore failures)
  ssh -o BatchMode=yes -o ConnectTimeout=15 tnr-1 'test -f /workspace/dualprim_paper_r52.log' >/dev/null 2>&1 \
    && rsync -az -e ssh tnr-1:/workspace/dualprim_paper_r52.log "$BASE/" >> "$WATCH_LOG" 2>&1

  # Ask Thunder for r52 process state
  REMOTE_STATE="$(ssh -o BatchMode=yes -o ConnectTimeout=15 tnr-1 'if test -f /workspace/r52.done; then echo done; elif test -f /workspace/r52.failed; then echo failed; elif test -f /workspace/r52.pid && ps -p $(cat /workspace/r52.pid) >/dev/null 2>&1; then echo running; else echo stopped; fi' 2>>"$WATCH_LOG")"
  REMOTE_STATE="${REMOTE_STATE:-ssh_failed}"
  echo "[watch] state=$REMOTE_STATE" >> "$WATCH_LOG"

  case "$REMOTE_STATE" in
    done|failed|stopped)
      echo "[watch] terminal state — final sync $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
      rsync -az -e ssh tnr-1:/workspace/dualprim_paper_r52.log "$BASE/" >> "$WATCH_LOG" 2>&1 || true
      rsync -az -e ssh tnr-1:/workspace/dualprim_k100_paper_camera_r52/ "$BASE/dualprim_k100_paper_camera_r52/" >> "$WATCH_LOG" 2>&1 || true
      echo "[watch] deleting instance $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
      "$TNR" delete 0 --yes >> "$WATCH_LOG" 2>&1 || true
      "$TNR" status --json > "$BASE/tnr_status_after_delete.json" 2>>"$WATCH_LOG" || true
      echo "[watch] exit clean $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
      exit 0
      ;;
  esac

  sleep 180
done
