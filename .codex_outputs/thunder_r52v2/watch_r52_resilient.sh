#!/usr/bin/env bash
# Fixed watcher v2: Only declare terminal state when r52.done OR r52.failed
# flag exists (not just on PID-not-running). The PID check ALONE is unsafe
# because the launcher may transition between processes during setup.
# Also: requires r52.running flag absent AND a terminal flag before cleanup.
BASE="/Users/Ashar/Documents/GitHub/clearmesh/.codex_outputs/thunder_r52v2"
WATCH_LOG="$BASE/watch_r52.log"
TNR="/Users/Ashar/.tnr/bin/tnr"
INSTANCE_NAME="${INSTANCE_NAME:-bbml1h2c}"
mkdir -p "$BASE"
echo "[watch] start pid=$$ $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
trap 'echo "[watch] EXIT trap pid=$$ $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"' EXIT

# Wait grace period before any termination logic — give r52 time to actually
# come up and write its sentinel files.
GRACE_SECONDS=300
START_EPOCH=$(date -u +%s)

while true; do
  echo "[watch] tick $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"

  STATUS="$("$TNR" status --json 2>>"$WATCH_LOG" || true)"
  printf '%s\n' "$STATUS" > "$BASE/tnr_status_latest.json"
  if ! printf '%s\n' "$STATUS" | grep -q "$INSTANCE_NAME"; then
    echo "[watch] instance $INSTANCE_NAME not found; exiting" >> "$WATCH_LOG"
    exit 0
  fi

  # Always sync the log
  ssh -o BatchMode=yes -o ConnectTimeout=15 tnr-1 'test -f /workspace/dualprim_paper_r52.log' >/dev/null 2>&1 \
    && rsync -az -e ssh tnr-1:/workspace/dualprim_paper_r52.log "$BASE/" >> "$WATCH_LOG" 2>&1

  # Read sentinel flags from Thunder
  REMOTE_FLAGS="$(ssh -o BatchMode=yes -o ConnectTimeout=15 tnr-1 'echo running=$(test -f /workspace/r52.running && echo 1 || echo 0); echo done=$(test -f /workspace/r52.done && echo 1 || echo 0); echo failed=$(test -f /workspace/r52.failed && echo 1 || echo 0)' 2>>"$WATCH_LOG" || echo "ssh_failed=1")"
  echo "[watch] flags: $REMOTE_FLAGS" >> "$WATCH_LOG"

  HAS_DONE=$(echo "$REMOTE_FLAGS" | grep -E "^done=1" | head -1)
  HAS_FAILED=$(echo "$REMOTE_FLAGS" | grep -E "^failed=1" | head -1)

  # Only declare terminal if a terminal flag is set
  if [ -n "$HAS_DONE" ] || [ -n "$HAS_FAILED" ]; then
    echo "[watch] terminal flag detected — final sync $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
    rsync -az -e ssh tnr-1:/workspace/dualprim_paper_r52.log "$BASE/" >> "$WATCH_LOG" 2>&1 || true
    rsync -az -e ssh tnr-1:/workspace/dualprim_k100_paper_camera_r52/ "$BASE/dualprim_k100_paper_camera_r52/" >> "$WATCH_LOG" 2>&1 || true
    rsync -az -e ssh tnr-1:/workspace/r52.done "$BASE/" >> "$WATCH_LOG" 2>&1 || true
    rsync -az -e ssh tnr-1:/workspace/r52.failed "$BASE/" >> "$WATCH_LOG" 2>&1 || true
    echo "[watch] deleting instance $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
    "$TNR" delete 0 --yes >> "$WATCH_LOG" 2>&1 || true
    "$TNR" status --json > "$BASE/tnr_status_after_delete.json" 2>>"$WATCH_LOG" || true
    echo "[watch] exit clean $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCH_LOG"
    exit 0
  fi

  # Defensive: if past grace period AND running flag is absent AND no terminal
  # flag, that's an abnormal exit (process killed outside our control).
  NOW_EPOCH=$(date -u +%s)
  ELAPSED=$((NOW_EPOCH - START_EPOCH))
  if [ "$ELAPSED" -gt "$GRACE_SECONDS" ]; then
    HAS_RUNNING=$(echo "$REMOTE_FLAGS" | grep -E "^running=1" | head -1)
    if [ -z "$HAS_RUNNING" ] && [ -z "$HAS_DONE" ] && [ -z "$HAS_FAILED" ]; then
      echo "[watch] WARN past grace, no running/done/failed — assuming abnormal exit, syncing + exiting (no instance delete)" >> "$WATCH_LOG"
      rsync -az -e ssh tnr-1:/workspace/dualprim_paper_r52.log "$BASE/" >> "$WATCH_LOG" 2>&1 || true
      rsync -az -e ssh tnr-1:/workspace/dualprim_k100_paper_camera_r52/ "$BASE/dualprim_k100_paper_camera_r52/" >> "$WATCH_LOG" 2>&1 || true
      exit 0
    fi
  fi

  sleep 180
done
