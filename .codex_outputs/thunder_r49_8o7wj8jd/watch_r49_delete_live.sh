#!/usr/bin/env bash
set -u
OUT="/Users/Ashar/Documents/GitHub/clearmesh/.codex_outputs/thunder_r49_8o7wj8jd"
LOG="$OUT/watch_live.log"
TNR="/Users/Ashar/.tnr/bin/tnr"
MAX_SECONDS=$((4 * 60 * 60))
START=$(date +%s)
MISSES=0
echo "[watch2] start $(date) max_seconds=$MAX_SECONDS" >> "$LOG"
while true; do
  NOW=$(date +%s)
  if [ $((NOW - START)) -gt "$MAX_SECONDS" ]; then
    echo "[watch2] hard timeout $(date); copying then deleting" >> "$LOG"
    break
  fi
  if ssh -o BatchMode=yes -o ConnectTimeout=10 tnr-1 'test -f /workspace/R49_DONE' >/dev/null 2>&1; then
    echo "[watch2] R49_DONE $(date)" >> "$LOG"
    break
  fi
  if ssh -o BatchMode=yes -o ConnectTimeout=10 tnr-1 'pgrep -af "run_r49_depth|run_canary.py.*depth_camera_r49" >/dev/null' >/dev/null 2>&1; then
    MISSES=0
    echo "[watch2] still running $(date) elapsed=$((NOW - START))s" >> "$LOG"
    sleep 60
    continue
  fi
  MISSES=$((MISSES+1))
  echo "[watch2] process check miss=$MISSES $(date)" >> "$LOG"
  if [ "$MISSES" -ge 3 ]; then
    echo "[watch2] process gone/crashed; copying then deleting $(date)" >> "$LOG"
    break
  fi
  sleep 60
 done
rsync -az -e ssh tnr-1:/workspace/dualprim_depth_r49.log "$OUT/" >> "$LOG" 2>&1 || true
rsync -az -e ssh tnr-1:/workspace/dualprim_k008_depth_camera_r49/ "$OUT/dualprim_k008_depth_camera_r49/" >> "$LOG" 2>&1 || true
"$TNR" delete 0 --yes >> "$LOG" 2>&1 || true
echo "[watch2] delete issued $(date)" >> "$LOG"
