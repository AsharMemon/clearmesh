#!/usr/bin/env bash
set -u
OUT="/Users/Ashar/Documents/GitHub/clearmesh/.codex_outputs/thunder_r49_8o7wj8jd"
LOG="$OUT/watch_simple.log"
TNR="/Users/Ashar/.tnr/bin/tnr"
START=$(date +%s)
MAX=$((3*60*60))
echo "[simple] start $(date)" >> "$LOG"
while true; do
  NOW=$(date +%s)
  if [ $((NOW-START)) -gt "$MAX" ]; then echo "[simple] timeout $(date)" >> "$LOG"; break; fi
  ssh -o BatchMode=yes -o ConnectTimeout=10 tnr-1 'test -f /workspace/R49_DONE' >/dev/null 2>&1 && { echo "[simple] done $(date)" >> "$LOG"; break; }
  ssh -o BatchMode=yes -o ConnectTimeout=10 tnr-1 'pgrep -af "run_r49_depth|run_canary.py.*depth_camera_r49" >/dev/null' >/dev/null 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then echo "[simple] running $(date)" >> "$LOG"; sleep 60; continue; fi
  echo "[simple] not running rc=$rc $(date)" >> "$LOG"; break
done
rsync -az -e ssh tnr-1:/workspace/dualprim_depth_r49.log "$OUT/" >> "$LOG" 2>&1 || true
rsync -az -e ssh tnr-1:/workspace/dualprim_k008_depth_camera_r49/ "$OUT/dualprim_k008_depth_camera_r49/" >> "$LOG" 2>&1 || true
"$TNR" delete 0 --yes >> "$LOG" 2>&1 || true
echo "[simple] delete issued $(date)" >> "$LOG"
