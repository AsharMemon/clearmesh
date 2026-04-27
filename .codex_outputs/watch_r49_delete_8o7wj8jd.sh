#!/usr/bin/env bash
set -u
OUT="/Users/Ashar/Documents/GitHub/clearmesh/.codex_outputs/thunder_r49_8o7wj8jd"
LOG="$OUT/watch.log"
TNR="/Users/Ashar/.tnr/bin/tnr"
echo "[watch] start $(date -Is)" >> "$LOG"
while true; do
  if ssh -o ConnectTimeout=10 tnr-1 'test -f /workspace/R49_DONE' >/dev/null 2>&1; then
    echo "[watch] R49_DONE $(date -Is)" >> "$LOG"
    break
  fi
  if ssh -o ConnectTimeout=10 tnr-1 'pgrep -af "run_r49_depth|run_canary.py.*depth_camera_r49" >/dev/null' >/dev/null 2>&1; then
    echo "[watch] still running $(date -Is)" >> "$LOG"
    sleep 60
    continue
  fi
  echo "[watch] r49 process not found; treating as stopped $(date -Is)" >> "$LOG"
  break
 done
mkdir -p "$OUT"
rsync -az -e ssh tnr-1:/workspace/dualprim_depth_r49.log "$OUT/" >> "$LOG" 2>&1 || true
rsync -az -e ssh tnr-1:/workspace/dualprim_k008_depth_camera_r49/ "$OUT/dualprim_k008_depth_camera_r49/" >> "$LOG" 2>&1 || true
$TNR delete 0 --yes >> "$LOG" 2>&1 || true
echo "[watch] delete issued $(date -Is)" >> "$LOG"
