#!/usr/bin/env bash
# Outer wrapper: respawns watch_r52_resilient.sh if it exits non-zero.
BASE="/Users/Ashar/Documents/GitHub/clearmesh/.codex_outputs/thunder_r52v2"
SUPERVISOR_LOG="$BASE/supervisor.log"
WATCH="$BASE/watch_r52_resilient.sh"
mkdir -p "$BASE"
echo "[supervisor] start pid=$$ $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUPERVISOR_LOG"
attempt=0
while true; do
  attempt=$((attempt+1))
  echo "[supervisor] attempt=$attempt launching watch $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUPERVISOR_LOG"
  bash "$WATCH" >> "$SUPERVISOR_LOG" 2>&1
  rc=$?
  echo "[supervisor] watch exited rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUPERVISOR_LOG"
  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] clean exit; supervisor done" >> "$SUPERVISOR_LOG"
    exit 0
  fi
  echo "[supervisor] sleeping 30s before restart" >> "$SUPERVISOR_LOG"
  sleep 30
done
