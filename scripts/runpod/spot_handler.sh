#!/usr/bin/env bash
# RunPod Spot pod termination handler.
#
# RunPod sends SIGTERM before stopping a Spot pod.
# This script traps SIGTERM and forwards SIGUSR1 to the training process
# for emergency checkpoint saving.
#
# Usage: Run alongside training:
#   ./scripts/runpod/spot_handler.sh &
#   python -m clearmesh.stage2.train --config ...

set -euo pipefail

PID_FILE="/tmp/train.pid"

echo "RunPod spot handler started"

# Wait for training process to start
while [ ! -f "${PID_FILE}" ]; do
    echo "Waiting for training process PID file..."
    sleep 2
done

TRAIN_PID=$(cat "${PID_FILE}")
echo "Monitoring training process: PID ${TRAIN_PID}"

# Trap SIGTERM (sent by RunPod before spot termination)
handle_sigterm() {
    echo ""
    echo "!!! RUNPOD SPOT TERMINATION DETECTED !!!"
    echo "Sending SIGUSR1 to training process (PID ${TRAIN_PID})..."
    kill -USR1 "${TRAIN_PID}" 2>/dev/null || true

    # Wait for checkpoint save
    echo "Waiting for emergency checkpoint..."
    sleep 20

    echo "Spot handler done."
    exit 0
}

trap handle_sigterm SIGTERM

# Keep running (wait is interruptible by signals)
while true; do
    # Check if training is still alive
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "Training process (PID ${TRAIN_PID}) finished. Exiting."
        exit 0
    fi
    sleep 5
done
