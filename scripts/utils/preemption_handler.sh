#!/usr/bin/env bash
# Spot/preemptible VM handler for ClearMesh training.
#
# Supports both GCP and RunPod:
#   - GCP:    Polls metadata endpoint for 30-second preemption warning
#   - RunPod: Traps SIGTERM sent before spot pod termination
#
# Sends SIGUSR1 to the training process for emergency checkpoint save.
#
# Usage: Run alongside training:
#   ./preemption_handler.sh &
#   python -m clearmesh.stage2.train --config ...

set -euo pipefail

POLL_INTERVAL="${1:-5}"  # seconds between checks (GCP only)
PID_FILE="/tmp/train.pid"

# Auto-detect platform
if [ -n "${RUNPOD_POD_ID:-}" ]; then
    PLATFORM="runpod"
else
    PLATFORM="gcp"
fi

echo "Preemption handler started (platform: ${PLATFORM})"

# Wait for training process to start and write PID
while [ ! -f "${PID_FILE}" ]; do
    echo "Waiting for training process PID file..."
    sleep 2
done

TRAIN_PID=$(cat "${PID_FILE}")
echo "Monitoring training process: PID ${TRAIN_PID}"

# Signal training to save checkpoint
trigger_save() {
    echo ""
    echo "!!! PREEMPTION/TERMINATION DETECTED !!!"
    echo "Sending SIGUSR1 to training process (PID ${TRAIN_PID})..."
    kill -USR1 "${TRAIN_PID}" 2>/dev/null || true
    echo "Waiting for emergency checkpoint..."
    sleep 20
    echo "Preemption handler done."
    exit 0
}

if [ "${PLATFORM}" = "runpod" ]; then
    # RunPod: trap SIGTERM
    trap trigger_save SIGTERM

    while true; do
        if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
            echo "Training process (PID ${TRAIN_PID}) no longer running. Exiting."
            exit 0
        fi
        sleep 5
    done
else
    # GCP: poll metadata endpoint
    while true; do
        if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
            echo "Training process (PID ${TRAIN_PID}) no longer running. Exiting."
            exit 0
        fi

        STATUS=$(curl -s -H "Metadata-Flavor: Google" \
            "http://metadata.google.internal/computeMetadata/v1/instance/preempted" \
            2>/dev/null || echo "UNKNOWN")

        if [ "${STATUS}" = "TRUE" ]; then
            trigger_save
        fi

        sleep "${POLL_INTERVAL}"
    done
fi
