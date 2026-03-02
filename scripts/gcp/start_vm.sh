#!/usr/bin/env bash
# Start a stopped ClearMesh VM.
# Usage: ./start_vm.sh [vm-name] [zone]

set -euo pipefail

VM_NAME="${1:-clearmesh-train}"
ZONE="${2:-us-central1-a}"

echo "Starting VM: ${VM_NAME}..."
gcloud compute instances start "${VM_NAME}" --zone="${ZONE}"

echo "Waiting for VM to be ready..."
gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" \
    --format='get(status)'

echo ""
echo "VM started. SSH: gcloud compute ssh ${VM_NAME} --zone=${ZONE}"
