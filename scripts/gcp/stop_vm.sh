#!/usr/bin/env bash
# Stop the ClearMesh VM to save money. Disk data is preserved.
# Usage: ./stop_vm.sh [vm-name] [zone]

set -euo pipefail

VM_NAME="${1:-clearmesh-train}"
ZONE="${2:-us-central1-a}"

echo "Stopping VM: ${VM_NAME}..."
gcloud compute instances stop "${VM_NAME}" --zone="${ZONE}"
echo "VM stopped. You are only paying for disk storage now."
