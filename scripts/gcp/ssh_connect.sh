#!/usr/bin/env bash
# SSH into the ClearMesh VM.
# Usage: ./ssh_connect.sh [vm-name] [zone]

set -euo pipefail

VM_NAME="${1:-clearmesh-train}"
ZONE="${2:-us-central1-a}"

gcloud compute ssh "${VM_NAME}" --zone="${ZONE}"
