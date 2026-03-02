#!/usr/bin/env bash
# Create a Spot A100 80GB VM on Google Cloud for ClearMesh training/inference.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GPU quota approved for NVIDIA A100 80GB in target zone
#
# Usage: ./create_vm.sh [vm-name] [zone] [machine-type]

set -euo pipefail

VM_NAME="${1:-clearmesh-train}"
ZONE="${2:-us-central1-a}"
MACHINE_TYPE="${3:-a2-ultragpu-1g}"  # 1x A100 80GB

echo "=== Creating Spot VM: ${VM_NAME} ==="
echo "Zone: ${ZONE}"
echo "Machine type: ${MACHINE_TYPE}"
echo ""

gcloud compute instances create "${VM_NAME}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator=count=1,type=nvidia-a100-80gb \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --provisioning-model=SPOT \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes=default,storage-rw

echo ""
echo "=== VM created successfully ==="
echo "SSH:  gcloud compute ssh ${VM_NAME} --zone=${ZONE}"
echo "Stop: gcloud compute instances stop ${VM_NAME} --zone=${ZONE}"
echo ""
echo "Next steps:"
echo "  1. Run scripts/gcp/create_storage.sh to create persistent disk"
echo "  2. SSH in and run scripts/setup/setup_all.sh"
