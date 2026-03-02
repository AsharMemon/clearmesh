#!/usr/bin/env bash
# Create and attach a persistent disk for model weights, datasets, and checkpoints.
# This disk survives VM deletion — reattach to new VMs after Spot preemption.
#
# Usage: ./create_storage.sh [disk-name] [size-gb] [zone] [vm-name]

set -euo pipefail

DISK_NAME="${1:-clearmesh-data}"
SIZE_GB="${2:-500}"
ZONE="${3:-us-central1-a}"
VM_NAME="${4:-clearmesh-train}"

echo "=== Creating persistent disk: ${DISK_NAME} (${SIZE_GB}GB) ==="

# Create the disk
gcloud compute disks create "${DISK_NAME}" \
    --size="${SIZE_GB}GB" \
    --zone="${ZONE}" \
    --type=pd-balanced

# Attach to VM
echo "Attaching to VM: ${VM_NAME}..."
gcloud compute instances attach-disk "${VM_NAME}" \
    --disk="${DISK_NAME}" \
    --zone="${ZONE}"

echo ""
echo "=== Disk created and attached ==="
echo ""
echo "SSH into the VM and run these commands to format and mount:"
echo ""
echo "  sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0 /dev/sdb"
echo "  sudo mkdir -p /mnt/data"
echo "  sudo mount -o discard,defaults /dev/sdb /mnt/data"
echo "  sudo chown \$USER:\$USER /mnt/data"
echo ""
echo "  # Add to fstab for auto-mount on reboot:"
echo "  echo '/dev/sdb /mnt/data ext4 discard,defaults,nofail 0 2' | sudo tee -a /etc/fstab"
echo ""
echo "Store all models, datasets, and checkpoints under /mnt/data/"
