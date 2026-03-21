# ClearMesh 50K Pilot: Vast.ai Deployment Guide

## Prerequisites
- Backblaze B2 account (free tier covers this)
- Vast.ai account with credits (~$400 for 8×A100, 3 days)
- RunPod pod with shard JSONs already prepared

## Step 1: Create Backblaze B2 Bucket (~5 min)

1. Go to https://www.backblaze.com/cloud-storage
2. Create account (free, no CC needed for first 10GB)
3. Create bucket: `clearmesh-pairs` (private)
4. Go to App Keys → Create New Key:
   - Name: `clearmesh-rw`
   - Bucket: `clearmesh-pairs`
   - Capabilities: Read, Write, List
5. Save the `keyID` and `applicationKey`

## Step 2: Upload to B2 from RunPod (~20 min)

```bash
# SSH into RunPod pod
ssh -i ~/.ssh/id_ed25519 -p 39875 root@195.26.233.9

# Install rclone
curl -sSL https://rclone.org/install.sh | bash

# Configure B2
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << 'EOF'
[b2]
type = b2
account = YOUR_B2_KEY_ID
key = YOUR_B2_APP_KEY
EOF

# Test access
rclone ls b2:clearmesh-pairs/ --max-depth 1

# Upload shard JSONs (~10MB total, instant)
rclone copy /workspace/data/trellis500k/shards/ b2:clearmesh-pairs/shards/ --progress

# Upload TRELLIS weights (~18GB, ~5 min)
rclone copy /workspace/models/trellis2-4b/ b2:clearmesh-pairs/models/trellis2-4b/ --progress

# Upload ClearMesh repo as tarball (~1MB)
cd /workspace && tar czf /tmp/clearmesh.tar.gz clearmesh/
rclone copy /tmp/clearmesh.tar.gz b2:clearmesh-pairs/ --progress

# Verify
rclone ls b2:clearmesh-pairs/shards/ | head
rclone ls b2:clearmesh-pairs/models/trellis2-4b/ | head
```

## Step 3: Launch Vast.ai Pods (~10 min)

### Option A: Via Vast.ai CLI
```bash
pip install vastai
vastai set api-key YOUR_VAST_API_KEY

# Search for A100 80GB instances with ≥500GB disk
vastai search offers 'gpu_name=A100_SXM4 num_gpus=1 disk_space>=500 reliability>0.95' --order 'dph_total'

# Launch 8 pods
for i in {0..7}; do
    vastai create instance OFFER_ID \
        --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
        --disk 500 \
        --env "SHARD_ID=$i NUM_SHARDS=8 B2_KEY_ID=YOUR_KEY_ID B2_APP_KEY=YOUR_APP_KEY HF_TOKEN=YOUR_HF_TOKEN" \
        --onstart-cmd "apt-get update -qq && apt-get install -y -qq rclone curl && curl -sSL https://rclone.org/install.sh | bash 2>/dev/null; mkdir -p ~/.config/rclone && echo '[b2]' > ~/.config/rclone/rclone.conf && echo 'type = b2' >> ~/.config/rclone/rclone.conf && echo 'account = '$B2_KEY_ID >> ~/.config/rclone/rclone.conf && echo 'key = '$B2_APP_KEY >> ~/.config/rclone/rclone.conf && rclone copy b2:clearmesh-pairs/clearmesh.tar.gz /workspace/ && cd /workspace && tar xzf clearmesh.tar.gz && bash /workspace/clearmesh/scripts/data/setup_vastai_pod.sh && bash /workspace/clearmesh/scripts/data/run_pairs_vastai.sh"
done
```

### Option B: Via Vast.ai Web UI
1. Go to https://cloud.vast.ai/create/
2. Select: A100 80GB, 500GB disk, pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
3. Set environment variables:
   - `SHARD_ID=0` (change for each pod: 0-7)
   - `NUM_SHARDS=8`
   - `B2_KEY_ID=your_key_id`
   - `B2_APP_KEY=your_app_key`
   - `HF_TOKEN=your_hf_token` (for DINOv3 gated model)
4. Set onstart script (same as above)
5. Repeat for shards 1-7

## Step 4: Monitor Progress

### From any machine with rclone configured:
```bash
# Check B2 for progress
for i in {0..7}; do
    echo -n "Shard $i: "
    rclone cat b2:clearmesh-pairs/progress/shard_$i/progress.json 2>/dev/null | \
        python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "not started"
done
```

### SSH into a pod:
```bash
# Check pair count
find /workspace/data/training_pairs -name "coarse_voxels.npy" | wc -l

# Check logs
tail -50 /workspace/data/training_pairs/shard_0.log
```

## Step 5: Collect Results (~30 min)

After all pods complete:
```bash
# On RunPod training pod (or any machine):
mkdir -p /workspace/data/training_pairs_50k

# Download all pairs from B2
rclone copy b2:clearmesh-pairs/pairs/ /workspace/data/training_pairs_50k/ --progress

# Build manifest
cd /workspace/clearmesh
python3 scripts/data/build_manifest.py \
    --pairs_dir /workspace/data/training_pairs_50k \
    --output /workspace/data/training_pairs_50k/manifest_train.json

# Check count
python3 -c "import json; m=json.load(open('/workspace/data/training_pairs_50k/manifest_train.json')); print(f'Trainable pairs: {len(m)}')"
```

## Step 6: Train on 50K Pairs

```bash
cd /workspace/clearmesh
python -m clearmesh.stage2.train \
    --config configs/train_stage2_flexicubes_runpod.yaml \
    --data_dir /workspace/data/training_pairs_50k
```

## Expected Timeline

| Phase | Duration | Cost |
|-------|----------|------|
| B2 setup + upload | 30 min | $0 |
| Vast.ai pod boot + setup | 30 min | ~$10 |
| Model download per pod | 20-40 min | ~$15 |
| Pair generation (6,250/pod) | ~3 days | ~$370 |
| SDF conversion + upload | 2 hours | ~$5 |
| Collect + train | 12 hours | RunPod |
| **Total** | **~3.5 days** | **~$400** |

## Expected Yield

- 50,000 models attempted across 8 pods
- ~25,000-35,000 TRELLIS successes (50-70% on quality-filtered models)
- ~14,000-25,000 trainable pairs (after SDF conversion)
- That's **10-18× more data** than current 1,386 pairs

## Troubleshooting

### Pod crashed / preempted
Progress saves every 25 models. Launch a new pod with the same SHARD_ID — it resumes automatically.

### Disk full on pod
Each pod needs ~64GB models + ~18GB weights + ~75GB output ≈ ~160GB. 500GB disk is sufficient.

### B2 sync failing
Check rclone config: `rclone ls b2:clearmesh-pairs/`. If auth fails, re-create the app key.

### Low success rate
Check `failure_details.json` in the shard output. Common failures:
- OOM: reduce MAX_MODELS_PER_RUN to 100
- Mesh loading errors: expected, these are filtered out
- TRELLIS inference errors: usually bad geometry, expected
