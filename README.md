# ClearMesh

Unified 3D generation system: single image to high-fidelity, print-ready 3D asset.

## Architecture

```
Image → Background Removal → TRELLIS.2 (coarse) → Stage 2 DiT (refine)
      → Part Decomposition → Geometry Super-Res → NDC/FlexiCubes (sharp edges)
      → Mesh Repair → Retopology → Scale → Textures → Auto-Rig → Export
```

| Component | Role | Source |
|-----------|------|--------|
| TRELLIS.2 4B | Stage 1 coarse generation | microsoft/TRELLIS.2 (MIT) |
| Custom DiT | Stage 2 refinement with FlexiCubes | Trained on 120K objects |
| PartCrafter | Part decomposition (optional) | VAST-AI-Research/PartCrafter |
| SuperCarver / CraftsMan3D | Geometry super-resolution (optional) | CraftsMan3D fallback available |
| NDC / FlexiCubes | Sharp-edge mesh extraction | czq142857/NDC, NVIDIA Kaolin |
| BPT | Retopology — clean topology (optional) | CVPR 2025 |
| PyMeshFix + Trimesh | Mesh repair + print readiness | PyPI |
| Puppeteer | Auto-rigging + animation (optional) | Seed3D/Puppeteer (MIT) |
| UniRig | Auto-rigging fallback (optional) | VAST-AI-Research/UniRig (MIT) |

## Quick Start

### 1. Set up GCP VM

```bash
# Create Spot A100 80GB VM
./scripts/gcp/create_vm.sh

# Create persistent storage
./scripts/gcp/create_storage.sh

# SSH in
./scripts/gcp/ssh_connect.sh
```

### 2. Install everything

```bash
# On the VM:
git clone https://github.com/AsharMemon/clearmesh.git
cd clearmesh
chmod +x scripts/**/*.sh

# Full install (all optional components + rigging)
./scripts/setup/setup_all.sh

# Core only (skip optional components and rigging)
./scripts/setup/setup_all.sh /mnt/data --skip-rigging --skip-optional
```

### 3. Prepare training data

```bash
conda activate clearmesh

# Download Objaverse (start with 1K for testing)
python scripts/data/download_objaverse.py --limit 1000

# Filter dataset
python scripts/data/filter_dataset.py \
    --manifest /mnt/data/objaverse/manifest.json

# Generate coarse/fine pairs
python scripts/data/generate_pairs.py \
    --input_json /mnt/data/filtered/high_quality_models.json

# Convert to O-Voxel format
python scripts/data/convert_ovoxel.py \
    --input_dir /mnt/data/training_pairs
```

### 4. Train Stage 2

```bash
# Start preemption handler (for Spot VM resilience)
./scripts/utils/preemption_handler.sh &

# Train with FlexiCubes in the loop (Option C)
python -m clearmesh.stage2.train \
    --config configs/train_stage2_flexicubes.yaml

# Monitor progress
python scripts/utils/monitor_training.py \
    --checkpoint_dir /mnt/data/checkpoints/clearmesh_stage2 --watch
```

### 5. Generate 3D models

```bash
# Basic generation
python -m clearmesh.pipeline --input photo.png --output model.glb

# Print-ready STL at 32mm scale with drain holes
python -m clearmesh.pipeline \
    --input photo.png \
    --output mini.stl \
    --format stl \
    --scale 32mm \
    --add-base \
    --drain-holes

# Full pipeline: decomposition + super-res + retopology + textures
python -m clearmesh.pipeline \
    --input photo.png \
    --output model.glb \
    --decompose \
    --super-res \
    --retopo \
    --textures

# With auto-rigging for animation (optional)
python -m clearmesh.pipeline \
    --input character.png \
    --output rigged.fbx \
    --format fbx \
    --rig

# Fast mode (12 diffusion steps instead of 50)
python -m clearmesh.pipeline --input photo.png --fast
```

## Pipeline Stages

| # | Stage | Default | Flag |
|---|-------|---------|------|
| 1 | Background removal | on | `--skip-bg-removal` |
| 2 | TRELLIS.2 coarse generation | on | — |
| 3 | Stage 2 DiT refinement | on | `--no-refine` |
| 4 | Part decomposition (PartCrafter) | off | `--decompose` |
| 5 | Geometry super-resolution | off | `--super-res` |
| 6 | NDC mesh extraction | on | — |
| 7 | Mesh repair + print prep | on | — |
| 8 | Retopology (BPT) | off | `--retopo` |
| 9 | Scale to miniature size | off | `--scale 32mm` |
| 10 | PBR textures | off | `--textures` |
| 11 | Auto-rigging | off | `--rig` |
| 12 | Export (STL/GLB/OBJ/FBX) | on | `--format glb` |

## Project Structure

```
clearmesh/
├── scripts/
│   ├── gcp/              GCP VM management (create, start, stop, SSH)
│   ├── setup/            Environment and dependency installation
│   ├── data/             Dataset download, filtering, pair generation
│   └── utils/            Preemption handler, training monitor
├── configs/              Training and inference YAML configs
├── clearmesh/            Main Python package
│   ├── pipeline.py       ClearMeshPipeline (end-to-end, 12 stages)
│   ├── stage2/           Refinement DiT model + training loop
│   ├── mesh/             Extraction, repair, export
│   ├── partcrafter/      Part decomposition (PartCrafter)
│   ├── supercarver/      Geometry super-resolution
│   ├── retopology/       BPT retopology
│   ├── texture/          PBR texture handling
│   ├── rigging/          Auto-rigging (Puppeteer/UniRig)
│   └── utils/            Background removal, scaling
└── requirements.txt
```

## Cost Estimate (Spot A100 80GB)

| Phase | GPU Hours | Cost |
|-------|-----------|------|
| Environment setup | ~5 hrs | ~$4 |
| Data preparation | ~10 hrs | ~$7 |
| Training (Option C) | 120-200 hrs | $88-$146 |
| Integration testing | ~30 hrs | ~$22 |
| **Total** | | **$120-$180** |

## Environments

Two conda environments handle dependency conflicts:

- **clearmesh**: PyTorch 2.6 + CUDA 12.4 (TRELLIS.2, training, inference)
- **rigging**: PyTorch 2.1.1 + CUDA 11.8 (UniRig, Puppeteer)

## Spot VM Resilience

Training on Spot VMs saves 60-80% but VMs can be preempted with 30s notice:

- Checkpoints saved every 1000 steps to persistent disk
- `preemption_handler.sh` polls GCP metadata and sends SIGUSR1
- Training auto-resumes from latest checkpoint
- Persistent disk survives VM deletion

## License

MIT
