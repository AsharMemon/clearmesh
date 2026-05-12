# Production GPU Dependency Fixes

This file captures the non-obvious dependency and runtime fixes needed to spin
up a fresh production GPU instance for the ClearMesh route:

```text
TRELLIS.2
  -> UltraShape paper-setting reference refinement
  -> normalization / retopo / quad sidecars
  -> MeshRipple optional mesh head
  -> gates / export
```

Use this as the source of truth when recreating the Thunder/RunPod-style host.

## One-Command Thunder Setup

From the local repo:

```bash
export THUNDER_TOKEN=...
export HF_TOKEN=... # or HUGGINGFACE_HUB_TOKEN, if needed

THUNDER_INSTANCE_ID=0 scripts/thunder/setup_production_instance.sh
```

The setup script runs:

```text
sync_repo.sh
sync_hf_token.sh, when HF_TOKEN exists
bootstrap_remote.sh
install_trellis2_env.sh
install_ultrashape.sh
install_mesh_heads.sh
install_meshripple_env.sh
download_meshripple_checkpoints.sh
install_quad_baselines.sh
trellis2_preflight.sh
meshripple_preflight.sh
```

Skip optional pieces with:

```bash
INSTALL_MESHRIPPLE=0
INSTALL_QUAD=0
INSTALL_ULTRASHAPE=0
INSTALL_TRELLIS=0
RUN_PREFLIGHTS=0
SYNC_HF_TOKEN=0
```

## TRELLIS.2 Runtime Fixes

Script: `scripts/setup/install_trellis2_env.sh`

Important pins/fixes:

```text
venv: /home/ubuntu/trellis2-venv
repo: /home/ubuntu/TRELLIS.2
torch: 2.6.0 + cu124
torchvision: 0.21.0
CUDA toolkit: /usr/local/cuda-12.4
transformers: 4.57.5
huggingface_hub: >=0.33.5,<2.0
flash-attn: 2.7.3
extensions: nvdiffrast, nvdiffrec, cumesh, o-voxel, flexgemm
```

Why:

```text
Thunder can expose CUDA 13 at /usr/local/cuda, but TRELLIS.2 uses cu124 Torch.
Native extension builds need CUDA Toolkit 12.4, so PATH/CUDA_HOME must point to
/usr/local/cuda-12.4 during builds and runs.

TRELLIS.2 currently expects the Transformers 4.57 DINOv3 object layout. Newer
Transformers versions can break pipeline.run(...) with missing DINOv3 fields.

Interrupted TRELLIS extension installs can leave partial repos under
/tmp/extensions. The installer removes /tmp/extensions before rebuilding.
```

Run-time environment:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/ubuntu/TRELLIS.2:${PYTHONPATH:-}
export ATTN_BACKEND=flash_attn
```

## ClearMesh Lightweight Runtime Fixes

The shared `clearmesh-venv` / API-worker environment now needs
`fast-simplification>=0.1.13` in addition to `trimesh`. It is used only for
deterministic fixture/control-mesh decimation before FACE-target checks; it is
not a replacement for manifoldization.

Install from the repo requirements:

```bash
python -m pip install -r requirements-product.txt
```

If recreating an existing Thunder `clearmesh-venv`, patch it directly:

```bash
/home/ubuntu/clearmesh-venv/bin/python -m pip install fast-simplification>=0.1.13
```

## UltraShape Runtime Fixes

Scripts:

```text
scripts/thunder/install_ultrashape.sh
scripts/setup/install_ultrashape.sh
scripts/product/run_ultrashape_refinement.py
clearmesh/stage2/ultrashape_refiner.py
```

Important layout:

```text
venv: /home/ubuntu/ultrashape-venv
python: 3.10
repo: /workspace/UltraShape-1.0
checkpoint: /workspace/checkpoints/ultrashape_v1.pt
config: /workspace/UltraShape-1.0/configs/infer_dit_refine.yaml
```

Why:

```text
UltraShape and TRELLIS.2 should not run in the same Python process. They can
both register cuBVH/extension types. ClearMesh therefore invokes UltraShape as
a command hook/subprocess.

UltraShape's Python dependency stack is better isolated in Python 3.10. Do not
install it into /home/ubuntu/clearmesh-venv, which is the lightweight product
worker environment.
```

Paper-setting defaults now locked in ClearMesh:

```text
steps: 50
num_latents: 32768
octree_resolution: 1024
chunk_size: 8000
normalize_scale: 0.99
seed: 42
surface samples: 204800 uniform + 204800 sharp-edge
remove background: true for production parity
```

Use unbuffered Python for readable logs:

```bash
/home/ubuntu/ultrashape-venv/bin/python -u \
  /home/ubuntu/clearmesh/scripts/product/run_ultrashape_refinement.py \
  --mesh {input_mesh} \
  --image {reference_image} \
  --output-dir {output_dir} \
  --ultrashape-dir /workspace/UltraShape-1.0 \
  --checkpoint /workspace/checkpoints/ultrashape_v1.pt \
  --num-steps 50 \
  --octree-resolution 1024 \
  --num-latents 32768 \
  --chunk-size 8000 \
  --scale 0.99 \
  --seed 42 \
  --remove-bg
```

## MeshRipple Runtime Fixes

Script: `scripts/setup/install_meshripple_env.sh`

Important pins/fixes:

```text
venv: /home/ubuntu/meshripple-venv
repo: /home/ubuntu/mesh-heads/MeshRipple
torch: 2.8.0 + cu128
torchvision: 0.23.0
torchaudio: 2.8.0
flash-attn wheel: 2.7.3 cp312/cu12/torch2.8
checkpoint dir: /home/ubuntu/mesh-heads/MeshRipple/ckpt
```

Why:

```text
MeshRipple's public runtime wants a newer Python/Torch stack than TRELLIS.2 or
UltraShape. Keep it in its own venv.

Use the prebuilt flash-attn wheel in install_meshripple_env.sh; building from
source on a fresh rented GPU is slow and fragile.
```

## LATTICE/FACE Research Runtime Fixes

Scripts:

```text
scripts/thunder/face_conditioned_tiny_smoke.sh
scripts/thunder/face_conditioned_overfit.sh
scripts/thunder/face_level_conditioned_overfit.sh
scripts/thunder/face_level_conditioned_smoke.sh
scripts/thunder/lattice_vdf_tiny_smoke.sh
scripts/research/train_face_conditioned_tiny.py
scripts/research/train_face_level_conditioned_tiny.py
scripts/research/eval_face_level_conditioned_tiny.py
scripts/research/train_lattice_vdf_tiny.py
clearmesh/mesh_heads/face_level.py
```

Important pins/fixes:

```text
venv: /home/ubuntu/clearmesh-venv
torch: 2.8.0 + cu128
checkpoint metadata: JSON-safe only; no raw pathlib.Path objects
sync archives: tar --no-xattrs to avoid macOS extended-header noise
```

Why:

```text
PyTorch 2.6+ defaults torch.load(..., weights_only=True). If checkpoints store
raw argparse Path objects, safe loading fails with an unsupported-global error.
ClearMesh research trainers now save only simple metadata via
clearmesh.utils.checkpoint.args_to_json_safe(...).

Tiny overfit runs can hit a better loss before the final step, then drift.
The FACE and LATTICE research trainers now save the best observed model state,
plus best_step/best_loss, instead of blindly saving final weights.
```

Verified 2026-05-03 on Thunder instance `0`:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/lattice_vdf_tiny_smoke.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/face_conditioned_overfit.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/face_level_conditioned_overfit.sh
```

Results:

```text
LATTICE VDF tiny overfit:
  best_loss: 2.774e-10
  predicted: 12 faces, 8 vertices, watertight=true
  target:    12 faces, 8 vertices, watertight=true
  max VDF abs error: 8.82e-05

FACE conditioned overfit:
  best_loss: 0.00112
  generated: 12 faces, 8 vertices, watertight=true
  teacher:   12 faces, 8 vertices, watertight=true

FACE-level conditioned overfit:
  autoregressive steps: 12 faces instead of 108 coordinate tokens
  best_loss: 0.01457
  generated: 12 faces, 8 vertices, watertight=true
  teacher:   12 faces, 8 vertices, watertight=true

FACE-level count-head proof:
  face count argument: omitted
  requested_face_count_from_model: 12
  generated: 12 faces, 8 vertices, watertight=true

FACE-level 16-box smoke:
  attempted: 16
  watertight_rate: 1.0
  mean_chamfer_l2: 0.00502
  mean_normal_consistency: 0.92575

FACE-level mixed primitive smoke:
  attempted: 10
  max_ar_steps: 256
  max_coordinate_tokens: 2304
  watertight_rate: 1.0
  mean_chamfer_l2: 0.00394
  mean_normal_consistency: 0.95470

FACE-level adapter smoke:
  entrypoint: scripts/product/run_mesh_head.py --head face-level
  generated: 12 faces, 8 vertices, watertight=true
  boundary_edges: 0
  nonmanifold_edges: 0
```

Durable local proof meshes were downloaded to:

```text
artifacts/research_proofs/2026-05-03/
```

## Quad/Retopo Runtime Fixes

Scripts/modules:

```text
scripts/setup/install_quad_baselines.sh
clearmesh/retopology/quad_remesh.py
```

Important fix:

```text
pyinstantmeshes file API only accepts .ply/.obj/.aln. The production worker can
hand it .glb control meshes. ClearMesh now routes unsupported containers through
the array API or materializes a temporary OBJ for CLI remeshers before falling
back to the template cage.
```

This matters because the 2026-05-03 Thunder run revealed that pyinstantmeshes
was installed but silently fell back to `template_cage` after receiving `.glb`.

## Shell Quoting Fixes

Scripts:

```text
scripts/thunder/production_path_smoke.sh
scripts/thunder/trellis2_smoke.sh
scripts/thunder/pipeline_trellis2_meshripple_smoke.sh
scripts/thunder/pipeline_trellis2_meshripple_batch.sh
scripts/thunder/meshripple_quality_sweep.sh
```

Important fix:

```text
Remote heredocs must not over-escape shell variables. The working form is:

export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/ubuntu/TRELLIS.2:${PYTHONPATH:-}
METADATA_JSON=$(mktemp ...)
JOB_ID=$(...)

Not:

export PATH="\$CUDA_HOME/bin:\$PATH"
METADATA_JSON=\$(mktemp ...)
```

The broken form caused `rm: command not found` and `syntax error near unexpected
token '('` on Thunder because PATH and command substitution were passed
literally.

## Verified Smoke

Verified on 2026-05-03:

```bash
THUNDER_INSTANCE_ID=0 REFERENCE_MODE=ultrashape \
  scripts/thunder/production_path_smoke.sh
```

Job:

```text
job_5fe8693c8fd14558a8d06d7bfde95ff3: succeeded
```

Important quality caveat:

```text
The route completed, but the gate refused promotion because the raw TRELLIS ->
UltraShape reference remained fragmented. This setup proves dependency/runtime
correctness, not final asset quality.
```

## FACE Thunder Runtime Fixes (2026-05-07)

Scripts/modules:

```text
clearmesh/mesh_heads/face_paper.py
scripts/research/train_face_paper_faithful.py
scripts/thunder/launch_face_paper_existing_split_on_instance.sh
scripts/thunder/face_paper_existing_split_gate.sh
scripts/thunder/face_paper_train_eval_job.sh
```

Important fixes:

```text
Thunder A6000 shells may not expose /dev/nvidia*, even when nvidia-smi and
Torch CUDA work. The FACE launch preflight should treat nvidia-smi + torch.cuda
as the functional GPU source of truth instead of failing on missing device-node
listing.
```

```text
The paper-faithful Shape2VecSet attention path now uses PyTorch scaled dot
product attention when available. This preserves the same attention equation but
allows fused CUDA kernels instead of manual einsum/softmax/einsum.
```

```text
No-augmentation FACE runs can cache deterministic FPS query indices with
CACHE_FPS_INDICES=1 / --cache-fps-indices. This does not change the encoder
queries for fixed point clouds; it avoids recomputing the same 2048-step FPS
selection on every forward pass.
```

Measured on Thunder A6000 instance `2nhex24x`, strict 64-sample FACE closure
split, 128 bins, 8192 points, 2048 VecSet tokens, latent 64, hidden 384,
4 encoder layers, 8 decoder layers, Muon, BF16, no online augmentation:

```text
baseline batch=1, no FPS cache:
- selection eval at step 1: ~39.2 sec
- training throughput after warmup: ~0.88 steps/sec
- 30k-step ETA: ~9-11 hours

optimized batch=4, FPS cache, SDPA:
- one-time FPS cache build: ~39.6 sec for 56 samples
- selection eval at step 1: ~0.84 sec
- training throughput after warmup: ~2.4 steps/sec
- 30k-step ETA: ~3.5 hours on A6000
```

Use this profile for bounded no-augmentation closure diagnostics. Do not use
FPS cache with online rotation/flip/per-axis scaling augmentation; augmented
point geometry can alter FPS query selection.
