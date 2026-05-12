# Thunder/GPU Production Runbook

This runbook is for the first real GPU pass of the new ClearMesh pipeline:

```text
TRELLIS.2 proxy -> point cloud bridge -> MeshRipple bake-off -> eval harness -> export package
```

Easy3E remains part of the product path as an optional editing mode before final export. Autorigging remains optional after repair/export.

## Local status

The current repo has the product/API/job scaffolding, mesh-head adapter layer, and Thunder helper scripts. This laptop has the Thunder CLI available at `/Users/Ashar/.tnr/bin/tnr`; helper scripts default to Thunder instance `0` unless `THUNDER_INSTANCE_ID` is set.

For a clean production instance setup checklist, see
`docs/production_gpu_dependency_fixes.md`.

Do not print API tokens in logs. Pass tokens through environment variables or the provider secret store.

## LATTICE/FACE Research Smokes

These commands validate the new learned-topology research path on Thunder:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/lattice_vdf_tiny_smoke.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/face_conditioned_tiny_smoke.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/face_conditioned_overfit.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/face_level_conditioned_overfit.sh
```

The tiny FACE smoke is only a path check and is not expected to generalize at
80 steps. The overfit command is the stronger circuit test: it should emit a
watertight generated cube with 12 faces and 8 vertices.

The LATTICE VDF smoke proves the deterministic face-VDF representation and
export path. It should emit watertight predicted and target GLBs with 12 faces
and 8 vertices.

The FACE-level overfit is closer to the paper than the coordinate-token
decoder: it performs 12 autoregressive steps for the box instead of 108
coordinate-token steps, with the 9 coordinates inside each triangle predicted
in parallel.

New FACE-level checkpoints also train a small face-count head. If
`sample_face_level_conditioned_tiny.py` or `sample_face_level_from_mesh.py` is
called without `--face-count`, the sampler uses the predicted count when the
checkpoint advertises `has_count_head=true`; otherwise it falls back to
`max_faces` for backward compatibility.

Paper-faithful FACE ARAE gates use `scripts/thunder/face_paper_faithful_smoke.sh`.
For real mesh folders, upload with the clean archive helper first; it disables
macOS resource forks and extended attributes so Thunder does not see broken GLB
headers:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/upload_mesh_dir.sh 0 /path/to/local_meshes /tmp/clearmesh_input_meshes
```

Useful FACE ARAE run modes:

```bash
# Full teacher-forced export/eval. Fast, verifies reconstruction learning.
RUN_DIR=/tmp/clearmesh_face_paper_real_gate \
MESH_DIR=/tmp/clearmesh_input_meshes \
SYNTHETIC_COUNT=0 \
MAX_FACES=2048 \
POINT_SAMPLES=2048 \
TRAIN_POINT_SAMPLES=2048 \
STEPS=2500 \
BATCH_SIZE=1 \
HIDDEN_SIZE=192 \
ENCODER_HIDDEN_SIZE=192 \
ENCODER_LAYERS=2 \
DECODER_LAYERS=3 \
HEADS=4 \
VECSET_TOKENS=128 \
DISABLE_AUGMENT=1 \
GENERATION_MODE=teacher_forced \
DATASET_GATE_PROFILE=strict \
DATASET_GATE_TOKEN_FAMILY=paper \
SELECTION_EVAL_EVERY=200 \
SELECTION_EVAL_BATCH_SIZE=1 \
LOG_EVERY=100 \
scripts/thunder/face_paper_faithful_smoke.sh 0

# Capped free-running AR probe. Slow without cache, but catches rollout drift.
RUN_DIR=/tmp/clearmesh_face_paper_real_gate \
MESH_DIR=/tmp/clearmesh_input_meshes \
SYNTHETIC_COUNT=0 \
MAX_FACES=2048 \
GENERATION_MODE=autoregressive \
GENERATION_FACE_LIMIT=128 \
EVAL_LIMIT=2 \
scripts/thunder/face_paper_faithful_smoke.sh 0
```

Verified 2026-05-03 on one RTX A6000:

```text
real seven-mesh no-augmentation overfit: best_loss=0.000131 at step 2496
teacher-forced full export: mean_teacher_forced_accuracy=1.0
capped AR-128 probe, original no-cache loop: ~16-17 seconds/sample for 128 faces
capped AR-128 probe, incremental + one-pass greedy coordinate decode: 6.9 seconds total for two samples
strict voxel-shell paper overfit v2: dataset gate 7/7 passing; selected checkpoint step 2600 by full-dataset loss 0.0001087; teacher-forced export 7/7 watertight, boundary edges 0, edge pairing 1.0
strict voxel-shell paper full-count AR: 7/7 watertight on training-set overfit; total eval 125s; 3016-face mesh 46s
strict voxel-shell paper holdout v1: train 5/5 watertight, held-out 0/2 watertight; mean held-out boundary edges 41
strict mixed15 aug60 grouped holdout v1: grouped split 48 train / 12 held-out shards, strict gate 60/60 passing; 3200-step 192-hidden run undertrained with train AR sample 1/12 watertight and held-out 3/12 watertight, best selection loss 1.2087 and still falling
```

Overnight ladder checkpoint, 2026-05-04:

```text
A6000 ladder lab: /tmp/clearmesh_face_ladder_20260504_041315
local artifact cache: .codex_outputs/face_ladder_30k_bin512
long run: bin512_legacy_causal_30000
settings: 512 bins, 2048 point samples, 256 VecSet tokens, latent 64, AdamW, legacy CausalMLP, no online augmentation
train teacher-forced: accuracy 0.9985, mean boundary edges 4.74, watertight 3/110
train autoregressive: accuracy 0.9988 under teacher-forced scoring, mean boundary edges 170.8, watertight 0/5
test teacher-forced: accuracy 0.0402, mean boundary edges 981.0, watertight 0/27
test autoregressive: accuracy 0.0252, mean boundary edges 148.6, watertight 0/5
verdict: do not scale this run family. It memorizes teacher-forced reconstruction but free-running AR drifts into fragmented geometry.
next gate: A100 paper-knob probe with 128 bins, 8192 points, 2048 VecSet tokens, latent 64, Muon, bf16, online augmentation, and legacy_concat CausalMLP, which is the closest cited/public-code match to FACE's under-specified CausalMLP.
```

Do not use raw arbitrary decimated meshes as long-run targets. Run cleanup /
manifoldization first; the paper tokenizer now drops duplicate and colinear
quantized faces, which is correct but exposes holes in dirty sources.

For FACE paper targets, use the strict voxel-shell path before dataset building.
The current safe pilot recipe is to control density with voxel resolution and
disable simplification, because post-voxel quadric simplification can introduce
non-manifold edges:

```bash
python scripts/research/prepare_face_strict_targets.py \
  --input-dir /tmp/raw_proxy_meshes \
  --output-dir /tmp/face_strict_targets \
  --engine voxel_shell \
  --target-faces 0 \
  --sample-points 30000 \
  --voxel-resolution 16 \
  --voxel-dilate 2 \
  --voxel-close 1 \
  --fallback ''

python scripts/research/build_face_token_dataset.py \
  --mesh-dir /tmp/face_strict_targets/meshes \
  --output-dir /tmp/face_strict_targets/dataset \
  --max-faces 4096 \
  --point-samples 2048

python scripts/research/check_face_dataset_targets.py \
  --manifest /tmp/face_strict_targets/dataset/manifest.jsonl \
  --output /tmp/face_strict_targets/paper_dataset_gate.json \
  --profile strict \
  --token-family paper \
  --fail-on-violations
```

Local pilot result on seven real proxy meshes: 7/7 strict targets accepted,
all watertight, one component, zero non-manifold edges; paper-token dataset
gate passed 7/7.

The current strongest paper-faithful FACE proof uses that strict voxel-shell
pilot set with full-dataset checkpoint selection:

```bash
RUN_DIR=/tmp/clearmesh_face_paper_strict_voxel16_overfit_v2 \
DOWNLOAD_DIR=artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_overfit_v2 \
MESH_DIR=/tmp/clearmesh_face_strict_targets_real_voxel16_meshes \
SYNTHETIC_COUNT=0 \
MAX_FACES=4096 \
POINT_SAMPLES=2048 \
TRAIN_POINT_SAMPLES=2048 \
STEPS=3200 \
BATCH_SIZE=1 \
HIDDEN_SIZE=192 \
ENCODER_HIDDEN_SIZE=192 \
ENCODER_LAYERS=2 \
DECODER_LAYERS=3 \
HEADS=4 \
VECSET_TOKENS=128 \
LATENT_DIM=64 \
LR=6e-4 \
WEIGHT_DECAY=0.1 \
DISABLE_AUGMENT=1 \
DATASET_GATE_PROFILE=strict \
DATASET_GATE_TOKEN_FAMILY=paper \
SELECTION_EVAL_EVERY=200 \
SELECTION_EVAL_BATCH_SIZE=1 \
GENERATION_MODE=teacher_forced \
EVAL_LIMIT=7 \
SEED=61 \
scripts/thunder/face_paper_faithful_smoke.sh 0
```

Observed result:

```text
dataset gate: 7/7 passing
best checkpoint: step 2600, full-dataset loss 0.0001086935
teacher-forced export: 7/7 watertight, mean_boundary_edges=0, mean_edge_pairing_ratio=1.0
mean_teacher_forced_accuracy=0.9999999915
```

Full-count autoregressive follow-up from the same checkpoint:

```bash
python scripts/research/eval_face_paper_faithful.py \
  --checkpoint /tmp/clearmesh_face_paper_strict_voxel16_overfit_v2/face_paper_faithful.pt \
  --dataset-dir /tmp/clearmesh_face_paper_strict_voxel16_overfit_v2/dataset \
  --output /tmp/clearmesh_face_paper_strict_voxel16_overfit_v2_ar_full/eval_report_ar_full.json \
  --export-dir /tmp/clearmesh_face_paper_strict_voxel16_overfit_v2_ar_full/meshes \
  --limit 7 \
  --point-samples 2048 \
  --face-count-mode gt \
  --generation-mode autoregressive \
  --generation-face-limit 0 \
  --pair-samples 500 \
  --log-every 1
```

Observed result:

```text
full-count AR: 7/7 watertight, mean_boundary_edges=0, mean_edge_pairing_ratio=1.0
runtime: 125s total for 7 meshes on A6000; 3016-face mesh took 46s
```

Held-out split wrapper:

```bash
RUN_DIR=/tmp/clearmesh_face_paper_strict_voxel16_holdout_v1 \
DOWNLOAD_DIR=artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_holdout_v1 \
MESH_DIR=/tmp/clearmesh_face_strict_targets_real_voxel16_meshes \
SYNTHETIC_COUNT=0 \
MAX_FACES=4096 \
MODEL_MAX_FACES=4096 \
POINT_SAMPLES=2048 \
TRAIN_POINT_SAMPLES=2048 \
TEST_COUNT=2 \
SPLIT_SEED=73 \
STEPS=2600 \
BATCH_SIZE=1 \
HIDDEN_SIZE=192 \
ENCODER_HIDDEN_SIZE=192 \
ENCODER_LAYERS=2 \
DECODER_LAYERS=3 \
HEADS=4 \
VECSET_TOKENS=128 \
LATENT_DIM=64 \
DISABLE_AUGMENT=1 \
DATASET_GATE_PROFILE=strict \
DATASET_GATE_TOKEN_FAMILY=paper \
SELECTION_EVAL_EVERY=200 \
TRAIN_EVAL_MODE=autoregressive \
TEST_EVAL_MODE=autoregressive \
scripts/thunder/face_paper_holdout_smoke.sh 0
```

For detached long runs, prefer launching the remote job helper on Thunder rather
than keeping a local SSH process alive:

```bash
scripts/thunder/run_remote.sh 0 <<'REMOTE'
set -euo pipefail
cd /home/ubuntu/clearmesh
. /home/ubuntu/clearmesh-venv/bin/activate
RUN_NAME=clearmesh_face_paper_strict_mixed15_aug60_holdout_v2
LOG=/tmp/$RUN_NAME.log
PID=/tmp/$RUN_NAME.pid
ARCHIVE=/tmp/$RUN_NAME.tar.gz
nohup env \
  RUN_DIR=/tmp/$RUN_NAME \
  MESH_DIR=/tmp/clearmesh_face_strict_mixed15_aug60_meshes \
  SYNTHETIC_COUNT=0 \
  MAX_FACES=4096 \
  MODEL_MAX_FACES=4096 \
  POINT_SAMPLES=2048 \
  TRAIN_POINT_SAMPLES=2048 \
  TEST_COUNT=3 \
  SPLIT_SEED=101 \
  SPLIT_SHUFFLE=1 \
  SPLIT_GROUP_FIELD=source_name \
  SPLIT_GROUP_REGEX='^(?P<group>.*)_(?:base|aug[0-9]+)$' \
  STEPS=10000 \
  BATCH_SIZE=1 \
  HIDDEN_SIZE=192 \
  ENCODER_HIDDEN_SIZE=192 \
  ENCODER_LAYERS=2 \
  DECODER_LAYERS=3 \
  HEADS=4 \
  VECSET_TOKENS=128 \
  LATENT_DIM=64 \
  DISABLE_AUGMENT=1 \
  DATASET_GATE_PROFILE=strict \
  DATASET_GATE_TOKEN_FAMILY=paper \
  TRAIN_EVAL_MODE=autoregressive \
  TEST_EVAL_MODE=autoregressive \
  TRAIN_EVAL_LIMIT=12 \
  TEST_EVAL_LIMIT=0 \
  ARCHIVE_PATH=$ARCHIVE \
  bash scripts/thunder/face_paper_holdout_remote_job.sh > "$LOG" 2>&1 &
echo $! > "$PID"
REMOTE
```

Poll it with:

```bash
scripts/thunder/run_remote.sh 0 'tail -n 80 /tmp/clearmesh_face_paper_strict_mixed15_aug60_holdout_v2.log'
```

After the remote job prints and creates the archive, fetch it with:

```bash
scripts/thunder/fetch_remote_artifact.sh \
  0 \
  /tmp/clearmesh_face_paper_strict_mixed15_aug60_holdout_v2.tar.gz \
  artifacts/research_proofs/2026-05-03/face_paper_strict_mixed15_aug60_holdout_v2
```

Observed result:

```text
train full-count AR: 5/5 watertight, mean_boundary_edges=0
held-out full-count AR: 0/2 watertight, mean_boundary_edges=41, mean_teacher_forced_accuracy=0.038
```

Interpretation: this is a negative gate for tiny real-data generalization, not
a code failure. The next honest step is a 16-64 cleaned target split, then
1K-5K cleaned targets if the held-out topology curve improves.

Grouped 60-shard strict holdout recipe:

```bash
RUN_DIR=/tmp/clearmesh_face_paper_strict_mixed15_aug60_holdout_v1 \
DOWNLOAD_DIR=artifacts/research_proofs/2026-05-03/face_paper_strict_mixed15_aug60_holdout_v1 \
MESH_DIR=/tmp/clearmesh_face_strict_mixed15_aug60_meshes \
SYNTHETIC_COUNT=0 \
MAX_FACES=4096 \
MODEL_MAX_FACES=4096 \
POINT_SAMPLES=2048 \
TRAIN_POINT_SAMPLES=2048 \
TEST_COUNT=3 \
SPLIT_SEED=101 \
SPLIT_SHUFFLE=1 \
SPLIT_GROUP_FIELD=source_name \
SPLIT_GROUP_REGEX='^(?P<group>.*)_(?:base|aug[0-9]+)$' \
STEPS=3200 \
BATCH_SIZE=1 \
HIDDEN_SIZE=192 \
ENCODER_HIDDEN_SIZE=192 \
ENCODER_LAYERS=2 \
DECODER_LAYERS=3 \
HEADS=4 \
VECSET_TOKENS=128 \
LATENT_DIM=64 \
LR=6e-4 \
WEIGHT_DECAY=0.1 \
DISABLE_AUGMENT=1 \
DATASET_GATE_PROFILE=strict \
DATASET_GATE_TOKEN_FAMILY=paper \
FAIL_ON_DATASET_GATE=1 \
SELECTION_EVAL_EVERY=400 \
SELECTION_EVAL_BATCH_SIZE=1 \
TRAIN_EVAL_MODE=autoregressive \
TEST_EVAL_MODE=autoregressive \
TRAIN_EVAL_LIMIT=12 \
TEST_EVAL_LIMIT=0 \
GENERATION_FACE_LIMIT=0 \
PAIR_SAMPLES=500 \
LOG_EVERY=400 \
EVAL_LOG_EVERY=1 \
SEED=101 \
scripts/thunder/face_paper_holdout_smoke.sh 0
```

Observed result:

```text
dataset gate: 60/60 passing
split: 15 groups -> 48 train shards / 12 held-out shards
best checkpoint: step 3200, selection loss 1.2087 and still improving
train AR sample: 1/12 watertight, mean boundary edges 662.5
held-out AR: 3/12 watertight, mean boundary edges 982.2
runtime: ~58-61 generated faces/sec on one A6000
```

Interpretation: this run validates the grouped harness, not the model quality.
The model is undertrained because train AR has not reached the previous strict
overfit regime. Repeat the same grouped split at 10k-12k steps before deciding
whether architecture or data scale is the bottleneck.

To test the smallest reference-to-topology handoff after both overfit smokes:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/run_remote.sh 0 '
cd /home/ubuntu/clearmesh &&
. /home/ubuntu/clearmesh-venv/bin/activate &&
python scripts/research/sample_face_level_from_mesh.py \
  --checkpoint /tmp/clearmesh_face_level_conditioned_overfit/face_level_conditioned_overfit.pt \
  --mesh /tmp/clearmesh_lattice_vdf_smoke/predicted.glb \
  --output /tmp/clearmesh_face_level_from_lattice/generated_from_lattice.glb \
  --point-samples 256
'
```

Verified result: watertight generated GLB with 12 faces and 8 vertices.

Product-facing FACE-level adapter smoke:

```bash
cat > /tmp/clearmesh_face_level_adapter_config.json <<'JSON'
{
  "repo_dir": "/home/ubuntu/clearmesh",
  "python": "/home/ubuntu/clearmesh-venv/bin/python",
  "checkpoint": "/tmp/clearmesh_face_level_count_overfit/face_level_conditioned_overfit.pt",
  "point_samples": 256
}
JSON

python scripts/product/run_mesh_head.py \
  --head face-level \
  --case-id adapter_smoke \
  --point-cloud /tmp/clearmesh_lattice_vdf_smoke/predicted.glb \
  --proxy-mesh /tmp/clearmesh_lattice_vdf_smoke/predicted.glb \
  --output-dir /tmp/clearmesh_face_level_adapter_smoke \
  --config-json /tmp/clearmesh_face_level_adapter_config.json
```

Verified result: `adapter_smoke_face_level.glb` is watertight with 12 faces,
8 vertices, one component, zero boundary edges, and zero non-manifold edges.

For the current multi-shape FACE-level stress tests:

```bash
# Easy variable-extents box curriculum.
THUNDER_INSTANCE_ID=0 scripts/thunder/face_level_conditioned_smoke.sh

# Harder mixed primitive curriculum.
RUN_DIR=/tmp/clearmesh_face_level_mixed_smoke \
SYNTHETIC_KIND=mixed_cycle \
SYNTHETIC_COUNT=12 \
MAX_FACES=256 \
EVAL_LIMIT=10 \
STEPS=1200 \
THUNDER_INSTANCE_ID=0 \
scripts/thunder/face_level_conditioned_smoke.sh
```

Verified 2026-05-03:

```text
16-box smoke: watertight_rate=1.0, mean_chamfer_l2=0.00502, mean_normal_consistency=0.92575
mixed primitive smoke: watertight_rate=1.0, mean_chamfer_l2=0.00394, mean_normal_consistency=0.95470
```

## GPU host layout

Recommended generic GPU layout:

```text
/workspace/clearmesh
/workspace/mesh-heads/MeshRipple
/workspace/mesh-heads/Mesh-Silksong
/workspace/mesh-heads/DeepMesh
/workspace/mesh-heads/TreeMeshGPT
/workspace/mesh-heads/MeshMosaic
/workspace/mesh-heads/FastMesh
/workspace/artifacts
/workspace/state
```

## Bootstrap

```bash
cd /workspace
if [ ! -d clearmesh ]; then
  git clone <your-clearmesh-repo-url> clearmesh
fi
cd /workspace/clearmesh
python -m pip install -r requirements-product.txt
MESH_HEAD_ROOT=/workspace/mesh-heads INSTALL_ENV=1 bash scripts/setup/install_mesh_heads.sh
```

MeshRipple checkpoints are not committed to this repo. Download them following the public MeshRipple README and place them in:

```text
/workspace/mesh-heads/MeshRipple/ckpt
```

On Thunder, use the reproducible helper path:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/sync_repo.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/bootstrap_remote.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/install_meshripple_env.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/download_meshripple_checkpoints.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/meshripple_preflight.sh
```

## Optional Quad Baselines

Install the public pyinstantmeshes/Instant Meshes Python baseline into the
Thunder MeshRipple venv:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/install_quad_baselines.sh
# optional QuadriFlow build:
INSTALL_QUADRIFLOW=1 THUNDER_INSTANCE_ID=0 scripts/thunder/install_quad_baselines.sh
```

Run the current quad sidecar against the fixed Poisson control surface:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/run_remote.sh 0 "
source /home/ubuntu/meshripple-venv/bin/activate
cd /home/ubuntu/clearmesh
python scripts/product/analyze_retopology.py \
  --input /tmp/clearmesh_surface_normalization_fixed/control_poisson.obj \
  --report /tmp/clearmesh_quad_pyinstant/retopology_plan.json \
  --target-quads 5000
python scripts/product/quad_remesh.py \
  --input /tmp/clearmesh_surface_normalization_fixed/control_poisson.obj \
  --output /tmp/clearmesh_quad_pyinstant/quad.obj \
  --report /tmp/clearmesh_quad_pyinstant/report.json \
  --engine pyinstantmeshes \
  --target-faces 5000
python scripts/product/remesh_charts.py \
  --input /tmp/clearmesh_surface_normalization_fixed/control_poisson.obj \
  --plan /tmp/clearmesh_quad_pyinstant/retopology_plan.json \
  --output-dir /tmp/clearmesh_quad_pyinstant/charts \
  --manifest /tmp/clearmesh_quad_pyinstant/chart_manifest.json \
  --engine pyinstantmeshes \
  --max-charts 8
python scripts/product/stitch_chart_remesh.py \
  --manifest /tmp/clearmesh_quad_pyinstant/chart_manifest.json \
  --output /tmp/clearmesh_quad_pyinstant/stitched_chart_quads.obj \
  --report /tmp/clearmesh_quad_pyinstant/chart_stitch.json
python scripts/product/compare_chart_remesh_engines.py \
  --input /tmp/clearmesh_surface_normalization_fixed/control_poisson.obj \
  --plan /tmp/clearmesh_quad_pyinstant/retopology_plan.json \
  --output-dir /tmp/clearmesh_quad_engine_compare \
  --report /tmp/clearmesh_quad_engine_compare/report.json \
  --engines pyinstantmeshes,quadriflow_cli,template_cage
"
```

## Reference Refinement / Manifoldization

UltraShape can be invoked through the worker via `metadata.reference_refinement_command`:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/install_ultrashape.sh

/home/ubuntu/ultrashape-venv/bin/python -u scripts/product/run_ultrashape_refinement.py \
  --mesh /tmp/trellis_proxy.glb \
  --image /tmp/input.png \
  --output-dir /tmp/clearmesh_reference \
  --output-name ultrashape_reference.glb \
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

The product worker should call UltraShape as a command hook, not by importing
TRELLIS.2 and UltraShape into the same Python process. The shared
`UltraShapeRefiner` also defaults to subprocess isolation for Python callers
because TRELLIS.2 and UltraShape can both register `cuBVH`.

On Thunder, UltraShape uses `/home/ubuntu/ultrashape-venv` with Python 3.10.
Do not install it into the lightweight `clearmesh-venv`: upstream pins packages
such as `numpy==1.24.4`, which do not build correctly on Python 3.12.

Before enabling UltraShape in a commercial hosted product, do a license review
of the upstream UltraShape/Hunyuan terms. Keep `REFERENCE_MODE=poisson` or
`REFERENCE_MODE=manifoldplus` available as deterministic fallback profiles.

ManifoldPlus can be used as a deterministic watertight reference fallback:

```bash
scripts/setup/install_manifoldplus.sh
python scripts/product/run_manifoldplus.py \
  --input /tmp/trellis_proxy.obj \
  --output-dir /tmp/clearmesh_reference \
  --binary /tmp/clearmesh-manifold-tools/ManifoldPlus/build/manifold \
  --depth 8
```

Compare available reference builders:

```bash
python scripts/product/compare_reference_surfaces.py \
  --input /tmp/trellis_proxy.obj \
  --image /tmp/input.png \
  --output-dir /tmp/clearmesh_reference_compare \
  --report /tmp/clearmesh_reference_compare/report.json \
  --manifoldplus-binary /tmp/clearmesh-manifold-tools/ManifoldPlus/build/manifold \
  --ultrashape-dir /workspace/UltraShape-1.0 \
  --ultrashape-checkpoint /workspace/checkpoints/ultrashape_v1.pt \
  --ultrashape-remove-bg
```

Full production route smoke:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/production_path_smoke.sh
REFERENCE_MODE=poisson THUNDER_INSTANCE_ID=0 scripts/thunder/production_path_smoke.sh
```

Verified 2026-05-03 on Thunder instance `0` with `REFERENCE_MODE=ultrashape`:

```text
job_5fe8693c8fd14558a8d06d7bfde95ff3: succeeded
TRELLIS.2 -> UltraShape paper settings -> control surface -> chart/quad sidecars -> cleanup -> gate/export
```

Important caveat: this validates the production route and paper-setting handoff,
not final quality. The gate reported `preview_or_repair_required` because the
UltraShape reference still had hundreds of connected/tiny components when
conditioned on the raw TRELLIS output. The watertight pure-quad output in this
run was a sidecar/control cage, not the promoted final artist mesh.

## TRELLIS.2 GPU Runtime

TRELLIS.2 has a few sharp dependency edges on Thunder-style images:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/sync_repo.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/install_trellis2_env.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/trellis2_preflight.sh
```

Notes from the verified Thunder path:

- Thunder exposed CUDA 13.0 at `/usr/local/cuda`, while TRELLIS.2 pins PyTorch `2.6.0+cu124`. The installer now installs/uses CUDA Toolkit 12.4 at `/usr/local/cuda-12.4` for native extension builds.
- Flash-attn works when installed as `flash-attn==2.7.3 --no-build-isolation` after PyTorch is already installed, with `psutil` available in the build env.
- TRELLIS.2's extension setup stages repos in `/tmp/extensions`; the installer clears that temp directory before extension install so interrupted builds do not poison retries.
- TRELLIS.2 currently needs `transformers==4.57.5`; newer Transformers versions can raise `AttributeError: 'DINOv3ViTModel' object has no attribute 'layer'`.
- Image-to-3D inference needs Hugging Face access to the gated `facebook/dinov3-vitl16-pretrain-lvd1689m` model. Put an authorized `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` in the remote environment without printing it.

For a direct TRELLIS.2 proxy smoke:

```bash
cd /home/ubuntu/TRELLIS.2
source /home/ubuntu/trellis2-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export ATTN_BACKEND=flash_attn
python /home/ubuntu/clearmesh/scripts/product/run_trellis2_proxy.py \
  --input /home/ubuntu/TRELLIS.2/assets/example_image/0e4984a9b3765ce80e9853443f9319ecedf90885c74b56cccfebc09402740f8a.webp \
  --output-dir /tmp/clearmesh_trellis2_smoke \
  --output-name trellis_proxy.glb \
  --decimation-target 250000 \
  --texture-size 1024
```

Or use the repeatable helpers:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/sync_hf_token.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/trellis2_smoke.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/pipeline_trellis2_meshripple_smoke.sh
```

## Smoke-test product state

```bash
scripts/product/pipeline_scaffold_smoke.sh
```

Or manually:

```bash
python scripts/product/create_local_job.py \
  --state-root /workspace/state \
  --team-id team_dev \
  --user-id user_dev \
  --input-uri local:///workspace/inputs/example.png \
  --grant-credits 20 \
  --enable-rigging

python scripts/product/run_local_worker.py \
  --state-root /workspace/state \
  --artifact-root /workspace/artifacts \
  --once
```

This validates job state, credits, artifacts, and optional autorigging step bookkeeping. It does not run TRELLIS.2 or MeshRipple.

## MeshRipple adapter smoke

After generating or copying a point cloud:

```bash
python scripts/product/run_mesh_head.py \
  --head meshripple \
  --case-id smoke_001 \
  --point-cloud /workspace/pointclouds/smoke_40960.ply \
  --proxy-mesh /workspace/proxies/smoke_proxy.glb \
  --output-dir /workspace/artifacts/smoke_001/meshripple \
  --config-json /workspace/clearmesh/configs/meshripple.gpu.example.json
```

MeshRipple's public inference path consumes meshes from its generated `eval_dataset_path`, so the adapter requires `--proxy-mesh` even though ClearMesh also tracks the point cloud bridge.

For a fast Thunder integration smoke:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/sync_repo.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/meshripple_smoke.sh
```

The smoke config at `configs/meshripple.thunder.smoke.json` is intentionally tiny. It proves adapter execution and eval harness compatibility, not production mesh quality.

## Bake-off loop

1. Generate TRELLIS.2 proxy meshes for the fixed benchmark set.
2. Sample `16k`, `40k`, and `100k` point clouds:

```bash
python scripts/data/sample_point_clouds.py \
  --manifest manifests/pointcloud_bridge.gpu.csv \
  --output-dir artifacts/pointclouds \
  --output-manifest artifacts/pointcloud_manifest.csv \
  --budgets 16384 40960 100000
```

3. Run MeshRipple first, then Mesh Silksong, DeepMesh, TreeMeshGPT, and MeshMosaic as adapters are validated.
4. Evaluate every output:

```bash
python scripts/eval/evaluate_meshes.py \
  --manifest manifests/mesh_bakeoff.gpu.csv \
  --output artifacts/eval_report.json
```

## Production promotion gate

Do not promote a mesh head until it clears these thresholds on the fixed set:

- no worker crashes across the benchmark set
- low boundary-loop and non-manifold counts versus TRELLIS.2 proxy
- lower tiny-component count than the raw proxy on topology stress cases
- Blender import/export roundtrip succeeds
- artist-edit review passes for at least 10 representative assets

## API deployment shape

- `clearmesh.api.server` is the public control plane.
- GPU workers should poll job state, claim pending jobs, and emit artifacts.
- Billing starts with credit reservations, then migrates to Stripe checkout/subscriptions once model cost is measured.
- Auth starts with hashed API keys, then migrates to hosted identity once the app has real users.


## Thunder helper scripts

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/sync_repo.sh
THUNDER_INSTANCE_ID=0 INSTALL_MESH_HEAD_ENVS=0 scripts/thunder/bootstrap_remote.sh
scripts/thunder/run_remote.sh 0 "cd /home/ubuntu/clearmesh && . /home/ubuntu/clearmesh-venv/bin/activate && python scripts/product/run_pipeline_worker.py --once"
```

The remote virtualenv lives at `/home/ubuntu/clearmesh-venv` so repo syncs can replace `/home/ubuntu/clearmesh` without deleting installed runtime packages.

## Verified Thunder Smoke

The helper path has been validated on Thunder instance `0` with an RTX A6000:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/sync_repo.sh
THUNDER_INSTANCE_ID=0 INSTALL_MESH_HEAD_ENVS=0 scripts/thunder/bootstrap_remote.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/heavy_smoke.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/pipeline_meshripple_smoke.sh
```

The smoke uses the checked-in mug proxy as both `proxy_mesh_path` and `artist_mesh_path`. It verifies:

```text
job creation -> credit reservation -> proxy registration -> 40,960 point cloud sampling -> mesh registration -> mesh eval report -> export package -> credit consumption
```

`pipeline_meshripple_smoke.sh` runs the same product worker path with real MeshRipple inference through the adapter.

## Verified MeshRipple Smoke

MeshRipple has also been validated on Thunder instance `0` with an RTX A6000:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/install_meshripple_env.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/download_meshripple_checkpoints.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/meshripple_preflight.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/meshripple_smoke.sh
```

Verified artifacts:

```text
/tmp/clearmesh_meshripple_smoke_micro/mesh_head/meshripple/_val_generate_k20_p0.9_t0.9/test_mug_y.obj
/tmp/clearmesh_meshripple_smoke_micro/eval_report.json
```

The smoke result is intentionally tiny: `12` vertices and `10` faces, non-watertight, with one connected component. That is expected for the capped smoke config. Run `configs/meshripple.thunder.example.json` for a real quality pass.

## Verified TRELLIS.2 + MeshRipple Pipeline Smoke

The full product-worker pipeline has been validated on Thunder instance `0`:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/sync_hf_token.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/install_trellis2_env.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/trellis2_preflight.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/trellis2_smoke.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/pipeline_trellis2_meshripple_smoke.sh
```

Verified product path:

```text
uploaded/local image -> TRELLIS.2 proxy GLB -> 40,960 point cloud -> MeshRipple smoke adapter -> mesh eval report -> export package
```

The validated smoke job succeeded with these steps:

```text
input_validation: succeeded
trellis_proxy: succeeded
point_cloud_bridge: succeeded
part_structure: skipped
mesh_head: succeeded
repair_validation: succeeded
export_package: succeeded
```

The MeshRipple output in this smoke uses `configs/meshripple.thunder.smoke.json`, so it is intentionally capped to a tiny mesh. Use this smoke for pipeline correctness only; switch to `configs/meshripple.thunder.example.json` for quality benchmarking.

## Verified FACE-Level Holdout Smoke

FACE-lite now has a Thunder holdout wrapper for honest train/test validation:

```bash
THUNDER_INSTANCE_ID=0 scripts/thunder/sync_repo.sh
DOWNLOAD_DIR=/Users/Ashar/Documents/GitHub/clearmesh/artifacts/research_proofs/2026-05-03/face_level_holdout \
  scripts/thunder/face_level_conditioned_holdout.sh 0
```

Default settings:

```text
train_count: 48 synthetic mixed primitives, filtered to 44 shards
test_count: 24 synthetic mixed primitives, filtered to 23 shards
max_faces: 256
steps: 3000
model: hidden 192, layers 3, heads 6, condition tokens 8
```

The wrapper reports three views:

```text
gt_count: generated with teacher face count; isolates token quality
predicted_count: generated with model-predicted face count; production-like path
predicted_count_cleanup: predicted-count output after conservative cleanup/fill-holes
predicted_count_token_repair: predicted-count output after conservative token repair
predicted_count_token_repair_cleanup: token-repaired output after cleanup/fill-holes
```

Validated result on Thunder RTX A6000:

```text
gt_count watertight: 11/23, mean Chamfer L2 0.03617700868389408
predicted_count watertight: 14/23, mean Chamfer L2 0.03160356321306253
predicted_count_cleanup watertight: 18/23, mean Chamfer L2 0.03136991877360527
predicted_count_voxel_shell watertight: 23/23, mean Chamfer L2 0.03176261903287363, mean faces 28890.956521739132
```

Topology-weighted training can be enabled without changing the wrapper:

```bash
RUN_DIR=/tmp/clearmesh_face_level_topology_weight_holdout \
REUSE_VERTEX_LOSS_WEIGHT=0.5 \
EDGE_CLOSURE_LOSS_WEIGHT=1.0 \
DOWNLOAD_DIR=/Users/Ashar/Documents/GitHub/clearmesh/artifacts/research_proofs/2026-05-03/face_level_topology_weight \
  scripts/thunder/face_level_conditioned_holdout.sh 0
```

Validated topology-weighted result on the same split:

```text
gt_count watertight: 12/23, mean Chamfer L2 0.020536723682549104
predicted_count watertight: 15/23, mean Chamfer L2 0.02399260467306527
predicted_count_cleanup watertight: 19/23, mean Chamfer L2 0.02408016628884552
predicted_count_token_repair_cleanup watertight: 19/23, mean Chamfer L2 0.023771566799745503
mean token boundary edges after token repair: 2.217391304347826
```

The explicit topology auxiliary-head variant is available but did not win:

```bash
RUN_DIR=/tmp/clearmesh_face_level_topology_aux_holdout \
REUSE_VERTEX_LOSS_WEIGHT=0.5 \
EDGE_CLOSURE_LOSS_WEIGHT=1.0 \
TOPOLOGY_AUX_LOSS_WEIGHT=0.1 \
DOWNLOAD_DIR=/Users/Ashar/Documents/GitHub/clearmesh/artifacts/research_proofs/2026-05-03/face_level_topology_aux \
  scripts/thunder/face_level_conditioned_holdout.sh 0
```

Observed aux-head result:

```text
predicted_count watertight: 13/23, mean Chamfer L2 0.03279067156517142
predicted_count_cleanup watertight: 15/23, mean Chamfer L2 0.032685845149144535
```

Recommendation:

```text
Keep TOPOLOGY_AUX_LOSS_WEIGHT=0 by default.
Use REUSE_VERTEX_LOSS_WEIGHT=0.5 and EDGE_CLOSURE_LOSS_WEIGHT=1.0 for the current best tiny setting.
```

One rejected decoding trick:

```text
closure-extra decoding with 32 extra faces kept watertight at 14/23 and worsened Chamfer to 0.03399797346946442.
Do not use blind longer decoding as a production optimization.
```

Artifacts downloaded locally:

```text
artifacts/research_proofs/2026-05-03/face_level_holdout/holdout_summary.json
artifacts/research_proofs/2026-05-03/face_level_holdout/eval_gt_count_report.json
artifacts/research_proofs/2026-05-03/face_level_holdout/eval_predicted_count_report_v2.json
artifacts/research_proofs/2026-05-03/face_level_holdout/eval_predicted_count_voxel_shell_report.json
artifacts/research_proofs/2026-05-03/face_level_holdout/face_level_holdout_contact_sheet.svg
artifacts/research_proofs/2026-05-03/face_level_topology_weight/holdout_summary.json
artifacts/research_proofs/2026-05-03/face_level_topology_weight/face_level_topology_weight_contact_sheet.svg
artifacts/research_proofs/2026-05-03/face_level_topology_aux/holdout_summary.json
```

Real proxy fixture smoke:

```bash
python scripts/research/prepare_face_fixture_meshes.py \
  --input-list artifacts/research_proofs/2026-05-03/face_proxy_fixture_sources/candidates.txt \
  --output-dir artifacts/research_proofs/2026-05-03/face_proxy_fixtures_decimated \
  --decimate-to-faces 768 \
  --max-decimation-source-faces 500000 \
  --max-faces 1400

DOWNLOAD_DIR=/Users/Ashar/Documents/GitHub/clearmesh/artifacts/research_proofs/2026-05-03/face_level_proxy_fixture_smoke \
LOCAL_FIXTURE_DIR=/Users/Ashar/Documents/GitHub/clearmesh/artifacts/research_proofs/2026-05-03/face_proxy_fixtures_decimated \
RUN_DIR=/tmp/clearmesh_face_level_proxy_fixture_smoke \
FIXTURE_TEST_COUNT=3 \
SYNTHETIC_COUNT=24 \
MAX_FACES=1400 \
POINT_SAMPLES=512 \
TRAIN_POINT_SAMPLES=256 \
STEPS=600 \
BATCH_SIZE=2 \
HIDDEN_SIZE=128 \
LAYERS=2 \
HEADS=4 \
CONDITION_TOKENS=6 \
PAIR_SAMPLES=400 \
  scripts/thunder/face_level_proxy_fixture_smoke.sh 0
```

Observed result on three held-out MeshRipple-style fragmented proxy fixtures:

```text
gt_count watertight: 0/3, mean Chamfer L2 0.06385298777450342
predicted_count watertight: 0/3, mean Chamfer L2 0.06397313225285776
predicted_count_cleanup watertight: 0/3, mean Chamfer L2 0.06708656182925414
token_repair_cleanup watertight: 0/3, mean Chamfer L2 0.07711952171332169
decoded token edge-pairing after repair: 0.5095785440613027
```

Interpretation:

```text
Do not train FACE on raw fragmented generator outputs and expect watertight artist meshes.
The real-proxy fixture set is useful as a failure diagnostic, not as clean supervision.
FACE training targets must pass the strict dataset gate after UltraShape/manifoldization/remesh.
```

Strict target gate:

```bash
python scripts/research/check_face_dataset_targets.py \
  --manifest artifacts/research_proofs/2026-05-03/face_proxy_fixtures_decimated_face_tokens/manifest.jsonl \
  --output artifacts/research_proofs/2026-05-03/face_proxy_fixtures_decimated_face_tokens/strict_gate_report.json \
  --profile strict
```

The decimated real-proxy fixture tokens currently fail strict gate at 0/8 passing.

Promotion rule:

```text
Do not promote FACE-level as the sole production mesh head until held-out cleaned watertightness is near 100% on fixtures and real asset proxies.
Use it as a topology prior behind LATTICE/TRELLIS + UltraShape/manifoldization, cleanup, projection, and Blender gates.
Use voxel-shell recovery as a safety fallback only; it is watertight but not the desired editable final mesh.
```

## Verified FACE Strict-Target Smoke

Strict target preparation converts fragmented/proxy meshes into closed FACE
supervision candidates before tokenization:

```bash
python scripts/research/prepare_face_strict_targets.py \
  --input-dir artifacts/research_proofs/2026-05-03/face_proxy_fixtures_decimated/meshes \
  --output-dir artifacts/research_proofs/2026-05-03/face_strict_targets \
  --engine convex_hull \
  --target-faces 1400

python scripts/research/build_face_token_dataset.py \
  --mesh-dir artifacts/research_proofs/2026-05-03/face_strict_targets/meshes \
  --output-dir artifacts/research_proofs/2026-05-03/face_strict_targets/tokens \
  --max-faces 1400 \
  --point-samples 256 \
  --seed 23

python scripts/research/check_face_dataset_targets.py \
  --manifest artifacts/research_proofs/2026-05-03/face_strict_targets/tokens/manifest.jsonl \
  --output artifacts/research_proofs/2026-05-03/face_strict_targets/strict_gate_report.json \
  --profile strict
```

Observed local gate:

```text
accepted strict targets: 8/8
strict token gate: 8/8 passing
```

Thunder strict-target smoke:

```bash
DOWNLOAD_DIR=artifacts/research_proofs/2026-05-03/face_level_strict_target_smoke \
LOCAL_FIXTURE_DIR=artifacts/research_proofs/2026-05-03/face_strict_targets \
RUN_DIR=/tmp/clearmesh_face_level_strict_target_smoke \
FIXTURE_TEST_COUNT=3 \
SYNTHETIC_COUNT=24 \
MAX_FACES=1400 \
POINT_SAMPLES=512 \
TRAIN_POINT_SAMPLES=256 \
STEPS=800 \
BATCH_SIZE=2 \
HIDDEN_SIZE=128 \
LAYERS=2 \
HEADS=4 \
CONDITION_TOKENS=6 \
PAIR_SAMPLES=400 \
  scripts/thunder/face_level_proxy_fixture_smoke.sh 0
```

Observed Thunder result:

```text
train strict gate: 29/29 passing
test strict gate: 3/3 passing
best_loss: 0.1161209866
predicted_count watertight: 0/3
predicted_count_cleanup watertight: 1/3
predicted_count_token_repair_cleanup watertight: 1/3
```

This is a negative production gate for the current FACE-lite decoder. Keep it as
a research topology prior until the representation is upgraded to explicit
vertex-index reuse or another hard topology-closure mechanism.

## FACE-Lite v2 Indexed Smoke

Run the topology-indexed FACE-lite smoke on Thunder:

```bash
DOWNLOAD_DIR=artifacts/research_proofs/2026-05-03/face_indexed_v2_smoke \
STEPS=300 \
SYNTHETIC_COUNT=10 \
  scripts/thunder/face_indexed_v2_smoke.sh 0
```

Observed first result:

```text
best_loss: 0.3452322781
raw watertight: 2/6
cleanup watertight: 2/6
mean edge pairing ratio: 0.770321
```

The v2 representation fixes implicit vertex reuse, but the decoder still needs
edge-state constrained sampling before it can be considered a production mesh
head.

## FACE-Lite v2 Constrained Decode A/B

The indexed v2 evaluator now defaults to edge-constrained decoding:

```bash
python scripts/research/eval_face_indexed_conditioned_tiny.py \
  --checkpoint /tmp/clearmesh_face_indexed_v2_smoke/face_indexed_v2_tiny.pt \
  --dataset-dir /tmp/clearmesh_face_indexed_v2_smoke/dataset \
  --output /tmp/clearmesh_face_indexed_v2_smoke/best_constrained_eval/eval_report.json \
  --decode-mode edge_constrained \
  --constraint-top-k 24 \
  --closure-bonus 5.0 \
  --new-edge-penalty 0.1 \
  --require-boundary-closure-after 1
```

Observed A/B on the same checkpoint:

```text
unconstrained:    2/6 watertight, mean boundary edges 52.17
best constrained: 3/6 watertight, mean boundary edges 4.33
```

This is a real topology improvement, but still below production gate. Treat it
as proof that graph-aware decoding matters, then move the same edge-state logic
into training losses and/or a boundary-edge action decoder.

## FACE Paper Causal-Head Check

FACE's paper reports that CausalMLP inside each face is better than parallel coordinate decoding for its 9-coordinate face representation. We tested the analogous idea for our indexed topology representation:

```bash
CORNER_HEAD=causal \
DOWNLOAD_DIR=artifacts/research_proofs/2026-05-03/face_indexed_v2_corner_causal_smoke \
RUN_DIR=/tmp/clearmesh_face_indexed_v2_corner_causal_smoke \
STEPS=300 \
  scripts/thunder/face_indexed_v2_smoke.sh 0
```

Observed result:

```text
best_loss: 0.1332
raw watertight: 0/6
cleanup watertight: 1/6
mean boundary edges: 7.00
mean Chamfer L2: 0.05430
```

This underperforms the current parallel indexed head with edge-constrained decoding:

```text
parallel best constrained: 3/6 watertight, mean boundary edges 4.33
```

Keep `CORNER_HEAD=parallel` as the default for now. Use `CORNER_HEAD=causal` only for targeted experiments.

## FACE Objaverse++ 512-Bin Ladder

The production FACE corpus path now defaults to 512 coordinate bins because the same strict targets passed the token gate at 37/38 with 512 bins versus 27/38 with 128 bins.

Current background ladder:

```text
instance: 1
run_dir: /tmp/clearmesh_face_objpp200_512_20260504_020301
log: /tmp/clearmesh_face_objpp200_512_20260504_020301.nohup.log
```

Poll it with:

```bash
scripts/thunder/run_remote.sh 1 'RUN_DIR=$(cat /tmp/clearmesh_latest_face_objpp200_512_run.txt); tail -200 "$RUN_DIR.nohup.log"; [ -f "$RUN_DIR/pilot_summary.json" ] && cat "$RUN_DIR/pilot_summary.json"; [ -f "$RUN_DIR/summary.json" ] && cat "$RUN_DIR/summary.json"'
```

## FACE-Indexed Scale Gate

The current production candidate is the explicit vertex-table/indexed-face
route, not the original coordinate-only FACE token route. The key local proof is
the curated 8-mesh overfit:

```text
dataset: .codex_outputs/face_indexed_real50_boundary/dataset_4096_decoded_watertight
checkpoint: .codex_outputs/face_indexed_real50_boundary/local_real8_curated_teacher_2k/face_indexed.pt
free-run watertight: 8/8
mean normalized Chamfer: 0.000052
max normalized Chamfer: 0.000090
scale gate: promote to the next larger curated run
```

Use the machine gate rather than eyeballing screenshots:

```bash
python3 scripts/research/assess_face_indexed_scale_readiness.py \
  --curation-summary .codex_outputs/face_indexed_real50_boundary/dataset_4096_decoded_watertight/curation_summary.json \
  --teacher-eval .codex_outputs/face_indexed_real50_boundary/local_real8_curated_teacher_2k/eval_teacher_forced_normalized.json \
  --free-run-eval .codex_outputs/face_indexed_real50_boundary/local_real8_curated_teacher_2k/eval_free_run_eight_bonus1_normalized.json \
  --output .codex_outputs/face_indexed_real50_boundary/local_real8_curated_teacher_2k/scale_readiness_normalized.json
```

Next GPU ladder, once Thunder exposes a real GPU device:

```bash
RUN_STAMP=face_indexed_16_probe \
TRAIN_LIMIT=16 \
EVAL_LIMIT=16 \
STEPS=3000 \
BATCH_SIZE=2 \
HIDDEN_SIZE=128 \
LAYERS=3 \
HEADS=4 \
CONDITION_TOKENS=8 \
TRAIN_POINT_SAMPLES=1024 \
PAIR_SAMPLES=10000 \
GPU=a100 \
MODE=production \
CREATE_INSTANCE=1 \
RUN_SCALE_READINESS=1 \
FAIL_ON_SCALE_NOT_READY=0 \
scripts/thunder/launch_face_indexed_scale_ladder.sh
```

Autopilot wrapper for the full bounded ladder:

```bash
RUN_STAMP_ROOT=face_indexed_curated_ladder \
LADDER_LIMITS="16 32 47" \
STEPS_BY_LIMIT_JSON='{"16":3000,"32":5000,"47":7000}' \
GPU=a100 \
MODE=production \
CREATE_INSTANCE=1 \
WAIT_ATTEMPTS=360 \
WAIT_INTERVAL_SEC=10 \
WAIT_TIMEOUT_SEC=3600 \
scripts/thunder/launch_face_indexed_scale_ladder_supervisor.sh
```

The supervisor writes `supervisor_status.jsonl`, stops on the first rung whose
`scale_readiness.json` does not promote, and delegates every rung to the guarded
single-rung launcher so GPU preflight/deletion remains fail-closed.

Promotion order:

```text
8 curated meshes: pass locally
16 curated meshes: next
32 curated meshes: next
47 curated meshes: next
larger Objaverse++ curated corpus: only after the 47-mesh ladder is clean
```

Thunder caveat: multiple Thunder instances have shown an A100/A6000 banner
while missing `/dev/nvidia*` inside the container. The guarded launcher now
requires `CLEARMESH_GPU_PREFLIGHT_OK` after `nvidia-smi`; do not trust
`tnr connect` exit status alone.
