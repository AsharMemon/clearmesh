# FACE Paper Gap Closure Plan

This file tracks the eight active gaps from the hostile audit. The full FACE image-to-VecSet DiT is intentionally out of scope for now; ClearMesh uses TRELLIS/LATTICE/UltraShape as the geometry source and FACE as the learned artist-mesh/remeshing head.

## Current Decision

Use the paper-faithful ARAE path as the production topology learner:

```text
curated public mesh corpus
  -> UltraShape/manifoldized strict targets
  -> FACE paper-token dataset
  -> 3DShape2VecSet-style VecSet encoder
  -> causal face decoder + CausalMLP
  -> EOS/predicted-count generation
  -> strict mesh gates / Blender promotion
```

## Gap Closure Matrix

| Gap | Status | Code / plan |
| --- | --- | --- |
| 1. Exact 3DShape2VecSet internals | Addressed structurally | `clearmesh/mesh_heads/face_paper.py` now has `encoder_backend=shape2vecset`, following the public 3DShape2VecSet pattern: sinusoidal point embedding, FPS queries, pre-norm cross-attention, GEGLU FFN, latent self-attention/FFN blocks, bottleneck. We keep `native` for old checkpoints. |
| 2. Exact CausalMLP internals | Addressed by reasoned implementation | The paper does not publish CausalMLP code. `legacy_concat` is the closest public-code match to TreeMeshGPT: separate coordinate heads conditioned on concatenated embeddings of previous coordinate tokens. `paper_chain` remains available as an experimental lower-parameter variant. |
| 3. EOS / variable face count | Implemented | The paper-faithful model now has an optional EOS head. Training adds final-face BCE via `--eos-loss-weight`; eval supports `--face-count-mode predicted`, `--eos-threshold`, and `--min-generated-faces`. |
| 4. Paper-scale conditioning | Profiled | `configs/face_paper_profiles.json` defines smoke, A6000 ladder, and paper-scale profiles. Script defaults now use `8192` train point samples unless a smoke wrapper overrides them. |
| 5. Curated 130K mesh corpus | Tooling added | `scripts/data/build_face_training_corpus.py` ranks downloaded Objaverse/TRELLIS/Objaverse++ candidates using High/Superior quality annotations, semantic reject flags, geometry prefilters, and curation score. The output feeds UltraShape/strict FACE target generation. |
| 6. Paper augmentation | Default-on for paper path | Training supports SO3 rotation, flips, and independent axis scaling. Holdout wrappers now default to augmentation on; set `DISABLE_AUGMENT=1` only for tiny overfit debugging. |
| 7. FACE image DiT | Deferred | Not needed for the ClearMesh remeshing route right now. |
| 8. Exact preprocessing / filtering | Addressed as a stricter pipeline | Use Objaverse++ quality/trait filters, local mesh geometry filters, UltraShape/manifoldization, `prepare_face_strict_targets.py`, `build_face_token_dataset.py`, then `check_face_dataset_targets.py --profile strict --token-family paper`. |
| 9. Controlled ablations | Tooling added | `scripts/thunder/face_paper_ab_remote_job.sh` runs matched `causal` vs `parallel` head jobs and produces `ab_comparison.json`. Config also tracks native-vs-shape2vecset and GT-vs-predicted count comparisons. |

## Data Curation Recipe

Primary sources:

- Objaverse 1.0 / Objaverse-XL for scale.
- Objaverse++ High/Superior annotations for quality and trait filtering.
- TRELLIS-500K metadata/candidate pool when available.
- UltraShape-style watertight processing for strict training targets.

Recommended command ladder:

```bash
python scripts/data/download_objaverse.py \
  --limit 250000 \
  --output_dir /workspace/data/objaverse

python scripts/data/build_face_training_corpus.py \
  --candidates /workspace/data/objaverse/manifest.json \
  --objaversepp-annotations /workspace/data/objaversepp/annotations.json \
  --output /workspace/data/face_corpus/candidates_130k.jsonl \
  --rejects-output /workspace/data/face_corpus/rejects.json \
  --target 130000 \
  --min-quality 2

python scripts/research/prepare_face_strict_targets.py \
  --mesh-dir /workspace/data/face_corpus/source_meshes \
  --output-dir /workspace/data/face_corpus/strict_targets \
  --max-faces 4000

python scripts/research/build_face_token_dataset.py \
  --mesh-dir /workspace/data/face_corpus/strict_targets/meshes \
  --output-dir /workspace/data/face_corpus/tokens \
  --max-faces 4000 \
  --point-samples 8192

python scripts/research/check_face_dataset_targets.py \
  --manifest /workspace/data/face_corpus/tokens/manifest.jsonl \
  --output /workspace/data/face_corpus/strict_gate.json \
  --profile strict \
  --token-family paper \
  --fail-on-violations
```

Important: `build_face_training_corpus.py` selects candidates, not final training examples. The final count that matters is the number of meshes that pass strict target generation and the paper-token gate.

## Thunder Corpus Pilot

Fresh Thunder instances need the FACE data dependencies before Objaverse++ curation:

```bash
scripts/thunder/install_face_training_env.sh
```

The end-to-end pilot command is:

```bash
scripts/thunder/run_remote.sh 0 <<'EOF'
cd /home/ubuntu/clearmesh
RUN_DIR=/tmp/clearmesh_face_objpp50 \
SELECT_TARGET=50 \
SCAN_LIMIT=12000 \
OVERSAMPLE_FACTOR=4 \
MESH_TIMEOUT_SECONDS=30 \
TARGET_FACES=512 \
TOKEN_MAX_FACES=1024 \
NUM_BINS=128 \
POINT_SAMPLES=8192 \
FAIL_ON_GATE=0 \
bash scripts/thunder/face_objaversepp_corpus_pilot.sh
EOF
```

The pilot performs:

```text
Objaverse++ High/Superior selection
  -> Objaverse GLB download
  -> timeout-safe raw mesh curation
  -> voxel-shell strict watertight target creation
  -> FACE paper-token shards at 128 bins
  -> strict paper-token edge-graph gate
```

Do not use `NUM_BINS=16` for real meshes. It is only a smoke setting; it can quantize distinct vertices together and create artificial boundary edges. Real FACE token gates should use 128 bins unless a deliberate higher/lower-resolution experiment is being run.

The strict-target adapter now rejects outputs above `target_faces * max_target_face_ratio`, so a missing decimator cannot silently produce a high-face accepted target. The required decimator dependency is `fast-simplification`.

Raw GLB curation runs each mesh inspection in a child process with both a timeout and a memory cap (`MESH_TIMEOUT_SECONDS`, `MESH_MEMORY_LIMIT_GB`). Keep those enabled on public corpora; a small scanned GLB can still expand to tens of GB during scene/material processing.

## Thunder Paper-Ladder Run

A6000 iteration profile:

```bash
RUN_DIR=/tmp/clearmesh_face_paper_ladder_a6000 \
MESH_DIR=/workspace/data/face_corpus/strict_targets/meshes \
SYNTHETIC_COUNT=0 \
MAX_FACES=4000 \
MODEL_MAX_FACES=4000 \
POINT_SAMPLES=8192 \
TRAIN_POINT_SAMPLES=8192 \
HIDDEN_SIZE=384 \
ENCODER_HIDDEN_SIZE=384 \
ENCODER_LAYERS=4 \
DECODER_LAYERS=8 \
HEADS=8 \
VECSET_TOKENS=512 \
ENCODER_BACKEND=shape2vecset \
CAUSAL_MLP_VARIANT=legacy_concat \
DISABLE_AUGMENT=0 \
TRAIN_FACE_COUNT_MODE=predicted \
TEST_FACE_COUNT_MODE=predicted \
scripts/thunder/face_paper_holdout_remote_job.sh
```

Full paper-scale run should use `hidden=1024`, `encoder_hidden=768`, `encoder_layers=8`, `decoder_layers=24`, `vecset_tokens=2048`, `8192` points, `4000` face cap, Muon `lr=6e-4`, `weight_decay=0.1`, and roughly `100K` steps on multi-GPU hardware.

## Acceptance Gates

Do not promote a FACE checkpoint until all of these pass:

- Strict target gate pass rate at least 99% on curated training shards.
- Held-out teacher-forced coordinate accuracy trends high without AR collapse.
- Predicted-count AR outputs match reference face count distribution within tolerance.
- Watertightness and boundary-edge metrics improve over raw TRELLIS/LATTICE proxy.
- Blender import/export and subdivision smoke pass on promoted meshes.
- Causal head beats or ties parallel head in controlled A/B on topology and Chamfer.

## 512-Bin Production Token Update

The 50-candidate Objaverse++ pilot exposed a production-relevant quantization issue:

```text
same strict watertight targets, paper-token strict gate
128 bins: 27/38 passing, pass_rate 0.7105
256 bins: 33/38 passing, pass_rate 0.8684
512 bins: 37/38 passing, pass_rate 0.9737
```

Failure analysis showed mostly small quantized edge cracks rather than dirty target meshes: at 128 bins the median failing shard had 3 boundary edges, no degenerate faces, no non-manifold edges, and edge-pairing around 0.993. Therefore the production corpus default is now `NUM_BINS=512`; use `NUM_BINS=128` only for paper-faithful ablations.

Current Thunder ladder:

```text
instance: 1 / A6000
run_dir: /tmp/clearmesh_face_objpp200_512_20260504_020301
stage: Objaverse++ 200-candidate 512-bin corpus build, then 8k-step FACE train/eval
```

## Reproducible FACE Ladder Supervisor

The current 1-6 ladder is now encoded as a reusable Thunder script:

```bash
scripts/thunder/face_paper_ladder_supervisor.sh
```

It runs:

```text
1. strict target gallery
2. 128 / 256 / 512 paper-token dataset builds
3. strict paper-token gates and pass-only promotion
4. deterministic train/test splits
5. matched short ablations:
   - 128 causal CausalMLP
   - 256 causal CausalMLP
   - 512 causal CausalMLP
   - 128 parallel decode
   - 256 rotate-min-ZYX within-face order
   - 256 sort-ZYX within-face order
6. 512-bin longer run plus AR contact sheets and `ladder_summary.json`
```

Important implementation note:

```text
status logging must stay off stdout inside helper functions.
Only path-returning helpers may write paths to stdout, because Bash command
substitution captures stdout exactly.
```

The first ad hoc supervisor violated that rule and contaminated train/test paths
with JSON status lines. The committed supervisor writes status to stderr and
`status.jsonl`, preventing that failure mode.

Current launch:

```text
lab_root: /tmp/clearmesh_face_ladder_20260504_041315
status: /tmp/clearmesh_face_ladder_20260504_041315/status.jsonl
pid file: /tmp/clearmesh_face_ladder_supervisor.pid
```

## A100 Paper-Knob Probe

The next validation gate is encoded as:

```bash
scripts/thunder/face_paper_a100_probe.sh
scripts/thunder/launch_face_a100_probe_instance.sh 2
```

This probe restores the paper implementation details that matter most before
judging FACE quality:

```text
coordinate bins: 128
point samples: 8192 XYZ+normal samples
VecSet latent: 2048 tokens
bottleneck dim: 64
optimizer: Muon, lr 6e-4, weight decay 0.1
augmentation: SO(3) rotation, flips, and per-axis scaling enabled
causal head: legacy_concat, matching the TreeMeshGPT-style staged coordinate head
precision: bf16 autocast on A100
```

The only deliberate non-paper constraint is the current target corpus cap:

```text
current pilot target cap: 512 faces
paper target cap: fewer than 4000 faces
```

So this A100 probe is a paper-knob validation run, not the final paper-scale
reproduction. If it is healthy, the next corpus build must use curated
`<4000`-face targets before launching the real 100K-step run.

The local launcher performs the reproducible setup steps:

```text
1. sync the current repo to the selected Thunder instance
2. upload the strict target tarball
3. extract it to /tmp/clearmesh_face_objpp200_512_20260504_020301
4. write /tmp/clearmesh_latest_face_objpp_run.txt
5. run the FACE unit tests on CUDA
6. launch the A100 probe under nohup
```
