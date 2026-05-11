# FACE Reproduction and Scale Handoff

Date: 2026-05-10

This document captures the current ClearMesh FACE reproduction state, the model details that mattered, the failures we observed, and the production-scale path. It is intentionally practical: if this thread disappears, a new agent should be able to recover the current direction without re-learning the same lessons.

## Current Goal

Build a production-quality artist-mesh generator path that can produce editable triangle meshes, with watertightness/editability measured by our own gates rather than assumed from paper visuals.

The active FACE track is not the whole ClearMesh product, but it is the current best candidate for a learned artist-mesh head. The near-term question is whether scale fixes the remaining autoregressive failures.

## Architecture We Are Reproducing

FACE is an end-to-end autoregressive mesh autoencoder:

1. Input mesh is sampled into a point cloud with coordinates and normals.
2. A Shape Encoder compresses the point set into a compact VecSet latent `C`.
3. A causal autoregressive face decoder generates one mesh face per step, conditioned on `C` by cross-attention.
4. Each face is represented as one latent face token internally.
5. A face decoding head emits the 9 quantized coordinate tokens for that face.
6. Training is end-to-end cross-entropy over quantized face coordinates, not a separately trained VAE tokenizer.

Important paper knobs we use:

- Coordinate quantization range: `[0, 127]`, so `NUM_BINS=128`.
- VecSet latent: `VECSET_TOKENS=2048`, `LATENT_DIM=64`.
- Point samples: `POINT_SAMPLES=8192` for current reproduction runs.
- Optimizer: native `torch.optim.Muon`, learning rate and weight decay as configured in the training script.
- Precision: `bf16` on A100/H100-class GPUs.
- Online augmentation: random rotations/flips/per-axis scaling, matching the paper's training spirit.
- Current production gate face cap: `MODEL_MAX_FACES=512`, `TOKEN_MAX_FACES=512`, `TARGET_FACES=512`.

The paper trained on roughly 130k Objaverse meshes with fewer than 4000 faces for 100k steps. Our current 512-face rung is a lower-complexity reproduction/scale assessment, not a final FACE-equivalent training run.

## Smoking-Gun Lessons

### 1. Use `rotate_min_zyx` inside each face

Set:

```bash
PAPER_WITHIN_FACE_ORDER=rotate_min_zyx
```

This is the biggest implementation-level insight so far. A triangle has cyclically equivalent vertex orders. If the same geometric face can appear as any cyclic rotation, the model sees a multi-modal target for the same face. That damages teacher-forced learning and makes free-running AR brittle.

Our corrected run with `rotate_min_zyx` produced a dramatically better selection-loss curve than the earlier 10k attempt:

```text
5k   2.3075
10k  1.9603
15k  1.5014
20k  1.1131
25k  0.9671
30k  0.9123
```

Earlier non-rotated or compatibility runs did not show the same clean trajectory. Treat `rotate_min_zyx` as mandatory for strict FACE gates.

### 2. Use strict token-concat face embedding

Set:

```bash
FACE_EMBEDDING_VARIANT=token_concat_project
STRICT_FACE_PAPER_GATE=1
ALLOW_DEPRECATED_FACE_EMBEDDING=0
```

The paper's diagram and text imply one face token is formed from all 9 coordinate tokens. The closest implementation path is token concatenation followed by projection into the transformer hidden size. Older variants such as continuous MLP or discrete-sum style embeddings are useful ablations, but they should not be considered strict FACE.

### 3. Keep `legacy_concat` CausalMLP for current scale runs unless deliberately ablating

Current scale runs use:

```bash
CAUSAL_MLP_VARIANT=legacy_concat
DECODE_HEAD=causal
```

We explored `paper_chain` CausalMLP. It is still a useful paper-faithfulness ablation, but it was not the smoking gun. The much stronger signal came from face-internal vertex ordering plus strict token-concat face embedding.

Do not silently switch CausalMLP during a scale run. If testing `paper_chain`, name the run as an ablation and do not compare it directly to `legacy_concat` checkpoints.

### 4. Online augmentation means cached FPS must be off

Set:

```bash
DISABLE_AUGMENT=0
CACHE_FPS_INDICES=0
```

If online augmentation changes the point cloud, cached FPS indices are invalid. We hit this failure directly. For no-augmentation closure diagnostics, cached FPS can be enabled. For paper-like online augmentation, keep it off.

### 5. Use leakage/dedup gates before training

The 10k corpus initially had train/test token-hash leakage and duplicate token hashes. We created deduped splits before training. This is mandatory for credible scale assessment.

Checks to preserve:

- No leaked token hashes across train/test.
- No duplicate token hashes inside train.
- No duplicate token hashes inside test.
- All train/test samples pass strict FACE dataset gate.

### 6. Skip initial selection eval on large runs

Set:

```bash
SKIP_INITIAL_SELECTION_EVAL=1
SELECTION_EVAL_EVERY=10000
```

Initial selection eval costs time and tells us little before training. For 10k/100k and larger, skip it and evaluate every 10k steps. For production-scale corpora, selection eval must use a fixed subset; full selection eval over hundreds of thousands of samples is too slow.

### 7. Preserve checkpoints while running

Use:

```bash
scripts/thunder/preserve_face_paper_run_artifacts.sh
scripts/thunder/watch_face_paper_run_preserve.sh
```

The previous 30k to 100k continuation was lost when the Thunder instance disappeared around total step 94.3k. It had selection loss around `0.727` at 90k, which was a positive undertraining signal, but no final checkpoint was harvested. We should never again rely on a live instance as the only copy of a promising checkpoint.

## What The Latest 10k/30k Result Means

The corrected 10k/30k run gave mixed but useful evidence:

Positive:

- Selection loss improved smoothly to `0.9123` at 30k.
- Continued 30k->100k run improved selection loss further to roughly `0.727` by 90k before the instance disappeared.
- This strongly suggests undertraining was real.

Negative:

- 30k teacher-forced accuracy was still only about `0.753`.
- Free-running AR generated-token accuracy was only about `0.059` train / `0.070` test.
- Train/test watertight rates were about `10/32` and `7/32`.
- Edge pairing was weak, around `0.566` train / `0.447` test.
- Scale readiness was false.

Interpretation:

- The run is a positive signal for the scale hypothesis, not proof of production readiness.
- The strongest failure is autoregressive exposure/first-face divergence, not just mesh repair.
- We should continue to 10k/100k and inspect free-running AR, topology, and prefix diagnostics before launching a huge training job.

## Failure Modes To Watch

### First-face divergence

Zero-prefix AR often chooses a poor first face. Teacher-prefix probes improve output, which means the model has learned conditional continuation better than unconditional start-of-mesh generation.

Action:

- Keep teacher-prefix diagnostics enabled.
- Compare zero-prefix AR, 1-face prefix, 4-face prefix, and 16-face prefix.
- If prefix probes are strong but zero-prefix remains weak, consider start-token/count conditioning or a paper-faithful curriculum before huge scale.

### Exposure bias

Teacher-forced loss can look decent while free-running AR collapses. The model may recover when given correct prefixes but drift when sampling its own previous faces.

Action:

- Do not judge by loss alone.
- Always inspect full free-running train/test meshes.
- Track generated-token accuracy, boundary edges, edge pairing, nonmanifold edges, watertight rate, chamfer, and normals.

### Late coordinate slots are weaker

Failures often concentrate in late within-face coordinate slots and in long sequences. This is consistent with compounding face-token and coordinate-token uncertainty.

Action:

- Keep slot diagnostics.
- Preserve within-face canonicalization.
- Do not raise face cap until 512-face AR is healthy.

### EOS/count behavior

Predicted-count runs may over- or under-generate. Fixed-count evaluation is useful for controlled diagnostics, but production inference needs robust EOS/count behavior.

Action:

- Keep predicted-count eval enabled.
- Report fixed-count and predicted-count separately.

## Watertightness Reality Check

FACE-style papers show high-quality artist meshes, but FACE itself should not be treated as a watertightness guarantee. Watertightness is our product requirement, not necessarily the paper's stated theorem.

Therefore:

- FACE scale readiness is not only teacher-forced loss.
- Watertightness/editability must be measured downstream.
- If FACE gets visual/editable topology mostly right but not perfectly watertight, we may still need a cleanup/manifoldization stage.
- Do not use mesh repair to hide model collapse. Use it only after AR outputs are coherent.

## Current 10k/100k Run Contract

Use these settings for the current scale assessment:

```bash
RUN_STAMP=20260510_rotate10k_100k_preserve_a100_v2
GPU=a100
MODE=production
PRIMARY_DISK=300
RUN_REMOTE_TESTS=0
SELECT_TARGET=10000
SCAN_LIMIT=300000
CURATION_TARGET=10000
MIN_QUALITY=2
TARGET_FACES=512
TOKEN_MAX_FACES=512
MODEL_MAX_FACES=512
NUM_BINS=128
PAPER_WITHIN_FACE_ORDER=rotate_min_zyx
POINT_SAMPLES=8192
VECSET_TOKENS=2048
LATENT_DIM=64
HIDDEN_SIZE=768
ENCODER_HIDDEN_SIZE=768
ENCODER_LAYERS=6
DECODER_LAYERS=12
HEADS=12
PRECISION=bf16
BATCH_SIZE=2
STEPS=100000
SELECTION_EVAL_EVERY=10000
SELECTION_EVAL_BATCH_SIZE=1
SKIP_INITIAL_SELECTION_EVAL=1
CHECKPOINT_EVERY=10000
SAVE_CURRENT_CHECKPOINT=1
PREFETCH_BATCHES=1
CACHE_FPS_INDICES=0
DISABLE_AUGMENT=0
TEACHER_FORCED_LIMIT=128
AR_LIMIT=32
PREDICTED_LIMIT=8
CAUSAL_MLP_VARIANT=legacy_concat
FACE_EMBEDDING_VARIANT=token_concat_project
STRICT_FACE_PAPER_GATE=1
```

Expected runtime on one A100:

- Corpus download/strict-prep/tokenization: roughly several hours for 10k selected assets.
- Training/eval to 100k: likely around one day, depending selection eval cost and Thunder throughput.
- Use hourly monitoring and checkpoint preservation.

## Corpus Scale Plan

We currently have Pool A:

```text
Objaverse++ high/superior selected candidates: 383,397
Quality 2: 156,142
Quality 3: 227,255
Estimated strict usable at 55%: 210,868
Estimated strict usable at 65%: 249,208
```

This pool is acceptable as the first production-scale corpus. It may not yield 350k or 500k usable strict samples by itself, but it is large enough to run a serious scale rung.

Pool B reserve exists:

```text
Objaverse++ medium+ selected candidates: 490,837
Estimated strict usable at 55%: 269,960
Estimated strict usable at 65%: 319,044
```

Use Pool B only if Pool A scale is promising and we need more data. Pool B may lower quality, so keep it separate unless intentionally mixed.

### Sharded prep design

Pool A is already split into 64 JSONL shards:

```text
.codex_outputs/face_source_pool_objpp600k_20260508_061413_managed/shards/shard_0000.jsonl
...
.codex_outputs/face_source_pool_objpp600k_20260508_061413_managed/shards/shard_0063.jsonl
```

Each shard is about 5,991 annotations. Shard workers should run:

```bash
scripts/thunder/launch_face_corpus_shard_instance.sh
```

Important worker settings:

```bash
MIN_QUALITY=2
TARGET_FACES=512
TOKEN_MAX_FACES=512
NUM_BINS=128
POINT_SAMPLES=8192
PAPER_WITHIN_FACE_ORDER=rotate_min_zyx
TEST_RATIO=0.02
LEAN_ARCHIVE_PATH=/tmp/.../lean_face_corpus.tar.gz
```

The corpus pilot now supports `LEAN_ARCHIVE_PATH`, which calls `scripts/research/package_face_corpus.py` and avoids archiving raw downloads.

After shards finish, merge passing tokens with:

```bash
python scripts/research/merge_face_corpus_shards.py \
  --input-list shard_roots_or_manifests.txt \
  --output-dir /path/to/merged_face_corpus \
  --source auto \
  --copy-mode hardlink \
  --test-ratio 0.02 \
  --seed 303
```

Then run token leakage checks before any large training job.

## Thunder Operational Gotchas

- Prototyping A6000 primary disk max appears to be 200GB.
- Prototyping A6000 requires `--vcpus`; valid values observed include `4` and `8`.
- In `tnr create`, put `--vcpus` before `--json`; otherwise the CLI may ignore it and fail non-interactively.
- `RUN_REMOTE_TESTS=1` can hang on remote pytest and waste A100 time. For known-good launches, use `RUN_REMOTE_TESTS=0` after local syntax/unit checks.
- Keep at most one A100 training run unless explicitly scaling training. Shard prep can use cheaper A6000 workers, but inspect disk pressure.
- Do not delete a promising run until metadata and checkpoints are preserved.

## Production Scale Decision Gate

Do not launch a 350k+/100k production run based on loss alone.

Promote only if the 10k/100k run shows:

- Selection loss continues improving or plateaus at a strong value.
- Held-out teacher-forced accuracy materially improves from the 30k run.
- Zero-prefix train AR is coherent.
- Zero-prefix held-out AR is at least recognizably mesh-like, not fragment soup.
- Teacher-prefix diagnostics show robust continuation.
- Watertight/boundary/nonmanifold metrics improve materially.
- No train/test leakage or duplicate token hashes.
- Visual contact sheets and exported meshes pass human sanity check.

If the 10k/100k result is mixed:

- Continue bounded diagnostics before scaling all the way.
- Prioritize first-face/count conditioning and exposure-bias probes.
- Keep paper-faithful settings unless the ablation is explicitly named.

## Inference Provider Portability

Switching cloud providers for production inference should not require retraining.

Requirements:

- Same model code and checkpoint format.
- Compatible PyTorch/CUDA image.
- `bf16` capable GPU or a tested fallback precision path.
- Deterministic preprocessing: normalization, quantization, face ordering, and point sampling must match training.
- Clear inference wrapper for fixed-count and predicted-count decoding.
- Postprocess/evaluation gate before returning assets.

A100/H100/B100/L40S-style providers should run the same weights, but containerize the environment and validate output parity before migration.

## What To Do Next

1. Let the 10k/100k A100 run complete with checkpoint preservation.
2. Inspect 10k/100k free-running AR, teacher-forced, teacher-prefix, predicted-count, and topology metrics.
3. Let two Pool A shard workers finish and measure strict usable yield per shard.
4. If shard yield/time is acceptable, launch the remaining Pool A shards in batches.
5. If 10k/100k passes visual/topology gates, merge Pool A strict shards and run the next scale training rung.
6. If 10k/100k fails, do not scale blindly; run the smallest ablation that targets the observed failure.
