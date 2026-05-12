# FACE Scale Profile And Missing Details

Date: 2026-05-11

Purpose: define the FACE-native path from the current 10k diagnostics to a serious paper-scale run without losing the implementation lessons we already paid for.

This is not a proposal to replace FACE. It is a contract for making our FACE implementation strict enough, large enough, and instrumented enough that a 130k/100k/8x A100 run is a meaningful experiment rather than an expensive shrug.

## Current Position

We have reproduced the broad FACE structure:

- Shape encoder: point cloud with XYZ+normal goes through FPS/query compression into a VecSet latent.
- Decoder: autoregressive face-by-face transformer conditioned on VecSet.
- Face tokenization: one face token represents nine quantized coordinate tokens.
- Face embedding: token-concat/project path, not the deprecated averaged/weak embedding.
- Face decoding head: causal coordinate decoder.
- Quantization: 128 bins, matching integer coordinates in `[0, 127]`.
- Data gate: strict token manifest, token-hash dedupe, train/test leakage check.
- Training: Muon, bf16, paper-style online augmentation when enabled.

The key learned implementation details are:

- `PAPER_WITHIN_FACE_ORDER=rotate_min_zyx` is required. The previous runs did not consistently use this, and that likely injected unnecessary within-face orientation entropy.
- `FACE_EMBEDDING_VARIANT=token_concat_project` is the strict FACE-like lane. Legacy concat/continuous variants are compatibility lanes only.
- `CACHE_FPS_INDICES=0` is required with online augmentation. Cached FPS is only valid for frozen/no-augmentation diagnostics.
- `FIRST_FACE_TIE_MARGINAL_LOSS=1` is a FACE-native diagnostic/curriculum tool for the quantized ZYX first-face tie problem.
- Prefix weighting is useful diagnostically: `FIRST_FACE_LOSS_WEIGHT>1` with `LOSS_FACE_PREFIX_COUNT>0` asks whether early AR closure is possible.
- Noisy teacher prefix is now implemented as a FACE-native exposure-bias diagnostic: `INPUT_FACE_TOKEN_NOISE_PROB`, `INPUT_FACE_TOKEN_NOISE_MAX_OFFSET`, `INPUT_FACE_NOISE_PREFIX_COUNT`. The `0.05` prefix-16 probe improved train metrics but did not improve held-out rollout topology, so it is not a default scale setting.
- Topology-rescored decoding alone did not rescue the model, so the issue is in the learned distribution, not just greedy decoding.

## What The Paper Specifies

Paper-specified details we should treat as fixed unless a named ablation says otherwise:

- Dataset: Objaverse subset, around 130k meshes, meshes with fewer than 4000 faces.
- Input points: 8192 sampled point-cloud points with normals.
- Shape latent: VecSet with 2048 latent tokens and bottleneck dimension 64.
- Coordinates: vertex positions normalized and quantized to integers in `[0, 127]`.
- Augmentation: random rotation, random flipping, and random independent axis scaling.
- Optimizer: Muon, learning rate `6e-4`, weight decay `0.1`.
- Training budget: 100k steps on 8x A100 80GB.
- Architecture scale: approximately 500M parameters, decoder-heavy.
- Face ordering: ZYX lexicographic ordering wins the paper ablation.
- CausalMLP: causal coordinate decoding wins the paper ablation.

## What The Paper Does Not Fully Specify

These are the missing details we must resolve logically while staying FACE-native:

- Within-face vertex rotation under ZYX sorting.
- How they handle tied minimum vertices / identical quantized anchors.
- Whether reconstruction eval uses ground-truth face count or learned stopping.
- Exact EOS/count handling.
- Exact train batch size / gradient accumulation per GPU.
- Exact distributed training implementation.
- Exact mesh curation criteria beyond Objaverse and face-count filtering.
- Exact point sampling and normal convention after augmentation.
- Whether there is an implicit curriculum or warmup not described.

## Deductive Resolutions

### Within-Face Order

The CausalMLP decodes a face as an ordered nine-token sequence. If the same triangle can appear in multiple cyclic rotations or reversed orientations, the decoder sees equivalent geometry as different labels. That increases conditional entropy without adding useful information.

Resolution:

```text
Canonicalize each triangle by rotating the lexicographically minimum ZYX vertex into slot 0, then order the remaining two vertices deterministically.
```

In our code this is:

```text
PAPER_WITHIN_FACE_ORDER=rotate_min_zyx
```

This is still FACE-native because it does not add topology tokens or change the one-face-one-token representation. It only removes label symmetry.

### First-Face Tie Ambiguity

ZYX ordering selects the first face by the minimum vertex in the mesh. In quantized coordinates, multiple faces can share the same minimum vertex. If the target serialization picks one arbitrarily, the model is punished for selecting another equally canonical first face.

Mathematically, the target for face 0 is sometimes a set:

```text
Y_0 = {face_i : min_vertex(face_i) = global_min_vertex}
```

Plain cross-entropy optimizes one arbitrary member of `Y_0`. Tie-marginal loss optimizes the set:

```text
L_0 = -log sum_{y in Y_0} p(y | C, BOS)
```

Resolution:

```text
FIRST_FACE_TIE_MARGINAL_LOSS=1
```

Use this as a curriculum/diagnostic unless the large run shows it remains beneficial. It is FACE-native because it preserves the same serialized targets and model architecture; only the loss accounts for valid label equivalence.

### Exposure Bias

Teacher-forced training conditions on exact previous faces. Inference conditions on generated previous faces. Exact coordinate meshes are brittle: a one-bin coordinate error can destroy vertex welding, and a wrong early face changes the whole prefix distribution.

Resolution:

```text
INPUT_FACE_TOKEN_NOISE_PROB=0.02-0.05
INPUT_FACE_TOKEN_NOISE_MAX_OFFSET=1
INPUT_FACE_NOISE_PREFIX_COUNT=16-64
```

This should be used as a bounded continuation/curriculum, not blindly from step 0. The first `0.05` probe did not pass held-out topology, so a future noisy-prefix experiment should be smaller or annealed, for example `0.01-0.02`, and only kept if edge pairing/watertightness improve.

### Topology Metrics

FACE is coordinate-token autoregression. It does not mathematically guarantee watertightness. Watertightness emerges only if the model learns exact coordinate reuse and paired boundary closure.

Therefore, production gates must include:

- exact first-face metrics
- rollout generated-token accuracy
- edge pairing ratio
- boundary edge count
- nonmanifold edge/vertex count
- watertight rate
- Chamfer and normal consistency
- visual contact sheets

Loss alone is not sufficient.

## 130k / 100k / 8x A100 Target Profile

Recommended first serious scale profile:

```text
data:
  target_usable_strict_samples: 130000
  source_pool: high-quality Objaverse++ / Objaverse-like assets
  max_input_faces: 4000
  train_test_split: token-hash deduped
  strict_gate: leakage check required

tokenization:
  num_bins: 128
  target_faces: 512 initially, then 1024/2048 if 512 passes
  model_max_faces: 512 initially
  paper_within_face_order: rotate_min_zyx

encoder:
  point_samples: 8192
  vecset_tokens: 2048
  latent_dim: 64
  encoder_layers: 8
  encoder_hidden_size: 768

decoder:
  decoder_layers: 24
  hidden_size: 1024
  heads: 16
  face_embedding_variant: token_concat_project
  causal_mlp_variant: legacy_concat / causal paper chain, selected by validation
  decode_head: causal

training:
  optimizer: Muon
  lr: 6e-4
  weight_decay: 0.1
  precision: bf16
  steps: 100000
  hardware: 8x A100 80GB
  launch: torchrun, TORCHRUN_NPROC_PER_NODE=8
  augmentation: paper online augmentation
  cache_fps_indices: 0
```

Expected parameter scale:

```text
decoder self-attn + cross-attn + FFN:
  ~16 * hidden^2 per layer
  ~16 * 1024^2 * 24 = ~402M decoder params

encoder and embeddings:
  ~70M-110M depending exact cross-attn/FFN layout

total:
  roughly 480M-530M
```

This matches the paper-scale “~500M” target and is very different from our 768/12-layer diagnostics.

## Curriculum Recommendation

Do not jump straight from random initialization into the full 100k run with every diagnostic loss permanently enabled.

Use a staged but single-FACE curriculum:

```text
Stage A: paper strict warmup
  online augmentation on
  rotate_min_zyx
  token_concat_project
  full sequence CE
  monitor selection, teacher-forced, first-face beam

Stage B: early-prefix stabilization
  tie-marginal first-face loss on
  first 16-64 faces weighted moderately
  optional tiny/annealed noisy teacher prefix only if a bounded probe improves held-out rollout topology
  continue only if rollout metrics improve

Stage C: paper objective consolidation
  reduce prefix/noise weights
  keep rotate_min_zyx and token_concat_project
  full sequence CE dominates

Stage D: scale-readiness evaluation
  full AR train/test
  teacher-prefix ablations
  exact first-face beam
  topology/editability gate
```

This is not “ditching paper magic.” It is a FACE-compatible way to make the autoregressive prefix distribution learnable before asking scale to solve every instability at once.

## Current Barriers To 8x Scale

### 1. Distributed Training

Initial DDP/torchrun wiring now exists in the FACE trainer and Thunder wrappers:

- `--distributed auto|off|on`
- `--distributed-backend nccl`
- `TORCHRUN_NPROC_PER_NODE`
- rank-0 checkpoint/eval writes
- per-rank data/noise RNG offsets

Before a paid 8x production run, this still needs a real multi-GPU smoke:

- 2x or 8x A100 torchrun launch
- rank-safe checkpoint verification
- selection eval on rank 0 while other ranks barrier correctly
- throughput and memory measurement
- resume-from-checkpoint test
- B2/checkpoint upload under multi-GPU output volume

### 2. Corpus Scale

We do not yet have 130k strict passing tokenized samples in hand. We have source-pool and sharded-prep tooling, but production scale needs:

- parallel download
- strict mesh conversion/tokenization
- dedupe by token hash
- durable upload to B2 or another object store
- resumable manifest assembly

### 3. Evaluation Cost

Selection eval is expensive. Full AR is more expensive. For scale:

- use sparse selection eval during training
- run first-face and short-rollout probes at checkpoints
- run full topology eval only at milestone checkpoints

### 4. Exact Count / EOS

For reconstruction research gates, using ground-truth face count is acceptable because it isolates face quality. For production generation, we still need a robust count/EOS strategy.

Recommended split:

```text
research reconstruction gate:
  face_count_mode=gt

production inference gate:
  predicted count or EOS/count head
```

### 5. First-Face And Prefix AR

The first-face problem is not purely a decoder-choice problem. Beam/topology/geometric rescoring did not rescue the 100k checkpoint. The learned distribution itself must improve.

Minimum before full 8x launch:

- first-face exact/top-k improves on held-out
- first-face same-min oracle improves
- rollout-64 edge pairing improves
- rollout token accuracy improves

## What Would Count As A Positive Scale Signal

The current 10k/100k results are a positive signal for teacher-forced learning and for the rotate/tie/prefix diagnosis, but not yet enough to prove production-scale AR.

A convincing scale signal would be:

```text
selection/teacher-forced:
  held-out selection loss keeps improving
  held-out teacher-forced coordinate accuracy > 0.85 at 512 faces

first face:
  held-out first-face exact in beam512 > 30%
  held-out same-min oracle > 50%
  first-face divergence clearly below prior gates

rollout:
  generated-token accuracy materially above prior ~0.18-0.19 short-rollout range
  edge pairing > 0.75 on rollout64
  boundary edge count drops, not just Chamfer

topology:
  watertight rate improves alongside edge pairing
  no spike in nonmanifold edges/vertices

visual:
  contact sheets show coherent shapes, not only fragments/rings
```

## Immediate Next Steps

1. Finish the noisy teacher-prefix probe and compare it against prefix-16 no-noise.
2. If it improves rollout metrics, run a longer FACE-native continuation using moderate noise/prefix weighting.
3. If it worsens rollout metrics, do not use noise at scale; keep tie-marginal/prefix weighting only as a short curriculum.
4. Add DDP support before any 8x A100 launch.
5. Continue corpus prep toward at least 130k strict passing samples.
6. Only then launch the 130k/100k/8x profile.

## Bottom Line

The FACE paper results are plausible at their data/capacity/training scale, but our current smaller runs show that exact autoregressive mesh topology is much more brittle than teacher-forced loss suggests.

The right move is not to abandon FACE and not to blindly scale. The right move is:

```text
strict FACE implementation
+ rotate_min_zyx canonicalization
+ token-concat face embedding
+ paper-scale capacity
+ paper-scale data
+ FACE-native first-prefix curriculum if probes validate it
+ topology-aware gates
```

That gives us the best chance of reaching paper-like results while preserving the architecture’s central idea.
