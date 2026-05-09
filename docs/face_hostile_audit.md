# FACE Hostile Audit

Date: 2026-05-03
Paper: `/Users/Ashar/Downloads/2603.01515v2.pdf`
ArXiv: https://arxiv.org/abs/2603.01515
HTML: https://arxiv.org/html/2603.01515v2
Source inspected: `/tmp/face_src`

## Executive Decision

FACE is real enough to pursue, but not as a wholesale replacement for LATTICE.

My recommendation:

```text
Do not pivot from LATTICE to PixARMesh.
Do add a FACE-like reconstruction head as the learned artist-mesh stage after LATTICE.
Do not train FACE image-to-mesh DiT first.
First reproduce the ARAE point-cloud-to-mesh model.
```

The strongest production architecture is:

```text
Input image / prompt
    -> TRELLIS.2 coarse asset
    -> LATTICE / UltraShape-style high-fidelity watertight reference
    -> sample points + normals from reference
    -> FACE-like ARAE reconstruction head
    -> compact editable triangle mesh
    -> feature-aware projection back to reference
    -> part/quad/autorig gates as optional promotion layers
```

Why this is the right compromise:

```text
LATTICE gives geometry truth.
FACE gives compact explicit triangle structure.
Projection gives fidelity recovery.
Blender/eval gates decide whether the mesh is production-editable.
```

FACE should be treated as a learned remesher/topology head first, not a full image-to-3D generator.

## Current Implementation Status

Verified on 2026-05-03:

```text
clearmesh/mesh_heads/face_tokens.py
clearmesh/mesh_heads/face_tiny.py
clearmesh/mesh_heads/face_arae.py
scripts/research/build_face_token_dataset.py
scripts/research/train_face_tiny.py
scripts/research/train_face_conditioned_tiny.py
scripts/research/sample_face_conditioned_tiny.py
scripts/research/train_face_level_conditioned_tiny.py
scripts/research/sample_face_level_conditioned_tiny.py
scripts/research/sample_face_level_from_mesh.py
scripts/research/eval_face_level_conditioned_tiny.py
clearmesh/mesh_heads/face_level.py
scripts/thunder/face_conditioned_tiny_smoke.sh
scripts/thunder/face_conditioned_overfit.sh
scripts/thunder/face_level_conditioned_overfit.sh
scripts/thunder/face_level_conditioned_smoke.sh
```

What works now:

```text
- deterministic FACE-style coordinate tokenization
- decode/weld roundtrip for watertight cubes
- stable encoding under vertex permutation
- surface point + normal conditioning shards
- tiny point-conditioned autoregressive decoder
- tiny FACE-like face-level autoregressive decoder
- lightweight face-count prediction head for profile/model-selected output length
- batch evaluator with topology and pair metrics
- mixed synthetic primitive curriculum for controlled stress tests
- product mesh-head adapter registered as `face-level`
- safe PyTorch checkpoint loading for new checkpoints
- best-weight checkpointing instead of final-step checkpointing
- paper-faithful ARAE path with strict paper-token dataset gate
- full-dataset checkpoint selection for tiny real overfit runs
```

Thunder results:

```text
16-shape smoke:
  command: scripts/thunder/face_conditioned_tiny_smoke.sh
  best_loss: 2.080
  sampled generated mesh: 11 faces, 20 vertices, watertight=false
  interpretation: training/inference path works, but this tiny model/run is undertrained.

1-shape overfit:
  command: scripts/thunder/face_conditioned_overfit.sh
  best_loss: 0.00112
  generated: 12 faces, 8 vertices, watertight=true
  teacher:   12 faces, 8 vertices, watertight=true
  interpretation: tokenizer + conditioned decoder + sampler can emit a valid closed artist-style mesh when the learning problem is solved.

1-shape FACE-level overfit:
  command: scripts/thunder/face_level_conditioned_overfit.sh
  autoregressive_steps: 12
  coordinate_tokens: 108
  best_loss: 0.01457
  generated: 12 faces, 8 vertices, watertight=true
  teacher:   12 faces, 8 vertices, watertight=true
  interpretation: the paper-aligned sequence compression works in our scaffold; each AR step predicts a full triangle face.

1-shape FACE-level count-head proof:
  command: RUN_DIR=/tmp/clearmesh_face_level_count_overfit STEPS=500 scripts/thunder/face_level_conditioned_overfit.sh
  face_count_argument: omitted
  requested_face_count_from_model: 12
  generated: 12 faces, 8 vertices, watertight=true
  interpretation: the decoder can now select output face count when the caller does not specify a profile count.

LATTICE-to-FACE bridge:
  command: scripts/research/sample_face_level_from_mesh.py
  input: /tmp/clearmesh_lattice_vdf_smoke/predicted.glb
  checkpoint: /tmp/clearmesh_face_level_conditioned_overfit/face_level_conditioned_overfit.pt
  generated: 12 faces, 8 vertices, watertight=true
  interpretation: the reference-mesh-to-learned-topology handoff is now executable, albeit only on the toy box proof.

16-box FACE-level smoke:
  command: scripts/thunder/face_level_conditioned_smoke.sh
  best_loss: 0.00688
  attempted: 16
  watertight_rate: 1.0
  mean_chamfer_l2: 0.00502
  mean_normal_consistency: 0.92575
  interpretation: the tiny face-level decoder can use point conditioning across a small variable-extents box set.

Mixed primitive FACE-level smoke:
  command: RUN_DIR=/tmp/clearmesh_face_level_mixed_smoke SYNTHETIC_KIND=mixed_cycle SYNTHETIC_COUNT=12 MAX_FACES=256 EVAL_LIMIT=10 STEPS=1200 scripts/thunder/face_level_conditioned_smoke.sh
  included: boxes, cylinders, cones, capsules, tori
  max_ar_steps: 256
  coordinate_tokens_at_max_faces: 2304
  best_loss: 0.01079
  attempted: 10
  watertight_rate: 1.0
  mean_chamfer_l2: 0.00394
  mean_normal_consistency: 0.95470
  interpretation: variable topology/face-count synthetic shapes survive the face-level representation; this is still controlled data, not real asset generalization.

Product adapter smoke:
  command: scripts/product/run_mesh_head.py --head face-level
  input_proxy: /tmp/clearmesh_lattice_vdf_smoke/predicted.glb
  checkpoint: /tmp/clearmesh_face_level_count_overfit/face_level_conditioned_overfit.pt
  generated: 12 faces, 8 vertices, watertight=true
  boundary_edges: 0
  nonmanifold_edges: 0
  interpretation: the FACE-level topology head is now callable through the same adapter registry as external mesh heads.

Paper-faithful strict voxel-shell overfit:
  command: scripts/thunder/face_paper_faithful_smoke.sh with DATASET_GATE_PROFILE=strict and SELECTION_EVAL_EVERY=200
  targets: 7 cleaned real voxel-shell meshes, 220-3016 faces
  dataset gate: 7/7 passing, boundary edges 0, edge pairing 1.0
  selected checkpoint: step 2600 by full-dataset loss 0.0001086935
  teacher-forced export: 7/7 watertight, mean boundary edges 0, mean edge pairing 1.0
  full-count AR export: 7/7 watertight, zero boundary edges, 125s total, 46s for the 3016-face mesh
  interpretation: the paper-faithful ARAE scaffold can exactly reconstruct and roll out cleaned real training targets; this is not yet held-out generalization.

Paper-faithful strict voxel-shell holdout:
  command: scripts/thunder/face_paper_holdout_smoke.sh with 5 train / 2 held out strict real targets
  train full-count AR: 5/5 watertight, zero boundary edges
  held-out full-count AR: 0/2 watertight, mean boundary edges 41
  interpretation: tiny cleaned real-data generalization is not solved; scale the cleaned corpus before treating FACE as a production topology head.

Paper-faithful strict mixed15 aug60 grouped holdout:
  command: scripts/thunder/face_paper_holdout_smoke.sh with grouped split by source_name
  corpus: 15 strict source meshes, 4 offline variants per source, 60 shards total
  strict dataset gate: 60/60 passing
  split: 48 train shards / 12 held-out shards, no source-variant leakage
  training: 3200 steps, 192 hidden, best selection loss 1.2087 at final step and still improving
  train AR sample: 1/12 watertight, mean boundary edges 662.5
  held-out AR: 3/12 watertight, mean boundary edges 982.2
  interpretation: the grouped harness works, but this is undertrained. Do not use it as a final held-out verdict until train AR reaches the strict overfit regime.
```

Durable local proof meshes:

```text
artifacts/research_proofs/2026-05-03/face_conditioned_overfit_generated.glb
artifacts/research_proofs/2026-05-03/face_conditioned_overfit_teacher.glb
artifacts/research_proofs/2026-05-03/face_level_conditioned_overfit_generated.glb
artifacts/research_proofs/2026-05-03/face_level_conditioned_overfit_teacher.glb
artifacts/research_proofs/2026-05-03/face_level_from_lattice_generated.glb
artifacts/research_proofs/2026-05-03/face_level_count/generated.glb
artifacts/research_proofs/2026-05-03/face_level_16shape/eval_report.json
artifacts/research_proofs/2026-05-03/face_level_mixed/eval_report.json
artifacts/research_proofs/2026-05-03/face_level_adapter/eval.json
artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_overfit_v2/PROOF_SUMMARY.md
artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_overfit_v2/face_paper_strict_voxel16_teacher_forced_contact_sheet.png
artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_overfit_v2_ar_full/face_paper_strict_voxel16_ar_full_contact_sheet.png
artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_holdout_v1/HOLDOUT_SUMMARY.md
artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_holdout_v1/face_paper_strict_voxel16_holdout_test_ar_contact_sheet.png
artifacts/research_proofs/2026-05-03/face_paper_strict_mixed15_aug60_holdout_v1/HOLDOUT_SUMMARY.md
artifacts/research_proofs/2026-05-03/face_paper_strict_mixed15_aug60_holdout_v1/face_paper_strict_mixed15_aug60_train_ar_contact_sheet.png
artifacts/research_proofs/2026-05-03/face_paper_strict_mixed15_aug60_holdout_v1/face_paper_strict_mixed15_aug60_test_ar_contact_sheet.png
```

## Why FACE Is Tempting

FACE directly attacks the problem that bothered us with LATTICE/SDF/marching-cubes pipelines: dense, irregular triangle soup. It models explicit triangle faces and reports much cleaner reconstruction metrics than MeshAnything, MeshAnythingV2, TreeMeshGPT, and BPT on Objaverse, Toys4K, and Famous.

Its main idea is beautifully simple:

```text
Old AR mesh models:
    one coordinate token at a time
    sequence length = 9 x face_count

FACE:
    one triangle face token at a time
    sequence length = face_count
    CausalMLP decodes the 9 coordinate values inside the face
```

This is why it visually feels closer to editable mesh creation than SDF extraction. It is compact and explicit. It can output something closer to an artist triangle mesh, whereas LATTICE alone outputs a high-fidelity surface reference.

## What FACE Actually Builds

FACE has two stages.

### Stage 1: ARAE Reconstruction Model

```text
surface point cloud + normals
    -> Shape Encoder / VecSet
    -> autoregressive face decoder
    -> triangle mesh face sequence
```

Important details from the paper/source:

| Detail | Value |
| --- | --- |
| Model family | Autoregressive Autoencoder (ARAE) |
| Input | Surface point cloud with normals |
| Face representation | Each face is a flattened 9D vector: three vertices x xyz |
| Face order | Lexicographic ZYX order of the face minimum-coordinate vertex |
| Encoder | 8 layers, hidden dim 768 |
| Decoder | 24 layers, hidden dim 1024 |
| Total params | 500M |
| Input samples | 8192 surface points with normals |
| VecSet latent | 2048 tokens, bottleneck dim 64 |
| Training data | ~130k Objaverse meshes with fewer than 4000 faces |
| Quantization | vertex coordinates in [0, 127] |
| Augmentations | random rotation, flipping, individual axis scaling |
| Optimizer | Muon, LR 6e-4, weight decay 0.1 |
| Training compute | 100k steps on 8 x A100 80GB |
| Inference | deterministic top-1 autoregressive sampling |

### Stage 2: Image-To-Mesh DiT

```text
image
    -> DINOv3 image features
    -> flow-matching DiT over FACE VecSet latent
    -> FACE decoder
    -> mesh
```

Important details:

| Detail | Value |
| --- | --- |
| DiT params | 350M |
| Objective | flow matching |
| Image features | pretrained DINOv3 |
| Dataset | 50k meshes from ARAE training set |
| Renders | 10 per mesh with random lighting/cameras |
| Training compute | 400k steps on 32 x A100 80GB |
| Inference | 100 Euler steps |

### Scaling Result

The paper also trains a larger ARAE:

| Detail | Value |
| --- | --- |
| Params | 1.2B |
| Input points | 65,536 |
| Quantization | [0, 1023] |
| Dataset | 380k internal high-quality meshes |

This large model is exactly the part that makes the images look so impressive, and it is also the least reproducible part because the dataset is internal.

## Hostile Feasibility Audit

### 1. Public Code Status

I did not find an official public FACE repo or pretrained weights in the audit. The arXiv source is available, and the source confirms method/training details, but not code.

Implication:

```text
FACE reproduction is feasible engineering, but not a quick repo integration.
```

We would implement it ourselves using:

```text
- PyTorch / Transformers / x-transformers for decoder blocks
- Hunyuan/3DShape2VecSet-style VecSet encoder patterns
- PixARMesh/BPT utilities for mesh tokenization infrastructure
- our eval harness for topology/editability gates
```

### 2. The Core Model Is Reproducible

The ARAE is not exotic:

```text
- normalize mesh
- sample points + normals
- quantize vertices
- sort faces canonically
- embed previous face as one token
- causal transformer over face tokens
- cross-attend to VecSet latent
- CausalMLP predicts 9 coordinate tokens
```

This is implementable. The hardest parts are not the transformer blocks; they are data quality, sequence canonicalization, EOS/face-count handling, and preserving shared vertices after face-coordinate decoding.

### 3. The Paper Omits Critical Production Details

These are the hostile notes that matter.

| Missing / underspecified piece | Why it matters | Our mitigation |
| --- | --- | --- |
| EOS / variable face-count handling is not described clearly | AR generation needs to know when to stop. | For v0 reconstruction, condition on target face count; for generation, add learned EOS or predict face count separately. |
| Vertex sharing is under-explained | FACE predicts face coordinates; duplicate vertices can create seams unless merged robustly. | Quantized coordinates make merge possible; add deterministic vertex weld/merge gate and non-manifold tests. |
| Input sequence embedding is ambiguous | Paper says embed 9D face vector, but trains coordinate-token CE. | Use dequantized normalized 9D face vector for face embedding and quantized coordinate tokens for CausalMLP loss. |
| Data curation is hidden | Raw Objaverse is noisy; good topology needs good training meshes. | Build a curated artist-mesh subset; filter by face count, components, degenerates, non-manifold edges, and editability metrics. |
| Large model uses internal data | The prettiest result may depend on unavailable data. | Start with base reproduction; only claim large quality after building our own HQ dataset. |
| Image model is very expensive | 32 A100 x 400k steps is not sprint-zero. | Skip image DiT initially; condition FACE on LATTICE/TRELLIS point clouds. |
| Thin structures are a stated limitation | Point-cloud sampling can miss spokes, wires, cables. | Use higher point counts, feature/edge-biased sampling, and LATTICE reference projection. |
| It outputs triangles, not quads | Editable, but not the same as quad edge loops. | Treat as artist-triangle output; optionally promote through chart-level quad remesh. |
| No UV/material/part semantics | Production asset still needs metadata. | Keep OmniPart/material/UV/autorig stages. |

### 4. FACE Is Not DualPrim

FACE looks editable for a different reason than DualPrim.

```text
DualPrim:
    editable because geometry is decomposed into explicit primitives/patches
    risk: primitive underfit, generic coverage, brittle fitting

FACE:
    editable because it learns compact triangle face structure
    risk: triangles are not semantic parts, not quads, and may still need welding/projection
```

FACE is much more generic than DualPrim. It is not constrained to tubes/boxes/spheres. That is good. But it also means editability emerges statistically from the training mesh distribution; it is not guaranteed by construction.

### 5. Runtime Is Better Than Old AR, But Not Automatically Production-Fast

FACE reduces face-level sequence length by 9x compared with coordinate-token AR and reports a 0.11 compression ratio. That is excellent.

But it is still autoregressive over faces.

Rough production expectations, assuming a well-optimized 500M decoder with KV cache and flash attention:

| Target faces | Expected feel | Risk |
| ---: | --- | --- |
| 512 | likely interactive-ish preview, seconds to low tens of seconds | should be fine |
| 2k | likely tens of seconds | feasible for standard output |
| 4k | likely around 1-3 minutes unless heavily optimized | paper training target, but no reported runtime |
| 8k+ | possible but may become uncomfortable | needs chunk/speculative/parallel decoding or LATO-style flow |

This is still probably much better than MeshRipple/Silksong style high-face AR in our earlier runs, but LATO-style flow remains the better speed north star.

## LATTICE vs FACE vs PixARMesh

| Direction | Best at | Weakness | My recommendation |
| --- | --- | --- | --- |
| LATTICE | high-fidelity watertight/reference geometry | not editable topology by itself | keep as geometry truth stage |
| FACE | compact explicit triangle mesh reconstruction | no code; AR latency; no quads/parts/materials | build as learned remesher after LATTICE |
| PixARMesh | complete scene-level repo; pixel-aligned conditioning; layout tokens | scene-specific and AR; not our asset core | borrow conditioning patterns only |
| LATO | explicit topology with flow-like speed | no code; harder model | keep as longer-term learned topology target |

The choice should not be:

```text
LATTICE or FACE
```

It should be:

```text
LATTICE for reference fidelity
FACE for learned compact triangle topology
```

PixARMesh should not become the core route. It is useful, but mostly as a public implementation reference for conditioning, token streams, two-stage training, and scene context.

## Reproduction Plan

### Phase 0: Tokenizer And Roundtrip

Goal: prove the representation is not brittle before training.

Deliverables:

```text
clearmesh/mesh_heads/face_tokens.py
    - normalize vertices to [-1, 1]
    - quantize to 128 bins
    - sort vertices/faces by ZYX
    - rotate each face so canonical vertex is first
    - encode faces as [num_faces, 9] integer tokens
    - decode/weld/merge back to mesh
```

Tests:

```text
- cube roundtrip
- disconnected components roundtrip
- non-manifold and degenerate-face rejection
- face order deterministic across repeated runs
- merge/weld does not create accidental collapsed faces
```

Exit gate:

```text
Chamfer near quantization error floor on fixtures.
Boundary/non-manifold counts do not worsen unexpectedly.
```

### Phase 1: Tiny Overfit Model

Goal: prove our architecture is wired correctly.

```text
Model: 10M-30M
Data: 1 mesh, then 20 meshes
Quantization: 64 or 128 bins
Max faces: 256-512
Training: single GPU
```

Exit gate:

```text
Can reconstruct training meshes exactly or near-exactly after weld.
No broken EOS/face-count logic.
```

### Phase 2: Small FACE Reconstruction

Goal: prove generalization on a small public subset.

```text
Model: 50M-150M
Data: 1k-10k curated meshes
Input: 8192 points + normals
VecSet: start 512-1024 latent tokens, then scale to 2048
Max faces: 2k, then 4k
```

Exit gate:

```text
Beats BPT/TreeMeshGPT/simple decimation on our editability harness.
Does not fragment handles, limbs, branches, holes.
Runtime acceptable at 512/2k/4k profiles.
```

### Phase 3: LATTICE-To-FACE Path

Goal: turn FACE into our production remesher.

```text
LATTICE/TRELLIS reference mesh
    -> sample 8k/16k/65k points + normals
    -> FACE reconstruction head
    -> weld/repair
    -> projection to LATTICE reference
    -> Blender/editability gate
```

Exit gate:

```text
FACE output is more editable than marching-cubes reference and more stable than MeshRipple/Silksong at similar face counts.
```

### Phase 4: Base-Scale ARAE

Goal: serious reproduction of paper-level base model.

```text
Model: around 500M
Data: around 130k meshes under 4k faces
Compute: about 8 x A100/H100 class GPUs
Schedule: 100k steps
```

Exit gate:

```text
Held-out Objaverse/Toys4K/Famous-like results approach paper trend.
Our ClearMesh stress set passes production gates.
```

### Phase 5: Image DiT Only If Needed

Do not start here.

Train image-to-VecSet only after ARAE is good:

```text
Model: 350M DiT
Data: 50k meshes x 10 renders
Compute: paper used 32 x A100 80GB for 400k steps
Inference: 100 Euler steps
```

In our product, this may be unnecessary because TRELLIS.2/LATTICE already give us image-conditioned geometry. FACE can be the final topology head instead.

## Dataset Strategy

FACE quality lives or dies on mesh targets.

Recommended dataset tiers:

```text
Tier 0: fixtures
    cube, torus, sphere, mug, chair, animal toy, branching object

Tier 1: public clean subset
    Toys4K, Objaverse filtered under 4k faces, ABO, 3D-FUTURE/3D-FRONT objects

Tier 2: ClearMesh HQ set
    assets that pass editability gates after cleanup/retopo

Tier 3: synthetic paired data
    LATTICE high-fidelity reference paired with cleaned artist triangle target
```

Required filters:

```text
- face count <= profile cap
- no zero-area faces
- no extreme component noise
- bounded non-manifold edge/vertex count
- stable normalization
- reasonable valence histogram
- no catastrophic duplicate vertices
- no broken scale or scene outliers
```

## What Success Would Mean

A successful FACE reproduction would give us something extremely valuable:

```text
high-fidelity reference surface from LATTICE
+ learned compact triangle topology from FACE
+ projection/revalidation from ClearMesh
= editable artist-triangle production mesh
```

It would not automatically give us:

```text
- universal quads
- semantic parts
- UVs/materials
- rig-ready topology
- guaranteed edge loops
```

But it could become the best triangle-mesh head we have seen so far.

## Final Recommendation

I would pursue FACE, but in a disciplined way.

Decision:

```text
Keep LATTICE as the primary high-fidelity reference path.
Start FACE-lite immediately as the learned artist-triangle remeshing head.
Do not pivot to PixARMesh as the product core.
Do not train FACE image-DiT until the ARAE reconstruction head is proven.
```

Next engineering move:

```text
1. Implement FACE tokenization/roundtrip.
2. Add FACE metrics to the eval harness.
3. Train a tiny overfit ARAE on fixtures.
4. Train a small FACE reconstruction head on a curated public subset.
5. Feed LATTICE/TRELLIS reference point clouds into FACE and compare against our chart-retopo path.
```

If tiny/small FACE works, it becomes the core ClearMesh topology head. If it fails, we have still gained a strong tokenizer, mesh dataset, and editability evaluation path that helps LATO-lite and any future mesh head.

## Implementation Update: FACE-Level Held-Out Smoke

Date: 2026-05-03

The first FACE-lite reproduction milestone is now implemented in the repo:

```text
surface points + normals
    -> pooled condition tokens
    -> face-level autoregressive decoder
    -> 9 coordinate-bin logits per triangle
    -> optional predicted face-count head
```

Important distinction:

```text
This is not paper-scale FACE.
This is the smallest useful FACE-style reconstruction head for proving the topology route.
```

### Validated Pieces

Implemented:

```text
- deterministic FACE-style mesh tokenization
- face-level, not coordinate-level, autoregressive decoder
- point/normal conditioning
- face-count prediction head
- token-level topology diagnostics
- conservative token repair
- topology-weighted teacher-forced training
- product mesh-head adapter: "face-level"
- Thunder overfit and mixed-primitive smoke wrappers
- held-out train/test wrapper
- raw-vs-cleaned evaluation reporting
```

Local verification:

```bash
.venv/bin/python -m pytest tests/test_face_tokens.py tests/test_face_level_adapter.py tests/test_lattice_utils.py tests/test_mesh_cleanup.py
```

Result:

```text
23 passed
```

Thunder command:

```bash
DOWNLOAD_DIR=/Users/Ashar/Documents/GitHub/clearmesh/artifacts/research_proofs/2026-05-03/face_level_holdout \
  scripts/thunder/face_level_conditioned_holdout.sh 0
```

Training setup:

```text
train shards: 44 mixed synthetic primitives
test shards: 23 held-out mixed synthetic primitives
max faces: 256
point samples saved: 512
point samples used by model: 256
model: hidden 192, 3 layers, 6 heads, 8 condition tokens
steps: 3000
GPU: Thunder RTX A6000
best loss: 0.0022515568416565657 at step 2904
```

### Held-Out Result

Ground-truth face-count mode isolates token quality:

```json
{
  "attempted": 23,
  "mesh_ok_rate": 1.0,
  "watertight_rate": 0.4782608695652174,
  "mean_chamfer_l2": 0.03617700868389408,
  "mean_hausdorff_l2": 0.25146015029334506,
  "mean_normal_consistency": 0.8614573033291314
}
```

Predicted-count mode is the production-like setting:

```json
{
  "attempted": 23,
  "mesh_ok_rate": 1.0,
  "watertight_rate": 0.6086956521739131,
  "mean_chamfer_l2": 0.03160356321306253,
  "mean_hausdorff_l2": 0.26558871859799543,
  "mean_normal_consistency": 0.8477667050309572
}
```

Predicted-count plus conservative cleanup:

```json
{
  "attempted": 23,
  "mesh_ok_rate": 1.0,
  "watertight_rate": 0.782608695652174,
  "mean_chamfer_l2": 0.03136991877360527,
  "mean_hausdorff_l2": 0.26216617433399064,
  "mean_normal_consistency": 0.8360561386388283
}
```

Predicted-count plus voxel-shell manifold recovery:

```json
{
  "attempted": 23,
  "mesh_ok_rate": 1.0,
  "watertight_rate": 1.0,
  "mean_faces": 28890.956521739132,
  "mean_chamfer_l2": 0.03176261903287363,
  "mean_hausdorff_l2": 0.2728665717178185,
  "mean_normal_consistency": 0.8189079468221113
}
```

Artifacts:

```text
artifacts/research_proofs/2026-05-03/face_level_holdout/holdout_summary.json
artifacts/research_proofs/2026-05-03/face_level_holdout/eval_gt_count_report.json
artifacts/research_proofs/2026-05-03/face_level_holdout/eval_predicted_count_report_v2.json
artifacts/research_proofs/2026-05-03/face_level_holdout/eval_predicted_count_voxel_shell_report.json
artifacts/research_proofs/2026-05-03/face_level_holdout/face_level_holdout_contact_sheet.svg
```

### Hostile Read

This is a real milestone, but it is not production-ready yet.

What worked:

```text
- the decoder learned boxes, capsules, tori, cones, and cylinders beyond a one-shape overfit
- predicted face count often matched the teacher count
- every raw predicted-count output loaded as a mesh
- cheap cleanup improved watertightness from 14/23 to 18/23
- Chamfer stayed roughly stable after cleanup
```

What failed:

```text
- raw watertightness is still too low for promotion
- cleanup still leaves 5/23 non-watertight
- voxel-shell recovery reaches 23/23 watertight, but at high face count and with lower normal consistency
- failures cluster around cones/cylinders/capsules where boundary and non-manifold edges appear
- one cone was catastrophically over-counted: predicted 256 faces vs 24 target faces
- this tiny model has no explicit topological legality constraint
```

Interpretation:

```text
FACE-lite is promising as a learned topology prior.
FACE-lite should not replace the watertight reference path yet.
Production should still treat LATTICE/TRELLIS + UltraShape/manifoldization as the surface truth.
FACE should become the editable triangle topology proposal, then pass through cleanup, projection, and gates.
Voxel-shell recovery is a safety fallback, not the final editable topology.
```

### Next Required Fix

The next model change should not just scale parameters. The top issue is structural:

```text
independent coordinate-bin logits can produce geometrically close triangles
without guaranteeing manifold edge pairing.
```

Next experiment:

```text
Add topology legality bias:
- train/eval on larger held-out primitive and fixture sets
- add vertex reuse / edge-closure auxiliary losses
- add EOS/count calibration loss for over-count failures
- add post-decode topology repair before projection
- gate promotion on cleaned watertight rate, not raw screenshots
```

## Implementation Update: Topology-Weighted FACE-Lite

Date: 2026-05-03

Added token-level topology tools:

```text
clearmesh/mesh_heads/face_topology.py
```

These report, before floating-point decode:

```text
- degenerate face count
- duplicate face count
- boundary edge count
- nonmanifold edge count
- edge pairing ratio
- token-level watertight edge graph
```

Also added:

```text
- conservative token repair: none / dedupe / manifold
- topology-coordinate training weights
- explicit topology auxiliary heads
- closure-extra decoding experiment
```

### Topology Repair Result

Post-decode token repair alone is not enough:

```json
{
  "attempted": 23,
  "watertight_rate": 0.6086956521739131,
  "mean_chamfer_l2": 0.03141074098276787,
  "mean_boundary_edge_count": 15.826086956521738,
  "mean_nonmanifold_edge_count": 0.0,
  "mean_edge_pairing_ratio": 0.9567466865928455
}
```

Interpretation:

```text
The repair pass removes illegal overused edges.
The remaining failures are mostly missing edge pairs / open boundaries.
```

### Closure-Extra Decoding Result

Tested allowing up to 32 extra faces beyond predicted count until token edges close.

Result:

```json
{
  "attempted": 23,
  "watertight_rate": 0.6086956521739131,
  "mean_chamfer_l2": 0.03399797346946442,
  "mean_hausdorff_l2": 0.31804134620352004,
  "mean_boundary_edge_count": 28.217391304347824
}
```

Interpretation:

```text
Blindly decoding longer is worse.
Do not use closure-extra decoding as a production trick.
Missing closure needs training-time supervision or an explicit closure policy.
```

### Topology-Weighted Training Result

Ran the same 44-train / 23-test split with:

```text
reuse_vertex_loss_weight = 0.5
edge_closure_loss_weight = 1.0
```

Command:

```bash
RUN_DIR=/tmp/clearmesh_face_level_topology_weight_holdout \
REUSE_VERTEX_LOSS_WEIGHT=0.5 \
EDGE_CLOSURE_LOSS_WEIGHT=1.0 \
DOWNLOAD_DIR=/Users/Ashar/Documents/GitHub/clearmesh/artifacts/research_proofs/2026-05-03/face_level_topology_weight \
  scripts/thunder/face_level_conditioned_holdout.sh 0
```

Compared with the unweighted baseline:

| Setting | Raw watertight | Cleaned watertight | Mean Chamfer L2 |
| --- | ---: | ---: | ---: |
| baseline predicted-count | 14/23 | 18/23 | 0.03160 raw / 0.03137 cleaned |
| topology-weighted predicted-count | 15/23 | 19/23 | 0.02399 raw / 0.02408 cleaned |
| topology-weighted token-repair + cleanup | 15/23 | 19/23 | 0.02399 raw / 0.02377 cleaned |

Token topology improved substantially:

```json
{
  "watertight_edge_graph_rate": 0.6521739130434783,
  "mean_boundary_edge_count": 2.217391304347826,
  "mean_nonmanifold_edge_count": 0.0,
  "mean_edge_pairing_ratio": 0.9757293731663551
}
```

This is the first training-side lever that helps:

```text
geometry improved
boundary edges dropped sharply
cleaned watertightness improved from 18/23 to 19/23
```

But it is still not enough:

```text
19/23 cleaned watertight is not production-grade.
The model still needs a stronger topology representation or closure head.
```

Next architecture change:

```text
Add an explicit topology/closure head:
- predict whether each new face closes 0/1/2/3 existing boundary edges
- predict reused-vertex vs new-vertex source for each face vertex
- optionally generate from a vertex table + face indices, not raw XYZ bins only
```

### Aux-Head Result

Implemented explicit auxiliary heads:

```text
reuse_vertex_logits: per-face, per-corner reused-vs-new labels
edge_closure_logits: per-face 0/1/2/3 closed-edge label
```

Ran with:

```text
reuse_vertex_loss_weight = 0.5
edge_closure_loss_weight = 1.0
topology_aux_loss_weight = 0.1
```

Result:

| Setting | Raw watertight | Cleaned watertight | Mean Chamfer L2 |
| --- | ---: | ---: | ---: |
| topology-coordinate weights only | 15/23 | 19/23 | 0.02399 raw / 0.02408 cleaned |
| topology aux heads, weight 0.1 | 13/23 | 15/23 | 0.03279 raw / 0.03269 cleaned |

Interpretation:

```text
The auxiliary labels are learnable, but the auxiliary loss hurt generation in this tiny decoder.
Do not enable topology_aux_loss_weight by default.
Keep the aux heads in code as an experimental scaffold.
The winning lightweight setting so far is coordinate topology weighting without aux loss.
```

Artifacts:

```text
artifacts/research_proofs/2026-05-03/face_level_topology_weight/holdout_summary.json
artifacts/research_proofs/2026-05-03/face_level_topology_aux/holdout_summary.json
```

### Real Proxy Fixture Result

Added a conservative fixture-prep bridge:

```text
scripts/research/prepare_face_fixture_meshes.py
scripts/thunder/face_level_proxy_fixture_smoke.sh
scripts/research/check_face_dataset_targets.py
clearmesh/mesh_heads/face_dataset_gate.py
```

The fixture prep can decimate very dense TRELLIS/UltraShape/MeshRipple proxy
outputs into bounded FACE-token test meshes without mutating the source assets.
This is a diagnostic bridge, not a claim that decimated fragments are good
artist-mesh supervision.

Smoke command:

```bash
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

Result:

| Evaluation | Watertight | Mean Chamfer L2 |
| --- | ---: | ---: |
| held-out proxy, GT count | 0/3 | 0.06385 |
| held-out proxy, predicted count | 0/3 | 0.06397 |
| predicted + cleanup | 0/3 | 0.06709 |
| token repair + cleanup | 0/3 | 0.07712 |

The contact sheet shows a coarse collapsed large-triangle shape rather than a
faithful repair of the fragment ring:

```text
artifacts/research_proofs/2026-05-03/face_level_proxy_fixture_smoke/face_level_proxy_fixture_contact_sheet.png
```

This is the important conclusion:

```text
FACE-lite does not convert fragmented raw generator outputs into clean artist meshes by itself.
The target side of FACE training must be a manifoldized/editable reference.
Raw TRELLIS/MeshRipple/UltraShape fragments can be conditioning diagnostics, not supervision.
```

The new strict dataset gate confirms the current decimated proxy fixtures are
not valid artist-mesh targets:

```text
face_proxy_fixtures_decimated_face_tokens strict gate: 0/8 passing
proxy diagnostic profile: 8/8 reporting-only
```

Production rule added:

```text
Strict FACE target gate before training/promotion:
- decoded_watertight = true
- token_watertight_edge_graph = true
- token_boundary_edge_count = 0
- token_nonmanifold_edge_count = 0
- token_edge_pairing_ratio >= 0.999
```

This moves us toward the right unified path:

```text
TRELLIS.2 proxy
  -> UltraShape/manifoldization/reference builder
  -> strict FACE target gate
  -> FACE topology proposal
  -> feature-aware projection
  -> Blender promotion gate
```

## Implementation Update: Strict FACE Targets

Added a strict target builder so FACE supervision no longer has to learn from
fragmented generator proxies:

```text
scripts/research/prepare_face_strict_targets.py
```

Local strict-target preparation from the decimated proxy fixtures:

```text
input: artifacts/research_proofs/2026-05-03/face_proxy_fixtures_decimated/meshes
output: artifacts/research_proofs/2026-05-03/face_strict_targets
engine: convex_hull
target_faces: 1400
accepted: 8/8
strict dataset gate: 8/8 passing
```

Thunder strict-target FACE smoke:

```text
DOWNLOAD_DIR=artifacts/research_proofs/2026-05-03/face_level_strict_target_smoke
LOCAL_FIXTURE_DIR=artifacts/research_proofs/2026-05-03/face_strict_targets
RUN_DIR=/tmp/clearmesh_face_level_strict_target_smoke
FIXTURE_TEST_COUNT=3
SYNTHETIC_COUNT=24
MAX_FACES=1400
POINT_SAMPLES=512
TRAIN_POINT_SAMPLES=256
STEPS=800
BATCH_SIZE=2
HIDDEN_SIZE=128
LAYERS=2
HEADS=4
CONDITION_TOKENS=6
PAIR_SAMPLES=400
scripts/thunder/face_level_proxy_fixture_smoke.sh 0
```

Observed result:

```text
train strict gate: 29/29 passing
test strict gate: 3/3 passing
best_loss: 0.1161209866
predicted_count mean_chamfer_l2: 0.0000317999
predicted_count watertight: 0/3
predicted_count_cleanup watertight: 1/3
predicted_count_token_repair_cleanup watertight: 1/3
token boundary edges after repair: 10.67 mean
token nonmanifold edges after repair: 0 mean
```

Contact sheet:

```text
artifacts/research_proofs/2026-05-03/face_level_strict_target_smoke/face_level_strict_target_contact_sheet.png
```

Interpretation:

```text
Strict targets fix the training-data contract.
They do not by themselves fix topology closure.
The current raw-XYZ face decoder can be geometrically close while still leaving
boundary holes or producing visually wrong held-out topology.
```

Next FACE reproduction step:

```text
FACE-lite v2 should not merely increase steps/model size.
It should switch to a topology-indexed decoder:
  quantized vertex table
  triangle index reuse
  constrained edge-pair decoding
  deterministic final weld/project gate
```

This is the smoking gun from the strict smoke: the current model knows where the
surface is, but it does not yet know the mesh graph strongly enough. Production
FACE must make graph closure part of the representation, not only a post-process.

## Implementation Update: FACE-Lite v2 Indexed Topology

Added a topology-indexed FACE-lite representation:

```text
clearmesh/mesh_heads/face_indexed.py
scripts/research/train_face_indexed_conditioned_tiny.py
scripts/research/eval_face_indexed_conditioned_tiny.py
scripts/thunder/face_indexed_v2_smoke.sh
tests/test_face_indexed.py
```

The v2 representation stores:

```text
quantized vertex table: V x 3
triangle index sequence: F x 3
```

This changes vertex reuse from an implicit coordinate-welding hope into an
explicit graph-token contract. It is not enough by itself to guarantee perfect
surfaces, but it gives the decoder the right object to learn: topology.

First Thunder smoke on 2026-05-03:

```text
synthetic_count: 10 mixed_cycle shapes
steps: 300
hidden_size: 96
layers: 2
heads: 4
best_loss: 0.3452322781
best_step: 300
eval limit: 6
watertight raw: 2/6
watertight cleanup: 2/6
mean_chamfer_l2 raw: 0.039936
mean_edge_pairing_ratio raw: 0.770321
```

Artifact contact sheet:

```text
artifacts/research_proofs/2026-05-03/face_indexed_v2_smoke/face_indexed_v2_contact_sheet.png
```

Interpretation:

```text
The indexed representation is the right direction: simple cases close quickly.
The current decoder is still not production quality on higher-topology shapes.
Failures now come from illegal face combinations over legal vertex indices,
not from coordinate-weld misses.
```

Next representation upgrade:

```text
edge-state-constrained decoding
  - maintain boundary/nonmanifold edge counts during sampling
  - mask candidate face triples that create nonmanifold edges
  - bias next faces toward closing existing boundary edges
  - optionally decode from edges/corners rather than independent triangle indices
```

## Implementation Update: Edge-Constrained Indexed Decoding

Added edge-state constrained decoding for FACE-lite v2 indexed meshes:

```text
IndexedDecodeState
select_constrained_indexed_face
```

The constrained selector evaluates whole triangle candidates instead of accepting
independent per-corner argmax indices. It tracks current edge use and:

```text
- rejects degenerate triangles
- rejects duplicate faces
- avoids edges already used by two faces
- rewards faces that close existing boundary edges
- lightly penalizes opening new edges
```

A/B on the same Thunder checkpoint and dataset:

```text
unconstrained:
  watertight: 2/6
  mean_boundary_edges: 52.17
  mean_nonmanifold_edges: 60.33
  mean_edge_pairing_ratio: 0.7703
  mean_chamfer_l2: 0.03994

edge constrained, first setting:
  watertight: 2/6
  mean_boundary_edges: 9.17
  mean_nonmanifold_edges: 9.17
  mean_edge_pairing_ratio: 0.9567
  mean_chamfer_l2: 0.05160

edge constrained, best small sweep:
  top_k: 24
  closure_bonus: 5.0
  new_edge_penalty: 0.1
  require_boundary_closure_after: 1
  watertight: 3/6
  mean_boundary_edges: 4.33
  mean_nonmanifold_edges: 4.33
  mean_edge_pairing_ratio: 0.9537
  mean_chamfer_l2: 0.04818
```

Best constrained contact sheet:

```text
artifacts/research_proofs/2026-05-03/face_indexed_v2_best_constrained_eval/face_indexed_v2_best_constrained_contact_sheet.png
```

Interpretation:

```text
Constrained decoding materially improves topology on the same tiny checkpoint.
It does not solve production quality by itself.
The remaining problem is not vertex reuse; it is global face ordering and shape
coverage under an independent-corner face-index decoder.
```

Next model change should be training-time alignment with the constrained decode:

```text
- add edge-state teacher labels
- add closure/nonmanifold candidate losses
- optionally decode boundary-edge + third-vertex actions instead of independent triples
```

## FACE Paper Encoding/Decoding Clues Checked

Re-read the FACE methods and ablations from `/Users/Ashar/Downloads/2603.01515v2.pdf`.
Useful clues:

```text
1. Face order matters, and simple ZYX spatial ordering beats DFS/BFS.
2. FACE embeds each full 9D face as one transformer token.
3. FACE does not decode all 9 coordinates independently.
4. Its CausalMLP decodes coordinates inside each face autoregressively.
5. Its ablation says CausalMLP beats parallel and attention-based coordinate heads.
6. Inference is deterministic top-1 autoregressive sampling.
7. The reported production-ish setup uses 8192 points+normals, 2048 VecSet tokens, 128 coordinate bins, <4000-face meshes, and 100K steps on 8 A100 80GB GPUs.
```

What maps cleanly to ClearMesh:

```text
- Keep ZYX ordering. We already do this.
- Keep deterministic top-1 evaluation. We already do this.
- Avoid graph traversal as the primary ordering. The paper's ablation argues against BFS/DFS for FACE-style face tokens.
- Test within-face causal decoding rather than assuming independent corner logits are enough.
```

What does not map directly:

```text
FACE predicts 9 coordinate tokens, not explicit indexed topology.
Our indexed v2 variant is intentionally off-paper because production needs hard vertex reuse and topology gates.
```

Added an optional corner-causal indexed head inspired by FACE's CausalMLP:

```text
--corner-head causal
--corner-decode auto|causal|parallel
```

Thunder mini-test, same 10 mixed synthetic shapes and 300 steps:

```text
parallel indexed + edge constraints, previous best:
  best_loss: 0.3452
  watertight: 3/6
  mean_boundary_edges: 4.33
  mean_chamfer_l2: 0.04818

corner-causal indexed + edge constraints:
  best_loss: 0.1332
  raw watertight: 0/6
  cleanup watertight: 1/6
  mean_boundary_edges: 7.00
  mean_chamfer_l2: 0.05430
```

Interpretation:

```text
The paper clue is real for FACE's coordinate-token decoder, but it did not help our indexed topology variant at tiny scale.
Lower teacher-forced loss did not translate into better closed meshes.
Do not promote corner-causal indexed decoding as the default yet.
Keep it as an experimental switch for larger runs or a redesigned boundary-edge action decoder.
```

Default remains:

```text
corner_head: parallel
sampling: edge_constrained
constraint_top_k: 24
closure_bonus: 5.0
new_edge_penalty: 0.1
require_boundary_closure_after: 1
```
