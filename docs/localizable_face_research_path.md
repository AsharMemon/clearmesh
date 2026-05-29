# Localizable FACE Research Path

This document captures the active ClearMesh research lane for importing the
most useful LATTICE insight into FACE without rerunning the old FACE-Q baseline.

## Current Direction

The selected direction is **VoxSet-conditioned FACE**, not voxel-patch FACE.

The reviewer correction is important: FACE is not denoising a global latent in
the dark. FACE already has global shape context and an autoregressive face
stream. The tractable LATTICE-style improvement is to make the conditioning
tokens spatially grounded, then make decoder cross-attention prefer locally
relevant geometry.

Concretely:

1. Keep FACE autoregression intact.
2. Keep one face token per triangle.
3. Keep the existing boundary-growth / `rotate_min_zyx` face ordering.
4. Replace the global point-sampled VecSet condition with voxel-anchored VoxSet
   condition tokens.
5. Add spatial bias or top-k gating inside decoder cross-attention based on the
   current causal face-position proxy.

This is a smaller, cleaner ablation than splitting the output into independent
voxel patches. It asks one direct question:

> Does spatially grounded conditioning help FACE choose locally correct vertices
> and preserve thin/detail structures without disrupting its global topology
> stream?

## Implementation

Active implementation:

- `clearmesh/mesh_heads/face_arae.py`
- `scripts/research/train_face_indexed_conditioned_tiny.py`
- `scripts/research/eval_face_indexed_conditioned_tiny.py`
- `scripts/lambda/bootstrap_lambda_faceq_voxset_gate_sweep.sh`
- `scripts/vast/bootstrap_vast_faceq_smoke.sh`
- `tests/test_face_arae_voxset.py`

Added model options:

- `condition_backend=voxset`
- `decoder_backend=spatial_cross_attn`
- `decoder_backend=spatial_modulated_cross_attn`
- `voxset_resolution`
- `spatial_gate_sigma`
- `spatial_gate_top_k`

The spatial gate is causal. At step `t`, the decoder uses the existing prefix
face position as a proxy for the next face's location. It does not inspect the
target face at `t`. A regression test verifies that changing future input faces
does not change earlier logits.

The newer `spatial_modulated_cross_attn` backend keeps the same causal spatial
mask, but also computes a soft local VoxSet summary from nearby voxel tokens and
injects it into the face token before decoder cross-attention. This is the next
candidate because the first fixed-mask version appeared too weak: it biased
attention, but it did not give the decoder an explicit local geometry feature.

## Deprecated Patch Lane

The earlier voxel-patch FACE branch is no longer the recommended first lane.
Those files may still exist as research artifacts, but they should not be used
as evidence for the current Localizable FACE hypothesis:

- `clearmesh/mesh_heads/localizable_face.py`
- `scripts/research/build_localizable_face_patch_dataset.py`
- `scripts/research/train_localizable_face_patch_tiny.py`
- `scripts/research/eval_localizable_face_patch_tiny.py`
- `scripts/research/verify_localizable_face_patch_dataset.py`

Why deprecated for now:

- It changes the generation problem too much.
- It introduces stitching as a new major failure mode.
- It is less directly comparable to FACE.
- The reviewer-corrected VoxSet/spatial-cross-attention lane is a cleaner,
  lower-risk ablation.

## Completed Matched Smoke

Source shard:

- `face-corpora/paper-large-65k1024-repack8k128-v2/objpp-minquality1/shard0010/lean_face_corpus.tar.gz`

Run prefix:

- `face-runs/faceq-voxset-spatial-ablation/lambda-20260527T053216Z`

Configuration:

- 65,536 point samples
- 1,024 coordinate bins
- `TRAIN_LIMIT=384`
- `STEPS=2000`
- hidden 256
- 4 decoder layers
- 8 heads
- 256 condition tokens
- Muon
- bf16

Compared arms:

- Baseline: `condition_backend=vecset`, `decoder_backend=cross_attn`
- Localizable: `condition_backend=voxset`, `decoder_backend=spatial_cross_attn`,
  `spatial_gate_sigma=0.35`, `spatial_gate_top_k=48`

Training result:

- Baseline final loss: `2.3138`
- VoxSet final loss: `2.3559`
- Baseline best loss: `0.6527` at step 1750
- VoxSet best loss: `0.6268` at step 1750

Held-out teacher-forced result on 12 small test meshes:

- Baseline token accuracy: `0.4781`
- VoxSet token accuracy: `0.4824`
- Baseline face-exact ratio: `0.1589`
- VoxSet face-exact ratio: `0.1383`
- Baseline edge pairing: `0.2948`
- VoxSet edge pairing: `0.3006`
- Baseline normalized Chamfer: `0.00846`
- VoxSet normalized Chamfer: `0.00767`
- Baseline normalized Hausdorff: `0.1825`
- VoxSet normalized Hausdorff: `0.1579`
- Both arms raw watertight: `0/12`

Interpretation:

- The signal is promising but mixed.
- VoxSet/spatial attention slightly improved held-out token accuracy, edge
  pairing, Chamfer, and Hausdorff.
- It did not improve held-out face-exact ratio or raw watertightness in this
  tiny 2k-step setting.
- This supports continued ablation, not immediate production replacement.

## Active Gate Sweep

The first gate sweep tested whether the spatial gate was too narrow or too
broad.

Run prefix:

- `face-runs/faceq-voxset-spatial-gate-sweep/lambda-20260527T055950Z`

Compared arms:

- `baseline_vecset`: VecSet + cross-attention
- `voxset_soft_all`: VoxSet + spatial bias, no hard top-k
- `voxset_tight_k24`: VoxSet + `sigma=0.25`, `top_k=24`
- `voxset_broad_k96`: VoxSet + `sigma=0.50`, `top_k=96`

Decision rule:

- If one VoxSet arm consistently improves held-out token accuracy, geometry
  distance, and edge pairing without worsening topology too much, promote that
  configuration to a longer/larger ablation.
- If gains remain mixed, keep the production FACE-Q path separate and treat
  VoxSet spatial attention as research-only until it proves out.

Result:

- The 2k-step sweep preserved the earlier weak positive signal for held-out
  distance metrics.
- `voxset_tight_k24` had the best held-out token accuracy: `0.4842` versus
  baseline `0.4776`.
- `voxset_broad_k96` had the best held-out Chamfer/Hausdorff:
  `0.00756` / `0.1578` versus baseline `0.00854` / `0.1827`.
- All VoxSet variants had worse held-out face-exact ratio than baseline.
- All variants remained `0/12` raw watertight.

Interpretation:

- The 2k sweep was promising enough to justify one longer promotion gate.
- It was not strong enough to justify production adoption.

## 5k Promotion Gate

Run prefix:

- `face-runs/faceq-voxset-spatial-promote-5k/lambda-20260527T063727Z`

Compared arms:

- `baseline_vecset`: VecSet + cross-attention
- `voxset_soft_all`: VoxSet + spatial bias, no hard top-k

Configuration:

- 65,536 point samples
- 1,024 coordinate bins
- `TRAIN_LIMIT=768`
- `STEPS=5000`
- `EVAL_LIMIT=24`

Held-out teacher-forced result on 24 small test meshes:

- Baseline token accuracy: `0.47018`
- VoxSet token accuracy: `0.46959`
- Baseline face-exact ratio: `0.14875`
- VoxSet face-exact ratio: `0.15137`
- Baseline raw watertight: `1/24`
- VoxSet raw watertight: `0/24`
- Baseline boundary edges: `77.0`
- VoxSet boundary edges: `75.92`
- Baseline nonmanifold vertices: `25.67`
- VoxSet nonmanifold vertices: `26.25`
- Baseline edge pairing: `0.3343`
- VoxSet edge pairing: `0.2914`
- Baseline normalized Chamfer: `0.00554`
- VoxSet normalized Chamfer: `0.00561`
- Baseline normalized Hausdorff: `0.1297`
- VoxSet normalized Hausdorff: `0.1339`

Conclusion:

- The longer 5k gate did **not** validate VoxSet-soft as a better production
  path.
- The earlier 2k held-out geometry gains did not persist under the promoted
  setting.
- Keep VoxSet spatial attention as a research branch, but do not replace the
  production FACE-Q path with it yet.

## Current Code-Only v2

The next architecture has been implemented and tested with a small GPU gate:

- Backend: `spatial_modulated_cross_attn`
- Mechanism: causal face-position proxy -> voxel-distance weights -> local
  weighted VoxSet latent + local relative offset + local distance -> learned
  projection -> residual injection into the face token.
- Safety: the implementation keeps target faces out of the location proxy and
  passes causality tests.

Local smoke checks passed:

- forward/count-head unit tests
- future-face causality tests
- CPU 2-step train smoke
- CPU teacher-forced eval smoke from the saved checkpoint

### 2k Modulated Gate

Run prefix:

- `face-runs/faceq-voxset-spatial-modulated-gate/lambda-20260527T073115Z-a10`

Compared arms:

- `baseline_vecset`
- `voxset_modulated_soft`
- `voxset_modulated_tight_k24`
- `voxset_modulated_broad_k96`

Held-out teacher-forced result on 12 small test meshes:

- Baseline token accuracy: `0.47765`
- Best modulated token accuracy: `0.50128` (`tight_k24`)
- Baseline face-exact ratio: `0.15893`
- Best modulated face-exact ratio: `0.19933` (`tight_k24`)
- Baseline normalized Chamfer: `0.00868`
- Best modulated normalized Chamfer: `0.00712` (`soft`)
- Baseline normalized Hausdorff: `0.1841`
- Best modulated normalized Hausdorff: `0.1613` (`soft`)
- Baseline raw watertight: `0/12`
- Modulated raw watertight: `0/12`

Interpretation:

- This was the first actually positive Localizable FACE result.
- The modulated local residual appears more useful than the fixed spatial mask.
- Topology still did not become clean; boundary/nonmanifold metrics remained
  weak.

### 5k Modulated Promotion Gate

Run prefix:

- `face-runs/faceq-voxset-spatial-modulated-promote-5k/lambda-20260527T082612Z-a10`

Compared arms:

- `baseline_vecset`
- `voxset_modulated_tight_k24`

Held-out teacher-forced result on 24 small test meshes:

- Baseline token accuracy: `0.47057`
- Modulated token accuracy: `0.48178`
- Baseline face-exact ratio: `0.15259`
- Modulated face-exact ratio: `0.15898`
- Baseline raw watertight: `1/24`
- Modulated raw watertight: `0/24`
- Baseline boundary edges: `77.33`
- Modulated boundary edges: `78.38`
- Baseline nonmanifold vertices: `25.63`
- Modulated nonmanifold vertices: `26.50`
- Baseline edge pairing: `0.33839`
- Modulated edge pairing: `0.33423`
- Baseline normalized Chamfer: `0.00558`
- Modulated normalized Chamfer: `0.00568`
- Baseline normalized Hausdorff: `0.1297`
- Modulated normalized Hausdorff: `0.1400`

Conclusion:

- The 5k gate preserved small held-out token/face accuracy gains.
- It did **not** improve watertightness, boundary/nonmanifold topology, edge
  pairing, Chamfer, or Hausdorff.
- This is a real research signal, but not a production promotion signal.
- Stop small single-shard GPU spends here unless the next change directly
  targets topology/geometry consistency.

## Expected Quality Impact

This can plausibly help:

- thin structures
- railings
- windmill blades
- handles
- spokes
- fins
- local protrusions

The mechanism is not magic. It helps only when the conditioning geometry already
contains enough evidence for those structures. It does not directly solve:

- quad topology
- artist edge loops
- raw watertightness
- final repair
- semantic failures from TRELLIS/text-to-image

The realistic target is a small-to-moderate geometry/detail improvement after
tuning and scale, not a guaranteed step-function jump.

## Next Gates

1. Do not spend more GPU time on the old fixed-mask VoxSet-soft setting.
2. Do not promote `spatial_modulated_cross_attn` to production yet.
3. If revisiting, the next change should target topology/geometry consistency:
   topology-aware local residuals, local edge-pairing loss, or spatially gated
   topology heads rather than more generic token loss training.
4. Keep the current implementation and tests available for future ablations.
5. Keep production FACE-Q on the proven baseline unless a future matched
   ablation beats it on held-out token accuracy, topology, and distance metrics.

## Active Topology-Supervised Gate

Run prefix:

- `face-runs/faceq-voxset-spatial-topology-gate/lambda-20260527T092149Z-a10`

Purpose:

- Re-test the best modulated VoxSet setting with the topology auxiliaries turned
  on, instead of judging it mostly from token loss.
- This is intentionally small and bounded: it should answer whether localizable
  conditioning can coexist with FACE-Q's edge/topology supervision before we
  design another architecture.

Compared arms:

- `baseline_vecset`: VecSet + cross-attention
- `voxset_modulated_tight_k24`: VoxSet + spatial-modulated cross-attention,
  `sigma=0.25`, `top_k=24`

Important settings:

- `STEPS=2000`
- `TRAIN_LIMIT=384`
- `TOPOLOGY_LOSS_WEIGHT=0.2`
- `EDGE_ACTION_LOSS_WEIGHT=1.0`
- `EDGE_CHOICE_LOSS_WEIGHT=0.5`
- small teacher-forced eval plus tiny free-run topology eval

Decision rule:

- If VoxSet keeps the token/face gains and improves or matches boundary,
  nonmanifold, edge-pairing, and free-run topology metrics, promote to a longer
  topology-supervised gate.
- If VoxSet still wins token accuracy but loses topology, the next code move is
  not more generic training. It should be a topology-aware local head or local
  edge-pairing objective.

Result:

- Baseline test teacher-forced token accuracy: `0.46160`
- VoxSet-modulated test teacher-forced token accuracy: `0.47896`
- Baseline test teacher-forced face-exact ratio: `0.15900`
- VoxSet-modulated test teacher-forced face-exact ratio: `0.17096`
- Baseline test teacher-forced edge pairing: `0.28668`
- VoxSet-modulated test teacher-forced edge pairing: `0.31103`
- Baseline test teacher-forced normalized Chamfer/Hausdorff:
  `0.00853` / `0.18415`
- VoxSet-modulated test teacher-forced normalized Chamfer/Hausdorff:
  `0.00811` / `0.17184`
- Baseline test teacher-forced raw watertight: `0/12`
- VoxSet-modulated test teacher-forced raw watertight: `0/12`
- Baseline tiny free-run raw watertight: `4/4`
- VoxSet-modulated tiny free-run raw watertight: `4/4`
- Baseline tiny free-run normalized Chamfer/Hausdorff:
  `0.02325` / `0.25392`
- VoxSet-modulated tiny free-run normalized Chamfer/Hausdorff:
  `0.00926` / `0.17424`

Interpretation:

- Turning on edge-action and edge-choice supervision did not erase the
  Localizable FACE gains. That is the strongest evidence so far that the
  reviewer-corrected VoxSet conditioning direction is worth continuing.
- The win is not production-proof yet: teacher-forced raw topology still has
  boundary/nonmanifold errors in both arms, and the free-run eval is intentionally
  tiny.
- Next bounded gate should be a longer topology-supervised run, not a new giant
  production run. If the same trend holds, the next architecture change should
  make topology heads explicitly local-aware.

## 5k Topology-Supervised Promotion Gate

Run prefix:

- `face-runs/faceq-voxset-spatial-topology-promote-5k/lambda-20260527T095449Z-a10`

Compared arms:

- `baseline_vecset`
- `voxset_modulated_tight_k24`

Important settings:

- `STEPS=5000`
- `TRAIN_LIMIT=768`
- `EVAL_LIMIT=24`
- `TOPOLOGY_LOSS_WEIGHT=0.2`
- `EDGE_ACTION_LOSS_WEIGHT=1.0`
- `EDGE_CHOICE_LOSS_WEIGHT=0.5`

Held-out teacher-forced result on 24 test meshes:

- Baseline token accuracy: `0.46767`
- VoxSet-modulated token accuracy: `0.47179`
- Baseline face-exact ratio: `0.14956`
- VoxSet-modulated face-exact ratio: `0.15259`
- Baseline raw watertight: `1/24`
- VoxSet-modulated raw watertight: `0/24`
- Baseline edge pairing: `0.34528`
- VoxSet-modulated edge pairing: `0.34239`
- Baseline normalized Chamfer/Hausdorff: `0.00576` / `0.13571`
- VoxSet-modulated normalized Chamfer/Hausdorff: `0.00575` / `0.13800`

Tiny free-run topology result on 6 held-out meshes:

- Baseline raw watertight: `6/6`
- VoxSet-modulated raw watertight: `6/6`
- Baseline generated-face ratio: `0.8651`
- VoxSet-modulated generated-face ratio: `1.0`
- Baseline normalized Chamfer/Hausdorff: `0.01918` / `0.24733`
- VoxSet-modulated normalized Chamfer/Hausdorff: `0.00892` / `0.16209`

Train teacher-forced result on 24 train meshes:

- Baseline token accuracy: `0.85880`
- VoxSet-modulated token accuracy: `0.82234`
- Baseline face-exact ratio: `0.65799`
- VoxSet-modulated face-exact ratio: `0.56771`

Conclusion:

- The held-out token/face gains persisted, but they are small at 5k.
- The strongest VoxSet win is in the tiny constrained free-run geometry probe:
  full watertightness for both arms, but much better generated-face ratio and
  distance metrics for VoxSet.
- The weakest signal is train memorization/topology: baseline still fits the
  train set better and gets one more teacher-forced watertight mesh.
- Do not promote `spatial_modulated_cross_attn` as-is to production.
- The next worthwhile code change is to make local VoxSet context explicitly
  available to topology/edge heads or add a local topology-pairing objective,
  then rerun the same 5k topology gate.

## Topology-Aware Local Head Probe

Implemented backend:

- `decoder_backend=spatial_topology_modulated_cross_attn`

Mechanism:

- Keep the same face-token hidden stream used by the modulated VoxSet backend.
- Feed the local VoxSet residual explicitly into topology, edge-action, and
  edge-choice heads through a small learned projection.
- Keep face logits unchanged so this is a topology-head ablation, not a new
  face-token decoder.

Validation:

- Python compile passed.
- VoxSet unit tests passed: `7 passed`.
- CPU train/eval smoke passed.
- One remote free-run eval shape bug was fixed: when decoding one next face
  from a longer prefix, local topology context is sliced to the hidden length.

Run prefix:

- `face-runs/faceq-voxset-spatial-topology-head-gate/lambda-20260527T105557Z-a10`

Compared arms:

- `baseline_vecset`
- `voxset_modulated_tight_k24`
- `voxset_topology_modulated_tight_k24`

Held-out teacher-forced result on 12 test meshes:

- Baseline token / face-exact: `0.46108` / `0.15900`
- VoxSet modulated token / face-exact: `0.47814` / `0.17096`
- VoxSet topology-modulated token / face-exact: `0.47559` / `0.13774`
- Baseline edge pairing: `0.28384`
- VoxSet modulated edge pairing: `0.31344`
- VoxSet topology-modulated edge pairing: `0.30268`
- Baseline normalized Chamfer/Hausdorff: `0.00847` / `0.18165`
- VoxSet modulated normalized Chamfer/Hausdorff: `0.00815` / `0.17201`
- VoxSet topology-modulated normalized Chamfer/Hausdorff: `0.00823` / `0.18030`

Tiny held-out free-run result on 4 test meshes:

- All three arms raw watertight: `4/4`
- Baseline generated-face ratio: `0.9167`
- VoxSet modulated generated-face ratio: `1.0`
- VoxSet topology-modulated generated-face ratio: `1.0`
- Baseline normalized Chamfer/Hausdorff: `0.02325` / `0.25392`
- VoxSet modulated normalized Chamfer/Hausdorff: `0.00926` / `0.17424`
- VoxSet topology-modulated normalized Chamfer/Hausdorff: `0.01127` / `0.20225`

Conclusion:

- The topology-aware head is technically viable and safe to keep.
- It did not beat the simpler `spatial_modulated_cross_attn` arm on held-out
  token, face-exact, edge pairing, or free-run geometry.
- Current best research arm remains `spatial_modulated_cross_attn`.
- Next improvement should likely be a loss/decoding objective around local
  edge-pairing or boundary closure, not merely injecting local context into the
  existing topology heads.

## Corner Closure Presence Probe

Implemented training knob:

- `--corner-closure-presence-loss-weight`

Mechanism:

- Keep FACE autoregression, face order, and corner-token decoder unchanged.
- For teacher faces that close a currently open boundary edge, compute an
  order-invariant presence loss on the main corner logits.
- The target vertex set is the two endpoints of the teacher boundary edge plus
  the teacher third vertex.
- This connects topology supervision directly to the logits used by AR decode,
  instead of relying only on separate edge-action/edge-choice heads.

Validation:

- Python compile passed.
- Lambda launcher shell check passed.
- VoxSet/localizable unit tests passed: `9 passed`.
- CPU train/eval smoke passed with the new logged metric
  `corner_closure_presence_loss`.

Run prefix:

- `face-runs/faceq-voxset-spatial-corner-closure-gate/lambda-20260527T115400Z-a10`

Compared arms:

- `baseline_vecset`
- `voxset_modulated_tight_k24`

Important settings:

- `STEPS=2000`
- `TRAIN_LIMIT=384`
- `EVAL_LIMIT=12`
- `TOPOLOGY_LOSS_WEIGHT=0.2`
- `EDGE_ACTION_LOSS_WEIGHT=1.0`
- `EDGE_CHOICE_LOSS_WEIGHT=0.5`
- `CORNER_CLOSURE_PRESENCE_LOSS_WEIGHT=0.2`

Held-out teacher-forced result on 12 test meshes:

- Baseline token / face-exact: `0.43880` / `0.13266`
- VoxSet modulated token / face-exact: `0.44854` / `0.13500`
- Baseline raw watertight: `0/12`
- VoxSet modulated raw watertight: `0/12`
- Baseline edge pairing: `0.32821`
- VoxSet modulated edge pairing: `0.27222`
- Baseline normalized Chamfer/Hausdorff: `0.00854` / `0.18144`
- VoxSet modulated normalized Chamfer/Hausdorff: `0.00874` / `0.17991`

Tiny held-out free-run result on 6 test meshes:

- Baseline raw watertight: `6/6`
- VoxSet modulated raw watertight: `6/6`
- Baseline generated-face ratio: `1.0`
- VoxSet modulated generated-face ratio: `0.97222`
- Baseline normalized Chamfer/Hausdorff: `0.00867` / `0.16879`
- VoxSet modulated normalized Chamfer/Hausdorff: `0.01168` / `0.20164`

Train teacher-forced result on 12 train meshes:

- Baseline token / face-exact: `0.64583` / `0.26389`
- VoxSet modulated token / face-exact: `0.67130` / `0.32292`
- Baseline edge pairing: `0.30231`
- VoxSet modulated edge pairing: `0.35326`

Conclusion:

- The closure-presence loss is learnable and mechanically safe.
- It improved VoxSet train token/face/edge metrics and slightly improved
  held-out token/face accuracy.
- It did not improve held-out teacher-forced topology, raw watertightness, or
  tiny held-out free-run geometry in this 2k gate.
- Do not promote this objective as a default production loss yet.
- Keep it as an optional research knob. The best confirmed Localizable FACE
  arm remains the simpler `spatial_modulated_cross_attn` without extra closure
  loss, pending a larger confirmation.

## Boundary-Action Decode Bug Fix

Bug:

- The boundary-edge causal decoder builds a candidate from an open edge plus a
  proposed third vertex.
- It then canonicalizes the candidate with rotate-min face ordering.
- The edge-action bonus was incorrectly applied to `face[2]`.
- After canonical rotation, `face[2]` is not guaranteed to be the original
  third vertex; it can be one of the boundary-edge endpoints.
- Result: the trained edge-action head could be scoring the wrong vertex during
  free-run decode.

Fix:

- Track `candidate_thirds` alongside `candidate_edge_rows`.
- Apply edge-action logits to the original third vertex, not to the rotated
  face slot.
- Patched both single-candidate and beam candidate boundary-edge decode paths.

Validation:

- Added a regression test with boundary edge `(5, 6)` and true third vertex
  `1`, where rotate-min canonicalization produces a face whose slot `2` is an
  endpoint. The patched decoder selects `{1, 5, 6}`.
- Python compile passed.
- VoxSet/localizable unit tests passed: `10 passed`.
- A tiny CPU free-run evaluator smoke with edge-action bonuses completed through
  the patched path; it produced `1/2` raw watertight on an intentionally weak
  local smoke checkpoint, with no decode crash.

Conclusion:

- This is a real decoding correctness fix and should stay.
- It does not require retraining or retokenizing data.
- The next meaningful check is to rerun free-run eval on an existing stronger
  checkpoint with edge-action bonuses enabled before spending on another train
  gate.

## Topology-Local Edge Head Eval Fix

Bug:

- `spatial_topology_modulated_cross_attn` trains closure, edge-action, and
  edge-choice heads on a topology-local hidden state:
  `hidden + spatial_topology_scale * f(hidden, local_vox_context)`.
- Earlier free-run eval already used the public topology-local closure helper,
  but edge-action and edge-choice decode still called the private raw hidden
  helpers directly.
- Result: the topology-aware arm was partially evaluated with a different edge
  head input than the one it trained, especially when boundary-edge decode used
  edge-action and edge-choice bonuses.

Fix:

- Added public `topology_hidden_from_hidden`, `edge_action_logits_from_hidden`,
  and `edge_choice_logits_from_hidden` helpers on the FACE-Q indexed decoder.
- Updated free-run boundary-edge evaluation to pass `point_features`,
  `vertex_table`, and `input_faces` into those helpers.
- Updated `edge_action_next_logits` and `edge_choice_next_logits` so future
  probes also use the topology-local projection.

Validation:

- Python compile passed for the model, train script, and eval script.
- VoxSet/localizable unit tests passed: `11 passed`.
- Added a regression test that checks the next-edge helpers match the
  topology-local public helper outputs for the spatial-topology backend.

Conclusion:

- This is another eval/decode correctness fix, not a new training result.
- It makes the prior topology-aware head comparison more trustworthy only after
  rerunning free-run eval with the patched code.
- The least wasteful next step is an eval-only rerun on the saved topology-head
  checkpoint before launching a fresh training gate.

## Topology Eval-Fix Rerun

Run prefix:

- `face-runs/faceq-voxset-topology-evalfix-rerun/lambda-20260527T124853Z-a10`

Local summary:

- `.codex_outputs/lambda_faceq_voxset_topology_evalfix_rerun_20260527T124853Z_a10/fetched/ablation_summary.json`

Settings:

- A10, bounded gate, then instance terminated.
- `STEPS=2000`
- `TRAIN_LIMIT=384`
- `POINT_SAMPLES=65536`
- `EDGE_ACTION_LOSS_WEIGHT=1.0`
- `EDGE_CHOICE_LOSS_WEIGHT=0.5`
- `RUN_FREE_RUN_EVAL=1`
- Patched boundary-action third-vertex scoring.
- Patched topology-local edge-action/edge-choice eval helpers.

Compared arms:

- `baseline_vecset`: global VecSet, vanilla cross-attention.
- `voxset_modulated_tight_k24`: LATTICE-style VoxSet anchors plus spatially
  modulated cross-attention.
- `voxset_topology_modulated_tight_k24`: same VoxSet/local gate plus
  topology-local edge/closure heads.

Held-out teacher-forced result on 12 test meshes:

- Baseline token / face-exact: `0.46160` / `0.15900`
- VoxSet modulated token / face-exact: `0.47875` / `0.17277`
- VoxSet topology-modulated token / face-exact: `0.47568` / `0.13619`
- Baseline edge pairing: `0.28566`
- VoxSet modulated edge pairing: `0.31557`
- VoxSet topology-modulated edge pairing: `0.30272`
- Baseline normalized Chamfer/Hausdorff: `0.00853` / `0.18415`
- VoxSet modulated normalized Chamfer/Hausdorff: `0.00816` / `0.17201`
- VoxSet topology-modulated normalized Chamfer/Hausdorff: `0.00839` / `0.18507`

Tiny held-out free-run result on 4 test meshes:

- Baseline raw watertight: `4/4`
- VoxSet modulated raw watertight: `4/4`
- VoxSet topology-modulated raw watertight: `4/4`
- Baseline generated-face ratio: `1.0`
- VoxSet modulated generated-face ratio: `0.91667`
- VoxSet topology-modulated generated-face ratio: `1.0`
- Baseline normalized Chamfer/Hausdorff: `0.00991` / `0.20434`
- VoxSet modulated normalized Chamfer/Hausdorff: `0.01147` / `0.17489`
- VoxSet topology-modulated normalized Chamfer/Hausdorff: `0.01082` / `0.21332`

Train teacher-forced result on 12 train meshes:

- Baseline token / face-exact: `0.64468` / `0.26389`
- VoxSet modulated token / face-exact: `0.67940` / `0.32292`
- VoxSet topology-modulated token / face-exact: `0.67708` / `0.33333`
- Baseline edge pairing: `0.32583`
- VoxSet modulated edge pairing: `0.37707`
- VoxSet topology-modulated edge pairing: `0.39570`

Conclusion:

- The decoder/eval fixes are correct and should stay.
- The simple Localizable FACE arm remains the best promoted research arm:
  `condition_backend=voxset`, `decoder_backend=spatial_modulated_cross_attn`,
  tight local gate `k=24`, `sigma=0.25`.
- The topology-local head improved train edge-pairing but did not improve
  held-out token accuracy, held-out edge-pairing, or tiny held-out free-run
  geometry enough to promote.
- Do not spend on another topology-head train gate unless the next test changes
  the question, for example larger data, better local anchors, or a metric that
  specifically rewards edge-flow/topology.
