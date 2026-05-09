# FACE Watertightness Hostile Audit

Date: 2026-05-06

Scope:

- Paper-faithful FACE ARAE lane:
  - `clearmesh/mesh_heads/face_paper.py`
  - `clearmesh/mesh_heads/face_tokens.py`
  - `scripts/research/train_face_paper_faithful.py`
  - `scripts/research/eval_face_paper_faithful.py`
  - `scripts/research/build_face_token_dataset.py`
  - `scripts/research/prepare_face_strict_targets.py`
  - `scripts/research/assess_face_paper_scale_readiness.py`
  - `scripts/thunder/face_paper_train_eval_job.sh`
  - `configs/face_paper_profiles.json`
- Existing evidence docs:
  - `docs/face_scale_readiness.md`
  - `docs/face_paper_faithfulness_checklist.md`
  - `docs/face_full_hostile_audit_2026-05-04.md`
  - `docs/face_watertight_iteration.md`
- Paper source checked:
  - https://arxiv.org/html/2603.01515v2

## Bottom Line

The implementation is structurally close enough to the FACE paper that it is
worth continuing, but the current watertightness misses are not mysterious.
They are the expected outcome of asking an unconstrained coordinate-token
autoregressive model to learn exact edge pairing from a small, filtered,
distribution-shifted corpus under reduced capacity and often non-paper
evaluation shortcuts.

The most important sentence:

```text
FACE has no explicit watertightness guarantee; watertightness emerges only if
the learned sequence distribution repeatedly emits matching quantized edges.
```

That means one wrong coordinate bin on either end of one edge can turn a closed
mesh into an open mesh. Current results show the model can memorize clean
targets, but has not learned a robust held-out edge-pairing manifold.

## Evidence Snapshot

Healthy plumbing signals:

- Tokenizer roundtrips cubes and paper-token fixtures as watertight.
- Incremental hidden generation matches full causal hidden in unit tests.
- Greedy CausalMLP slotwise decode matches slotwise logits in unit tests.
- Muon/AdamW grouping test passes.
- No-augmentation tiny and real overfit probes can reach watertight output.

Current negative signals:

- Strict110 paper-augmentation A100 probe:
  - train teacher-forced watertight: 0/88
  - train AR watertight: 0/10
  - test AR watertight: 0/10
- Strict110 no-augmentation A100 probe:
  - train teacher-forced accuracy: 0.99898
  - train teacher-forced watertight: 39/88
  - train full-face AR watertight: 2/10
  - test full-face AR watertight: 0/10
  - test mean boundary edges: 983.2
- Objaverse++ 200 / 512-bin ladder:
  - train teacher-forced accuracy after correction: about 0.4239
  - test teacher-forced accuracy: about 0.0314
  - train/test AR watertight: 0/5 and 0/5

Interpretation:

```text
The model can learn the circuit.
The current run settings do not yet learn the mesh-topology distribution.
```

## High-Confidence Reasons Watertightness Is Missing

### 1. The paper representation does not encode topology explicitly

FACE predicts each triangle as nine quantized coordinates. Shared vertices and
paired edges are implicit consequences of emitting the exact same coordinate
triples later.

Code:

- `decode_paper_face_tokens_to_mesh` welds identical quantized coordinates with
  `np.unique`, then builds faces from the inverse map.
- No vertex table, edge table, half-edge state, or closure constraint exists in
  the paper lane.

Why this hurts:

```text
Coordinate equality is the topology carrier.
Coordinate mistakes are topology mistakes.
```

### 2. A one-bin coordinate error can create a boundary edge

The decoder welds only exactly identical quantized vertices. If one reused
corner is off by one bin, the intended shared edge is no longer shared.

The paper's 128-bin setup makes this statistically easier than 512 bins, but
128 bins also collapses nearby vertices in dirty or fine targets. That is why
the repo saw higher token pass rates at 512 bins, while training became harder.

### 3. The loss has no direct edge-pairing term

`_compute_loss` is coordinate-token cross entropy. It does not know whether a
predicted coordinate closes an existing boundary edge, starts a new island, or
creates a duplicate face.

This is paper-faithful, but it explains the metric gap: optimizing CE can lower
local coordinate error while still leaving many open edges.

### 4. Teacher forcing hides exposure bias

Teacher-forced eval gives the model the true previous face. AR eval feeds the
model its own previous face.

The gap matters because a single early wrong triangle changes the prefix. Every
later face is then conditioned on an off-manifold mesh history.

This is visible in the repo evidence:

- no-aug train TF can be near-perfect while full-face train AR is only 2/10
  watertight.
- held-out AR collapses even when train TF is high.

### 5. Held-out accuracy is far too low for topology closure

Watertightness requires many exact local decisions. A held-out teacher-forced
coordinate accuracy of 3-7% is not a near miss; it is a signal that the model
does not know the held-out sequence distribution.

Even 42% train coordinate accuracy is nowhere near enough for exact edge
pairing over hundreds of faces.

### 6. Current paper-lane scale is below the reported paper regime

The FACE ARAE paper reports:

- 500M parameters
- 8192 point samples with normals
- 2048 VecSet tokens
- encoder 8 layers at hidden 768
- decoder 24 layers at hidden 1024
- about 130k Objaverse meshes under 4000 faces
- 100k steps on 8 A100 80GB GPUs

Current bounded profiles often use:

- hidden 128-384
- encoder 2-4 layers
- decoder 2-8 layers
- 64-512 VecSet tokens except strict paper-knob probes
- 512 face cap
- 110-285 passing samples in the recent bounded rungs

That is not a small difference. It changes the statistical regime.

### 7. The corpus is too small for the topology prior being requested

FACE has no hard closure rule, so it learns closure from data. A 109/27 or
228/57 split can prove the code path and training curve, but not robust
topological generalization.

The readiness gate is right to require held-out AR improvements before scale
promotion.

### 8. The target distribution is off-paper

The current strict target path creates legal training meshes through
manifoldization / voxel shell / cleanup / fallback. This is practical for
ClearMesh, but it is not necessarily the same distribution as the paper's
curated artist-style meshes under 4000 faces.

Likely effects:

- voxel-shell topology can be more regular but less artist-like.
- convex-hull fallback can erase real topology.
- simplification can introduce non-manifold artifacts.
- the model may learn the target-conversion style, not a general mesh prior.

### 9. Simplification can break watertight targets before training

The docs already record that voxel shells were watertight before simplification,
while aggressive quadric simplification introduced non-manifold edges and forced
fallbacks.

That means some failures are target-prep failures, not model failures.

### 10. The tokenizer drops faces after quantization

`canonicalize_mesh_faces_paper_zyx` removes:

- duplicate-index faces
- quantized duplicate-vertex faces
- colinear quantized faces

This is sensible, but if removed faces are not topologically redundant, the
decoded teacher target can expose holes. The model then trains/evals on a
different topology than the input source mesh.

### 11. 128-bin paper faithfulness and production watertightness conflict

The paper uses 128 bins for the base model. The repo found 512 bins improved
strict token pass rate by avoiding quantized cracks.

But 512 bins increases the per-coordinate vocabulary and initial CE from
log(128) to log(512). That makes learning exact coordinate reuse harder under
the same data and step budget.

So:

```text
128 bins can create quantization cracks.
512 bins can create learning sparsity.
Both can hurt watertightness in different ways.
```

### 12. Online augmentation churns the target sequence

Paper augmentation applies random rotation, flipping, and per-axis scaling.
Because FACE ordering is spatial ZYX, online SO3/flips can radically reorder the
face sequence every step.

At large paper scale this may be regularization. At tiny scale it is a moving
target. The strict110 probe shows exactly this: paper-augmentation run is a
clean no-scale signal, while no-augmentation memorization proves capacity.

### 13. Augmentation is built from quantized tokens, not original meshes

`_augment_sample` reconstructs face vertices from the quantized token sequence,
applies affine transforms, renormalizes, then re-tokenizes.

That means augmentation compounds quantization/dequantization artifacts. It is
paper-intent-aligned, but it is not necessarily identical to augmenting the
source mesh before tokenization.

### 14. Mirrored augmentation changes winding and face identity pressure

The augmentation path flips face winding when the affine determinant is
negative. That protects normals, but it also means the within-face vertex order
can change under augmentation while the paper leaves within-face order
under-specified.

This is a plausible source of extra entropy for the CausalMLP.

### 15. Within-face order remains under-specified

The code defaults to `within_face_order="preserve"` for the paper-token path.
The paper defines face sorting by minimum-coordinate vertex, but does not define
the order of the three vertices inside each face.

Preserving exporter order keeps winding, but may inject arbitrary per-asset
noise. Rotating each face to the minimum ZYX vertex may lower entropy, but is
still an ablation, not a known paper fact.

### 16. Face ordering is simple spatial ZYX, not boundary-growth

The paper's ZYX order is faithful, but not guaranteed to be closure-friendly.
The indexed sidecar found boundary-growth order eliminated zero-closure jumps
for constrained decoding. The paper lane deliberately does not use that.

So a faithful paper run can be worse for our watertight metric than an
off-paper indexed/boundary-growth route.

### 17. Coordinate CE does not weight topology-critical coordinates

All valid coordinates receive equal loss weight. A coordinate that closes two
open edges is not treated as more important than a coordinate on a new isolated
triangle.

This is faithful to Eq. 6, but hostile to watertightness as a metric.

### 18. EOS / face-count handling is auxiliary and still not paper-core

The paper under-specifies termination. The current eval commonly uses
ground-truth face count to isolate reconstruction quality.

That is fair for paper-lane diagnostics, but product watertightness also needs
termination to be correct. Stopping too early creates holes; stopping too late
creates duplicate/non-manifold faces.

### 19. GT face-count AR is not production inference

The scale-readiness script explicitly warns when eval uses `face_count_mode=gt`.
GT-count AR tells us whether the model can complete a known-length sequence; it
does not tell us whether it can decide when a mesh is complete.

### 20. Truncated AR metrics can be falsely optimistic or pessimistic

Older AR-128 probes did not evaluate full meshes. The readiness gate now blocks
truncated samples. That is correct because watertightness is a global property:
prefix closure says little about the full mesh.

### 21. Point sampling can miss thin structures

The paper itself lists thin/fine structures as a limitation of point-cloud
conditioning. If surface sampling misses cables, spokes, handles, or small
holes, the decoder has no information source for them.

This affects both geometry and topology closure.

### 22. The build script uses the first sampled points

Training and eval cap point count via `points[:point_samples]`. The dataset
builder samples once per shard. If the saved point order is not already
stratified enough, smaller point caps inherit a fixed prefix bias instead of a
fresh random sample.

This is especially relevant to smaller ladder profiles.

### 23. Normals inherit source/target quality

Surface normals are sampled from mesh face normals. Dirty winding,
manifoldization artifacts, or simplification artifacts can give the encoder a
confusing local signal.

The decoder may then learn plausible geometry without exact edge reuse.

### 24. Shape2VecSet implementation is dependency-light, not exact official code

The code follows the 3DShape2VecSet topology, but it is local:

- deterministic pure PyTorch FPS
- additive normal embedding
- local attention implementation
- local FFN/LayerNorm details

This should not explain catastrophic failure alone, but it is still a parity
gap before claiming paper reproduction.

### 25. The "native" encoder path remains available

The strict path defaults to `shape2vecset`, but the implementation still has a
`native` encoder. Any run accidentally using `native` weakens paper parity and
can reduce topology quality because the paper relies heavily on VecSet
conditioning.

### 26. VecSet token count is a major conditioning bottleneck

The paper uses 2048 VecSet tokens. Ladder profiles use fewer for cost. Every
decoder layer cross-attends to VecSet; weak VecSet scale weakens every face
decision.

Topology closure is not a local-only problem. It needs global shape context.

### 27. The CausalMLP is an approximation of unpublished FACE code

The code uses `legacy_concat` as the closest public TreeMeshGPT-style match.
That is defensible, but not exact. If CausalMLP is the paper's key ablation
winner, implementation details matter.

### 28. The CausalMLP can still optimize local coordinates without global edge closure

The CausalMLP only sees hidden face state and previous coordinates within the
same face. It does not directly score whether the new face closes an existing
edge elsewhere in the prefix.

### 29. Greedy top-1 has no recovery path

The paper uses deterministic top-1 AR inference. That means one wrong face is
accepted permanently. There is no beam, repair-aware reranking, or closure
constraint in the paper-faithful lane.

Top-1 is faithful; it is also brittle before the learned distribution is very
strong.

### 30. The model has no hard duplicate-face prevention

Duplicate faces are only discovered after generation by topology reporting or
mesh decode. The model can emit duplicates, which can create non-manifold edge
uses or leave intended regions uncovered.

### 31. The model has no hard non-manifold prevention

Edges can be used more than twice. The paper lane does not track edge counts
during decode, so the sampler cannot forbid this.

### 32. The model has no hard component/connectivity tracking

The sampler can jump to a separate region at any face step. Sometimes that is
valid for multi-component targets; often it creates islands or incomplete
shells.

### 33. Decoded watertightness and token edge-graph watertightness can diverge

The token report checks quantized edge pairing. `trimesh.is_watertight` checks
the decoded floating mesh. Degenerate removal during decode can change the
actual mesh after token-level reporting.

Both metrics are useful, but mismatches should be treated as failure cases, not
noise.

### 34. `trimesh` processing is deliberately minimal

Decode uses `process=False` by default and only removes nondegenerate faces and
unreferenced vertices. That is good for paper evidence because it avoids repair
masking, but it means raw model output gets little help.

### 35. Evaluation compares generated output to decoded teacher, not original source

`eval_face_paper_faithful.py` decodes both generated tokens and teacher tokens.
Pair metrics are generated-vs-teacher-token-mesh, not generated-vs-original raw
asset. This isolates token learning, but can hide target-prep drift from source
geometry.

### 36. Curation accepts non-watertight raw candidates

`build_face_training_corpus.py` scores raw watertightness positively but does
not require it. Downstream strict target generation is expected to repair. That
is a rational pipeline, but it means raw corpus distribution and strict target
distribution can differ sharply.

### 37. Cheap metadata mode marks watertightness false

The cheap GLB/GLTF report path sets `watertight_raw=false` and
`winding_consistent_raw=false`. If used heavily, curation scoring may under-rank
actually clean assets or choose assets on incomplete geometry signals.

### 38. Strict target fallback can simplify the task too much

`prepare_face_strict_targets.py` defaults to `engine=convex_hull` in the script,
while Thunder jobs may override it. A convex hull target is watertight, but it
destroys concavities and holes. It can help topology legality while harming the
shape/topology prior FACE needs.

### 39. The script-level defaults are not paper defaults

`face_paper_train_eval_job.sh` defaults to:

- point samples 1024
- hidden 128
- decoder layers 2
- VecSet tokens 64
- AdamW
- fp32
- augmentation disabled

These are smoke defaults, not paper defaults. Any result from this wrapper must
be labeled by profile.

### 40. The A100 paper-knob probe still caps faces at 512

`paper_knob_a100_probe` restores many paper settings, but keeps max faces 512.
That is useful for bounded validation, not equivalent to the paper's under-4000
face training regime.

### 41. Batch size and step budget affect topology rhythm

Many runs use batch size 1-2 and 5k-18k steps. The paper uses 100k steps on a
large corpus. The model may learn local coordinate statistics before it learns
global face-order rhythm and closure.

### 42. Selection loss is unaugmented dataset loss

When `selection_eval_every` is enabled, checkpoint selection uses unaugmented
dataset loss. That helps memorize clean targets, but it may select checkpoints
that are not best under online paper augmentation or free-running topology.

### 43. Loss-only checkpointing is not topology-aware

Best checkpoint is selected by CE-style loss, not AR watertightness. This is
reasonable for training cost, but it can choose a lower-loss checkpoint with
worse closure.

### 44. Current scale gate admits reduced capacity as warning

`assess_face_paper_scale_readiness.py` warns, rather than blocks, if capacity is
below the paper-scale profile. That is appropriate for rung promotion, but any
watertightness failure at reduced capacity has an easy explanation: the model is
not the paper model.

### 45. Post-repair watertightness is not FACE evidence

Indexed sidecar experiments can produce watertight meshes after boundary fill or
unpinch cleanup. The paper lane correctly quarantines these as product
experiments. If a metric uses post-repair watertightness, it is not measuring
raw FACE paper performance.

### 46. Boundary fill can hide hundreds of raw boundary edges

The indexed sidecar found held-out outputs that were watertight after fill but
had hundreds of raw boundary edges before fill. That is a warning for every
FACE-style pipeline: repair can make the headline metric look green while the
decoder has not learned closure.

### 47. Edge-watertightness is weaker than vertex-link manifoldness

The indexed experiments found bow-tie/pinched vertices even when edge
watertightness was green. The paper-token topology report currently focuses on
edge counts. A true production watertight/editability gate needs vertex-link
checks too.

### 48. Visual geometry can fail after topology passes

The sidecar showed closed but spiky/collapsed outputs. This matters because
watertightness alone is not the product metric; a watertight blob with bad
normal consistency is still a failure.

### 49. Thin structures are a known paper limitation

The FACE paper explicitly notes that reliance on input point clouds can miss
extremely fine or thin structures. ClearMesh stress cases include handles,
cables, holes, branches, and small appendages. These are exactly hostile cases
for point-conditioned topology generation.

### 50. The paper's strongest visual claims use even larger internal scale

The paper's scaling section reports a 1.2B model, 65,536 points, 1024-bin
quantization, and 380k internal high-quality meshes. We should not expect the
small public-corpus lane to match those visuals or topology behavior.

## Code-Level Findings

### Finding A: paper-token topology is structurally unconstrained

Files:

- `clearmesh/mesh_heads/face_tokens.py`
- `clearmesh/mesh_heads/face_paper.py`

Impact:

The decode path welds equal coordinates after the fact. The model does not
sample from a topology state. This is the single biggest logical reason raw FACE
outputs miss watertightness.

### Finding B: augmentation likely makes tiny runs harder than necessary

File:

- `scripts/research/train_face_paper_faithful.py`

Impact:

Online augmentation re-tokenizes every sample, which changes ZYX face order. At
small data/model scale, this can prevent the model from ever learning a stable
face sequence. No-aug memorization passing while paper-aug probes fail is exactly
consistent with this.

### Finding C: smoke wrapper defaults are easy to misread as FACE defaults

File:

- `scripts/thunder/face_paper_train_eval_job.sh`

Impact:

The default settings are smoke settings. They should not be used to explain
paper-level failure except as "we did not run paper-level settings."

### Finding D: target prep can silently change the topology task

Files:

- `scripts/research/prepare_face_strict_targets.py`
- `scripts/research/build_face_token_dataset.py`

Impact:

Strict target generation, simplification, and quantized face dropping can change
the supervised target. This is useful for legality, but it makes the learned
topology prior different from the raw asset topology.

### Finding E: scale-readiness gate is doing the right thing

File:

- `scripts/research/assess_face_paper_scale_readiness.py`

Impact:

The gate blocks for the right reasons: paper knob mismatch, truncated AR, low
train AR watertightness, low held-out edge pairing, and weak held-out
teacher-forced accuracy. Do not weaken it to make metrics look better.

## What Not To Claim

Do not claim:

- FACE is disproven.
- The tokenizer is dead.
- The incremental AR cache is broken.
- The CausalMLP is bad because an indexed sidecar failed.
- Post-repair watertightness is paper-faithful FACE success.
- A no-augmentation memorization run is paper-scale evidence.
- A 512-bin production run is paper-faithful base-model evidence.

Safe claim:

```text
Our implementation can memorize and reconstruct clean targets, but raw
paper-faithful FACE generation has not yet learned robust held-out edge pairing
under current data, capacity, augmentation, and target-prep regimes.
```

## Best Next Diagnostic Ladder

1. Keep strict paper lane separate from indexed/product lane.
2. Run one bounded paper-knob no-aug overfit until train full-face AR is near
   perfect.
3. Add frozen/offline augmentation copies and verify train full-face AR remains
   watertight.
4. Reintroduce online paper augmentation only after the above passes.
5. Compare 128 vs 512 bins on the same strict targets with the same model/steps.
6. Compare `preserve` vs `rotate_min_zyx` within-face order.
7. Require full-face AR, not AR prefixes.
8. Log token edge pairing, decoded watertightness, non-manifold vertices, and
   repair burden separately.
9. Build a 1k-5k curated watertight corpus before jumping to 130k.
10. Treat post-repair or indexed boundary-fill wins as product research, not
    FACE paper evidence.

## Final Hostile Verdict

The watertightness misses have many legitimate explanations:

```text
implicit topology + exact-coordinate welding
+ small corpus
+ reduced capacity
+ target distribution drift
+ quantization tradeoffs
+ online ordering churn
+ exposure bias
+ no edge closure loss
+ no constrained decoding
+ GT-count/truncated eval caveats
= raw outputs that often do not close.
```

That is a coherent failure story. It is also a constructive one: the code path
is not hopeless, but the current metrics are failing for reasons that are
logically aligned with the paper's assumptions and the implementation choices.
