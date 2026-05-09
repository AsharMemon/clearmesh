# FACE Full Hostile Audit

Date: 2026-05-04
Primary paper: `/Users/Ashar/Downloads/2603.01515v2.pdf`
Related source: [official 3DShape2VecSet repo](https://github.com/1zb/3DShape2VecSet)
Implementation audited:

- `clearmesh/mesh_heads/face_paper.py`
- `clearmesh/mesh_heads/face_tokens.py`
- `scripts/research/train_face_paper_faithful.py`
- `scripts/research/eval_face_paper_faithful.py`
- `scripts/research/build_face_token_dataset.py`
- `scripts/research/prepare_face_strict_targets.py`
- `scripts/data/build_face_training_corpus.py`
- `scripts/thunder/face_paper_train_eval_job.sh`
- `configs/face_paper_profiles.json`

## Bottom Line

The FACE paper does point us toward the current trust-ladder direction, but not toward declaring victory from one fix.

The paper's quality claim depends on the entire coupled system:

```text
clean curated mesh target
  -> surface point cloud with normals
  -> downsampled/FPS VecSet encoder
  -> causal face-level decoder with per-layer VecSet cross-attention
  -> true coordinate-token CausalMLP
  -> end-to-end CE reconstruction training
  -> deterministic top-1 AR sampling after the reconstruction map is learned
```

So the right conclusion from the latest 200-asset run is:

```text
The no-augmentation/AdamW correction proved the training loop is no longer obviously poisoned.
It did not prove FACE mesh quality.
The next proof must be a broad ladder: train-overfit, AR-overfit, heldout improvement, then scale.
```

The paper strongly supports this ladder because its central hypothesis is not "a decoder head will magically close topology." It is that end-to-end ARAE training makes the encoder learn a structured latent `C` from which the decoder can reconstruct meshes. If train teacher-forced and train AR are weak, the latent/decoder contract is not learned yet and heldout mesh quality is not a meaningful test.

## Current Experimental Reality

Latest meaningful Thunder result:

```text
run_dir: /tmp/clearmesh_face_objpp200_512_20260504_020301/face_512bins_noaug_adamw_8000_20260504_030910
corpus: Objaverse++ 200 raw assets -> 156 curated -> 154 strict targets -> 136 strict 512-bin token passes
split: 109 train / 27 test
model: 512 bins, max 512 faces, 2048 point samples, hidden 256, encoder 3 layers, decoder 6 layers, vecset 256, latent 64
training: 8000 steps, AdamW, no online augmentation
```

Training improved sharply versus the online-augmented run:

```text
old online-augmented run:
  final/best selection loss: ~4.668
  train teacher-forced accuracy: ~0.0547

no-augment AdamW run:
  step 1000 selection loss: 4.7985
  step 8000 selection loss: 2.5763
  train teacher-forced accuracy: ~0.4239
```

But mesh quality is still not proven:

```text
train teacher-forced watertight: 0/109
test teacher-forced watertight: 0/27
train AR-128 watertight: 0/5
test AR-128 watertight: 0/5
test teacher-forced accuracy: ~0.0314
```

Hostile interpretation:

```text
This is a recovery signal, not a quality signal.
```

## Paper Variables And Code Crosswalk

| Paper symbol / claim | Paper meaning | Current code | Audit verdict |
| --- | --- | --- | --- |
| `P` | input shape as point cloud | `surface_points + surface_normals` loaded in `train_face_paper_faithful.py` | Structurally faithful; paper method text says `P in R^{m x 3}`, implementation details say 8192 points with normals. We use 6D features. |
| `M` | target triangle mesh | strict/manifoldized GLB -> paper tokens | Conceptually faithful, but our strict voxel-shell targets may not match paper's artist-mesh target distribution. |
| `C` | compact latent VecSet | `Shape2VecSetEncoder` output `(B, k, 64)` | Faithful topology, but local implementation rather than exact imported official code. |
| `F=(f_1,...,f_N)` | ordered face sequence | `paper_tokens` shape `(F, 9)` | Structurally faithful. |
| `f_i=(v_i^0,v_i^1,v_i^2) in R^9` | one face as one token | each face is nine quantized coordinate tokens | Faithful. |
| face ordering | lexicographic ZYX of minimum-coordinate vertex | `canonicalize_mesh_faces_paper_zyx` | Mostly faithful; tie-break and within-face vertex order are under-specified by the paper. |
| Eq. 1 `C' = CrossAttn(Q,K_P,V_P)` | downsampled/FPS query points attend to full point set | `Shape2VecSetEncoder.cross_attend(...)` | Faithful to FACE text and official 3DShape2VecSet encoder style. |
| Eq. 2 `C = TransformerEncoder_LE(C')` | refine query latents | `Shape2VecSetEncoder.layers` then bottleneck | Faithful structure. |
| Eq. 3 `t_{i-1}=MLP_embed(f_{i-1})` | previous face becomes one transformer token | `face_pooling` MLP | Faithful structure; exact MLP depth is under-specified. |
| Eq. 4 `H'_l=CausalSelfAttn(H_l)` | face-level causal self-attention | `FaceDecoderBlock.self_attn` with causal mask | Faithful. |
| Eq. 5 `H_{l+1}=CrossAttn(Q=H'_l,K=C,V=C)` | inject VecSet every decoder layer | `FaceDecoderBlock.cross_attn` | Faithful. |
| `h_i` | latent face vector | decoder hidden at position `i` | Faithful. |
| CausalMLP | intra-face AR coordinate decoder | `LegacyConcatCausalCoordinateMLP` | Closest public-code match to the TreeMeshGPT coordinate head cited by FACE: separate coordinate heads conditioned on previous coordinate-token embeddings. Exact FACE code is still unpublished. |
| Eq. 6 | mean CE over all faces and nine coordinate tokens | `_compute_loss` over `(B,F,9)` | Faithful. |
| top-1 inference | deterministic autoregressive argmax | `_generate_tokens`, `greedy_face_from_hidden` | Faithful for fixed face count; EOS/termination remains our extension. |
| 128 bins | base coordinate quantization `[0,127]` | profiles support 128; production ladder uses 512 | 512 is a deliberate production deviation, useful for quantization cracks but much harder statistically. |
| 8192 points | base encoder input | profiles support 8192; latest run used 2048 | Latest run was under-conditioned versus paper. |
| 2048 VecSet tokens | base VecSet size | profiles support 2048; latest run used 256 | Latest run was much smaller than paper. |
| 500M params | base ARAE scale | latest run tiny | Latest run cannot be judged as paper-quality. |
| 100K steps / 8x A100 | base training budget | latest run 8K steps / 1x A6000 | Not comparable. |
| 130K curated meshes | base data scale | latest strict split 109/27 | Not comparable. |

## Equation-Level Audit

### Problem Formulation

Paper:

```text
p(M | P)
```

Meaning:

```text
Given point-cloud shape P, reconstruct/generate explicit triangle mesh M.
```

Our product implication:

```text
TRELLIS/LATTICE/UltraShape can provide geometry, but FACE is not an image model by itself.
FACE must see a point-cloud condition whose sampled geometry contains the structures we expect it to reconstruct.
```

Risk:

```text
If upstream point sampling misses thin structures, FACE has no magic information source.
The paper itself names thin structures as a limitation.
```

### Encoder Equation 1

Paper:

```text
C' = CrossAttn(Q = Q, K = K_P, V = V_P)
```

Variables:

- `Q`: projections/embeddings of FPS/downsampled query points.
- `K_P`, `V_P`: projections of the full point set.
- `C'`: initial query-token VecSet.

Our code:

```text
query_indices = FPS(point_features[..., :3], vecset_tokens)
sampled_embeddings = point_embed(sampled XYZ) + normal_embed(sampled normals)
point_embeddings = point_embed(all XYZ) + normal_embed(all normals)
x = cross_attend(sampled_embeddings, context=point_embeddings) + sampled_embeddings
```

Verdict:

```text
This is directionally paper-faithful and close to the official 3DShape2VecSet encoder pattern.
```

Open risk:

```text
FACE says it adopts 3DShape2VecSet, but does not publish FACE code.
The official 3DShape2VecSet repo uses torch_cluster.fps, point embeddings, cross-attention, and self-attention layers.
Our FPS is deterministic pure PyTorch and may differ numerically. That should not explain catastrophic quality by itself, but it is a parity gap for final reproduction.
```

### Encoder Equation 2

Paper:

```text
C = TransformerEncoder_LE(C')
```

Our code:

```text
for self_attn, self_ff in layers:
    x = self_attn(x) + x
    x = self_ff(x) + x
C = bottleneck(norm(x))
```

Verdict:

```text
Faithful in topology.
```

Deep risk:

```text
The paper's latent C is not merely a compressed point-cloud descriptor; it is the interface the decoder learns against end-to-end.
If we train too small or on too few meshes, C can become a memorization/codebook shortcut rather than a structured reusable shape latent.
That exactly matches the pattern: train improves, heldout stays bad.
```

### Face Ordering

Paper:

```text
Sort all faces by lexicographical ZYX order of their minimum-coordinate vertex.
```

Our code:

```text
q_vertices_xyz = quantize(vertices)
face_vertex_zyx = q_vertices_xyz[faces][:, :, [2,1,0]]
min_offsets = lexsort inside each face
min_vertices = face_vertex_zyx[face, min_offset]
order = lexsort(min_vertex z,y,x plus full flattened tie-break)
tokens = sorted face vertices emitted as z,y,x
```

Verdict:

```text
Mostly faithful.
```

Risks:

```text
The paper does not define tie-breaks between faces with identical minimum-coordinate vertex.
The paper does not define whether the three vertices inside a face are rotated to start at the minimum vertex.
Our current paper path preserves source within-face order, only sorting the face sequence.
That preserves winding, but it may inject arbitrary local entropy if mesh exporters order triangle vertices inconsistently.
```

Recommended test:

```text
Run a matched tokenizer ablation:
A. preserve within-face order, current path
B. rotate face to min ZYX vertex while preserving winding
C. sort the three vertices ZYX without preserving winding, only as a diagnostic

Judge teacher-forced loss and AR edge-pairing on the same tiny overfit and 109-shard split.
```

### Face Pooling Equation 3

Paper:

```text
t_{i-1} = MLP_embed(f_{i-1})
```

Our code:

```text
face_pooling = Linear(9,H) -> GELU -> LayerNorm(H) -> Linear(H,H)
```

Verdict:

```text
Faithful enough; exact depth is under-specified.
```

Risk:

```text
At 512 bins, we normalize coordinate tokens to continuous [-1,1] before the MLP.
This is reasonable, but it means the face embedding sees continuous-ish bin positions, not learned discrete coordinate embeddings.
The paper says face vector is projected by an MLP, so this is aligned.
```

### Decoder Equations 4 And 5

Paper:

```text
H'_l = CausalSelfAttn(H_l)
H_{l+1} = CrossAttn(Q=H'_l, K=C, V=C)
```

Our code:

```text
self_out = self_attn(norm(x), norm(x), norm(x), causal_mask)
x = x + self_out
cross_out = cross_attn(norm(x), vecset, vecset)
x = x + cross_out
x = x + ff(norm(x))
```

Verdict:

```text
Faithful.
```

Deep risk:

```text
The paper explicitly says global shape context is injected at every decoder layer.
This means weak encoder scale, few VecSet tokens, or low point count directly weakens every face decision.
A model can lower CE locally while still failing topology because each step may be plausible but not globally consistent.
```

### CausalMLP

Paper:

```text
p(c_{i,j} | h_i, c_{i,<j})
```

Our code:

```text
for slot j:
    prefix = pooled embeddings of previous coordinate tokens c_{i,<j}
    logits_j = MLP([h_i, slot_embedding(j), prefix])
```

Verdict:

```text
Correct causal factorization; not guaranteed exact implementation.
```

Important correction from prior work:

```text
The failed indexed CausalMLP was not a valid FACE CausalMLP test.
FACE predicts coordinate bins inside a face.
Our indexed experiment predicted vertex-table indices.
Those distributions are radically different.
```

Open risk:

```text
The paper's Table 5 says CausalMLP massively beats parallel decode.
If our causal coordinate head does not beat parallel decode in a matched coordinate FACE run, then either our CausalMLP implementation differs in an important way, our training regime is too small/noisy, or our tokenization/data is mismatched.
```

### Objective Equation 6

Paper:

```text
L = (1/N) sum_i^N sum_j^9 CE(L_{i,j}, c_{i,j})
```

Our code:

```text
CE(logits.reshape(-1,num_bins), target_faces.reshape(-1), ignore_index=-100)
loss = weighted mean over valid face-coordinate tokens
optional EOS BCE term
```

Verdict:

```text
Coordinate loss is faithful.
EOS is a practical extension, not part of the specified FACE objective.
```

Risk:

```text
The paper under-specifies termination.
Using GT face count is valid for reconstruction-isolation experiments but cannot be our production inference story.
Predicted-count/EOS must be evaluated separately and not mixed with paper reconstruction metrics.
```

## Broad System Audit

### 1. Data Is Probably The Largest Remaining Gap

Paper regime:

```text
~130,000 curated Objaverse meshes
fewer than 4,000 faces
surface point clouds with normals
```

Our current meaningful real regime:

```text
109 train / 27 test strict shards
512-face cap
strict voxel-shell/manifoldized targets
```

Why this matters:

```text
FACE's topology coherence is learned statistically.
There is no explicit edge-pairing loss in the paper.
Therefore data diversity and target cleanliness are not just nice-to-have; they are the source of topology priors.
```

Hostile conclusion:

```text
A 109-shard run cannot prove heldout quality.
It can only prove the code path learns.
```

### 2. Target Distribution May Be Off-Paper

Paper target:

```text
artist-style triangle mesh under 4000 faces
```

Our strict target path:

```text
raw asset -> voxel_shell / cleanup / convex_hull fallback -> watertight strict target -> optional simplification -> FACE tokens
```

Good:

```text
This is production-rational because ClearMesh wants watertight legal meshes.
```

Risk:

```text
Voxel-shell targets can look like isosurface reconstructions, not artist meshes.
If we train FACE only on voxel shells, we may get watertight but not highly editable/crisp artist topology.
If we train on raw artist meshes, token topology may fail because public assets are dirty.
```

Practical answer:

```text
Use two target streams:
A. strict watertight voxel/manifold targets for topology legality
B. curated artist triangle meshes that already pass topology gates for editability/style

Do not train only on convex hull or over-smoothed voxel shells if the desired output is crisp and editable.
```

### 3. 512 Bins Is A Product-Deviation With Real Tradeoffs

Paper base:

```text
128 bins
```

Paper large:

```text
1024 bins, 1.2B params, 65,536 points, 380K internal high-quality meshes
```

Our product ladder:

```text
512 bins because strict token pass rate improved dramatically on real strict targets
```

Pros:

```text
Fewer quantization-induced vertex collapses.
Better chance of preserving watertight edge identity after decode.
```

Cons:

```text
Per-coordinate CE starts at log(512)=6.238 instead of log(128)=4.852.
The head vocabulary is 4x larger.
For the same parameter/data/step budget, learning is harder.
```

Hostile conclusion:

```text
512 is plausible for production, but a paper-faithful ablation must still run at 128.
If 128 learns clean AR topology on the same corpus and 512 does not, we need curriculum or staged bin upsampling.
```

### 4. Online Augmentation Is Paper-Faithful But Currently Dangerous

Paper:

```text
random rotation, flipping, independent axis scaling during training
```

Our online augmentation:

```text
reconstruct mesh from quantized tokens -> apply affine -> re-normalize -> re-tokenize and reorder every batch
```

Problem:

```text
FACE ordering is spatial. Strong online SO3/flips can radically change face order from step to step.
At tiny data/model scale, that turns memorization/early reconstruction into a moving target.
```

Does the paper contradict disabling augmentation first?

```text
No. The paper reports final training with augmentation, but the lab-style validation ladder should first prove no-aug reconstruction, then frozen/offline augmentation, then online augmentation.
```

Recommended ladder:

```text
1. no augmentation: prove code and target learnability
2. frozen/offline augmented copies: prove invariance without per-step sequence churn
3. mild online augmentation: only after train AR is healthy
4. full paper augmentation: only before large final run
```

### 5. Optimizer Fix Is Necessary, Not Sufficient

Paper:

```text
Muon optimizer, lr 6e-4, wd 0.1
```

Current fix:

```text
Use AdamW for embeddings/norms/bias/BOS/EOS and optionally Muon for hidden matrices.
Use AdamW for ladder runs to avoid native Muon uncertainty.
```

Verdict:

```text
This is a sensible engineering correction.
```

Hostile conclusion:

```text
The optimizer fix explains why loss started moving better.
It cannot explain the remaining topology failures by itself.
The remaining failures are more likely data scale, target distribution, model size, point/VecSet scale, and free-running exposure bias.
```

### 6. Free-Running AR Is A Separate Failure Mode

Teacher-forced reconstruction asks:

```text
Given true previous faces, can the model predict the next face?
```

Autoregressive rollout asks:

```text
Given its own previous mistakes, can the model stay on the mesh manifold for hundreds/thousands of faces?
```

The paper evaluates deterministic top-1 inference after a very large training run.

Our current AR failures are expected when teacher-forced accuracy is only ~42% on train and ~3% heldout.

Hostile conclusion:

```text
Do not debug AR topology deeply until train teacher-forced accuracy is high.
Once train TF is high but train AR is bad, then we debug exposure bias, caching equivalence, CausalMLP greedy decode, EOS, and constrained decoding.
```

### 7. Incremental Decoding Has A Subtle Cache Risk

Our incremental path caches `block_input` before self-attention rather than projected K/V tensors.

Good:

```text
It should reproduce full causal attention logically because key/value norms are recomputed over the same block inputs.
```

Risk:

```text
It is not a true KV cache and still grows memory/time with sequence length.
It must be unit-tested against full non-incremental logits at multiple lengths.
```

Required test:

```text
For a fixed checkpoint and sample, compare full hidden last-position logits vs incremental hidden logits for positions 1, 2, 16, 128.
Tolerance should be tight enough to catch causal-mask/cache mistakes.
```

### 8. FACE Alone Does Not Guarantee Quad Editability

FACE output:

```text
explicit triangle mesh
```

ClearMesh product goal:

```text
watertight, crisp, highly editable, ideally quad/chart structured
```

Truth:

```text
FACE can be an artist-triangle topology prior.
It is not automatically a quad-remesher.
```

Product implication:

```text
Even if FACE works, production still needs chart-level retopo / QuadriFlow / Instant Meshes / feature-aware projection / Blender gates for the universal editable quad story.
```

## Failure Hypotheses Ranked

### H1: Current run is under-scale and under-data

Confidence: very high.

Evidence:

```text
paper: 130K curated meshes, 500M params, 100K steps, 8x A100
ours: 109 train shards, tiny model, 8K steps, 1x A6000
```

Prediction:

```text
Longer training on the 109 split should improve train teacher-forced substantially.
It may still not improve heldout much.
```

### H2: 512-bin vocabulary makes the small run too hard

Confidence: high.

Evidence:

```text
512 improves token gate pass rate but increases CE/vocab difficulty.
Paper uses 128 for base model and only scales to 1024 with a 1.2B model and 380K meshes.
```

Prediction:

```text
128-bin matched run should learn faster and possibly achieve higher train TF/AR on the same data, but may suffer token cracks.
256 may be the best ladder compromise.
```

### H3: Online augmentation poisoned the earlier run

Confidence: high.

Evidence:

```text
same split no-aug AdamW step 1000 selection loss 4.7985 vs augmented run 5.5761.
```

Prediction:

```text
Offline/frozen augmentation should preserve learnability better than per-step SO3 reordering.
```

### H4: Target distribution is too voxel-shell / too simplified

Confidence: medium-high.

Evidence:

```text
strict target path emphasizes watertight legality, not necessarily artist topology.
```

Prediction:

```text
A gallery of strict targets may show over-smoothed or blocky topology. FACE trained on them will reproduce that style.
```

### H5: Within-face canonicalization adds entropy

Confidence: medium.

Evidence:

```text
paper under-specifies vertex order inside a face.
source mesh triangle winding/order may be arbitrary.
```

Prediction:

```text
Rotating each face to a canonical minimum vertex while preserving winding may reduce CE and improve AR edge pairing.
```

### H6: Our CausalMLP differs from paper's unpublished implementation

Confidence: medium.

Evidence:

```text
paper gives only causal factorization and cites TreeMeshGPT.
our legacy-concat variant now matches the public TreeMeshGPT coordinate-head pattern more closely, but exact FACE code is unpublished.
```

Prediction:

```text
A matched causal-vs-parallel coordinate ablation should reproduce Table 5 direction if implementation is good enough.
```

### H7: Point/VecSet scale is too low

Confidence: medium.

Evidence:

```text
latest run used 2048 points and 256 VecSet tokens.
paper base uses 8192 and 2048.
```

Prediction:

```text
Increasing to 8192/512 or 8192/1024 should improve heldout conditioning but slow training.
```

### H8: EOS/face count is under-specified and not solved

Confidence: medium.

Evidence:

```text
paper says variable generation but does not define termination in methods.
our eval mostly uses GT face count.
```

Prediction:

```text
Predicted face-count mode will be poor until reconstruction is strong and EOS has enough varied face-count training.
```

### H9: Strict token gate over-selects easy geometry

Confidence: medium.

Evidence:

```text
strict gates reject cracked/dirty assets; the passing set may skew toward simple shells.
```

Prediction:

```text
Heldout generalization may look deceptively low/high depending on geometry mix. Need category/grouped splits and target galleries.
```

## Does The Paper Point Us Toward This Direction?

Yes, in three explicit ways.

### 1. The ARAE hypothesis points to reconstruction-first validation

The paper's latent claim is that end-to-end ARAE training learns useful `C`.

Therefore, the correct lab order is:

```text
tokenizer roundtrip
-> no-aug teacher-forced overfit
-> no-aug AR overfit
-> frozen augmentation
-> heldout improvement
-> scale data/model
-> only then image/TRELLIS-conditioned production use
```

### 2. The ablations point away from single-bug thinking

The paper reports large effects from:

```text
face ordering
query choice
coordinate decoding head
```

That means a failure can come from any of those system-level choices. We should not prematurely close on Muon, augmentation, or CausalMLP alone.

### 3. The scaling section points to data/model/resolution coupling

The paper's high-resolution result changes multiple variables together:

```text
1.2B params
65,536 input points
1024 bins
380K high-quality meshes
```

So if we want 512/1024-bin crisp meshes, we should expect to increase model/data/point scale, not just vocabulary size.

## Immediate Validation Plan Before Another Expensive Run

### Test 1: Fixed no-aug overfit continuation

Purpose:

```text
Find whether the 109-shard 512-bin split can be memorized by the current architecture.
```

Run:

```text
continue or relaunch no-aug AdamW
steps: 30K to 50K
same split/model first
checkpoint/eval every 5K
```

Pass:

```text
train teacher-forced accuracy > 90%
train teacher-forced boundary edges sharply lower
train AR-128 visibly improves
```

Fail meaning:

```text
Architecture/tokenization/head is still wrong or too small even for train memorization.
```

### Test 2: 128/256/512 matched bin ladder

Purpose:

```text
Separate token-resolution difficulty from architecture failure.
```

Run:

```text
same train/test assets, same target prep where possible
128 bins, 256 bins, 512 bins
same small model and step budget
```

Pass:

```text
128 learns fastest; 256 is middle; 512 is hardest but best strict gate.
```

Decision:

```text
If 256 gives much better train AR with acceptable token gates, use 256 for medium ladder and reserve 512 for large models.
```

### Test 3: Causal-vs-parallel coordinate ablation

Purpose:

```text
Reproduce FACE Table 5 direction in our implementation.
```

Run:

```text
same dataset/model/steps
DECODE_HEAD=causal vs parallel
no topology aux, no indexed variant
```

Pass:

```text
causal has lower heldout teacher-forced loss and better AR edge-pairing than parallel.
```

Fail meaning:

```text
Our CausalMLP implementation or token order is suspect.
```

### Test 4: Within-face ordering ablation

Purpose:

```text
Test paper-under-specified vertex ordering inside each face.
```

Variants:

```text
A. preserve source within-face order
B. rotate to min ZYX vertex, preserve winding
C. canonical sort diagnostic only
```

Pass:

```text
one variant clearly lowers train/heldout CE and improves edge-pairing.
```

### Test 5: Full-vs-incremental decode equivalence

Purpose:

```text
Make sure speed optimization is not changing logits.
```

Pass:

```text
last-position logits match full causal path within numerical tolerance.
```

### Test 6: Target gallery/style audit

Purpose:

```text
Verify the training target is what we want the model to learn.
```

Run:

```text
render 50 strict targets, 50 decoded paper-token targets, 50 source assets
```

Questions:

```text
Are strict targets crisp or blobby?
Are they artist-editable or only watertight?
Are important handles/holes/branches preserved?
```

## Production-Relevant Conclusion

FACE is still worth pursuing, but it is not the immediate one-shot solution to universal editable watertight crisp meshes.

Current role in ClearMesh:

```text
research branch / learned topology prior / possible artist-triangle head
```

Near-term production path should still be:

```text
TRELLIS.2 or LATTICE geometry
  -> UltraShape/manifoldization for watertight reference
  -> OmniPart/MeshMosaic-style part structure
  -> MeshRipple/Silksong prior where useful
  -> chart-level quad remesh
  -> feature-aware projection
  -> Blender gate promotion
```

FACE becomes production-critical only if it clears this ladder:

```text
1. train-overfit high accuracy
2. train AR watertight/stable
3. heldout AR improves with data/model scale
4. output target style is editable, not just watertight
5. runtime is acceptable or can be used as async high-quality refinement
```

## Decisions For Next Work

1. Do not launch a huge FACE run yet.
2. Continue the no-aug overfit/ladder until train reconstruction is convincing.
3. Run 128/256/512 and causal/parallel matched ablations before scaling.
4. Add the incremental decode equivalence test.
5. Render target galleries before judging model output aesthetics.
6. Treat online augmentation as a later paper-scale step, not the first ladder run.
7. Keep Easy3E/editing and quad-remesh path intact; FACE is complementary, not a replacement.

## 2026-05-04 Paper-Parity Correction

I rechecked the FACE v2 paper, the arXiv code/media links, and the cited public
repos before launching the next ladder. The result is deliberately not phrased
as "paper identical":

```text
FACE paper code status: no official implementation located.
Paper source: arXiv v2, last revised 2026-03-03.
Public code cross-checks: 3DShape2VecSet and TreeMeshGPT.
Conclusion: equation-faithful and cited-repo closer, but not byte-identical.
```

The concrete mismatch found in our code was the CausalMLP default. The FACE
paper cites TreeMeshGPT for CausalMLP, and the TreeMeshGPT public implementation
uses separate staged coordinate heads conditioned by previous coordinate
embeddings. Our `paper_chain` variant was a reasonable causal interpretation,
but `legacy_concat` is closer to the cited public-code pattern. I changed the
FACE paper-ladder default to `legacy_concat` and kept `paper_chain` as an
experimental ablation only.

What is now directly checked:

```text
Eq. 1: FPS/downsampled query points cross-attend to point cloud.
Eq. 2: Transformer encoder refines the latent VecSet.
Eq. 3: each 9-token face is pooled/projected to one decoder token.
Eq. 4: causal face-level self-attention over previous faces.
Eq. 5: cross-attention from face tokens to encoder VecSet.
Eq. 6: coordinate-token cross-entropy over all faces/tokens.
```

What remains under-specified by the paper:

```text
exact Face Pooling MLP width/depth
exact CausalMLP implementation
within-triangle vertex ordering after face ZYX sort
EOS/termination target and inference face-count policy
exact mesh cleanup/curation pipeline for the 130K training subset
exact Muon parameter grouping
```

Code changes made for this audit:

```text
default causal_mlp_variant: paper_chain -> legacy_concat
paper within-face order ablations: preserve / rotate_min_zyx / sort_zyx
full-vs-incremental decode equivalence test
Thunder FACE ladder supervisor with reproducible token gates and galleries
```

Validation:

```text
local non-Torch tests: 18 passed, 3 skipped
Thunder Torch/CUDA parity tests: 22 passed, 3 warnings
```

Current ladder launch:

```text
script: scripts/thunder/face_paper_ladder_supervisor.sh
instance: Thunder instance 1 / RTX A6000
lab_root: /tmp/clearmesh_face_ladder_20260504_041315
status: /tmp/clearmesh_face_ladder_20260504_041315/status.jsonl
```

Current A100 paper-knob launch plan:

```text
script: scripts/thunder/face_paper_a100_probe.sh
first A100 attempt: Thunder instance 2 / A100 80GB
first A100 result: deleted because /dev/nvidia* was missing after allocation and CUDA probes timed out
second A100 attempt: Thunder instance 0 / A100 80GB prototyping
second A100 result: deleted because /dev/nvidia* was also missing after allocation
third A100 attempt: Thunder instance 0 / A100 80GB production mode, queued after replacement
kept A6000 instance: 1, active ladder
closed A6000 instance: 0, stale/OOM-restarted instance
expected post-provision setup: repo sync + strict target transfer, roughly 5-10 min
expected probe runtime: measure after first logged throughput; rough prior 5K-step estimate 1.5-4 hr
precision note: A100 probe uses bf16 autocast; this is a practical training-runtime knob, not a representation change
launcher hardening: install pytest if missing and fail fast when /dev/nvidia0 is absent
```
