# FACE Scaling Decision Memo

Date: 2026-05-07

## Decision

Do not launch the full 130k-350k mesh, 100k-step production FACE run yet.

Scaling is likely necessary for held-out generalization, but the latest bounded
A100 run is not strong enough evidence that scaling alone will fix the current
failure. The next rung should prove near-perfect train autoregressive closure on
a cleaned, paper-like subset, then show monotonic held-out improvement as data
and capacity increase.

## Latest Evidence

Run:

```text
local inspection:
.codex_outputs/face_noaug_exact512_setup_noaug_exact512_20260507_002158/face_gate_inspection.json

remote run:
/tmp/clearmesh_face_noaug_exact512_20260507_002158
```

Configuration:

```text
train/test: 228 / 57 strict token samples
bins: 128
point samples: 8192
VecSet tokens: 2048
latent dim: 64
hidden: 384
encoder layers: 4
decoder layers: 8
optimizer: Muon
precision: bf16
augmentation: disabled
steps: 30000
```

Final signals:

```text
selection loss best: 0.0388 at step 29000
train teacher-forced token accuracy: 0.9958
train autoregressive generated token accuracy: 0.9731
train autoregressive edge-set F1: 0.9513
train autoregressive edge pairing: 0.9269
train autoregressive watertight: 4/20

test teacher-forced token accuracy: 0.0902
test autoregressive generated token accuracy: 0.0341
test autoregressive edge-set F1: 0.0010
test autoregressive edge pairing: 0.0438
test autoregressive watertight: 0/20
```

Interpretation:

```text
The model can memorize much of the train distribution.
The model has not learned a reusable held-out topology prior.
The paper-faithful path is alive, but not scale-ready.
```

## What The FACE Paper Actually Supports

The paper's base reconstruction model is not close to our bounded diagnostic:

```text
paper base:
500M parameters
8192 surface points with normals
2048 VecSet tokens
64 bottleneck dimension
8-layer, hidden-768 encoder
24-layer, hidden-1024 decoder
around 130k Objaverse meshes with fewer than 4000 faces
128 coordinate bins
random rotation, flipping, independent axis scaling
Muon lr 6e-4, wd 0.1
100k steps on 8x A100 80GB

latest diagnostic:
38.4M parameters
8192 surface points with normals
2048 VecSet tokens
64 bottleneck dimension
4-layer, hidden-384 encoder
8-layer, hidden-384 decoder
285 strict samples
128 coordinate bins
augmentation disabled
Muon lr 6e-4, wd 0.1
30k steps on 1x A100
```

So yes, the paper gives real evidence that scaling can improve reconstruction
and generalization. It also reports a larger 1.2B / 1024-bin model trained on
380k high-quality meshes.

But the paper does not say small-data train AR can be ignored. Its central claim
is end-to-end ARAE training: the encoder learns a structured VecSet latent and
the decoder learns the face sequence distribution. If the model cannot close
the train split reliably, the encoder-decoder contract is not learned yet.

## What The Cited Sources Add

3DShape2VecSet:

- The official implementation uses FPS-sampled point queries, point embeddings,
  cross-attention to the full point cloud, then latent self-attention.
- This supports our `shape2vecset` encoder direction.
- It also reinforces that point-query details matter; learnable queries are not
  the default evidence path.

TreeMeshGPT:

- The public coordinate head predicts one coordinate axis, embeds it, then
  predicts the next axis from the latent plus previous coordinate embeddings.
- This supports using `legacy_concat` as the closest public-code CausalMLP fill.
- TreeMeshGPT also uses topology-aware sequencing, which FACE deliberately
  avoids. FACE asks scale and CausalMLP to learn exact coordinate reuse.

BPT:

- BPT's core lesson is not "bigger only"; it uses a more structured compressed
  tokenization to support more faces and more robust topology.
- That is a warning: if FACE is brittle, blindly scaling data may reduce but not
  eliminate topology errors.

Objaverse / Objaverse++:

- Large 3D corpora improve generalization, but quality curation matters.
- For ClearMesh, the corpus should be high-quality, manifoldized, token-pass,
  and deduplicated, not just large.

## Should We Perfect Train AR First?

Yes, but with a precise definition.

We do not need 100% train AR on every 512-face sample before any scaling. That
would overfit our gate. But before a large production run, we should require:

```text
train AR token accuracy >= 0.995
train AR edge-set F1 >= 0.99
train AR edge pairing >= 0.99
train AR mean boundary edges near zero
train AR watertight rate mostly passing on closed targets
held-out teacher-forced accuracy trending upward over checkpoints
held-out AR contact sheets structurally plausible, even if imperfect
```

The latest run is close on train token accuracy but not close on watertightness
or boundary edges, and held-out is not plausible.

## Will 300k+ High-Quality Meshes Generalize Better?

Likely yes, if all of these are also true:

```text
targets are clean and paper-like
model capacity is much closer to 500M
augmentation is paper-faithful but not destabilizing
face ordering and within-face ordering are stable
the data loader is not silently poisoning topology
held-out loss improves early in the rung
```

Likely no, or at least not efficiently, if:

```text
the strict corpus contains conversion artifacts
quantization creates topology cracks
within-face vertex order injects arbitrary entropy
augmentation changes ordering too violently at small scale
capacity remains 38M while data jumps to 300k
we judge with only Chamfer/visuals instead of AR topology metrics
```

## Recommended Next Rungs

### Rung 1: Train-AR Closure Probe

Use 32-64 high-quality, watertight, strict passing meshes.

Use paper knobs except model capacity can stay bounded:

```text
128 bins
8192 points with normals
2048 VecSet tokens
latent 64
Muon
bf16
no online augmentation first
```

Goal:

```text
prove train AR closure is not an accident of tiny toy sets
```

### Rung 2: Frozen Paper-Aug Probe

Create an offline expanded split from the same 32-64 base meshes.

Goal:

```text
test paper augmentation without online face-order churn
```

### Rung 3: 1k-5k Curated Corpus Probe

Use grouped train/test split with dedupe.

Goal:

```text
show held-out teacher-forced loss and AR topology improve with data
```

### Rung 4: Capacity Ladder

Run matched data with:

```text
38M current
100M-150M mid
300M-500M paper-near
```

Goal:

```text
measure whether capacity or data is the immediate bottleneck
```

### Rung 5: Production Scale Only After Curves Are Sane

Only launch 130k+ / 100k steps when:

```text
train AR closes
held-out metrics trend in the right direction
visuals are coherent
runtime profile is acceptable
corpus prep is reproducible from fresh Thunder instance
```

## Product Implication

FACE should not be our only watertightness guarantee. The paper does not provide
a hard watertightness guarantee; watertightness emerges only if generated
coordinate tokens exactly reuse vertices and pair edges. ClearMesh still needs a
separate production gate:

```text
FACE output
  -> topology metrics
  -> manifold/repair or reference-surface fallback when needed
  -> optional artist retopo / quad promotion
```

This does not mean FACE is a dead end. It means FACE is the learned artist-mesh
prior, not the final product warranty.
