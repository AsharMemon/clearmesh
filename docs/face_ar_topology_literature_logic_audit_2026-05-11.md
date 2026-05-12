# FACE AR Topology Literature + Logic Audit - 2026-05-11

## Question

We need to explain and fix the gap between:

```text
teacher-forced FACE reconstruction improves
but free-running autoregressive meshes still drift, open, or close into wrong topology
```

The specific symptoms are:

```text
- first-face ambiguity
- early autoregressive divergence
- coordinate-close but unwelded adjacent faces
- weak edge pairing / watertightness under free-run decoding
```

## Paper Constraints

### FACE

FACE is intentionally simple at the sequence level:

```text
input point cloud -> VecSet encoder -> autoregressive face decoder
```

The core paper claims are:

```text
- one triangle face is embedded as one decoder token
- each face is decoded to 9 coordinate tokens by a CausalMLP
- faces are sorted by lexicographic ZYX order of the minimum-coordinate vertex
- the model is trained end-to-end, not through a separate coordinate-token VAE
- inference uses deterministic top-1 autoregressive decoding
```

Implementation details that matter:

```text
- 8192 input points with normals
- 2048 VecSet tokens
- latent bottleneck dimension 64
- 128 coordinate bins for the base model
- <4000-face Objaverse subset, about 130k meshes
- random rotation, flipping, and per-axis scaling
- Muon, lr 6e-4, weight decay 0.1
- 100k steps on 8x A100 80GB
- base ARAE around 500M params
- larger model: 1.2B params, 380k high-quality meshes, 256 bins, 65,536 input points
```

FACE ablations constrain our choices:

```text
ordering:
  BFS: bad
  DFS: better, still behind
  ZYX-component: close to ZYX
  ZYX: best reported

query:
  downsampled point queries beat learnable queries

coordinate head:
  CausalMLP beats parallel and attention coordinate decoding
```

Therefore, a faithful fix should not immediately discard ZYX, VecSet, or CausalMLP.

## Adjacent Literature Signal

### MeshAnythingV2

MeshAnythingV2’s Adjacent Mesh Tokenization compresses by reusing a single vertex token where feasible. The important lesson for us is not the exact token format; it is that topology preservation improves when the token stream explicitly exploits adjacency/reuse rather than treating every triangle as an independent coordinate tuple.

### TreeMeshGPT

TreeMeshGPT uses an autoregressive tree sequencing rule. Its abstract says the next input token is retrieved from a dynamically growing tree built from triangle adjacency, so generation locally extends from the previous triangular face. The important lesson is that local frontier structure reduces training difficulty and improves mesh quality.

FACE’s own ablation says plain DFS/BFS is worse than ZYX for FACE reconstruction, so we should not copy TreeMeshGPT wholesale. But the failure mode we see is exactly the one tree/frontier methods try to prevent: the decoder generates a plausible face that is not connected to the target frontier, then all subsequent exact topology metrics collapse.

### Mesh Silksong

Mesh Silksong makes topology a first-class tokenization object. It uses connected-component and layer tokens plus matrix tokens representing intra-layer and inter-layer connections. Its project page explicitly frames prior AR mesh tokenizers as treating meshes like unordered triangle collections and lacking global topology awareness.

The lesson for FACE is again not “replace everything now.” It is:

```text
coordinate-only face tokens are too weak to guarantee topology
```

If we stay with FACE, we need either:

```text
- stronger training against early prefix drift
- topology-aware auxiliary losses / labels
- or a later FACE-compatible topology-indexed representation
```

## Pure Logic / Math

### 1. First-Face Ambiguity Is Label Noise

Let a mesh be encoded as an ordered face sequence:

```text
F = (f_1, f_2, ..., f_N)
```

FACE chooses:

```text
f_1 = lexicographically first face under ZYX-min-vertex ordering
```

But if several faces share the same minimum quantized vertex:

```text
G_1 = { f_j : min_vertex(f_j) = min_vertex(f_1) }
```

then the exact row-0 face is an arbitrary tie-break among several valid local starts.

Standard CE optimizes:

```text
L = -log p(f_1 | C, BOS)
```

But the semantically correct target is closer to:

```text
L_tie = -log sum_{f in G_1} p(f | C, BOS)
```

This is exactly why our tie-marginal loss improved train first-face metrics and same-min oracle rates.

### 2. Exact Coordinate Tokens Are Harsh For Welded Topology

For two adjacent faces to share an edge, two quantized vertices must match exactly:

```text
edge A = ((z1,y1,x1), (z2,y2,x2))
edge B = ((z1,y1,x1), (z2,y2,x2))
```

If one coordinate is off by one bin:

```text
(z1,y1,x1) != (z1,y1,x1 + 1)
```

the decoder creates a nearby duplicate vertex, not a shared welded vertex.

So Chamfer can stay low while topology breaks.

This means a visually plausible coordinate-token model still needs very high exact token accuracy to be editable/watertight.

### 3. Teacher Forcing Hides Exposure Bias

During training:

```text
p(f_i | C, f_<i true)
```

During inference:

```text
p(f_i | C, f_<i generated)
```

If the generated first face is plausible but off-target:

```text
f_1_hat != f_1
```

then every later conditional distribution is evaluated off the training manifold.

This explains our result:

```text
teacher-forced / beam first-face quality improves
but rollout still drifts
```

The model has learned to score target prefixes better, but not to recover from slightly wrong generated prefixes.

### 4. Decoding Constraints Alone Are Not Enough

Our hybrid/topology-constrained short rollouts did not fix the issue. That makes sense:

```text
post-hoc constraints can reject impossible local steps
but they cannot make the hidden state represent the correct target frontier
```

If the model’s prefix hidden state is already off-manifold, constrained decoding can produce cleaner wrong meshes, not faithful meshes.

## Current Empirical State

### Smoking Gun Found

```text
FACE first-face exact target is ambiguous under quantized ZYX ties.
Tie-marginal loss improves first-face train metrics and same-min oracle metrics.
Full-sequence continuation after tie-marginal improves held-out exact-in-beam.
```

### Smoking Gun Not Yet Found

```text
We have not yet found a training scheme that makes free-running rollout stable.
```

Prefix-16 continuation improved first-face train metrics and short-rollout token accuracy slightly, but did not yet improve edge pairing / watertightness enough to justify full scale.

Final prefix-16 probe:

| Split | Metric | Value |
| --- | --- | ---: |
| Train | first-face exact in beam | 43.75% |
| Train | first-face top-1 slot accuracy | 62.85% |
| Train | same-min oracle | 71.88% |
| Train rollout greedy | generated token accuracy | 18.07% |
| Train rollout greedy | edge pairing | 0.598 |
| Train rollout greedy | watertight edge graph | 12.50% |
| Test | first-face exact in beam | 15.62% |
| Test | first-face top-1 slot accuracy | 40.28% |
| Test | same-min oracle | 34.38% |
| Test rollout greedy | generated token accuracy | 19.13% |
| Test rollout greedy | edge pairing | 0.617 |
| Test rollout greedy | watertight edge graph | 25.00% |

Interpretation:

```text
Prefix-16 improves first-face ranking and slightly improves generated token accuracy.
It does not materially improve edge pairing or watertightness.
This supports the exposure-bias diagnosis but is not a scale green light.
```

## Fix Candidates

### Candidate A: Noisy Teacher Prefix

Purpose:

```text
train p(f_i | C, perturbed f_<i) while target remains f_i
```

Why it follows logically:

```text
Exposure bias means inference prefixes differ from teacher-forced prefixes.
Small decoder-input corruption creates training states near inference states.
```

FACE compatibility:

```text
high
```

Risk:

```text
too much noise may degrade exact teacher-forced reconstruction
```

Bounded test:

```text
init: best fullseq/prefix16 tie-marginal checkpoint
steps: 2k-4k
noise prob: 0.03-0.08
noise offset: 1 bin
noise prefix: first 16 or 64 faces
disable augment initially
pass criterion: rollout token accuracy + edge pairing improve without first-face beam collapse
```

Launched bounded falsification run:

```text
run: face0_tie_marginal_prefix16_noisy005_noaug_3k_20260512
init: face0_tie_marginal_prefix16_noaug_4k_20260512/checkpoint.pt
steps: 3,000
noise prob: 0.05
noise offset: 1 bin
noise prefix: first 16 faces
loss prefix: first 16 faces
augmentation: disabled
lr: 5e-5
```

### Candidate B: ZYX-Component Tie Marginal

Purpose:

```text
preserve FACE's spatial ordering but group connected components before ZYX
```

Why:

```text
FACE reported ZYX-component close to ZYX.
Mesh Silksong also treats connected components explicitly.
```

Risk:

```text
requires retokenizing / continuing from incompatible ordering
```

Bounded test:

```text
small corpus retokenization only
do not launch production run until loss curve and first-face probes compare favorably
```

### Candidate C: Boundary-Closure Auxiliary Head

Purpose:

```text
predict how many currently open boundary edges the next face should consume
```

Why:

```text
This preserves FACE coordinate tokens but teaches frontier awareness.
TreeMeshGPT and Mesh Silksong both encode topology/frontier information more explicitly.
```

Risk:

```text
paper-deviation; requires model-head change
```

Bounded test:

```text
train auxiliary loss only on early prefix
measure rollout edge pairing / boundary edges
```

### Candidate D: FACE-Compatible Indexed Topology Tokens

Purpose:

```text
factor face generation into vertex-table tokens + face-index tokens
```

Why:

```text
Welding becomes categorical index reuse rather than exact coordinate equality.
This directly attacks the topology/editability target.
```

Risk:

```text
larger representational change; less faithful to FACE
```

Recommendation:

```text
keep as backup lane, not immediate next step
```

## Recommendation

The next best experiment is Candidate A:

```text
noisy teacher prefix
```

Reason:

```text
- directly targets the observed exposure-bias failure
- preserves FACE architecture, ordering, VecSet, CausalMLP, and one-face-one-token setup
- can be enabled with default-off flags
- cheap to falsify
- if it works, it creates a real scaling gate
```

Scale decision rule:

```text
Do not scale to 350k/100k because first-face beam got better.
Only scale if noisy-prefix or another bounded curriculum produces held-out rollout improvement:
  generated token accuracy up
  edge pairing up
  boundary edges down
  first-face divergence down
  watertight graph rate up
```
