# FACE Discrete Topology Strategy

Date: 2026-05-12

Status: parallel research and implementation lane

Goal: keep FACE as the core model family, but stop asking a coordinate-only decoder to learn topology by accident.

## Executive Summary

Our strongest current diagnosis is not "the model needs only more data." More data should help, but the repeated failure mode is structurally discrete:

- FACE emits each triangle as 9 quantized coordinate tokens.
- Watertightness requires exact vertex reuse and exact edge pairing.
- Exact reuse is a quotient relation over coordinates, but the model currently has to infer that quotient relation implicitly.
- A visually close mesh can still have hundreds of boundary edges if shared vertices differ by even one bin.
- The first-face failure is a seed-localization problem plus a label-symmetry problem, not merely a face-0 loss-weight problem.

The clean solution is a FACE-compatible topology layer:

```text
VecSet shape encoder
  -> seed/anchor head
  -> topology-aware indexed/frontier decoder
  -> CausalMLP only for new geometry/refinement
  -> constrained topology product-of-experts
  -> optional coordinate projection/welding gate
```

This keeps the FACE spirit: shape encoder, one face step per autoregressive step, causal transformer, CausalMLP geometry head, end-to-end training. The change is that vertex identity and boundary state become explicit discrete state variables instead of being rediscovered from coordinates.

## Why The Current Coordinate FACE Path Breaks

### Mesh As A Finite Complex

Represent a triangle mesh as a finite 2-dimensional simplicial complex:

```text
K = (V, E, F)
```

where:

- `V` is a set of vertices.
- `E` is a set of unoriented edges.
- `F` is a set of triangular faces.

For a closed manifold triangle mesh:

```text
for every edge e in E: edge_count(e) = 2
for every vertex v in V: link(v) is a cycle
```

Boundary edges are exactly the edges with count 1. Non-manifold edges have count greater than 2. Pinched vertices occur when the vertex link is not a single cycle or path.

### FACE's Coordinate Quotient Problem

The coordinate decoder emits:

```text
face_t = (x0, y0, z0, x1, y1, z1, x2, y2, z2)
```

with each coordinate in `[0, B-1]`, currently `B = 128`.

The actual mesh topology depends on a quotient map:

```text
pi: [0, B-1]^3 -> V
```

Two face corners refer to the same vertex only if their quantized coordinate triples are exactly equal after decoding and repair. FACE currently does not directly model `pi`; it emits raw triples and hopes exact equality emerges.

This is the core discrete mismatch.

### Why Low Chamfer Can Still Mean Bad Mesh

Let per-coordinate token error probability be `epsilon`.

For one shared edge to weld correctly, the two endpoint coordinate triples must match exactly. That is 6 coordinate tokens. A crude independent approximation gives:

```text
P(edge welds) ~= (1 - epsilon)^6
```

If `epsilon = 0.05`, then:

```text
P(edge welds) ~= 0.735
```

That already implies many broken edges. If `epsilon = 0.20`, then:

```text
P(edge welds) ~= 0.262
```

At 512 faces, there are roughly 768 unique edges in a closed triangle mesh. The probability that all shared edges weld becomes astronomically small unless token accuracy is extremely high and errors are not concentrated on reused vertices.

So "teacher-forced Chamfer is low" and "watertightness is zero" are not contradictory. They are exactly what coordinate-only topology predicts.

### Boundary Recurrence

Let `b_t` be the number of open boundary edges after `t` generated faces.

When adding a triangle, suppose `c` of its edges are already open boundary edges that get paired/closed.

Then:

```text
b_{t+1} = b_t + 3 - 2c
```

Cases:

- `c = 0`: starts a new isolated triangle or disconnected island, boundary increases by 3.
- `c = 1`: grows a surface patch, boundary increases by 1.
- `c = 2`: fills a wedge, boundary decreases by 1.
- `c = 3`: fills a triangular hole, boundary decreases by 3.

For a remaining face budget `r = N - t`, a necessary closure condition is:

```text
b_t <= 3r
```

There is also a parity relation. Since each future triangle changes boundary count by an odd number, closure to `b_N = 0` requires:

```text
b_t == r mod 2
```

These are cheap admissible constraints for decoding. A decoder that violates them early is provably unable to end watertight with the remaining face budget.

## First Face Ambiguity

The paper-style ZYX face ordering imposes a canonical face, but quantized meshes can have ties:

```text
G0 = { faces sharing the same canonical anchor/min-vertex key }
```

Training with ordinary CE on one chosen face inside `G0` injects label noise. The mathematically correct loss for an equivalence class is:

```text
L_seed = -log sum_{f in G0} p(f | C, BOS)
```

However, our latest first-face probe was worse than a tie-only problem:

- the true first face was absent from high beam candidates;
- predicted candidates often did not even share the canonical min vertex.

That says the model is not only confused between equivalent first faces. It is failing seed localization. We need an explicit seed/anchor head.

## The FACE-Compatible Fix

### 1. Add A Seed Anchor Head

Train a direct head from the VecSet latent to predict the canonical seed region.

Targets:

```text
anchor vertex bin: (x_min, y_min, z_min)
anchor face equivalence class: G0
optional nearest FPS/query token index
```

Loss:

```text
L = L_FACE + lambda_anchor * L_anchor + lambda_seed * L_seed_tie_marginal
```

Purpose:

- decouple "where do I start" from "how do I emit all future faces";
- make face 0 less brittle;
- monitor same-anchor accuracy, not just exact face-0 accuracy.

Promotion signal:

```text
same-min-vertex@beam32 > 50% train
same-min-vertex@beam32 > 25% heldout
```

### 2. Replace Raw Coordinate Reuse With Explicit Vertex Identity

Introduce an indexed FACE representation:

```text
vertices: V_i = (x_i, y_i, z_i)
faces:    F_t = (i, j, k)
```

Autoregressive state:

```text
S_t = (V_seen, F_seen, edge_counts, boundary_edges, vertex_links)
```

Action:

```text
a_t = (edge_action, v_new_or_existing, geometry_refinement)
```

The action can be:

- start a seed face;
- extend from one boundary edge with a new vertex;
- extend from one boundary edge with an existing vertex;
- close a wedge;
- fill a triangular hole;
- start a new component only when allowed by part/component state.

This makes topology an explicit state machine. Coordinates still exist, but they are attached to vertex creation or refinement, not repeated for every face corner.

### 3. Frontier/Shelling Decoder

Use a frontier stack/queue of open boundary edges.

At each step:

```text
choose boundary edge e = (u, v)
choose third vertex w
emit face (u, v, w)
update edge counts and vertex links
```

Legal action constraints:

```text
edge_count(edge) <= 2
new face not duplicate
vertex link remains manifold-compatible
boundary budget remains closable
normal orientation consistent enough
```

This matches the mathematical object and aligns with recent topology-aware AR mesh work:

- TreeMeshGPT uses triangle adjacency and a growing tree structure rather than a flat face list.
- MeshRipple grows from an active frontier to preserve global coherence and reduce holes/fragments.
- Mesh Silksong explicitly targets manifoldness, watertightness, normal consistency, and part awareness through topology-aware tokenization.
- MeshAnything V2 uses adjacent mesh tokenization to compact the sequence and preserve adjacency structure.
- PolyGen separated vertex generation from face generation, which is an older but important clue that geometry and connectivity should not be collapsed into raw coordinate rows.

### 4. Topology Product Of Experts

At decode time:

```text
p_final(a_t | C, S_t) proportional to
    p_FACE(a_t | C, history)
    * exp(-E_topology(a_t, S_t))
    * exp(-E_surface(a_t, point_cloud))
    * exp(-E_budget(a_t, S_t, N))
```

Where:

- `E_topology` rejects or penalizes non-manifold edges, duplicate faces, pinched vertex links, disconnected starts, and irreversible boundary budgets.
- `E_surface` penalizes third vertices/faces far from the conditioned point cloud or proxy surface.
- `E_budget` applies the boundary recurrence and parity constraints.

Why our previous decode-only constraints did not solve it:

- The true first face and true early actions were not in the coordinate beam.
- A product-of-experts cannot rescue a candidate that the proposal distribution never offers.

Therefore the proposal distribution itself must be indexed/frontier-aware.

### 5. Coordinate Projection/Welding As A Bridge, Not The Core

We can still exploit coordinate FACE outputs through a deterministic projection:

1. Build candidate vertex clusters by quantized equality and near-neighbor distance.
2. Score possible welds by distance, normal compatibility, and point-cloud consistency.
3. Solve a constrained union-find / min-cost repair:

```text
minimize total vertex displacement
subject to edge_count(e) <= 2
           vertex links manifold-compatible
           no degenerate faces
           preserve surface error below threshold
```

This is useful if teacher-forced or AR geometry is close but unwelded. It is not enough if the rollout picked the wrong seed or wrong component.

## Novel Product Direction: FACE-Q

Working name: FACE-Q, for FACE with quotient topology.

FACE-Q keeps:

- VecSet shape encoder;
- one autoregressive face step per generated face;
- causal self-attention plus cross-attention to VecSet;
- CausalMLP for face/vertex geometry;
- end-to-end training.

FACE-Q changes:

- raw coordinate row output becomes an indexed/frontier action;
- vertex reuse is explicit;
- boundary state is a first-class input to the decoder;
- legal topology is enforced by construction where possible;
- coordinate CausalMLP predicts only new vertex coordinates/refinements.

Training losses:

```text
L_total =
    L_action
  + L_vertex_choice
  + L_new_vertex_coord
  + L_anchor
  + L_count
  + L_normal
  + L_aux_teacher_coordinate
```

The auxiliary teacher coordinate loss keeps paper FACE behavior alive while the topology losses make the product usable.

## Immediate Experiments

### Experiment A: Topology Projection On Existing Predictions

Question:

```text
Are the 10k/100k teacher-forced predictions geometrically close enough to weld?
```

Run:

- take saved teacher-forced predicted meshes;
- apply constrained projection/welding;
- measure Chamfer delta, boundary edges, non-manifold edges, watertight rate.

Interpretation:

- If watertightness jumps while Chamfer stays low, coordinate FACE has learned geometry but not quotient topology.
- If projection distorts badly, the model is not geometrically accurate enough yet.

### Experiment B: Indexed Frontier Decoder On The Current Strict Corpus

Use existing repo primitives:

- `clearmesh/mesh_heads/face_indexed.py`
- `clearmesh/mesh_heads/face_topology.py`
- `scripts/research/train_face_indexed_conditioned_tiny.py`
- `tests/test_face_indexed.py`

Target settings:

```text
FACE_ORDER=boundary_growth
DECODE_MODE=edge_constrained
EDGE_ACTION_LOSS_WEIGHT=1.0
EDGE_CHOICE_LOSS_WEIGHT=1.0
REQUIRE_BOUNDARY_CLOSURE_AFTER=0.75
BOUNDARY_FILL=centroid
```

Initial scale:

```text
512 to 2k strict meshes
512 faces
128 bins
8192 points
2048 VecSet tokens
```

Promotion criteria:

```text
train AR edge_pairing > 0.90
heldout AR edge_pairing > 0.75
boundary edges materially below coordinate FACE
same-anchor accuracy improves over coordinate FACE
visual contact sheets coherent
```

### Experiment C: Seed Anchor Head

Question:

```text
Can the VecSet latent identify the canonical seed before the AR history exists?
```

Train a small head:

- anchor coordinate bin;
- anchor face equivalence class;
- nearest FPS token or local point cluster.

Promotion criteria:

```text
same-anchor@32 > 50% train
same-anchor@32 > 25% heldout
```

If this fails at small scale, the encoder/query representation is not localizing mesh starts well enough.

### Experiment D: Boundary Budget Beam

Add admissible pruning to any coordinate or indexed beam:

```text
reject if b_t > 3 * remaining_faces
reject if b_t mod 2 != remaining_faces mod 2
reject if edge_count > 2
reject duplicate oriented or unoriented face
reject pinched vertex link
```

This is not the whole solution, but it is almost free and mathematically justified.

First local sanity check:

```text
checkpoint: face_indexed_topocausal47_a1
dataset: face_indexed_real50_boundary/dataset_4096
limit: 2

without budget:
  watertight: 0/2
  mean boundary edges: 208.0
  mean non-manifold edges: 211.0
  mean edge pairing: 0.3469
  mean normalized Chamfer: 0.01976

with budget:
  watertight: 0/2
  mean boundary edges: 203.0
  mean non-manifold edges: 206.0
  mean edge pairing: 0.3435
  mean normalized Chamfer: 0.02180
```

Interpretation: the boundary-budget rule is a useful guardrail, not a rescue mechanism. It can prune obviously impossible endings, but if the learned proposal distribution is already on the wrong topology manifold, the decoder needs explicit indexed/frontier actions and seed/edge heads.

## Scale Implications

Scaling from 10k to 100k/350k should help:

- seed localization;
- semantic diversity;
- long-tail shapes;
- robustness to augmentation;
- heldout teacher-forced loss.

But scaling alone does not remove the combinatorial burden of exact vertex reuse.

The expected pattern if we scale coordinate FACE only:

```text
selection loss improves
teacher-forced Chamfer improves
visual AR improves somewhat
watertightness remains brittle
edge pairing remains unstable
```

The expected pattern if we scale FACE-Q:

```text
selection/action loss improves
seed localization improves
edge pairing improves by construction
watertightness becomes a realistic objective
visual fidelity depends on geometry head quality
```

Recommendation:

- Continue corpus prep and the paper-faithful scale lane.
- Do not bet the production mesh product only on coordinate FACE.
- Build FACE-Q in parallel as the topology-native product lane.

## Why This Is Still FACE, Not A Pivot Away

This is not abandoning FACE. It is making FACE explicit about the discrete object it is generating.

FACE already says one face is one token and uses a learned internal face representation rather than an external VAE. FACE-Q preserves that, but changes what "one face token" contains:

```text
coordinate FACE:
    face token -> 9 coordinates

FACE-Q:
    face token -> topology action + optional new coordinates/refinement
```

The paper's central hypothesis remains intact:

```text
end-to-end training lets the shape encoder learn a structured latent space C
```

FACE-Q simply gives the decoder the right discrete algebra so that `C` does not have to solve vertex identity from scratch at every repeated coordinate.

## Current Best Lane

1. Keep sourcing and preparing the strict 100k+ corpus.
2. Run the 100k-step coordinate FACE continuation for scale evidence only.
3. Start FACE-Q on 512 to 2k meshes immediately.
4. Use projection/welding to test whether existing coordinate FACE checkpoints contain salvageable geometry.
5. Promote to the expensive 8x A100 run only when either:
   - coordinate FACE shows real first-anchor and AR topology recovery, or
   - FACE-Q shows small-scale topology closure and needs scale for fidelity.

The moat is not just "train FACE bigger." The moat is:

```text
paper-faithful FACE backbone
+ high-quality curated strict corpus
+ explicit quotient/topology decoder
+ mathematically constrained decode/repair
+ production gates that measure editability, not screenshots
```

## References Checked

- PolyGen: Autoregressive 3D mesh generation with separate vertex and face modeling. https://proceedings.mlr.press/v119/nash20a.html
- MeshAnything V2: Adjacent Mesh Tokenization reduces redundant face/vertex tokens and preserves adjacency structure. https://arxiv.org/abs/2408.02555
- TreeMeshGPT: Autoregressive Tree Sequencing over triangle adjacency. https://arxiv.org/abs/2503.11629
- Mesh Silksong: topology-preserved autoregressive mesh generation. https://arxiv.org/abs/2507.02477
- MeshRipple: frontier-aware BFS/ripple generation for topological completeness. https://arxiv.org/abs/2512.07514
