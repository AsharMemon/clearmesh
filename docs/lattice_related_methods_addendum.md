# LATTICE Related Methods Addendum

Date: 2026-05-03
Scope: focused audit of LATO, 3DILG, FACE, and PixARMesh for implementation details that help the ClearMesh LATTICE reproduction plan without derailing it.

Sources checked:
- LATO: https://arxiv.org/html/2603.06357v1
- 3DILG paper: `/Users/Ashar/Downloads/2205.13914v2.pdf`
- 3DILG repo: https://github.com/1zb/3DILG
- FACE paper: `/Users/Ashar/Downloads/2603.01515v2.pdf`
- FACE hostile audit: `/Users/Ashar/Documents/GitHub/clearmesh/docs/face_hostile_audit.md`
- PixARMesh paper: https://arxiv.org/abs/2603.05888
- PixARMesh repo: https://github.com/mlpc-ucsd/PixARMesh

## Executive Verdict

Do not change the main LATTICE reproduction path yet.

The main path remains:

```text
TRELLIS.2 coarse asset
    -> LATTICE/UltraShape-style high-fidelity watertight reference
    -> part structure
    -> chart retopo / projection
    -> Blender promotion gate
    -> optional autorigging
```

However, these methods add a very important topology lesson:

```text
LATTICE-style latents solve high-fidelity surface reference.
LATO-style VDF/T-Voxel supervision is the strongest path toward learned explicit artist topology.
FACE-style face tokens are a strong future explicit mesh head.
PixARMesh gives production conditioning/token-stream patterns.
3DILG gives simple irregular-grid implementation patterns for query-token smoke tests.
```

The best near-term change is to add a small topology sidecar to M0/M1, not to rewrite the LATTICE plan. Specifically, add VDF feature sampling and a topology-probe dataset path so we can later train a LATO-like explicit mesh decoder if the LATTICE + retopo pathway is not editable enough.

## Ranking Of Usefulness For ClearMesh

| Method | How useful now | Why |
| --- | ---: | --- |
| LATO | Very high | It directly attacks the same failure we care about: implicit/refined surfaces are high quality but do not expose artist topology. Its VDF/T-Voxel idea is the best explicit topology signal in this batch. |
| FACE | High | It gives a simple face-level autoregressive compression design that could become a future mesh head. It is slower than flow methods but much more direct than vertex-token AR. |
| PixARMesh | Medium-high | It is more scene-reconstruction than asset generation, but its pixel-aligned image conditioning, global context aggregation, token stream construction, and two-stage layout/object training are production-useful. |
| 3DILG | Medium | It is older neural-field work, but its FPS + KNN irregular latent grid is an excellent minimal pattern for query-token scaffolding and ablations. |

## LATO: The Most Important Add-On

### What It Does

LATO argues that vecset/sparse-voxel/SDF pipelines recover surface geometry but not explicit mesh topology. It introduces a Vertex Displacement Field (VDF), samples surface points, attaches displacement vectors from each sampled point to the three vertices of its containing triangle, then compresses these topology-rich features into sparse T-Voxels.

The core feature per sampled point is:

```text
x_k = [p_k, F(p_k), n(p_k)]
F(p_k) = {v - p_k | v is one of the three vertices of the face containing p_k}
```

Feature dimension for triangular meshes:

```text
p_k: 3
F(p_k): 9
n(p_k): 3
Total: 15
```

The decoder does not use marching cubes. It progressively subdivides and prunes latent voxels to instantiate vertex locations, then uses a connection head to predict edges between vertex pairs.

### Concrete Paper Details To Reuse

| Component | Paper detail | ClearMesh use |
| --- | --- | --- |
| Dense topology signal | Surface point gets displacement vectors to all three face vertices. | Add `VDFSampler` for topology-supervised datasets. |
| Sparse voxel encoder | PointNet per occupied voxel, mean pooling, sparse transformer. | Reuse as future LATO-like topology encoder once LATTICE smoke works. |
| Decoder | Initial pruning, then 3 stages of 8-way subdivision and pruning. | Future explicit topology decoder; not M0 mandatory. |
| Connection head | Positive samples are all GT edges; negatives are neighbor vertices plus random vertices. | Reuse exactly for edge prediction training. |
| Undirected edge invariance | Predict both orders `(i, j)` and `(j, i)`, average logits. | Low-risk implementation detail. |
| Imbalance handling | Asymmetric loss with negative down-weighting for final vertex occupancy. | Important for vertex voxel prediction. |
| Generative model | Two-stage flow: structure voxels, then topology features. | Matches our TRELLIS.2/LATTICE philosophy. |
| Conditioning | Topology feature generator conditions on `log(N_v)`. | Useful for vertex-count controllability/profile targets. |
| Scale | 819,200 sampled points per mesh, T-Voxels at 128^3 with 16 channels. | Full-scale target, not smoke target. |
| Runtime | H100 examples show roughly 5-10 seconds up to ~15k faces. | This is the production-speed north star if reproduced. |

### How This Changes Our Plan

Add one small sidecar to M0/M1:

```text
clearmesh/lattice/topology_vdf.py
    - sample mesh surface points with face indices
    - emit point, normal, face vertex displacements
    - voxel-bin VDF features for probes
    - save `.npz` topology sidecar beside LATTICE smoke assets
```

Add two tests:

```text
tests/test_topology_vdf.py
    - one triangle: displacement vectors reconstruct its vertices
    - two triangles sharing edge: positive edge set and negative sampler are stable
```

Do not build the full LATO decoder yet. First prove LATTICE-style geometry reproduction. But keep the VDF data path alive so we are not trapped later if retopology quality is not enough.

## 3DILG: Useful Irregular-Grid Pattern, Not A Replacement

### What It Does

3DILG represents shapes as fixed-length tuples `(x_i, z_i)` where each latent has an explicit 3D position. It constructs those positions with farthest point sampling (FPS), builds local KNN patches, encodes each patch with a PointNet-like module, then refines latents with a transformer.

The important implementation recipe:

```text
Input point cloud: N = 2048
Latent anchors: M = 512
Patch size: K = 32
Latent dictionary: D = 1024 for VQ runs
Coordinate quantization: 8-bit per axis
```

Repo details:

```text
/tmp/3DILG/modeling_vqvae.py
    Encoder.forward:
        fps(pos, batch, ratio=N/M)
        knn(pos, pos[idx], k=32)
        PointConv over local relative positions
        transformer with positional embeddings

/tmp/3DILG/modeling_prob.py
    ClassEncoder predicts x, y, z, latent components autoregressively
    model size: 24 layers, 16 heads, 1024 dim, 512 latent positions
```

### What To Reuse

Use 3DILG as the simplest sanity check for LATTICE query scaffolding:

```text
clearmesh/lattice/irregular_patches.py
    - FPS anchors
    - KNN local neighborhoods
    - sinusoidal 3D PE
    - optional VQ token ids for debug experiments
```

This helps answer:

```text
Can our query provider produce stable spatial anchors?
Can a small transformer learn local detail around those anchors?
Does irregular anchoring outperform regular voxel queries on toy assets?
```

### What Not To Reuse

Do not switch the LATTICE VAE to 3DILG's occupancy-field decoder. It still reconstructs via neural fields and is not a production artist-topology solution. Also, the official repo has training code but no pretrained models, so it is a reference implementation rather than a drop-in production component.

## FACE: Strong Future Mesh Head

### What It Does

FACE treats each triangle face as one autoregressive token instead of flattening a face into nine coordinate tokens. It uses:

```text
point cloud + normals
    -> VecSet encoder
    -> autoregressive face-token decoder
    -> CausalMLP predicts the 9 coordinate values inside each face
```

The important details from the paper:

```text
Face ordering: sort faces by lexicographic ZYX order of the minimum-coordinate vertex.
Input: 8192 surface points with normals.
VecSet: 2048 tokens, bottleneck dimension 64.
Quantization: [0, 127] for normal model, [0, 1023] for large model.
Model: 500M params for main model; 1.2B for large.
Training: around 130k meshes under 4000 faces; 100k steps on 8 A100 80GB.
Inference: deterministic top-1 AR sampling.
Large model: 65,536 input points, 380k high-quality meshes.
```

### What To Reuse Now

Add FACE-inspired utilities, not a full model:

```text
clearmesh/mesh_heads/face_tokens.py
    - canonical face sorting by ZYX min vertex
    - face-token sequence export
    - compression stats
    - roundtrip reconstruction sanity check
```

This will help evaluate any generated mesh as a potential training target for a future explicit mesh head.

### Why Not Build FACE First

FACE is promising but still autoregressive. It is much better compressed than vertex-token AR, but production latency is still likely worse than a LATO-style flow decoder. It also has no public repo in this audit, so a full reproduction would be from paper only.

## PixARMesh: Production Conditioning Lessons

### What It Does

PixARMesh is scene-level reconstruction, not our core asset pipeline. It directly predicts object pose and mesh tokens in one autoregressive stream, using EdgeRunner or BPT tokenizers. The repo is useful because it is complete and practical.

Repo implementation details worth stealing:

```text
/tmp/PixARMesh/src/data/mesh.py
    - back-project depth to point clouds
    - keep 2D pixel coordinates for each 3D point
    - sample object point cloud and context point cloud
    - normalize scene/object to unit cube

/tmp/PixARMesh/src/models/img_cond.py
    - frozen DINOv2 / DPT-style image feature encoder

/tmp/PixARMesh/src/models/edgerunner.py
/tmp/PixARMesh/src/models/bpt.py
    - grid_sample image features at point-projected 2D coordinates
    - cross-attend object point features to global scene context
    - inject condition embeddings into AR token stream

/tmp/PixARMesh/src/data/collator.py
    - construct `[condition prefix] + [pose sequence] + [mesh sequence] + eos`
    - token-type labels distinguish layout vs object loss
```

Paper/repo details:

```text
Object point clouds: 8192 for EdgeRunner, 4096 for BPT.
Global context point cloud: 16384.
Training: two stages, layout bootstrapping then full pose+mesh training.
Optimizer: AdamW, LR 1e-4 with 500 warmup steps and cosine decay to 1e-5.
Runtime: about 4.5 minutes per scene for EdgeRunner, 6.7 minutes for BPT on A100.
```

### What To Reuse

For our production visual UI and conditioning stack:

```text
- pixel-aligned feature sampling using `grid_sample`
- object/local point conditioning plus global context point conditioning
- separate condition-drop probabilities for robustness
- two-stage training when predicting structural tokens before geometry tokens
- token-type-aware loss scaling when a sequence mixes structure and mesh tokens
```

### What Not To Reuse

Do not make PixARMesh the main asset generator. It is scene-oriented, depends on segmentation/depth/layout assumptions, and the AR runtime is not aligned with our low-latency production path.

## Concrete Roadmap Delta

### M0 Delta: Add Three Tiny Utilities

Add these after the existing LATTICE query/voxel scaffold:

```text
1. VDF topology sidecar
   Path: clearmesh/lattice/topology_vdf.py
   Purpose: sample topology-supervised features for future LATO-like decoder.

2. Irregular patch provider
   Path: clearmesh/lattice/irregular_patches.py
   Purpose: 3DILG-style FPS/KNN anchors for query smoke tests.

3. Face-token canonicalizer
   Path: clearmesh/mesh_heads/face_tokens.py
   Purpose: FACE-style sequence stats and explicit mesh-head dataset prep.
```

### M1 Delta: Add Topology Probe Metrics

In VAE smoke runs, emit:

```text
- VDF reconstruction sanity on original mesh
- vertex occupancy recall at several voxel resolutions
- edge positive/negative sample balance
- connected component preservation after reference/retopo
- FACE token length / compression stats
```

### M2/M3 Delta: Choose Branch Based On Evidence

If LATTICE + retopo gives clean editable output:

```text
Keep VDF/FACE as dataset tooling only.
Prioritize production speed and projection quality.
```

If LATTICE + retopo is high fidelity but not editable enough:

```text
Start LATO-lite:
    - VDF encoder on 128^3 sparse voxels
    - hierarchical vertex occupancy decoder
    - connection head trained on existing artist meshes
    - condition on LATTICE/TRELLIS active voxels
```

If AR mesh heads remain too slow or fragmented:

```text
Do not chase MeshRipple/Silksong latency.
Move explicit-topology work toward LATO-lite or FACE-lite.
```

## Implementation Notes

### VDF Sampling

Recommended v0 behavior:

```text
mesh.sample(num_points, return_index=True)
face = mesh.faces[face_index]
verts = mesh.vertices[face]
feature = concat(point, verts[0] - point, verts[1] - point, verts[2] - point, normal)
```

For invariance experiments:

```text
- preserve face order for exact reconstruction target
- optionally sort the 3 displacement vectors by local angle/distance as an ablation
```

### Edge Training Samples

Recommended LATO-style edge candidate construction:

```text
positive_pairs = all unique mesh edges
negative_pairs = k nearest non-edge vertices per vertex + random non-edge vertices
logit(i, j) = 0.5 * (mlp(h_i || h_j) + mlp(h_j || h_i))
```

### FACE Ordering

Canonical face ordering for sequence export:

```text
1. quantize/sort vertices by ZYX
2. rotate each face so its minimum vertex index is first
3. sort faces by the ZYX position of their minimum-coordinate vertex
```

This is useful even if we never train FACE, because it gives stable sequence-level statistics for generated meshes.

## Risks

| Risk | Severity | Mitigation |
| --- | ---: | --- |
| LATO has no public code in this audit | High | Implement only VDF sampler now; defer full decoder until LATTICE smoke passes. |
| Full LATO training is expensive | High | Start with LATO-lite on curated artist meshes and low resolutions. |
| FACE is still autoregressive | Medium | Use it for tokenization and future high-quality head, not production v1. |
| PixARMesh is scene-specific | Medium | Borrow conditioning patterns only. |
| 3DILG uses implicit occupancy output | Medium | Borrow FPS/KNN anchoring only. |

## Updated Next Step

The next implementation step is still M0, but with a slightly richer scaffold:

```text
1. Build LATTICE query jitter and active voxel extraction.
2. Add VDF topology sidecar for the same assets.
3. Add 3DILG-style FPS/KNN irregular patch provider for ablations.
4. Add FACE-style face-token canonicalizer for mesh sequence stats.
5. Run all of the above on a tiny mesh fixture before using Thunder.
```

The high-level plan stays stable. The moat gets stronger: LATTICE gives us the high-fidelity watertight reference, and the VDF/FACE tooling gives us a credible path to learned explicit editable topology if chart retopo is not enough.
