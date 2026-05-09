# FACE Methods Parity Hostile Audit

Date: 2026-05-03
Paper audited: `/Users/Ashar/Downloads/2603.01515v2.pdf`
Relevant implementation files:

- `clearmesh/mesh_heads/face_tokens.py`
- `clearmesh/mesh_heads/face_indexed.py`
- `clearmesh/mesh_heads/face_arae.py`
- `scripts/research/train_face_indexed_conditioned_tiny.py`
- `scripts/research/eval_face_indexed_conditioned_tiny.py`

## Bottom Line

Do not interpret our failed indexed CausalMLP smoke as a failure of the FACE paper's CausalMLP.

Our tested CausalMLP is not the same object as the paper's CausalMLP. FACE uses CausalMLP to decode nine quantized coordinate tokens inside one face token. Our failed test used a corner-causal head to decode three vertex-table indices inside an explicit indexed-topology variant. That is a reasonable ClearMesh experiment, but it is off-paper.

The more likely explanation is method mismatch, not that CausalMLP is bad.

## What FACE Actually Specifies

From the methods and implementation details:

- Input shape is point cloud with normals.
- Encoder is 3DShape2VecSet-style.
- FPS/downsampled point cloud queries are used for VecSet.
- VecSet latent has 2048 tokens and bottleneck dimension 64.
- Face sequence is sorted by lexicographical ZYX order of each face's minimum-coordinate vertex.
- One face is one transformer token.
- Previous face is embedded as a 9D face vector by a Face Pooling MLP.
- Transformer decoder has causal self-attention over face tokens and cross-attention to VecSet at every layer.
- CausalMLP decodes each latent face token into nine quantized coordinate tokens autoregressively.
- Training loss is coordinate-token cross entropy over all faces and all nine coordinates.
- Coordinates are normalized and quantized to `[0, 127]` for the base model.
- Training uses random rotation, flipping, and independent axis scaling.
- Reported ARAE scale is roughly 500M parameters, 100K steps, 8 A100 80GB GPUs, around 130K meshes under 4000 faces.
- Inference uses deterministic top-1 autoregressive sampling.

## Parity Matrix

| Method Area | Paper | Our Current FACE-Lite v2 | Parity | Risk |
| --- | --- | --- | --- | --- |
| Face ordering | ZYX by minimum-coordinate vertex | Global vertex ZYX, rotate each face to min index, sort by face indices | Partial | Tie-breaking and within-face token order may differ from paper. |
| Coordinate order | Figures show per-vertex `z, y, x` token order | We store vertices as `x, y, z` arrays | Low/partial | Consistent train/eval still works, but it is not paper-exact. Causal within-face dependencies may change. |
| One face = one token | 9D coordinate face projected by MLP | Indexed v2 projects three vertex-index embeddings | Conceptual only | CausalMLP ablation is not transferable to indexed corner decoding. |
| CausalMLP target | Nine coordinate tokens | Three vertex-table indices | Not faithful | Main reason our CausalMLP result should not be read as paper evidence. |
| Shape encoder | FPS queries, cross-attention to full point set, transformer encoder VecSet | Mean/max point MLP plus learned condition tokens and vertex pooling | Not faithful | Huge. We do not have the paper's latent geometry interface. |
| Decoder conditioning | Cross-attention to VecSet at each decoder layer | Prefix condition tokens prepended to self-attention stream | Approximation | We inject global context weakly compared with paper. |
| Model scale | 500M params, decoder larger, hidden 1024, 24 decoder layers | Tiny smoke: hidden 96, 2 layers | Not comparable | CausalMLP benefit was reported at much larger capacity. |
| Point count | 8192 surface points plus normals | Smokes use 128 train points / 256 dataset points | Not comparable | Thin/topological structures are under-conditioned. |
| Dataset | Around 130K curated Objaverse meshes under 4000 faces | 10 synthetic primitives in tiny A/B | Not comparable | No reason to expect paper-level generalization or topology rhythm. |
| Augmentation | random rotation, flip, independent axis scaling | no epoch-level augmentation in indexed training | Missing | Ordering and invariance may be brittle. |
| Face count / EOS | Paper under-specifies exact termination; reports variable mesh generation | We often use GT face count for eval; count head is our addition | Partial/off-paper | GT-count eval is good for isolating token quality, but not production inference. |
| Inference | deterministic top-1 | parallel path can be top-1 or constrained top-k reranking; causal path enumerates top-k | Partial | Our constrained decoder optimizes topology, but it is not the paper's inference. |
| Token objective | CE over 9 coordinate bins per face | CE over 3 vertex indices, optional topology aux CE | Off-paper | Topology aux helps our representation, but it is not FACE. |

## Findings

### 1. CausalMLP Did Not Get A Fair Paper-Faithful Test

The FACE paper's CausalMLP is nested inside coordinate decoding:

```text
latent face token -> coordinate 1 -> coordinate 2 -> ... -> coordinate 9
```

Our failed smoke tested:

```text
latent face token -> vertex index 1 -> vertex index 2 -> vertex index 3
```

Those are different distributions. Coordinate tokens have strong local numeric dependencies inside a face. Vertex-table indices are arbitrary after canonical sorting and encode global identity, not local coordinate value. A causal dependency among arbitrary vertex IDs can easily lower teacher-forced loss without improving topology.

This matches our observed result:

```text
corner-causal indexed + edge constraints:
  best_loss: 0.1332
  raw watertight: 0/6
  cleanup watertight: 1/6
  mean boundary edges: 7.00

parallel indexed + edge constraints:
  raw watertight: 3/6
  cleanup watertight: 3/6
  mean boundary edges: 4.33
```

Lower teacher-forced index loss did not mean better mesh topology.

### 2. Our Encoder Is The Biggest Off-Paper Shortcut

FACE leans hard on a VecSet encoder:

```text
8192 points + normals
  -> FPS/downsampled point queries
  -> cross-attention to full point set
  -> transformer encoder
  -> 2048 VecSet tokens
```

Our tiny indexed model uses:

```text
128 point samples
  -> point MLP
  -> mean/max pooling
  -> add learned condition queries
  -> prepend condition tokens
```

That is useful for smoke tests, but it removes the central representation the paper claims is doing the heavy lifting.

If we want to test FACE honestly, the next model must replace pooled condition tokens with at least a small VecSet-style encoder and decoder cross-attention.

### 3. Our Face Tokenizer Is Close But Not Paper-Exact

Good:

- We quantize to 128 bins.
- We normalize to a centered cube.
- We sort vertices/faces deterministically.
- We weld repeated quantized vertices on coordinate-token decode.

Potential mismatch:

- The paper sorts faces by ZYX order of the minimum-coordinate vertex.
- Our coordinate tokenizer sorts vertices by ZYX, rotates each face to the minimum sorted vertex index, then sorts faces by the full rotated index triple.
- The paper figures show coordinate token order as `z, y, x` per vertex; our arrays are `x, y, z`.

This probably does not explain the indexed CausalMLP failure, but it matters for a paper-faithful coordinate CausalMLP reproduction.

### 4. Our Indexed v2 Representation Is Better For ClearMesh, But It Is Not FACE

The indexed representation is a product-driven deviation:

```text
quantized vertex table: V x 3
faces: F x 3 vertex indices
```

This makes vertex reuse explicit and gives us topology gates. That is good for ClearMesh.

But FACE's compression claim is based on processing one face token while predicting coordinate bins. It does not require a separate vertex table or index distribution. Therefore, CausalMLP and face-pooling details from FACE cannot be copied naively into indexed v2 and expected to work.

### 5. The New Topology-Target Head Is More Aligned With Our Indexed Variant Than CausalMLP

The latest topology-aux smoke gives useful evidence:

```text
previous parallel indexed + edge constraints:
  raw watertight: 3/6
  cleanup watertight: 3/6
  mean boundary edges: 4.33
  mean Chamfer L2: 0.04818

topology aux + closure-target bonus 2.0:
  raw watertight: 3/6
  cleanup watertight: 4/6
  mean boundary edges: 3.67
  mean Chamfer L2: 0.04476
```

Ablation on the same topology-trained checkpoint:

```text
closure target bonus 0.0:
  raw watertight: 2/6
  cleanup watertight: 2/6
  mean boundary edges: 6.17

closure target bonus 2.0:
  raw watertight: 3/6
  cleanup watertight: 4/6
  mean boundary edges: 3.67

closure target bonus 4.0:
  raw watertight: 4/6
  cleanup watertight: 4/6
  mean boundary edges: 4.33
  mean Chamfer worsened to 0.05096
```

Interpretation: our indexed representation benefits from explicit topology-state supervision. This is not in FACE, but it is a rational adaptation for production mesh editability.

## Likely Mis-Implementations Or Under-Specified Pieces

### High Confidence

1. CausalMLP was tested on the wrong target type.
2. We do not implement the paper's VecSet encoder.
3. We do not implement layerwise decoder cross-attention to VecSet.
4. We are far below paper scale and data regime.
5. We are not doing paper augmentation.
6. Our coordinate order is probably not paper-exact.

### Medium Confidence

1. Face ordering tie-breaks may differ from the paper.
2. Our use of GT face count hides termination errors.
3. Our point sampling is too small to test topology preservation.
4. Our synthetic primitives do not stress the same distribution as Objaverse/Toys4K/Famous.

### Low Confidence / Paper Under-Specified

1. Exact EOS/termination mechanism.
2. Exact Face Pooling MLP architecture.
3. Exact CausalMLP implementation details.
4. Exact mesh cleanup/validation before tokenization.

## Corrective Plan

### Step 1: Paper-Faithful Coordinate FACE Micro-Repro

Implement a separate coordinate-token FACE path, not indexed v2:

```text
point cloud + normals
  -> small VecSet encoder
  -> decoder with causal self-attn + cross-attn
  -> face-pooling MLP for previous 9D face
  -> CausalMLP over 9 coordinate tokens
```

Keep it tiny but structurally faithful:

```text
points: 1024 initially, then 8192
VecSet tokens: 128 initially, then 512/2048
decoder hidden: 256 initially, then 768+
faces: <=256 initially, then <=4000
quantization: 128 bins
coordinate order: z,y,x per vertex for parity test
face order: exact ZYX-min-coordinate face order
```

Success criterion:

```text
CausalMLP coordinate head beats parallel coordinate head on the same coordinate-token model.
```

If it does, the paper is vindicated and our previous CausalMLP test was simply not comparable.

### Step 2: VecSet Encoder Swap For Indexed v2

Once coordinate FACE parity is validated, bring only the useful part into ClearMesh indexed v2:

```text
replace pooled point MLP condition with VecSet encoder
replace prepended condition tokens with decoder cross-attention
keep explicit indexed topology
keep topology closure head
```

Success criterion:

```text
indexed v2 + VecSet + topology head beats current topology-aux smoke on watertightness, boundary edges, Chamfer, and visual contact sheet.
```

### Step 3: Stop Calling The Indexed Corner Head "CausalMLP" In Planning

Rename the experiment mentally and in docs:

```text
old: CausalMLP indexed
new: corner-causal index decoder
```

This avoids overloading the FACE paper term.

### Step 4: Bigger Validation Run Before More Architecture Speculation

Run with:

```text
synthetic_count: 100-500
steps: 5K-20K
point_samples: 1024+
hidden: 256+
layers: 6+
profiles:
  coordinate FACE parallel vs coordinate FACE CausalMLP
  indexed v2 baseline vs indexed v2 topology aux
  indexed v2 pooled condition vs indexed v2 VecSet condition
```

Use held-out meshes and visual sheets, not train-set smokes.

## Recommendation

Short term:

```text
Keep indexed v2 topology-aux as the ClearMesh production-facing branch.
Do not promote boundary-edge forced decoding.
Do not judge FACE CausalMLP from the indexed corner-causal smoke.
```

Next implementation target:

```text
Build paper-faithful coordinate FACE micro-repro with VecSet + true 9-coordinate CausalMLP.
Then transplant VecSet conditioning into indexed v2 if it proves useful.
```

## Full Methods Walkthrough: Beyond CausalMLP

This section audits the whole FACE method stack as a system. The screenshot/figure is important because the architecture is not just "a decoder head"; it is a coupled encoder-decoder-tokenization recipe.

### A. Input Representation: Point Cloud With Normals

Paper method:

```text
mesh surface -> 8192 sampled points with normals -> encoder input
```

The methods notation writes `P in R^{m x 3}`, but the figure and implementation details explicitly say XYZ + normal / points with normals. Treat the real input as six channels.

Our current parity:

```text
surface_points + surface_normals are stored in dataset shards
indexed v2 smoke trains with 128 point samples, dataset built with 256 samples
```

Audit result:

```text
Not paper-faithful. The data fields exist, but the point count is tiny and the encoder cannot preserve local structure the way the paper's VecSet encoder can.
```

Production implication:

```text
Thin structures, handles, spokes, holes, and branching objects cannot be judged from our 128-point smoke.
```

Corrective action:

```text
Move validation runs to 1024 points immediately, then 8192 for paper-faithful runs.
Add feature-biased sampling later, but first match uniform surface sampling + normals.
```

### B. Face Sequence Ordering

Paper method:

```text
sort all faces by lexicographical ZYX order of each face's minimum-coordinate vertex
```

Paper ablation:

```text
BFS:           HD 0.728, CD 0.528
DFS:           HD 0.171, CD 0.077
ZYX-component: HD 0.110, CD 0.045
ZYX:           HD 0.103, CD 0.047
```

Our current parity:

```text
coordinate tokenizer:
  quantize vertices
  sort vertices by ZYX
  rotate face to minimum sorted vertex index
  sort face rows by the rotated face-index triple

indexed tokenizer:
  quantize/weld vertices
  sort vertex table by ZYX
  rotate face to minimum sorted vertex index
  sort by index triple
```

Audit result:

```text
Partial parity, not exact parity.
```

Why this matters:

```text
The paper sorts by the minimum-coordinate vertex, not necessarily by the entire remapped index triple.
The paper figure presents each face token as z,y,x per vertex. Our token arrays are x,y,z.
A causal coordinate decoder is sensitive to within-face coordinate order.
```

Corrective action:

```text
Add a paper_exact_face_order mode:
  - derive each face's minimum-coordinate vertex in quantized coordinate space
  - sort faces by that vertex's z,y,x coordinate
  - use deterministic tie-breakers only after this primary key
Add a paper_exact_coord_order mode:
  - store per vertex as z,y,x for the coordinate FACE path
  - keep x,y,z only in indexed v2 if we want product convenience
```

### C. One Face, One Transformer Token

Paper method:

```text
previous face fi-1 is a 9D coordinate vector
Face Pooling MLP embeds that 9D face vector into one d_model token
Transformer autoregresses over face tokens, not coordinate tokens
```

Our current parity:

```text
coordinate FACE-lite path: closer conceptually, but tiny and not fully paper structured
indexed v2 path: face token is three vertex-index embeddings projected to one face token
```

Audit result:

```text
Coordinate path has conceptual parity.
Indexed v2 does not; it is a ClearMesh-specific representation.
```

Why this matters:

```text
The paper's compression ratio and CausalMLP ablation assume a coordinate-face token, not a vertex-index face token.
```

Corrective action:

```text
Keep two tracks clearly separated:
  1. paper-faithful coordinate FACE for reproduction/parity
  2. ClearMesh indexed FACE-v2 for production topology
Do not use paper ablations as proof for indexed v2 until re-tested there.
```

### D. VecSet Shape Encoder

Paper method:

```text
input points/normals
  -> FPS/downsampled query points
  -> CrossAttn(Q=query projections, K/V=full point projections)
  -> Transformer encoder stack
  -> VecSet C with 2048 tokens and bottleneck dimension 64
```

Paper ablation:

```text
learnable queries:   HD 0.132, CD 0.058
downsample queries:  HD 0.103, CD 0.047
```

Our current parity:

```text
point MLP -> mean/max pool -> condition projection -> learned condition queries
indexed v2 also pools vertex-table hidden states
```

Audit result:

```text
Major mismatch. This is probably the single biggest off-paper shortcut.
```

Why this matters:

```text
FACE's decoder is not asked to infer the whole shape from one pooled vector.
It receives a spatial latent set through cross-attention at every layer.
Our pooled condition can collapse local topology cues before decoding even begins.
```

Corrective action:

```text
Implement MiniVecSet first:
  - FPS/query selection from sampled points
  - cross-attention from query tokens to point tokens
  - transformer encoder over query tokens
  - start with 128 query tokens, hidden 256
  - scale to 512/2048 query tokens after smoke
```

### E. Decoder Conditioning Structure

Paper method:

```text
for each decoder layer:
  H' = causal self-attention over previous face tokens
  H_next = cross-attention(Q=H', K=VecSet, V=VecSet)
```

Our current parity:

```text
condition tokens are prepended to the causal token stream
condition tokens cannot inspect generated tokens, but generated tokens attend to them through self-attention
```

Audit result:

```text
Approximation, not parity.
```

Why this matters:

```text
Prepended condition tokens are a weaker and less structured version of layerwise encoder-decoder cross-attention.
FACE injects global shape information at every decoder layer.
```

Corrective action:

```text
Replace the current TransformerEncoder-as-decoder with an explicit decoder block:
  - causal self-attention over face tokens
  - cross-attention into VecSet tokens
  - MLP/residual/norm
```

### F. Coordinate Decoding Head

Paper method:

```text
latent face vector hi -> CausalMLP -> 9 coordinate logits
loss = mean CE over N faces x 9 coordinate tokens
```

Paper ablation:

```text
parallel decode:   HD 0.426, CD 0.239
attention-based:   HD 0.132, CD 0.064
CausalMLP:         HD 0.103, CD 0.047
```

Our current parity:

```text
failed smoke tested corner-causal index decoding, not 9-coordinate CausalMLP
```

Audit result:

```text
Not a valid test of the paper's CausalMLP claim.
```

Corrective action:

```text
Build true coordinate CausalMLP path:
  - target sequence length per face = 9
  - target vocabulary = coordinate bins, not vertex indices
  - input prefix uses previous quantized coordinate tokens in paper order z,y,x
```

### G. Training Objective

Paper method:

```text
single reconstruction objective: CE over coordinate bins
end-to-end encoder + decoder training
```

Our current parity:

```text
indexed v2: CE over vertex indices
optional topology closure-count auxiliary loss
optional count-head loss
```

Audit result:

```text
Off-paper, intentionally.
```

Why this matters:

```text
Our topology aux loss is a good ClearMesh adaptation, but it is not part of FACE.
It should be evaluated as a production topology improvement, not as FACE reproduction evidence.
```

Corrective action:

```text
For paper reproduction:
  - no topology aux in the first coordinate FACE parity run
  - compare parallel vs CausalMLP on exactly CE coordinate loss
For production indexed v2:
  - keep topology aux because it improved cleanup watertightness in tiny smoke
```

### H. Training Scale, Optimizer, Augmentation, Data

Paper base model:

```text
500M parameters
encoder: 8 layers, hidden 768
likely decoder: 24 layers, hidden 1024 (paper text appears to repeat "encoder" here; context says decoder)
8192 points with normals
VecSet: 2048 tokens, bottleneck dim 64
Objaverse subset: ~130K meshes, <4000 faces
quantization: [0,127]
augmentation: random rotation, flipping, independent axis scaling
optimizer: Muon, lr 6e-4, weight decay 0.1
training: 100K steps on 8 A100 80GB
```

Paper large model:

```text
1.2B parameters
65,536 sampled points
quantization [0,1023]
internal dataset: 380K high-quality meshes
```

Our current parity:

```text
hidden 96, 2 layers, 4 heads
10 synthetic train examples
300 steps
128 point samples
AdamW, lr 3e-4
no rotation/flip/axis augmentation loop
```

Audit result:

```text
Not comparable. Our smokes only validate plumbing and directionality.
```

Corrective action:

```text
Do not draw quality conclusions from 10-shape smokes.
Use them only to reject broken branches.
Next meaningful validation should be 100-500 meshes, 5K-20K steps, 1024+ points.
```

### I. Inference And Termination

Paper method:

```text
deterministic autoregressive top-1 sampling
```

Unclear/under-specified:

```text
EOS / face-count / termination details are not described clearly in the main paper.
```

Our current parity:

```text
for evaluation we often use GT face count to isolate token quality
count head exists but is not paper-specified
constrained top-k reranking is our production topology adaptation
```

Audit result:

```text
Partial/off-paper.
```

Corrective action:

```text
For paper-faithful reproduction, implement explicit EOS or face-count protocol and report both:
  - GT face count reconstruction quality
  - predicted/EOS face count quality
For production, keep constrained reranking but label it as ClearMesh-specific.
```

### J. Image-To-Mesh Latent Diffusion

Paper method:

```text
train ARAE first
freeze/use FACE decoder
train 350M DiT in VecSet latent space
condition DiT on DINOv3 image features
50K mesh subset, 10 renderings/model
400K steps on 32 A100 80GB
Muon lr 1e-4, no weight decay
Euler solver, 100 latent sampling steps
```

Our current parity:

```text
none for FACE; ClearMesh uses TRELLIS/LATTICE as upstream geometry providers
```

Audit result:

```text
Out of scope for immediate production unless the ARAE reconstruction head is strong.
```

Corrective action:

```text
Do not build FACE image DiT yet.
Use TRELLIS/LATTICE/UltraShape as geometry source, then run FACE-like topology reconstruction.
Only revisit image DiT after coordinate FACE/VecSet reproduction is proven.
```

## Revised Highest-Risk Method Gaps

Ranked by likely impact on our poor visual results:

```text
1. No VecSet encoder / no downsampled FPS query cross-attention.
2. No layerwise decoder cross-attention to VecSet.
3. Tiny data/scale/point count far below paper regime.
4. Coordinate order and face ordering are not paper-exact.
5. CausalMLP tested on index targets instead of coordinate targets.
6. Missing paper augmentations.
7. GT face-count eval hides termination difficulty.
8. Synthetic primitive distribution is too narrow.
```

## Updated Implementation Plan

### Milestone 1: Paper-Exact Tokenizer Mode

Add a coordinate FACE mode with:

```text
- z,y,x coordinate order
- face order by min-coordinate vertex in z,y,x lexicographic order
- deterministic tie-breaks
- quantization [0,127]
- decode weld by identical quantized coordinates
```

### Milestone 2: MiniVecSet Encoder

Implement:

```text
- input: point XYZ+normal
- FPS/downsampled queries
- query-to-point cross-attention
- transformer encoder over query tokens
```

Start small:

```text
1024 input points
128 VecSet tokens
hidden 256
4 encoder layers
```

Scale target:

```text
8192 input points
512 then 2048 VecSet tokens
hidden 768
8 encoder layers
```

### Milestone 3: True Decoder Blocks

Replace prepended condition-token transformer with:

```text
causal self-attention over face tokens
cross-attention to VecSet at each layer
feed-forward block
```

### Milestone 4: Coordinate CausalMLP A/B

Run faithful A/B:

```text
parallel coordinate decode vs attention decode vs CausalMLP
same tokenizer
same VecSet encoder
same train set
same inference
```

Success criterion:

```text
CausalMLP must beat parallel on held-out Chamfer/Hausdorff/topology metrics.
```

### Milestone 5: Bring Lessons Back To Indexed v2

If paper-faithful FACE works:

```text
- keep VecSet encoder
- keep decoder cross-attention
- swap coordinate CausalMLP for indexed topology head
- keep topology closure-target loss/reranking
```

This gives us the best of both worlds:

```text
FACE's spatial latent/decoder architecture
+ ClearMesh's explicit topology/eval gates
```

## Decision

The current evidence does not say "FACE methods are wrong." It says our FACE-lite v2 has only partial method parity. The most important thing to fix is not CausalMLP first; it is the encoder-decoder interface:

```text
FPS/downsampled query VecSet + decoder cross-attention
```

After that, we can test true coordinate CausalMLP honestly.
