# FACE-Q Paper Hostile Audit - 2026-05-15

This is a hostile implementation audit for the active ClearMesh FACE-Q run:

- Remote root: `/tmp/clearmesh_faceq_partial_gate_20260514_faceq_big510m_vecset_cross_geom_33k_100k_v2`
- Active run: `runs/faceq_vecset_small114m_lazy_33k_100k_20260514`
- Checkpoint under sidecar eval: `checkpoint.step040000.pt`
- Training target: indexed FACE-Q strict-token samples, not coordinate-bin paper FACE samples

Sources checked:

- FACE paper v2 local PDF: `/Users/Ashar/Downloads/2603.01515v2.pdf`
- arXiv primary record: `https://arxiv.org/abs/2603.01515`
- Active trainer: `scripts/research/train_face_indexed_conditioned_tiny.py`
- Active eval: `scripts/research/eval_face_indexed_conditioned_tiny.py`
- Current model implementation: `clearmesh/mesh_heads/face_arae.py`
- Paper-faithful coordinate trainer: `scripts/research/train_face_paper_faithful.py`

## Executive Finding

The active run has fixed the most serious earlier miss: it uses a real FPS-query VecSet conditioning path and decoder cross-attention. It is therefore much closer to the FACE paper than the old pooled-condition models.

But it is not a paper-identical FACE reproduction. It is a FACE-inspired indexed-topology variant, FACE-Q. That matters when interpreting topology metrics and scale expectations.

The current result should be treated as a promising scale signal, not as proof that we have replicated FACE.

## Paper Claims To Match

From the paper:

- Sequence unit: one triangle face as one autoregressive token.
- Target representation: each face is a flattened 9D coordinate vector, quantized to integer bins.
- Face order: lexicographic ZYX order of the face's minimum-coordinate vertex.
- Encoder: 3DShape2VecSet-style FPS query set, cross-attention from query points to full point cloud, Transformer encoder refinement.
- Decoder: causal self-attention over previous face tokens, cross-attention to the VecSet in decoder layers.
- Face embedding: lightweight MLP maps one face to one model token.
- Face decoding head: CausalMLP predicts the nine quantized coordinate tokens inside each face.
- Scale: approximately 500M parameters.
- Capacity: encoder 8 layers hidden 768, decoder 24 layers hidden 1024.
- Input: 8192 sampled surface points with normals.
- VecSet: 2048 latent tokens, bottleneck dimension 64.
- Data: around 130k Objaverse meshes with fewer than 4000 faces.
- Quantization: vertex positions normalized and quantized to integers in `[0, 127]`.
- Augmentation: random rotation, flipping, and independent per-axis scaling.
- Optimizer/run: Muon, lr 6e-4, weight decay 0.1, 100k steps on 8x A100 80GB.
- Inference: deterministic top-1 autoregressive decoding.

## Active Run Match/Mismatch Table

| Paper item | Active FACE-Q run | Hostile verdict |
| --- | --- | --- |
| One-face-one-token outer sequence | Yes: one indexed face step per AR position | Match in sequence granularity |
| 9 coordinate-bin target face | No: predicts indexed vertices from a per-sample vertex table | Major intentional deviation |
| Face order ZYX min vertex | Partially: strict data lineage uses `rotate_min_zyx` / boundary-growth indexed order | Needs per-run manifest proof |
| VecSet encoder | Yes: FPS query points, point cross-attention, Transformer encoder, bottleneck | Good, but dependency-light implementation |
| Decoder cross-attention | Yes for `decoder_backend=cross_attn` | Good |
| VecSet tokens | 512 | Paper uses 2048; current run is 4x smaller |
| VecSet bottleneck | 64 | Match |
| Surface points + normals | 8192 with `surface_points` + `surface_normals` | Match |
| Model scale | 106.7M params | Far below 500M |
| Encoder depth/width | 4 layers, hidden 640 | Below paper 8 layers, hidden 768 |
| Decoder depth/width | 12 layers, hidden 640 | Below paper 24 layers, hidden 1024 |
| Data scale | about 21.9k strict indexed samples | Below paper 130k |
| 100k steps | Active run targets 100k | Match if it completes |
| Muon/lr/wd | Muon, lr 6e-4, wd 0.1 | Match |
| Online augmentation | Not evident in active indexed trainer | Major mismatch unless upstream token data already encodes augmentation |
| Deterministic top-1 inference | Greedy/top-k constrained eval paths; boundary constraints added | Not paper-identical; product-topology-oriented |

## Highest-Risk Gaps

1. Target representation gap.

The paper predicts coordinate bins; FACE-Q predicts indexed vertex ids from an explicit vertex table. This is not just an implementation detail. It changes what the decoder must learn: coordinate reconstruction vs graph-index/topology selection. FACE-Q may be better for editability, but paper Chamfer results cannot be directly used as expected FACE-Q performance.

2. Capacity gap.

The active run is about 106.7M parameters. The paper reports about 500M parameters with a decoder-heavy architecture. This gap is large enough that a 40k or even 100k-step run may underfit exact token decisions despite looking promising on geometry.

3. VecSet token gap.

The active run uses 512 condition tokens. The paper uses 2048 VecSet tokens. This is a plausible bottleneck for high-detail reconstruction and first-face/early-face disambiguation.

4. Augmentation gap.

The paper trains with rotation, flipping, and per-axis scaling. The active indexed trainer does not visibly apply online augmentation/re-tokenization in the same way the paper-faithful coordinate trainer does. If indexed data is fixed, the model may learn a narrower canonicalization distribution.

5. Evaluation strictness gap.

Our topology/editability gates are stricter than the paper's headline Chamfer/Hausdorff reconstruction tables. A model can improve paper-style Chamfer while still failing watertight/editable production criteria.

6. Decode gap.

The paper says deterministic top-1 AR decoding. Our FACE-Q eval uses topology-aware constrained decoding and boundary budget / vertex-link constraints. That is appropriate for product topology, but it is not a clean paper inference reproduction.

## Good Signs So Far

- The current model now has a real VecSet path: FPS query selection, cross-attention from queries to the point set, Transformer encoder refinement, and decoder cross-attention.
- The 40k checkpoint has nonzero held-out teacher-forced watertight results. That is materially better than a totally dead topology circuit.
- Training is still improving and has not hit NaNs/OOM/stall after switching to the 106.7M lazy true-VecSet run.
- The 40k AR eval crash was eval-only: geometry-mode causal corner logits require `vertex_table`, and training already passed it correctly.

## Current Interpretation

The 40k checkpoint is a good sign, but not a green light by itself.

Realistic interpretation:

- Positive: held-out TF watertightness exists at 40k on only about 20k strict samples.
- Caution: TF token accuracy is still around two-thirds, and exact topology is brittle.
- Caution: free-run AR still has to be remeasured after the eval path fix.
- Caution: current model/data/VecSet size are below the paper regime.

Scaling should help if the failure is mostly capacity/data/conditioning. Scaling will not automatically fix a target-representation or decode-objective mismatch.

## Immediate Follow-Ups

1. Finish the 40k AR sidecar rerun after passing `vertex_table` into eval-only causal corner calls.
2. Compare 40k TF vs AR:
   - TF improves but AR fails: exposure bias / first-face ambiguity / constrained decoding remains dominant.
   - TF and AR both improve: scale signal is much stronger.
   - TF stagnates: capacity/objective/data representation issue.
3. Let the active 100k-step run continue unless it stalls or diverges.
4. Do not call this a FACE reproduction until we run at least one paper-closer configuration:
   - 2048 VecSet tokens
   - decoder 24 layers, hidden 1024
   - encoder 8 layers, hidden 768
   - 130k+ strict samples
   - online augmentation or a justified indexed equivalent
   - DDP/8x A100 smoke first

