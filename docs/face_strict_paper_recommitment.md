# FACE Strict Paper Recommitment

Date: 2026-05-05

## Decision

ClearMesh is returning FACE work to a strict paper-reproduction lane.

The indexed / half-edge / boundary-fill / geometry-edge-head experiments are
useful product research, but they are not FACE. They must not be used as
evidence for or against the FACE paper's central claim:

```text
end-to-end ARAE training makes the VecSet latent C rich, structured, and
semantically meaningful enough for autoregressive face generation.
```

## Paper Contract

The default FACE reproduction path should preserve these details unless a paper
detail is unavailable:

| Area | Required paper-faithful setting |
| --- | --- |
| Representation | one face token is nine quantized coordinate tokens |
| Coordinate bins | 128, integer range [0, 127] |
| Coordinate order | paper FACE ordering, tracked as z-y-x in tokenizer tests |
| Face order | lexicographic ZYX order by each face's minimum-coordinate vertex |
| Conditioning input | point cloud with normals |
| Point samples | 8192 for serious validation |
| Encoder | 3DShape2VecSet-style FPS query VecSet encoder |
| VecSet size | 2048 latent tokens for paper-knob validation |
| Bottleneck | latent dimension 64 |
| Decoder | causal face-token transformer |
| Decoder conditioning | cross-attention to VecSet in each decoder layer |
| Coordinate head | CausalMLP over the nine coordinate tokens |
| Optimizer | Muon, lr 6e-4, weight decay 0.1 |
| Augmentation | random rotation, flipping, independent axis scaling |
| Face cap | fewer than 4000 faces for paper-scale corpus |
| Training budget | 100K steps for real paper-scale comparison |
| Dataset | curated Objaverse-style mesh subset, target around 130K meshes |

## Allowed Fills

These are acceptable because the paper does not fully specify them or because
public code is unavailable:

- Exact CausalMLP internals: use the closest cited/public implementation, and
  preserve causal factorization over the nine coordinate slots.
- Exact 3DShape2VecSet implementation details: use public 3DShape2VecSet logic
  where available; otherwise keep FPS queries, cross-attention, latent
  self-attention, GEGLU FFN, and bottleneck structure intact.
- EOS / termination: implement as an auxiliary head only because the paper is
  under-specified; report GT-face-count metrics separately from predicted-count
  metrics.
- Mesh cleanup before tokenization: allowed only to produce a clean curated
  training subset, not to patch generated outputs during paper evaluation.
- Muon availability: if native `torch.optim.Muon` is unavailable, use the local
  Muon fallback for hidden matrices and report the fallback explicitly.

## Disallowed In The Paper Lane

These are ClearMesh production experiments and must stay out of paper-faithful
FACE results unless clearly labeled as ablations:

- explicit vertex table plus indexed faces
- 4096-bin or 512-bin coordinate vocabularies as primary FACE evidence
- constrained half-edge decoding
- edge-action or edge-choice heads
- boundary-fill or centroid caps during generated-output evaluation
- unpinch cleanup as a success criterion
- chart/quad remeshing as part of FACE quality reporting
- geometry-aware pointer heads over vertex tables
- topology auxiliary losses not described by FACE

## Next Strict Validation

The next meaningful run should be a FACE paper-knob validation, not another
indexed topology run:

```text
strict curated mesh targets
  -> 128-bin paper coordinate tokens
  -> point cloud + normals, 8192 samples
  -> Shape2VecSet-style VecSet, 2048 tokens, latent dim 64
  -> causal face decoder with layerwise cross-attention
  -> CausalMLP coordinate head
  -> Muon lr 6e-4 wd 0.1
  -> online paper augmentation
  -> train/eval with raw generated outputs, no boundary-fill promotion
```

Use `scripts/thunder/face_paper_a100_probe.sh` for the bounded A100 gate. Scale
to the full 130K / 100K-step run only after this paper-knob lane shows healthy
train learning, improving held-out metrics, and visually coherent free-running
AR samples.

