# FACE Scale and Generalization Reassessment

Date: 2026-05-07

## Question

Would FACE generalize if we scaled to 300k+ high-quality meshes, or should we
perfect train autoregressive closure first?

## Short Answer

Scaling is likely necessary for FACE to generalize, and the paper supports that.
But scaling is not sufficient by itself. The current run proves that the
paper-faithful AR path can nearly memorize clean training meshes, while also
showing that the current tiny no-augmentation split has no held-out
generalization signal.

Do not launch a 130k-350k production run from this evidence alone. The correct
next step is a bounded generalization rung on a larger curated split, with
paper-faithful knobs restored as far as memory allows.

## Current Run

Local archive:

```text
.codex_outputs/face_paper_existing_split_setup_rung1_closure64_noaug_cachefps_b4_20260507_042201/harvest/clearmesh_face_paper_existing_split_gate_rung1_closure64_noaug_cachefps_b4_20260507_042201
```

Thunder:

```text
instance: 0 / 2nhex24x
gpu: 1x A6000
status: harvested, then deleted
```

Settings:

```text
train samples: 56
test samples: 8
bins: 128
max faces: 512
points: 8192 XYZ+normal
VecSet tokens: 2048
latent dim: 64
hidden: 384
encoder layers: 4
decoder layers: 8
optimizer: Muon
precision: bf16
augmentation: disabled
FPS index cache: enabled
trained checkpoint selected at: step 2000
```

## Results

| Split / mode | Token accuracy | Edge F1 | Mean boundary edges | Watertight |
| --- | ---: | ---: | ---: | ---: |
| Train teacher-forced | 0.999707 | 0.998322 | 3.91 | 46 / 56 |
| Train autoregressive | 0.999931 | 0.999629 | 1.06 | 26 / 32 |
| Test teacher-forced | 0.066298 | n/a | 1347.63 | 0 / 8 |
| Test autoregressive | 0.044614 | n/a | 1090.88 | 1 / 8 |

Interpretation:

```text
Train AR is close enough to prove the model path is not broken.
Held-out generalization is absent at this tiny scale.
```

The train AR result matters because it rules out the worst failure mode: a
decoder/cache/tokenization mismatch that cannot even free-run on memorized
targets. The held-out result matters because it blocks any honest claim that we
are ready to scale blindly.

## What The Paper Says That Matters

FACE defines a face sequence sorted by lexicographic ZYX order of each face's
minimum-coordinate vertex, embeds each 9D face as one decoder token, then uses a
CausalMLP to decode the nine coordinate tokens inside each face. Its loss is
plain coordinate cross-entropy over face coordinates, trained end-to-end with the
VecSet encoder. Source: https://arxiv.org/html/2603.01515v2

The implementation details are much larger than our current rung:

```text
500M params
8192 points with normals
2048 VecSet tokens
encoder: 8 layers, hidden 768
decoder: 24 layers, hidden 1024
130k Objaverse meshes under 4000 faces
128 coordinate bins
random rotation, flipping, per-axis scaling
Muon, lr 6e-4, weight decay 0.1
100k steps on 8x A100 80GB
```

The scaling section is even larger:

```text
1.2B params
65,536 points
1024 bins
380k high-quality meshes
```

So yes, the paper explicitly points toward scale as the route to better
generalization. But it also means our current 56/8 run is only a plumbing and
closure diagnostic.

## What The Cited Sources Add

3DShape2VecSet supports the VecSet design: FPS/downsampled point queries,
cross-attention from sampled points to the full point set, then self-attention
refinement over the latent set. Our local Shape2VecSet encoder was cross-checked
against the official repo structure and is close, except for dependency-light
choices and the additive normal embedding needed by FACE.

TreeMeshGPT supports the importance of hierarchical/causal coordinate decoding.
Its ablation says simultaneous XYZ prediction is worse than hierarchical MLP
heads. FACE directly cites this as CausalMLP motivation.

TreeMeshGPT also filters for manifold meshes with no flipped normals. That is a
warning for us: mesh count alone is not enough. Quality, manifoldness, winding,
and tokenizer stability matter.

Meshtron supports the broader scaling thesis: high-quality artist-like mesh
generation needs higher face counts, higher coordinate resolution, and sampling
or inference strategies that preserve sequence order.

## Does FACE Promise Watertightness?

Not as a formal guarantee.

FACE claims better topological coherence and fewer holes/fragments than
baselines, but the method still predicts coordinates. Watertightness emerges
only if the model repeatedly emits exactly matching quantized vertices and edge
pairs. One wrong coordinate bin can create a boundary edge.

That means our product target should be:

```text
raw FACE/Mesh prior learns high-quality editable topology
+ strict topology gates
+ minimal repair / manifoldization
+ optional quad remesh / projection
```

We should not expect coordinate CE alone to guarantee watertight meshes.

## Why Selection Loss Is Not The Whole Story

Selection loss is unaugmented teacher-forced coordinate CE over the dataset.
It answers:

```text
Can the model predict the next coordinate when every previous face is correct?
```

It does not answer:

```text
Can the model free-run without compounding one early coordinate mistake?
Will the decoded mesh be watertight?
Will held-out meshes reconstruct?
```

This run shows the difference clearly: train AR was excellent but not perfectly
watertight, despite near-zero CE. Test teacher-forced was poor, so test AR had
no chance.

Operational note: this run evaluated the best selection checkpoint from step
2000. Later ordinary train loss dipped lower before the manual stop. I added a
`--save-current-checkpoint` option so future periodic checkpoints can preserve
both best-selection and current weights.

## Verdict On 300k Scaling

300k+ high-quality meshes should improve generalization if:

```text
the corpus is curated, manifold/watertight where possible, and topology-rich;
the target tokenization is stable after quantization;
the model is close enough to paper capacity;
augmentation is introduced at the right scale;
held-out teacher-forced and AR metrics improve on intermediate rungs.
```

300k meshes will not help if:

```text
targets are dirty or heavily convex-hull-like;
online augmentation keeps changing face order before the model has learned;
capacity is too small for the data distribution;
we only optimize CE and ignore topology gates;
we skip intermediate held-out AR validation.
```

## Next Rung

Run one of these two bounded validations before any production-scale job:

```text
Option A: closure polish
- same 64-sample noaug split
- 5k-10k steps
- save best-selection and current checkpoints
- selection eval every 250 steps
- goal: train AR 32/32 watertight or mean boundary edges ~= 0

Option B: first real generalization rung
- 512-1024 curated strict meshes
- 128 bins, <=512 or <=1024 faces
- 8192 points, 2048 VecSet tokens
- restore paper augmentation as frozen/offline copies first
- same train/test held-out gate
- goal: held-out teacher-forced accuracy rises materially above chance and held-out AR stops collapsing visually
```

My recommendation: do Option B next, while keeping Option A as a quick sanity
polish if we need confidence that the last 6 train AR failures are just checkpoint
selection granularity.

## Scale Gate For Production

Do not promote to 130k-350k until a mid-scale rung shows:

```text
train AR: >=95% watertight or boundary-edge mean near zero
held-out teacher-forced token accuracy: strong and rising across rungs
held-out AR edge pairing: strong and rising across rungs
visual contact sheets: coherent, not spiky/fractal
target prep: low repair burden, low degenerate face dropping
runtime: feasible per profile
```

This is the moat logic: we scale only when the evidence says the model is
learning topology, not merely memorizing token strings.
