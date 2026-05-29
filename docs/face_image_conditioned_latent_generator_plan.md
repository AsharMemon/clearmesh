# FACE/FACE-Q Image-Conditioned Latent Generator Plan

## Why this exists

The FACE paper does not use a TRELLIS mesh as an intermediate for its image-to-mesh result. It first trains a FACE autoregressive autoencoder (ARAE), then trains an image-conditioned latent diffusion model to generate the latent VecSet consumed by the pretrained FACE decoder.

Our current product path is different:

```text
text/image -> TRELLIS proxy mesh -> sampled proxy points + proxy vertex table -> FACE-Q indexed decoder
```

The paper-shaped path we should test is:

```text
image -> image-conditioned latent generator -> FACE/FACE-Q decoder -> mesh
```

This removes the TRELLIS proxy mesh as the geometry bottleneck.

## Paper reference points

From FACE arXiv 2603.01515:

- ARAE reconstructs mesh `M` from point cloud `P` by encoding `P` into a latent VecSet and decoding faces autoregressively.
- Image-to-mesh is a downstream latent-prior task: a DiT denoises/generates latent VecSets from image features, then the pretrained FACE decoder produces the mesh.
- The ARAE model is about 500M parameters.
- ARAE details: 8192 surface points with normals, 2048 VecSet latent tokens, bottleneck dimension 64, ~130k Objaverse meshes with <4000 faces, 100k steps on 8x A100 80GB.
- Image DiT details: 350M parameters, standard flow matching objective, 50k curated meshes, 10 random-lighting/camera renders per mesh, 400k steps on 32x A100 80GB, 100 Euler sampling steps at inference.
- Large ARAE variant: 1.2B parameters, 65,536 input points, higher quantization resolution.

Adjacent methods that support this design:

- 3DShape2VecSet: set-of-vectors latent shape representation; demonstrated image-conditioned generation over shape latents.
- Shap-E: two-stage encoder then conditional diffusion over encoded 3D latents.
- Hunyuan3D/TRELLIS/LATTICE family: image-conditioned latent generation is the dominant pattern; the key is the latent representation and decoder quality.

## ClearMesh implementation decision

We should not start by training a 350M image DiT from scratch against raw meshes. First, freeze the current best FACE-Q decoder and ask a narrower question:

```text
Can an image encoder/prior predict the exact FACE-Q condition latent that the decoder already uses?
```

If yes, then image-conditioned FACE-Q is credible. If no, a giant run is likely just expensive noise.

## Milestones

### M0: Latent target extraction

Status: initial utility added at `scripts/research/extract_faceq_condition_latents.py`.

Input:

```text
FACE-Q checkpoint + indexed FACE-Q npz samples
```

Output:

```text
condition_latents: [condition_tokens, latent_dim or hidden_dim]
condition_positions: optional [condition_tokens, 3] for VoxSet
manifest JSON
```

This gives us the supervised target for an image-conditioned prior.

### M1: Tiny paired-render smoke

Goal: prove data plumbing and loss can decrease.

Dataset:

- 256-1,024 meshes.
- 4-10 renders per mesh.
- Existing B2 mesh corpus is enough; no new mesh collection needed.

Model:

- Frozen image encoder: DINOv2/DINOv3/CLIP-like backbone depending on availability and license.
- Tiny latent predictor: 20M-80M transformer or MLP-token decoder.
- Target: MSE/flow-matching velocity to frozen FACE-Q condition latents.

Validation:

- Latent MSE drops materially.
- Generated latent fed into frozen FACE-Q decoder yields non-empty meshes.
- Teacher-latent decode remains good; predicted-latent decode is not collapsed.

Expected cost:

- 1x A100 or H100 for 2-8 hours.
- About $2-$20 depending provider/spot price.

### M2: Small image-prior gate

Goal: check whether image conditioning beats TRELLIS-proxy conditioning on held-out examples.

Dataset:

- 5k-20k meshes.
- 5-10 renders each.
- 25k-200k image/latent pairs.

Model:

- 100M-200M latent prior.
- Flow matching or diffusion over latent VecSet.
- Frozen FACE-Q decoder.

Validation:

- Held-out image -> latent -> FACE-Q mesh gives coherent object class and silhouette.
- Compare against current website path: HiDream/TRELLIS -> FACE-Q.
- If image-prior outputs are less fragmented and more semantically aligned, promote.

Expected cost:

- 1-4x A100/H100 for 12-48 hours.
- Roughly $20-$400 depending GPU availability.

### M3: Paper-like image DiT

Goal: serious reproduction/production candidate.

Dataset:

- 50k curated meshes minimum, 10 renders each, matching paper scale.
- Better: 100k+ meshes if B2 corpus quality supports it.

Model:

- 300M-400M DiT latent prior.
- Frozen best FACE/FACE-Q decoder.
- Flow matching objective.

Expected cost:

- Paper: 32x A100 80GB for 400k steps.
- Practical ClearMesh: use 8x H100/A100 and lower precision/checkpointing first.
- Conservative estimate: 3-10 days on 8x A100/H100 for a paper-ish run, depending batch size and latent count.
- Cost ballpark: $1k-$8k on cheap spot/marketplace GPUs; materially higher on premium cloud.

### M4: Only if M3 is strong

Train a paper-closer or larger FACE/FACE-Q ARAE and image prior together as a production model family. This is where 1B+ decoder scale makes sense.

## Key technical risks

- FACE-Q indexed decoding still needs a vertex table. An image prior can generate the condition latent, but if we keep indexed FACE-Q we also need a strategy for vertex-table generation or a canonical vertex-table prior. The paper avoids this by predicting coordinate-bin faces directly.
- If we want the most paper-faithful image-to-mesh path, we should revive coordinate FACE for the decoder, not indexed FACE-Q.
- If we keep FACE-Q, a practical compromise is a two-headed prior: image -> condition latent + vertex table latent/candidate vertices.
- Good renders matter. Bad single-view renders will train ambiguity into the prior.
- The current website pipeline can remain as fallback while the image latent prior matures.

## Recommended path

1. Extract condition latents from the current best FACE-Q checkpoint over a small B2 shard subset.
2. Build paired renders if not already present.
3. Train M1 tiny image-to-latent smoke.
4. Decode predicted latents through the frozen FACE-Q decoder and visually inspect.
5. If M1 works, run M2 against the current website examples before committing to M3.

This is the least expensive route that actually tests the paper's image-to-mesh idea.
