# ClearMesh — Full Pipeline Plan

Single image (or text prompt) to high-fidelity, print-ready, optionally rigged 3D asset.

---

## Input Modes

ClearMesh supports three ways to start generation:

**A. Image-to-3D (primary)**
A single photo or rendered image enters the pipeline at Stage 1.

**B. Text-to-3D (via FLUX.1)**
A text prompt is passed to FLUX.1, which generates a conditioning image. That image then enters the same pipeline at Stage 1. No additional 3D training required — FLUX.1 handles the text→image step and TRELLIS.2 handles image→3D. The `clearmesh/text_to_3d/` module wraps this. Ctrl-Adapter (~20 GPU-hours to train) can optionally give finer control over viewpoint, depth, and normal maps in the generated image.

**C. Text-to-3D with Easy3E editing**
Same as B, but after the initial mesh is generated (through Stage 6), Easy3E allows iterative text-guided edits: "make the helmet pointier," "add spikes to the shoulders," etc. Each edit re-runs a partial diffusion process on the TRELLIS.2 latents without regenerating from scratch.

---

## Pipeline Stages

```
Input (image or FLUX.1-generated image)
  │
  ├─ 1.  Background Removal
  ├─ 2.  TRELLIS.2 4B (coarse generation)
  ├─ 3.  Stage 2 DiT (SDF refinement)
  ├─ 4.  PartCrafter (part decomposition + classification)
  ├─ 5.  Geometry Super-Res (SuperCarver / CraftsMan3D)
  ├─ 6.  NDC / FlexiCubes (mesh extraction from SDF)
  ├─ 7.  Mesh Repair (PyMeshFix + Trimesh)
  ├─ 8.  Geometric Upgrade (selective B-rep refit for hard surfaces)   ← NEW
  ├─ 9.  Retopology (BPT)
  ├─ 10. Scale (miniature sizing, base, drain holes)
  ├─ 11. PBR Textures
  ├─ 12. Auto-Rigging (Puppeteer / UniRig)
  ├─ 13. Export (STL / GLB / OBJ / FBX)
  │
  └─ Easy3E Editing Loop (optional, returns to Stage 2–6)
```

---

## Stage Details

### Stage 1 — Background Removal
- **Default:** on
- **Flag:** `--skip-bg-removal`
- Removes background from input image to isolate the subject for TRELLIS.2 conditioning.

### Stage 2 — TRELLIS.2 4B (Coarse Generation)
- **Default:** on
- Generates coarse 3D shape as structured latents (SLAT) from the conditioned image.
- Outputs sparse voxel positions (3-dim) + 32-dim SLAT features + DINOv2/v3 conditioning features (1024-dim per patch token, 257 tokens).
- `guidance_strength=9.0` (sparse), `4.5` (shape).
- **Note:** DINOv3 (`dinov3-vitl16`) is a gated HuggingFace model (mirror: `tao-hunter/dinov3-vitl16-pretrain-lvd1689m`). Outputs 1024-dim features same as DINOv2-ViT-L but trained on LVD-1.689M dataset.

### Stage 3 — Stage 2 DiT (SDF Refinement)
- **Default:** on
- **Flag:** `--no-refine`
- **Architecture:** 527.5M-param DiT backbone (12 blocks from TRELLIS.2's 30-block shape transformer, dim=1536, 12 heads).
- Pretrained initialization from TRELLIS.2 weights — critical, model barely learns from scratch.
- Takes coarse SLAT (32-dim) + positions (3-dim) + optional DINOv2/v3 features (1024-dim cross-attention), predicts refined SDF field.
- **MLP output head:** `nn.Sequential(LayerNorm(1536), Linear(1536, 256), GELU(), Linear(256, 1))` — prevents weight collapse that occurs with single `nn.Linear`.
- **Epsilon prediction:** Predicts noise, not x0 or velocity. Linear noise schedule `alpha(t) = 1 - t`.
- **Differential LR:** 5e-5 for pretrained backbone, 2.5e-4 (5x) for fresh heads (sdf_proj, out_head).
- **SDF scale:** Ground-truth SDF multiplied by 10.0 during loading to match noise magnitude.
- **Inference modes:**
  - `refine()` — DDIM sampling (50 steps) with x0 clipping (±5.0) to prevent high-t divergence.
  - `predict_x0()` — Single-step direct prediction (faster, more stable, less diverse).
- Progressive token schedule: 2048 → 4096 → 8192 (at steps 0 → 30K → 70K).
- **Training status:** COMPLETE — 100K steps on 34,831 clean pairs (noise_mse=0.005 final, x0_corr=0.997 at t=0.1). Cup mesh e2e: 6,815 verts in 11.6s.

### Stage 4 — PartCrafter (Part Decomposition)
- **Default:** off (on when `--decompose` or `--geo-upgrade` is set)
- **Flag:** `--decompose`
- Decomposes the shape into semantic parts (head, torso, limbs, weapon, base, etc.).
- Classifies each part as **hard-surface** or **organic** based on normal variance and keywords.
- Part classifications propagate to Stage 8 for selective geometric upgrade.

### Stage 5 — Geometry Super-Resolution
- **Default:** off
- **Flag:** `--super-res`
- SuperCarver (primary) or CraftsMan3D (fallback).
- Adds fine geometric detail beyond what Stage 3 refinement captures.

### Stage 6 — NDC / FlexiCubes (Mesh Extraction)
- **Default:** on
- Extracts triangle mesh from the SDF field.
- NDC (Neural Dual Contouring) for sharp-edge preservation; FlexiCubes as differentiable alternative.

### Stage 7 — Mesh Repair
- **Default:** on
- PyMeshFix for watertight repair, Trimesh for degenerate triangle removal.
- Print-readiness checks (manifold, no self-intersections, minimum wall thickness).

### Stage 8 — Geometric Upgrade (Selective B-rep Refit) ← NEW
- **Default:** off
- **Flag:** `--geo-upgrade` (implies `--decompose`)
- **Only runs on parts PartCrafter classified as hard-surface.** Organic parts pass through untouched.
- Per mechanical part:
  1. **Face clustering:** Segment the part mesh into near-planar, near-cylindrical, near-conical, and freeform regions (using normal-based clustering or a lightweight Point2CAD-style segmentation).
  2. **Parametric surface fitting:** Fit exact geometric primitives (planes, cylinders, cones, tori) to each cluster. Use implicit neural surface fitting for freeform regions.
  3. **Edge sharpening:** Extend adjacent fitted surfaces, compute their analytical intersection curves, and use those as true sharp edges (the BrepDiff face-extension-and-intersection technique).
  4. **Retessellation:** Replace the original triangles in detected regions with tessellations of the fitted primitives. Sharp edges become actually sharp, flat faces become actually flat, cylinders become actually round.
  5. **Boundary feathering:** Blend the upgraded geometry back into the neural mesh at region boundaries with matched normals and watertight connectivity.
- Does **not** produce full B-rep/STEP output — stays in triangle mesh representation throughout.
- Estimated implementation: ~2–3 weeks using Point2CAD segmentation + classical surface fitting.

### Stage 9 — Retopology (BPT)
- **Default:** off
- **Flag:** `--retopo`
- BPT (CVPR 2025) for clean quad-dominant topology.
- Useful for animation-ready assets; less critical for print.

### Stage 10 — Scale
- **Default:** off
- **Flag:** `--scale 32mm`
- Scale to miniature/print size. Optional base plate and drain holes for resin printing.

### Stage 11 — PBR Textures
- **Default:** off
- **Flag:** `--textures`
- PBR material assignment (albedo, roughness, metallic, normal maps).

### Stage 12 — Auto-Rigging
- **Default:** off
- **Flag:** `--rig`
- Puppeteer (primary, MIT) or UniRig (fallback, MIT).
- Runs in separate `rigging` conda env (PyTorch 2.1.1 + CUDA 11.8).
- Produces skeleton + skinning weights for animation.

### Stage 13 — Export
- **Default:** on
- **Flag:** `--format glb` (also: stl, obj, fbx)
- Final output with all metadata, materials, and optional rig.

---

## Easy3E Editing (Post-Generation)

- **Flag:** `--edit "make the helmet pointier"`
- **Status:** Scaffolded in `clearmesh/editing/`, core methods raise `NotImplementedError`.
- **Approach:** Training-free editing via TRELLIS.2's flow-based latent space.
  - Encode existing mesh back to TRELLIS.2 latents (`_encode_ss_latent`, `_encode_shape_latent`).
  - Apply text-guided velocity field modifications (`_compute_velocity`).
  - Decode modified latents back to mesh (`decode`).
  - Re-run Stages 3–13 on the edited output.
- **Blocker:** Requires reverse-engineering TRELLIS.2's internal encode/decode APIs — the public API only exposes generation, not encoding of existing meshes. Needs investigation on RunPod with access to the model internals.
- **Text-to-3D + Easy3E combo:** Generate initial mesh from text via FLUX.1 → TRELLIS.2, then iteratively refine with text edits. Each edit is fast (~5–10s) because it's a partial re-diffusion, not a full generation.

---

## Text-to-3D (via FLUX.1)

- **Flag:** `--text "a medieval sword with ornate crossguard"`
- **Module:** `clearmesh/text_to_3d/`
- **How it works:** FLUX.1 generates a high-quality image from the text prompt. That image feeds into Stage 1 as if the user had provided a photo. No 3D-specific text encoder needed.
- **Ctrl-Adapter (optional):** ~20 GPU-hours to train. Adds control signals (depth, normal, edge maps) to FLUX.1 for more precise viewpoint and structural control over the generated image.
- **Quality note:** Output quality is bounded by FLUX.1's image quality and TRELLIS.2's ability to reconstruct 3D from that image. Works best with object-centric prompts; struggles with scenes or highly abstract descriptions.

---

## Development Phases

### Phase 1 — Data Preparation (DONE)
Download Objaverse, filter dataset, set up pair generation infrastructure.

### Phase 2 — Pair Generation (DONE — 34.8K clean pairs)
Generated ~38K coarse/fine mesh pairs on RunPod A100. After outlier filtering (max_sdf_std=0.5, removes 8.4% poison pairs), 34,831 clean pairs available. Each pair contains `coarse_voxels.npy` (N,32), `positions.npy` (N,3), `fine_sdf.npy` (N,1). Most pairs lack DINOv2/v3 conditioning features — a gap for future data generation.

### Phase 3 — Stage 2 Training (COMPLETE — 100K steps, 11.4 hrs)
Trained RefinementDiT (527.5M params) on 34.8K pairs on Vast.ai H100 SXM 80GB. Key results:
- **Noise prediction**: x0_corr=0.997 at t=0.1 (near-perfect reconstruction)
- **Loss curve**: 0.89 → 0.005 (stable, no overfitting)
- **E2E cup mesh**: 6,815 verts, 14,092 faces, bounded SDF, 11.6s total
- **DDIM unconditional**: Low GT correlation (0.034 avg) — expected since eval pairs lack DINOv2/v3 conditioning
- **Key gap**: Most training pairs missing cond_features — future data gen must save these
- See `TECHNICAL_LEARNINGS.md` for detailed failure mode analysis and all discoveries.

### Phase 4 — Pipeline Integration
Wire trained Stage 2 into `pipeline.py`. Benchmark ClearMesh vs standalone TRELLIS.2. End-to-end testing of all optional stages.

### Phase 5 — Geometric Upgrade (Option 3)
Implement Stage 8: face clustering, parametric surface fitting, edge sharpening via analytical intersection, retessellation with boundary feathering. Point2CAD segmentation as starting point. ~2–3 weeks.

### Phase 6 — Easy3E Editing
Investigate TRELLIS.2 internal APIs for encode/decode. Implement `_encode_ss_latent`, `_encode_shape_latent`, `decode`, `_compute_velocity`, `_generate_features`. Training-free — no GPU hours for training, but needs RunPod access for API exploration.

### Phase 7 — Text-to-3D
Wire FLUX.1 image generation into pipeline entry point. Implement prompt → image → 3D flow. Optional: train Ctrl-Adapter (~20 GPU-hours) for viewpoint/structure control.

### Phase 8 — Text-to-3D + Easy3E
Combine Phases 6 and 7: generate from text, then iteratively edit with text. This is the "conversational 3D modeling" workflow — describe what you want, see a result, describe changes, see updates.

### Future — BrepDiff Integration
Full parallel generation path: image-level classifier routes mechanical objects through image-conditioned BrepDiff for native B-rep output, organic objects through TRELLIS.2 → Stage 2. Requires training an image-conditioned BrepDiff variant and solving boundary stitching for mixed objects. Research-grade effort, 6–12 months.

---

## CLI Examples

```bash
# Basic image-to-3D
python -m clearmesh.pipeline --input photo.png --output model.glb

# Full pipeline with geometric upgrade
python -m clearmesh.pipeline \
    --input photo.png \
    --output model.glb \
    --decompose \
    --geo-upgrade \
    --super-res \
    --retopo \
    --textures

# Print-ready miniature with sharp mechanical edges
python -m clearmesh.pipeline \
    --input mech_part.png \
    --output part.stl \
    --format stl \
    --geo-upgrade \
    --scale 32mm \
    --add-base \
    --drain-holes

# Text-to-3D
python -m clearmesh.pipeline \
    --text "a medieval sword with ornate crossguard" \
    --output sword.glb

# Text-to-3D with iterative editing
python -m clearmesh.pipeline \
    --text "a sci-fi helmet" \
    --output helmet.glb \
    --edit "add a visor slit" \
    --edit "make it more angular"

# Rigged character from image
python -m clearmesh.pipeline \
    --input character.png \
    --output rigged.fbx \
    --format fbx \
    --decompose \
    --textures \
    --rig
```

---

## Component Table

| Component | Role | Source | Stage |
|---|---|---|---|
| TRELLIS.2 4B | Coarse generation | microsoft/TRELLIS.2 (MIT) | 2 |
| RefinementDiT (527.5M) | SDF refinement (dim=1536, MLP head) | Trained on 34.8K pairs | 3 |
| PartCrafter | Part decomposition + classification | VAST-AI-Research/PartCrafter | 4 |
| SuperCarver / CraftsMan3D | Geometry super-resolution | CraftsMan3D fallback | 5 |
| NDC / FlexiCubes | Sharp-edge mesh extraction | czq142857/NDC, NVIDIA Kaolin | 6 |
| PyMeshFix + Trimesh | Mesh repair + print readiness | PyPI | 7 |
| Point2CAD segmentation + classical fitting | Selective geometric upgrade | prs-eth/point2cad + custom | 8 |
| BPT | Retopology | CVPR 2025 | 9 |
| Puppeteer / UniRig | Auto-rigging | Seed3D / VAST-AI-Research (MIT) | 12 |
| FLUX.1 | Text-to-image for text-to-3D | Black Forest Labs | Input B |
| Ctrl-Adapter | Controllable image generation | Optional, ~20 GPU-hrs | Input B |
| Easy3E (TRELLIS.2 latent editing) | Text-guided mesh editing | Training-free, needs API work | Post-gen |

---

## Environments

| Env | PyTorch | CUDA | Used for |
|---|---|---|---|
| `clearmesh` | 2.6 | 12.4 | TRELLIS.2, Stage 2 training/inference, all mesh processing |
| `trellis2` | 2.6 | 12.4 | TRELLIS.2 standalone (legacy, may merge with clearmesh) |
| `rigging` | 2.1.1 | 11.8 | UniRig, Puppeteer |
