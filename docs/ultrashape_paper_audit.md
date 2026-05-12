# UltraShape Paper Audit

## Verdict

UltraShape did not fail because the repo or checkpoint was unusable. It failed
because our first TRELLIS -> UltraShape test fed the refiner an input that is
outside the distribution described by the paper and official inference path.

UltraShape is best understood as:

```text
Hunyuan3D-2.1-style coarse shape
    -> normalized surface + sharp-edge point sampling
    -> voxel query anchors
    -> image-conditioned DiT refinement
    -> SDF/marching-cubes surface
```

It is not, by itself, a universal repair operator for arbitrary shredded GLBs.

## Paper Assumptions That Matter

The paper frames UltraShape as a two-stage coarse-to-fine generator. The coarse
stage supplies global structure; the refinement stage uses voxel queries derived
from coarse geometry as fixed spatial anchors.

The report also says their data pipeline removes low-quality samples, fills
holes, thickens thin structures, filters fragmented/problematic geometry, and
trains on curated watertight meshes. That is a much cleaner supervision regime
than raw TRELLIS.2 postprocess triangle soup.

Implementation details from the released repo:

```text
coarse mesh source: Hunyuan3D-2.1 initial mesh
surface conditioning: 204,800 uniform + 204,800 sharp-edge samples
latent tokens: 32,768 at inference
voxel query resolution: 128
image size: 1022/1024 px DINO conditioning
official octree default: 1024
official steps default: 50
official image preprocessing: run rembg if input is non-RGBA, or if forced
```

## What We Actually Tested

The first direct UltraShape retry used the raw TRELLIS proxy as the coarse mesh.
That proxy was heavily fragmented/non-manifold. On the production-path sample,
the downstream baselines saw the same pathology:

| Engine | Runtime | Faces | Components | Boundary loops | Non-manifold edges | Pair Chamfer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Poisson | 33s | 1,210,321 | 2,095 | 461 | 12,563 | 0.000315 |
| ManifoldPlus | 130s | 3,840,426 | 7,501 | 0 | 13,687 | 0.000301 |
| UltraShape raw retry | 484s | 3,984,284 | 851 | 2 | 50 | 0.065843 |

UltraShape did close almost everything geometrically:

```text
boundary loops: down to 2
boundary edges: down to 10
non-manifold edges: down to 50
```

But it drifted badly:

```text
connected components: 851
surface area ratio vs raw proxy: 1.99
normal consistency: 0.516
Chamfer to raw proxy: 0.0658
```

That is the signature of a learned refiner receiving bad anchors: it makes a
mostly closed high-detail field, but the field is not faithful enough to the
intended object.

## Wrapper Parity Fix

We found one concrete mismatch with the official repo and fixed it:

```text
before:
  non-RGBA input image -> convert to all-opaque RGBA

official / now:
  non-RGBA input image -> run UltraShape BackgroundRemover
```

The reference comparison script now also supports:

```bash
--ultrashape-remove-bg
```

This should improve paper parity for image-conditioned runs, but it does not
solve the core distribution mismatch by itself.

## Production Implication

For paper-parity testing, ClearMesh intentionally swaps only the coarse stage:

```text
paper:      Hunyuan3D-2.1 coarse mesh -> UltraShape settings
ClearMesh:  TRELLIS.2 coarse mesh     -> identical UltraShape settings
```

Those locked settings are tracked in `docs/trellis_ultrashape_bridge.md`.

## TRELLIS Replacement Smoke: 2026-05-03

We ran the full Thunder production route with:

```text
TRELLIS.2 coarse mesh
  -> UltraShape infer_dit_refine.yaml
  -> --num-steps 50
  -> --num-latents 32768
  -> --octree-resolution 1024
  -> --chunk-size 8000
  -> --scale 0.99
  -> --seed 42
  -> --remove-bg
```

The worker completed end-to-end:

```text
trellis_proxy: succeeded
reference_refinement: succeeded
surface_normalization: succeeded
retopology_planning: succeeded
chart_remesh/chart_stitch: succeeded
quad_remesh/feature_projection: succeeded
point_cloud_bridge/part_structure: succeeded
repair_validation/export_package: succeeded
```

But the production gate correctly did not promote the cleaned triangle mesh:

```text
UltraShape reference:
  vertices: 8,624,861
  faces: 17,257,891
  connected components: 571
  tiny components: 570
  boundary loops: 12
  non-manifold edges: 373
  watertight: false

Cleaned/export mesh:
  vertices: 388,658
  faces: 771,238
  connected components: 862
  tiny components: 861
  boundary loops: 332
  non-manifold edges: 6,839
  watertight: false
  production gate: preview_or_repair_required
```

The whole-object quad sidecar did produce a watertight pure-quad control cage:

```text
quad sidecar:
  quads: 5,046
  connected components: 1
  boundary loops: 0
  non-manifold edges: 0
  watertight: true
```

That sidecar was not promoted as final because it was a fallback/template cage,
not a faithful artist mesh. The run also exposed a practical bug: whole-object
`pyinstantmeshes` was receiving a `.glb` control mesh, which its file API does
not accept. The quad remesher now routes unsupported containers through the
array API or temporary OBJ conversion instead of falling back immediately.

For production quality, UltraShape should stay in the pipeline, but we may still
need a TRELLIS coarse-mesh adapter if raw TRELLIS postprocess output remains too
fragmented.

Recommended path:

```text
TRELLIS.2 raw proxy
    -> component pruning / surface normalization / manifoldization
    -> part-aware split when available
    -> UltraShape on clean whole/part coarse meshes
    -> quad/cage retopology + feature-aware projection
    -> Blender gate
```

For the next fair audit, run:

```text
raw TRELLIS -> UltraShape
sanitized TRELLIS control surface -> UltraShape
official Hunyuan-style coarse sample -> UltraShape
```

with:

```text
--remove-bg
--num-steps 50
--num-latents 32768
--octree-resolution 1024
```

If the official/Hunyuan-style sample passes and raw TRELLIS fails, UltraShape is
healthy and our bridge is the problem. If the sanitized TRELLIS input passes,
UltraShape becomes a strong high-quality reference/detail stage. If sanitized
input still fails, keep UltraShape optional and lean on SDF/manifold reference
surfaces plus quad projection for production.

## Follow-Up Finding: Main Component Was Good

A deeper component probe of the Thunder UltraShape output changed the diagnosis:
the whole mesh was not production-ready, but the largest component was far better
than the aggregate metrics suggested.

```text
UltraShape raw output:
  vertices: 8,410,555
  faces: 16,824,890
  connected components: 778
  tiny components: 777
  boundary loops: 1
  non-manifold edges: 289
  watertight: false

largest component only:
  vertices: 8,392,889
  faces: 16,791,410
  connected components: 1
  tiny components: 0
  boundary loops: 0
  non-manifold edges: 0
  watertight: true
```

That means UltraShape was much healthier than the preview implied. The failed
right-most projected quad sidecar was caused by feeding all of the tiny bubbles
into later normalization/projection stages. ClearMesh now filters the dominant
UltraShape component immediately after reference refinement, before surface
normalization, chart remeshing, quad remeshing, or feature projection.
