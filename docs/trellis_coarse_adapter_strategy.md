# TRELLIS Coarse Adapter Strategy

## Problem

UltraShape's refinement stage expects a coherent coarse mesh. In the paper and
released repo, that coarse mesh comes from Hunyuan3D-2.1 and is then used to
derive voxel-query anchors for local geometric refinement.

ClearMesh intentionally replaces the Hunyuan coarse generator with TRELLIS.2.
That is the right product direction, but the 2026-05-03 Thunder smoke showed a
distribution mismatch:

```text
TRELLIS.2 -> UltraShape paper settings:
  UltraShape reference components: 571
  tiny components: 570
  boundary loops: 12
  non-manifold edges: 373
  watertight: false
```

The renderer preview looked fragmented because the downstream control surface
was built from that fragmented reference. The issue is not only screenshot
quality; the mesh topology really contains many disconnected islands.

## Root Cause

UltraShape uses coarse geometry as spatial anchors. If the coarse input is
fragmented, open, noisy, or full of tiny components, the refinement DiT receives
fragmented voxel-query support. It can synthesize detail around those anchors,
but it cannot reliably infer a clean global object from anchors that already
describe shredded geometry.

The failure path is:

```text
raw TRELLIS visual mesh
  -> many components / open boundaries / non-manifold regions
  -> fragmented voxel-query anchors
  -> UltraShape detailed but fragmented reference
  -> Poisson/control-surface normalization spreads/noises those defects
  -> production gate refuses promotion
```

## Fix

Add one mandatory internal stage before UltraShape:

```text
TRELLIS.2 visual mesh
  -> TRELLIS coarse adapter
  -> UltraShape paper-setting refinement
  -> retopo / projection / gates
```

This keeps the user-facing pathway singular while internally ensuring
UltraShape receives a coarse mesh closer to its expected distribution.

## Adapter Contract

Input:

```text
TRELLIS.2 GLB or O-Voxel mesh output
reference image
```

Output:

```text
coherent coarse proxy mesh
adapter report
optional preserved visual mesh for texture/detail reference
```

The proxy should be:

```text
single or few connected components
low tiny-component count
low boundary-loop count
low non-manifold count
bounded face count
roughly faithful to silhouette and major holes
```

## Concrete Algorithm

1. Mesh passport:

```text
measure components, tiny components, boundary loops, non-manifold edges,
surface area, bounds, face count, and density
```

2. Component filtering:

```text
remove components below a face/area threshold
keep components that contribute meaningfully to silhouette/bounds
cluster near-touching components before deciding they are debris
```

3. Occupancy/SDF proxy:

```text
sample oriented points from remaining geometry
voxelize into a fixed grid
morphological close to bridge sub-voxel cracks
remove floating occupancy islands
extract a manifold-ish surface
decimate to the coarse budget UltraShape likes
```

4. Feature preservation:

```text
preserve major handles/holes only if stable in the occupancy grid
preserve image-visible protrusions if they survive multi-view/silhouette checks
avoid preserving tiny TRELLIS debris
```

5. Gate before UltraShape:

```text
if proxy still has too many components or boundaries, use the conservative
watertight proxy rather than raw TRELLIS
```

6. After UltraShape:

```text
run the normal production gate
if the reference remains fragmented, do not promote
```

## Candidate Engines

The first implementation should try engines in this order:

```text
1. voxel occupancy + marching cubes / OpenVDB-style morphology
2. screened Poisson with aggressive component pruning
3. ManifoldPlus or manifold3d repair
4. template cage only as preview/control fallback, never as final detail mesh
```

## Promotion Criteria

Before UltraShape:

```text
connected_components <= 8
tiny_component_count <= 16
boundary_loop_count <= 32
nonmanifold_edge_count <= 128
face_count <= 250k
```

For production promotion after full pipeline:

```text
watertight: true
connected_components <= intended part count
boundary_loop_count == 0
nonmanifold_edge_count == 0
passes Blender import/export/deformation gate
artist-editable control or quad sidecar is faithful enough
```

## Key Principle

Do not tune UltraShape away from paper settings to compensate for raw TRELLIS
fragmentation. Fix the coarse conditioning distribution first.
