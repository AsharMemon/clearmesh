# TRELLIS -> UltraShape Bridge

## Contract

ClearMesh replaces UltraShape's first-stage Hunyuan3D-2.1 coarse mesh with a
TRELLIS.2 proxy mesh. Everything after the coarse mesh handoff should remain
aligned with UltraShape's released inference recipe.

```text
input image
  -> TRELLIS.2 coarse proxy mesh
  -> UltraShape official refinement settings
  -> reference/detail mesh
```

## ComfyUI Research Note

Public ComfyUI integrations appear to use the same practical seam we now use:
mesh object handoff, not latent handoff.

```text
TRELLIS.2 mesh/TRIMESH
  -> UltraShape Load Coarse Mesh From Trimesh
  -> SharpEdgeSurfaceLoader
  -> UltraShape voxel query conditioning
  -> UltraShape refined mesh
  -> TRIMESH output for Trellis2/export/cleanup nodes
```

The `ComfyUI-UltraShape1` bridge accepts a `TRIMESH` object directly, samples
204,800 sharp-edge points and 204,800 uniform points by default, voxelizes those
points into UltraShape's latent query count, and then calls the released
UltraShape pipeline. That means there is probably no hidden TRELLIS.2 O-Voxel
adapter in the public workflow; the bridge is just a mesh handoff plus careful
post-processing.

RunComfy's public TRELLIS.2 workflow documentation also routes generated meshes
through cleanup/reconstruction stages such as remesh, simplify, hole fill,
voxel-to-trimesh conversion, and high-poly to low-poly projection. So a good
ComfyUI result should not be read as "UltraShape alone produced an editable
artist mesh." It is a staged mesh-processing sandwich.

## Locked UltraShape Settings

```text
config: configs/infer_dit_refine.yaml
steps: 50
num_latents: 32768
octree_resolution: 1024
chunk_size: 8000
normalize_scale: 0.99
seed: 42
surface samples: 204800 uniform + 204800 sharp-edge
voxel query resolution: from UltraShape config, currently 128
image preprocessing: UltraShape BackgroundRemover for non-RGBA or forced remove-bg
device mode: regular GPU mode by default, low-vram only when explicitly requested
process boundary: subprocess by default to avoid TRELLIS/UltraShape cuBVH clashes
```

## Production Fix After Thunder Smoke

The 2026-05-03 Thunder smoke showed that UltraShape can produce a high-quality
closed main component plus hundreds of tiny closed bubbles. The raw output looked
fragmented in whole-mesh metrics, but filtering to the dominant component gave:

```text
vertices: 8,392,889
faces: 16,791,410
connected components: 1
tiny components: 0
boundary loops: 0
non-manifold edges: 0
watertight: true
```

So the correct production order is:

```text
TRELLIS.2 / Easy3E proxy
  -> coarse adapter
  -> UltraShape paper-setting refinement
  -> dominant-component filter
  -> surface normalization / charting / quad projection
  -> Blender gate
```

The component filter is intentionally separate from the UltraShape command. It
does not tune the model away from paper settings; it removes obvious debris from
the generated mesh before downstream retopology.

## Worker Command

```bash
/home/ubuntu/ultrashape-venv/bin/python -u \
  /home/ubuntu/clearmesh/scripts/product/run_ultrashape_refinement.py \
  --mesh {input_mesh} \
  --image {reference_image} \
  --output-dir {output_dir} \
  --ultrashape-dir /workspace/UltraShape-1.0 \
  --checkpoint /workspace/checkpoints/ultrashape_v1.pt \
  --num-steps 50 \
  --octree-resolution 1024 \
  --num-latents 32768 \
  --chunk-size 8000 \
  --scale 0.99 \
  --seed 42 \
  --remove-bg
```

## Invariant

Do not tune UltraShape to make a bad TRELLIS proxy look better during the paper
parity audit. If the result fails, the bridge problem is the TRELLIS coarse mesh
distribution, not the UltraShape inference recipe.

After parity is measured, production can add a separate TRELLIS-coarse adapter
or cleanup stage, but that should be reported as a deliberate ClearMesh
improvement rather than as "paper-identical UltraShape."
