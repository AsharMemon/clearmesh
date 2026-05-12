# Trusted Worker Commands

Public API requests intentionally strip executable metadata. These examples are for trusted server-side job creation, internal benchmarks, and GPU operators only.

## Easy3E Edit Job

Use `configs/easy3e.command.example.json` as a template for `metadata-json`:

```bash
python scripts/product/create_local_job.py \
  --state-root /tmp/clearmesh_easy3e_state \
  --input-uri local://uploads/edit/input.png \
  --mode edit_image \
  --metadata-json configs/easy3e.command.example.json \
  --grant-credits 20
```

The worker runs `scripts/product/run_easy3e_edit.py`, which supports either `--edit-image` or `--instruction`.

## Reference Refinement Job

The single production route can refine/manifoldize the TRELLIS.2 or Easy3E mesh before passporting:

```json
{
  "reference_refinement_enabled": true,
  "reference_refinement_command": "/home/ubuntu/ultrashape-venv/bin/python -u scripts/product/run_ultrashape_refinement.py --mesh {input_mesh} --image {reference_image} --output-dir {output_dir} --ultrashape-dir /workspace/UltraShape-1.0 --checkpoint /workspace/checkpoints/ultrashape_v1.pt --num-steps 50 --octree-resolution 1024 --num-latents 32768 --chunk-size 8000 --scale 0.99 --seed 42 --remove-bg",
  "reference_refinement_timeout_seconds": 7200
}
```

Run UltraShape as a command hook/subprocess boundary. TRELLIS.2 and UltraShape
can both register `cuBVH`, so importing both in one long-lived worker process is
fragile.

For a deterministic watertight manifold command hook:

```json
{
  "reference_refinement_enabled": true,
  "manifoldization_command": "python scripts/product/run_manifoldplus.py --input {input_mesh} --output-dir {output_dir} --binary /workspace/ManifoldPlus/build/manifold --depth 8"
}
```

For local/dev fallback without external repos:

```json
{
  "reference_refinement_enabled": true,
  "reference_refinement_engine": "poisson"
}
```

## OmniPart Structure Job

Use `configs/omnipart.command.example.json` once a real part mask exists:

```bash
python scripts/product/create_local_job.py \
  --state-root /tmp/clearmesh_omnipart_state \
  --input-uri local://uploads/parts/input.png \
  --metadata-json configs/omnipart.command.example.json \
  --grant-credits 20
```

For plumbing only, create a whole-object placeholder mask:

```bash
python scripts/product/create_single_part_mask.py \
  --image input.png \
  --output /tmp/omnipart_mask.exr
```

If OpenCV lacks EXR support, it writes `.npy`; that fallback is useful for our bookkeeping but not a substitute for OmniPart's expected EXR mask.

If OmniPart is not available yet, use the MeshMosaic-style connected-component fallback:

```json
{
  "part_structure_fallback": "meshmosaic_components",
  "component_part_max_parts": 8,
  "component_part_min_faces": 32,
  "component_part_point_count": 4096
}
```

## Quad And Production Gates

Trusted metadata for chart-level and whole-object quad sidecars:

```json
{
  "chart_remesh_enabled": true,
  "chart_remesh_engine": "quadriflow_cli",
  "chart_remesh_max_charts": 8,
  "quad_remesh_enabled": true,
  "quad_remesh_engine": "pyinstantmeshes",
  "quad_target_faces": 5000,
  "feature_projection_enabled": true,
  "feature_projection_as_final": false,
  "blender_gates_enabled": true,
  "production_require_blender": false
}
```

`feature_projection` preserves OBJ quad face arity by moving vertices instead of loading through Trimesh's triangle-only path.

## Cleanup Knobs

Trusted metadata can tune conservative cleanup:

```json
{
  "cleanup_enabled": true,
  "cleanup_min_component_faces": 8,
  "cleanup_min_component_face_ratio": 0.0,
  "cleanup_keep_largest_components": null,
  "cleanup_fill_holes": false
}
```

Keep the raw `artist_mesh` asset for bake-off analysis. The worker writes a separate `cleaned_mesh` asset and evaluates/exports the cleaned mesh.
