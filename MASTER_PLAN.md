# ClearMesh Master Plan

> Last updated: 2026-05-02
> Current decision: ditch DualPrim and avoid broad output branching. Build the evaluation harness first, then OmniPart-style structure, then one winning artist-mesh head.

## North Star

ClearMesh should turn an image or prompt into an editable, production-survivable triangle mesh with semantic parts.

The v1 architecture is intentionally narrow:

```text
Input image / prompt
  -> TRELLIS.2 visual asset / proxy
  -> evaluation harness
  -> OmniPart-style part decomposition
  -> MeshRipple or another winning artist-mesh head
  -> repair + validation + Blender roundtrip
  -> optional autorigging for eligible assets
  -> editable semantic mesh
```

## What Changed

Previous plans explored too many branches at once: PartCrafter, retopo, primitive fitting, B-Rep-ish upgrades, Easy3E editing, and DualPrim-style paths. That is too broad before we know the actual failure modes.

New rule:

```text
No new output branch earns engineering time until it wins in the harness.
```

## Do Not Build In V1

```text
DualPrim
custom LATO/SATO implementation
quad-native generation
B-Rep / STEP output
NURBS fitting
large custom SLAT-conditioned mesh transformer
many parallel mesh/CAD/primitive heads
```

## Build In Order

### Sprint 0: Evaluation Harness

Status: starter implemented in `clearmesh/eval/mesh_quality.py` and `scripts/eval/evaluate_meshes.py`.

The harness measures:

```text
Geometry: Chamfer, Hausdorff, normal consistency, area/volume ratios, tiny components
Topology: watertightness, boundary loops, non-manifold edges/vertices, components, genus estimate
Editability: face count, valence histogram, pole count, aspect ratios, degenerate faces, subdivision smoke test
Production: optional Blender import/export roundtrip
```

Next harness improvements:

```text
exact self-intersection counts
UV/material preservation
Blender decimation behavior
per-part separability
edge-loop continuity proxy
HTML/dashboard summary
```

### Sprint 1: TRELLIS.2 Point-Cloud Bridge

Run fixed TRELLIS.2 outputs through point budgets:

```text
16k points
40k points
100k points
```

Then test:

```text
MeshRipple
Mesh Silksong
DeepMesh
TreeMeshGPT
MeshMosaic, once components exist
FastMesh, if inference is clean
```

Decision question:

```text
Is point-cloud conditioning good enough for v1, or do thin/topological structures force O-Voxel/SLAT-native conditioning?
```

### Sprint 2: OmniPart-Style Part Structure

Use OmniPart-style structure because it is closest to TRELLIS-style sparse latent workflows. Treat this as promising but not drop-in for TRELLIS.2/O-Voxel until proven.

First experiment:

```text
TRELLIS.2 O-Voxel / mesh output -> OmniPart/TRELLIS SLat-compatible structure
```

Initial deliverables:

```text
part boxes
part masks
part IDs
part-level point clouds
part-level proxy meshes
```

### Sprint 3: Choose One Mesh Head

Current ranking for bake-off priority:

```text
1. MeshRipple
2. Mesh Silksong
3. DeepMesh
4. MeshMosaic
5. TreeMeshGPT
6. FastMesh, if repo/weights run cleanly
```

Selection criterion:

```text
Pick the method that minimizes topology and editability failures, not the one with the prettiest qualitative render.
```


## Optional Autorigging

Autorigging remains in scope for production, but it sits after mesh validation rather than inside the mesh-head bake-off.

```text
validated semantic mesh
  -> riggability classifier
  -> Puppeteer or UniRig backend
  -> skinning/skeleton QA
  -> FBX/GLB export
```

Initial rule:

```text
offer rigging for humanoid, creature, animal, and character assets
skip by default for props, tools, furniture, and mechanical objects
```

## Fixed Benchmark Cases

Use 50-100 stable cases.

```text
organic: animal, character, creature, hand, plant/tree
hard surface: chair, robot, vehicle, tool, appliance
topology stress: mug handle, headphones, eyeglasses, bicycle, lamp, branching tree
production: game prop, toy, furniture, product asset
```

## Existing Assets To Keep

```text
TRELLIS.2 setup and generation scripts
Stage 2 refinement learnings and checkpoints
mesh repair/export utilities
PBR/export/optional autorigging modules as downstream tools
text/image entry points
RunPod/Vast.ai operational scripts
```

## Current Repo Map

| Area | Path | Status |
|---|---|---|
| Evaluation harness | `clearmesh/eval/` | starter implemented |
| Batch eval CLI | `scripts/eval/evaluate_meshes.py` | starter implemented |
| New bake-off doc | `docs/mesh_head_bakeoff.md` | added |
| DualPrim audit | `docs/archive/dualprim_preoptimization_hostile_audit.md` | archived |
| Pipeline plan | `PIPELINE_PLAN.md` | pivoted |
| Stage 2 research | `clearmesh/stage2/` | keep as existing research asset |
| Mesh repair/export | `clearmesh/mesh/` | keep and reuse in validation |

## Immediate Next Steps

1. Build a `manifests/mesh_bakeoff.csv` with TRELLIS.2 baseline outputs and reference/proxy meshes.
2. Run `python scripts/eval/evaluate_meshes.py --manifest manifests/mesh_bakeoff.csv --output eval_results/mesh_bakeoff_report.json`.
3. Add point-cloud sampling/export helpers for 16k, 40k, and 100k conditioning sets.
4. Wire the first runnable MeshRipple inference path into an experiment script.
5. Repeat for Mesh Silksong and DeepMesh only after the harness report format feels useful.
