# ClearMesh Pipeline Plan

Single image or prompt to an editable, production-survivable 3D asset.

## Strategic Pivot

We are no longer building a broad tree of mesh, CAD, quad, primitive, and DualPrim branches. The v1 plan is deliberately narrower:

```text
TRELLIS.2 -> evaluation harness -> OmniPart-style part structure -> one artist-mesh head
```

Do not build in v1:

```text
DualPrim
B-Rep / STEP generation
NURBS
quad-native generation
custom LATO/SATO reimplementation
custom SLAT-conditioned mesh transformer
five parallel output branches
```

Build in v1:

```text
evaluation harness
TRELLIS.2 -> point-cloud bridge at 16k / 40k / 100k samples
OmniPart-style semantic part structure
MeshRipple / Mesh Silksong / DeepMesh / MeshMosaic / TreeMeshGPT bake-off
repair + validation + Blender roundtrip tests
optional autorigging after final mesh validation
```

## V1 Architecture

```text
Input image / prompt
    |
    v
TRELLIS.2 high-fidelity visual asset
    |
    v
Mesh evaluation harness baseline
    |
    v
OmniPart-style decomposition
    |
    +--> part boxes
    +--> part masks / IDs
    +--> part-level point clouds
    +--> part-level proxy meshes
    |
    v
Winning artist-mesh head per part or whole object
    |
    v
repair + validation + DCC roundtrip
    |
    +--> optional autorigging for eligible characters/creatures
    |
    v
editable triangle mesh with semantic parts
```

## Sprint 0: Evaluation Harness

This is the first serious deliverable. Screenshots are not enough; the harness should catch whether an artist can edit the mesh without immediately fighting holes, islands, poles, and import/export failures.

Implemented starter CLI:

```bash
python scripts/eval/evaluate_meshes.py \
  --manifest manifests/mesh_bakeoff.csv \
  --output eval_results/mesh_bakeoff_report.json
```

Manifest format:

```csv
case_id,method,mesh_path,reference_path
mug_handle,trellis2,outputs/mug_trellis.glb,refs/mug_proxy.glb
mug_handle,meshripple,outputs/mug_meshripple.obj,refs/mug_proxy.glb
```

Current harness metrics:

```text
Geometry:
- Chamfer L2 to reference/proxy
- Hausdorff L2 to reference/proxy
- normal consistency
- surface area / volume ratios
- tiny component count

Topology:
- watertightness
- boundary edge count
- boundary loop count
- non-manifold edge count
- non-manifold vertex count
- connected component count
- Euler number
- genus estimate for watertight meshes

Editability:
- face count
- vertex count
- vertex valence histogram
- pole vertex count
- triangle aspect ratios
- degenerate face count
- subdivision smoke test

Production:
- optional Blender import/export GLB roundtrip
- JSON report for dashboards and method comparison
```

Still to add:

```text
- exact self-intersection counts via Blender or libigl
- UV/material preservation checks
- decimation behavior checks
- per-part separability score
- edge-loop continuity proxy beyond valence/aspect heuristics
```

## Sprint 1: TRELLIS.2 Point-Cloud Bridge

The key empirical question:

```text
Does TRELLIS.2 -> 16k/40k/100k point cloud -> mesh head preserve enough structure?
```

Test all bake-off methods on the same fixed cases and point budgets. If the bridge fails on thin structures, handles, holes, fingers, cables, or branching forms, then we invest in O-Voxel/SLAT-native conditioning. Until then, do not build a custom conditioning path.

Target methods:

| Rank | Method | Use |
|---|---|---|
| 1 | MeshRipple | Top first candidate; topology completeness focus; inference weights available |
| 2 | Mesh Silksong | Strong topology-preserved baseline |
| 3 | DeepMesh | Strong artist-mesh baseline |
| 4 | MeshMosaic | Best when part/component inputs are available |
| 5 | TreeMeshGPT | Useful point-cloud-conditioned baseline |
| 6 | FastMesh | Try if repo/weights run cleanly |

## Sprint 2: OmniPart-Style Structure

OmniPart is the preferred part-structure direction because it is aligned with TRELLIS-style sparse latent structure. Treat it as TRELLIS-native, not guaranteed TRELLIS.2/O-Voxel drop-in.

First experiment:

```text
Can TRELLIS.2 O-Voxel / mesh output be aligned into the OmniPart/TRELLIS SLat workflow without losing too much structure?
```

Initial outputs should be structural, not final geometry:

```text
part boxes
part masks
part IDs
part-level point clouds
part-level proxy meshes
```

Then feed each part into the winning mesh head.

## Sprint 3: Choose One Mesh Head

Decision rule:

```text
Pick the method that wins on topology/editability failures, not the method with the prettiest curated screenshot.
```

Likely first bet: MeshRipple.

Strong baseline: Mesh Silksong.

Useful comparison: DeepMesh.

Part-aware candidate: MeshMosaic, once OmniPart-style components exist.

## Fixed Test Set

Use 50-100 frozen cases across:

```text
organic:
- animal
- character
- creature
- hand
- plant/tree

hard surface:
- chair
- robot
- vehicle
- tool
- appliance

topology stress:
- mug handle
- headphones
- eyeglasses
- bicycle
- lamp
- branching tree

production:
- game prop
- toy
- furniture
- product asset
```

## Existing ClearMesh Pieces

Keep:

```text
TRELLIS.2 generation
Stage 2 refinement research artifacts
mesh repair/export utilities
text-to-image/image-to-3D entry points
PBR/export/rigging as downstream optional modules
```

Deprioritize for v1:

```text
PartCrafter as primary decomposition
geometric primitive/B-Rep upgrade
retopology branch as a separate research bet
Easy3E editing until mesh-head bake-off is clearer
```

## Component Table

| Component | Role | Status |
|---|---|---|
| TRELLIS.2 | high-fidelity initial asset / proxy | keep |
| Evaluation harness | moat and sprint-zero deliverable | starter implemented |
| OmniPart-style stage | semantic structure and part latents | next integration target |
| MeshRipple | first mesh-head candidate | bake off |
| Mesh Silksong | topology-preserved baseline | bake off |
| DeepMesh | RL artist-mesh baseline | bake off |
| MeshMosaic | component-aware candidate | after parts exist |
| TreeMeshGPT | point-cloud baseline | bake off |
| DualPrim | primitive branch | archived / not v1 |

## Optional Autorigging In Production

Autorigging is still part of the full product, but it is not a v1 mesh-head selection criterion. It should run after the mesh has passed repair/export gates.

```text
final validated mesh
  -> riggability classifier
  -> Puppeteer primary / UniRig fallback
  -> weights and skeleton QA
  -> FBX or rigged GLB export
```

Failure should not fail the whole generation by default. If rigging fails, deliver the static mesh and report rigging as a separate failed optional step.
