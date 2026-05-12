# Quad Remeshing Strategy

## Recommendation

Do not promise universal artist-grade quads from raw TRELLIS output in v1.

Do build a staged, generic charted retopology lane behind the same ClearMesh
pathway:

```text
visual mesh
  -> mesh passport
  -> watertight/detail reference refinement when configured
  -> normalized control surface
  -> feature graph and chart decomposition
  -> chart merge/repair
  -> semantic parts where available
  -> per-chart quad layout or triangle fallback
  -> feature-preserving projection/detail bake
  -> subdivision and Blender QA
```

The key distinction:

```text
generic quad remesh: makes quads
artist quad topology: makes useful edge flow
```

## Practical Baselines

Use these as benchmarks, not as the final product promise:

```text
Instant Meshes / pyinstantmeshes:
  fast field-aligned quad-dominant remeshing
  good for previews and non-rigged assets
  weaker on semantic loops and animation topology

QuadriFlow:
  robust scalable quadrangulation baseline
  useful as another automatic reference point
  still not enough for characters/hands/faces by itself

QuadGPT:
  promising native quad/mixed-topology model
  keep as a watchlist/refinement candidate
  do not block production on it
```

## Current Repo Implementation

The production worker now has opt-in chart and whole-object quad sidecar stages:

```text
surface_normalization / shrinkwrap_projection output
  -> retopology_planning
  -> chart_remesh
  -> chart_stitch
  -> quad_remesh
  -> feature_projection
  -> production_gate
  -> preview_publish and final pipeline continue
```

Trusted job metadata:

```text
chart_remesh_enabled=true
chart_remesh_engine=auto | pyinstantmeshes | instant_meshes_cli | quadriflow_cli | template_cage
chart_remesh_max_charts=8
chart_remesh_min_chart_faces=16
chart_stitch_enabled=true
chart_stitch_prefer_for_projection=true
quad_remesh_enabled=true
quad_remesh_engine=auto | pyinstantmeshes | instant_meshes_cli | quadriflow_cli | template_cage
quad_target_faces=5000
quad_pure=true
instant_meshes_path=/path/to/InstantMeshes
quadriflow_path=/path/to/quadriflow
quadriflow_sharp=true
quad_remesh_as_final=false
feature_projection_enabled=true
feature_projection_as_final=false
blender_gates_enabled=true
production_require_blender=false
```

Generic chart planning smoke test:

```bash
python3 scripts/product/analyze_retopology.py \
  --input control.obj \
  --report retopology_plan.json \
  --crease-angle 45 \
  --target-quads 5000
```

Command-line smoke test:

```bash
python3 scripts/product/quad_remesh.py \
  --input control.obj \
  --output quad.obj \
  --report quad_report.json \
  --engine template_cage
```

Per-chart smoke test:

```bash
python3 scripts/product/remesh_charts.py \
  --input control.obj \
  --plan retopology_plan.json \
  --output-dir chart_remesh \
  --manifest chart_remesh_manifest.json \
  --engine template_cage \
  --max-charts 4
```

Stitch successful chart outputs into a single quad candidate:

```bash
python3 scripts/product/stitch_chart_remesh.py \
  --manifest chart_remesh_manifest.json \
  --output stitched_chart_quads.obj \
  --report chart_stitch.json \
  --weld-tolerance 1e-6
```

Compare chart engines on the same plan:

```bash
python3 scripts/product/compare_chart_remesh_engines.py \
  --input control.obj \
  --plan retopology_plan.json \
  --output-dir chart_engine_compare \
  --report chart_engine_compare.json \
  --engines pyinstantmeshes,quadriflow_cli,template_cage
```

Install pyinstantmeshes for the public Instant Meshes baseline:

```bash
scripts/setup/install_quad_baselines.sh
THUNDER_INSTANCE_ID=0 scripts/thunder/install_quad_baselines.sh
```

To also build QuadriFlow:

```bash
INSTALL_QUADRIFLOW=1 scripts/setup/install_quad_baselines.sh
INSTALL_QUADRIFLOW=1 THUNDER_INSTANCE_ID=0 scripts/thunder/install_quad_baselines.sh
```

The sidecar publishes:

```text
retopology_plan asset
chart_remesh_manifest asset
chart_stitched_mesh asset
quad_mesh asset
projected_quad_mesh asset
production_gate_report asset
quad_remesh.json report
per-chart quad reports
chart_stitch.json report
feature/chart/seam summary
quad_ratio / pure_quad stats
normal mesh quality metrics after OBJ import
postprocess vertex-weld stats
pre/post weld metrics and promotion hint
```

This is deliberately conservative. `template_cage` is not a finished remesher;
it is a deterministic fallback and a scaffold for part-parametric templates.
Instant Meshes or pyinstantmeshes should be used for automatic quad-dominant
baselines when available.

## Thunder pyinstantmeshes Result

The retopology planner runs on the current Poisson control surface in about
6 seconds and explains why whole-object quad promotion is unsafe:

```text
chart count: 6,383
feature edges: 48,356
boundary edges: 2,658
feature-curve components: 3,493
risk: high
operator recommendation:
  cleanup_or_merge: 6,292 charts
  cross_field_with_feature_constraints: 91 charts
```

On the current Poisson control surface from the real TRELLIS.2 proxy:

```text
input: /tmp/clearmesh_surface_normalization_fixed/control_poisson.obj
engine: pyinstantmeshes
target: 5,000 faces
```

Before the vertex weld postprocess, pyinstantmeshes produced a visually
quad-like OBJ that downstream topology treated as nearly one island per quad:

```text
quad faces: 17,971
quad ratio: 1.0
connected components: 17,953
non-manifold edges: 70,982
boundary loops: 17,707
```

After the ClearMesh OBJ weld postprocess:

```text
quad faces: 18,105
quad ratio: 1.0
vertices: 19,498 -> 19,435
connected components: 910
non-manifold edges: 4,924
boundary loops: 111
```

A tolerance sweep did not solve the core problem:

```text
1e-6: components 827, non-manifold 4,848, boundary loops 110
1e-5: components 804, non-manifold 5,058, boundary loops 108
1e-4: components 810, non-manifold 4,948, boundary loops 112
```

Interpretation:

```text
vertex welding is a cheap required postprocess for automatic quad baselines
generic whole-object quad remeshing still fails the artist-editable bar here
part-aware templates and feature constraints remain the production direction
```

## Vertex-Weld Success Criteria

Vertex welding is only meant to fix false fragmentation from duplicated OBJ
vertices. A good weld result should preserve shape and quads while improving
topology:

```text
quad ratio: unchanged or nearly unchanged
face count: unchanged except degenerate cleanup
components: large drop toward real logical parts
boundary loops: large drop when loops were duplicate seams
non-manifold edges: large drop, not a rise
promotion hint: eligible_for_deeper_quad_gates
```

The report now stores pre-weld metrics, post-weld metrics, deltas, component
reduction, boundary-loop reduction, and a promotion hint. If the output still has
many components or boundary loops, welding helped but the mesh is not promoted.

## Novel ClearMesh Direction

### Generic Feature/Chart Graph

The first principle is to avoid one global decision over a damaged whole-object
mesh.

```text
detect feature edges from:
- sharp creases
- boundary/open edges
- semantic part seams
- future silhouette/curvature/user-painted constraints

then:
- flood-fill smooth charts across non-feature edges
- build chart adjacency/seam graph
- classify chart risk and local shape
- choose operator per chart
```

Templates are only one possible operator. The generic fallback is chart-level
cross-field quadrangulation or triangle control topology.

### Part-Parametric Quad Atlas

Part templates are fast paths for common structures, not a coverage assumption.

```text
feature/part charts
  -> classify each part as tube / box / sheet / disk / sphere-like / branch
  -> generate a quad template for that primitive family
  -> deform template to the control surface
  -> stitch seams between adjacent parts
  -> project/bake detail from the visual mesh
```

Why this is better:

```text
tubes get circular edge loops
boxes get panel/grid loops
limbs/horns/handles get lengthwise loops
flat parts get sheet parameterizations
branching parts get explicit singularities at junctions
```

This targets editability directly instead of hoping a global remesher guesses
what artists want.

### Feature-Constrained Cross Field

For assets that do not fit templates, use a cross-field pipeline:

```text
detect feature curves from normals/curvature/silhouette/part seams
seed directional constraints along those curves
solve a smooth 4-RoSy cross field
place singularities intentionally
compute an integer-grid parameterization
extract quads
project back to the control/visual surface
```

This follows the proven family of mixed-integer quadrangulation methods, but
ClearMesh can improve it by adding semantic constraints from parts and visual
model priors.

### Learned Layout Prior

Longer term:

```text
train a small model to predict:
- feature curves
- singularity positions
- part template class
- seam graph
- target quad density map
```

This is more realistic than training a full native quad generator immediately.
The deterministic solver still creates the mesh; the model proposes artist-like
layout hints.

## Quality Gates

Quad output must pass more than `quad_ratio`.

```text
quad ratio
extraordinary vertex count and placement
edge-loop continuity
valence histogram
UV stretch
subdivision smoke test
deformation/rigging smoke test for characters
Blender roundtrip
artist edit-time review
```

## Production Stance

```text
v1:
  editable triangle control mesh, optional automatic quad preview/experiment

v1.5:
  part-parametric quad atlas for simple parts: handles, tubes, boxes, limbs,
  sheets, and branch junctions

v2:
  feature-constrained cross-field quadrangulation with learned layout priors

not now:
  promise perfect quads on arbitrary AI meshes
```

## References

- Instant Meshes: https://github.com/wjakob/instant-meshes
- pyinstantmeshes: https://pypi.org/project/pyinstantmeshes/
- Mixed-Integer Quadrangulation: https://www.vci.rwth-aachen.de/publication/0344/
- libigl cross-field / MIQ implementation notes: https://libigl.github.io/tutorial/
- xatlas chart parameterization baseline: https://github.com/jpcy/xatlas
- QuadriFlow: https://yichaozhou.com/publication/1805quadriflow/
- QuadGPT: https://arxiv.org/abs/2509.21420
