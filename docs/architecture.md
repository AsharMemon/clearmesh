# ClearMesh Architecture

## Product Shape

ClearMesh is a generation service for editable 3D assets. The core product promise is not merely "pretty 3D from an image"; it is a mesh that survives artist workflows, downstream DCC tools, export, optional rigging, and customer delivery.

## Core V1 Pipeline

```text
User input
  -> image upload or text prompt
  -> optional text-to-image conditioning
  -> TRELLIS.2 visual/proxy asset
  -> UltraShape or manifold reference refinement when configured
  -> mesh passport diagnostics
  -> normalized control surface
  -> generic retopology planning: features, charts, seams, risk
  -> chart merge/repair and optional chart-level remesh
  -> optional QuadriFlow/Instant Meshes quad sidecar
  -> feature-preserving projection to the reference surface
  -> early preview/export
  -> OmniPart-style semantic part structure
  -> point-cloud/proxy bridge from the control surface
  -> targeted MeshRipple/Silksong-style topology prior when the passport says it is safe
  -> mesh repair and validation
  -> Blender/DCC production promotion gates
  -> texture/material preservation or rebake
  -> export package
  -> optional autorigging
```

This is one product pathway. Internally it is adaptive, but users should not see
"choose MeshRipple route vs UltraShape route vs cleanup route." The invariant is:

```text
every asset becomes a visual mesh + reference surface + bounded control surface
every high-compute asset gets a retopology plan and production gate before promotion
```

The preview appears as soon as the visual mesh/control surface is ready. High
resolution continues in the same job and publishes a later asset revision.
Quad output is published as a scored sidecar asset until it consistently beats
the triangle/control mesh on editability gates.

The current worker exposes this as one route with optional engines:

```text
reference_refinement: UltraShape command, ManifoldPlus command, supplied mesh, or local Poisson/cleanup
part_structure: OmniPart command or connected-component MeshMosaic-style fallback
chart_remesh: per-chart QuadriFlow/Instant Meshes/template-cage sidecars
chart_stitch: concatenates, welds, scores, and promotes chart outputs when safe
feature_projection: OBJ vertex projection that preserves quad face arity
production_gate: mesh metrics plus optional Blender subdivision/deformation/roundtrip gates
```

## Generation Plane

### Input Layer

```text
image upload
text prompt -> image generator -> image upload path
edit_image / edit_text via Easy3E
future: multi-view images, sketches, depth/normal controls
```

Responsibilities:

```text
file validation
content and abuse checks
background removal
input normalization
job creation
asset lineage tracking
```

### TRELLIS.2 Proxy Layer

TRELLIS.2 remains the high-fidelity first-pass generator and proxy/reference asset.

Outputs to preserve:

```text
raw TRELLIS.2 mesh or GLB
O-Voxel / sparse structure artifacts when available
sampled point clouds at 16k / 40k / 100k
rendered previews
metadata needed for reproducibility
```

### Mesh Passport Layer

The mesh passport turns raw topology metrics into production decisions before
expensive refinement runs. It is inspired by the failure mode that MeshRipple and
Mesh Silksong are trying to solve: autoregressive mesh heads need coherent
conditioning structure, not shredded arbitrary triangle soup.

Passport inputs:

```text
connected components
tiny component count
boundary loops
non-manifold edges/vertices
watertightness
face and vertex counts
triangle quality
```

Passport outputs:

```text
risk level
whether surface normalization is required
whether topology-aware mesh heads are eligible
whether high-resolution publish is safe
estimated preview and high-resolution timing
```

### Surface Normalization Layer

The normalized control surface is the central production trick. It lets us
support arbitrary input vertex counts without asking MeshRipple or any other
artist-mesh head to repair broken generator output directly.

Preferred operator:

```text
sample dense oriented points from the visual mesh
  -> reconstruct a coherent surface with Poisson/SDF/voxel reconstruction
  -> crop and remove low-confidence density
  -> simplify to a bounded face budget
  -> evaluate again
```

Fallback operator:

```text
remove degenerate faces
remove tiny fragments
fill obvious holes
fix normals
simplify if dependencies are available
```

This layer is where UltraShape-like ideas remain useful, but not as
`TRELLIS -> UltraShape -> whole-object MeshRipple` by default. The worker now
has a `reference_refinement` stage before the passport so UltraShape,
ManifoldPlus, or a supplied manifold mesh can become the high-detail reference
surface. The normalized control surface and quad/chart outputs can then project
back toward that reference without inheriting its triangle soup.

Optional projection operator:

```text
clean control surface
  -> constrained shrink-wrap toward visual mesh
  -> preserve control topology
  -> bake visual normals/detail/materials
```

This is a benchmark candidate, not a required default. Shrink-wrap only helps
when the input topology is already clean; it should not be used to turn raw
triangle soup into topology.

### Evaluation Harness Layer

The harness is a first-class production component, not only a research script.

It should run at three points:

```text
baseline: TRELLIS.2 output
candidate: mesh-head output
release gate: final repaired/exported asset
```

It produces:

```text
machine-readable JSON metrics
pass/fail quality gates
customer-facing warnings when an output is usable but imperfect
internal dashboards for model bake-offs
```

### Part Structure Layer

Use OmniPart-style structure as the preferred direction.

Expected artifacts:

```text
part boxes
part masks
part IDs and labels
part-level point clouds
part-level proxy meshes
parent/child relationships where available
```

The part layer feeds:

```text
per-part mesh generation
part-aware repair
part-aware material assignment
part-aware editing
future autorigging hints
```

### Artist-Mesh Head Layer

V1 uses one product flow and keeps mesh heads as refinement operators behind the
mesh passport. MeshRipple remains valuable, but it is no longer the backbone that
every user must wait for.

Initial candidates:

```text
MeshRipple
Mesh Silksong
DeepMesh
MeshMosaic
TreeMeshGPT
FastMesh if runnable
```

Selection rule:

```text
topology/editability/production pass rate beats screenshot quality
```

Mesh head eligibility rule:

```text
only feed a mesh head a coherent control surface or part-level control surface
```

This matches the logic of the papers:

```text
MeshRipple: frontier-aware connected growth needs coherent conditioning
Mesh Silksong: layered topology assumes manifold-aware structure
OmniPart: part planning improves locality before detailed synthesis
UltraShape: refinement benefits from structured geometric latents
```

### Quad / Cage Roadmap

High-quality quads are a separate product promise from "clean editable triangles."
The likely path is not generic global quad remeshing alone and not templates
alone. It is charted retopology:

```text
control surface
  -> feature edges from boundaries/creases/part seams
  -> smooth chart decomposition
  -> seam graph and chart risk report
  -> per-chart operator choice:
       template when obvious
       cross-field quadrangulation when generic
       triangle control fallback when unsafe
  -> projection/detail baking from the visual mesh
  -> subdivision and Blender QA
```

The current implementation includes a `retopology_planning` job step plus an
opt-in `quad_remesh` sidecar step. `scripts/product/analyze_retopology.py`
emits the feature/chart/seam plan. `scripts/product/quad_remesh.py` tries
Instant Meshes bindings/CLI when available and otherwise emits a deterministic
pure-quad template cage for smoke tests and future part-template work. The quad
step is not enabled by default and does not replace the final mesh unless
trusted metadata explicitly sets `quad_remesh_as_final=true`.

See `docs/quad_remeshing_strategy.md`.

### Repair, Validation, Export Layer

The final asset gate includes:

```text
mesh repair
component cleanup
normal/winding checks
optional decimation
UV/material preservation checks
Blender import/export roundtrip
GLB/OBJ/STL/FBX export
asset manifest and preview renders
```

## Optional Autorigging Layer

Autorigging stays in the architecture, but after the mesh is validated.

```text
validated semantic mesh
  -> riggability classifier
  -> skeleton + skinning backend
  -> rig QA checks
  -> FBX/GLB export with skeleton
```

Preferred behavior:

```text
organic/humanoid/creature assets: offer autorigging
hard-surface props: skip by default
mixed assets: ask user or run only on eligible parts
```

Current backends in repo:

```text
Puppeteer: primary if installed
UniRig: fallback/generalist
HumanRig: future humanoid-specific scaffold
```

Autorigging quality gates:

```text
skeleton present
weights sum to 1 per vertex
no large unweighted vertex islands
joint count within expected range
basic pose smoke test
FBX/GLB import into Blender succeeds
```

## Service Architecture

```text
Web app / SDK / API clients
  -> API gateway
  -> auth + billing middleware
  -> job API
  -> queue
  -> GPU workers
  -> artifact storage
  -> evaluation service
  -> notification/webhook service
  -> admin + observability dashboards
```

## Runtime Components

### API Server

Responsibilities:

```text
users, teams, projects
API keys and OAuth/session auth
job submission and status
image-to-3D, text-to-3D, edit_image, and edit_text job modes
asset metadata
billing entitlements
webhooks
signed upload/download URLs
```

Suggested stack:

```text
FastAPI or Node/TypeScript API
Postgres for relational state
Redis for queues/cache/rate limits
S3/R2/GCS for artifacts
Stripe for billing
OpenTelemetry + Sentry + Prometheus/Grafana for observability
```

### Worker System

Worker classes:

```text
cpu-preprocess: validation, background removal, previews
gpu-generate: TRELLIS.2 and mesh-head inference
gpu-eval: heavy mesh metrics and Blender checks
cpu-package: export zips, metadata, thumbnails
optional-gpu-rig: autorigging jobs
gpu-edit: Easy3E edit jobs
```

Queue requirements:

```text
priority tiers
retry policy
idempotent job steps
checkpoint/resume for long jobs
dead-letter queue
GPU capacity-aware scheduling
per-customer concurrency limits
```

### Artifact Storage

Recommended object layout:

```text
projects/{project_id}/jobs/{job_id}/input/original.png
projects/{project_id}/jobs/{job_id}/trellis/proxy.glb
projects/{project_id}/jobs/{job_id}/points/points_40000.ply
projects/{project_id}/jobs/{job_id}/parts/parts.json
projects/{project_id}/jobs/{job_id}/mesh/final.glb
projects/{project_id}/jobs/{job_id}/mesh/final.obj
projects/{project_id}/jobs/{job_id}/rig/final.fbx
projects/{project_id}/jobs/{job_id}/reports/eval.json
projects/{project_id}/jobs/{job_id}/previews/*.png
```

## Data Model

Minimum production tables:

```text
users
teams
memberships
api_keys
projects
jobs
job_steps
assets
asset_versions
evaluation_reports
billing_customers
subscriptions
usage_events
webhook_endpoints
webhook_deliveries
audit_log
```

## Quality Gates

A production job should have explicit gates:

```text
input accepted
TRELLIS.2 proxy generated
mesh head generated
topology gate passed or warning issued
repair/export gate passed
optional rig gate passed
billing usage recorded
artifacts delivered
```

Failure modes should be user-visible and actionable:

```text
retryable infrastructure failure
unsupported input
low-confidence geometry
non-riggable asset
export failed
billing/entitlement blocked
```
