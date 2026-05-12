# Production Roadmap

## Phase 0: Repository And Research Reset

Goal: make the repo reflect the new plan.

Deliverables:

```text
archive/remove DualPrim artifacts
keep Stage 2/TRELLIS learnings as research assets
keep autorigging as optional downstream module
add architecture and production roadmap docs
keep generated output artifacts out of git
```

Done in this pass:

```text
tracked .DS_Store removed
tracked .codex_outputs DualPrim artifacts removed
ignored local generated outputs and transfer artifacts
DualPrim audit archived
harness docs added
API/job/billing scaffold added
point-cloud sampling bridge added
mesh passport/control-surface contract added
preview/final API progress state added
worker emits preview_image and control_mesh assets before slow refinement
generic retopology planning report added before quad sidecar
quad remesh sidecar added behind `quad_remesh_enabled`
reference refinement stage added for UltraShape/ManifoldPlus/supplied manifold meshes
component-based MeshMosaic-style part manifest fallback added
chart-level remesh manifest added behind `chart_remesh_enabled`
chart stitch promotion candidate added behind `chart_stitch_enabled`
QuadriFlow CLI support added alongside pyinstantmeshes/Instant Meshes
feature-preserving OBJ projection added for quad sidecars
production gate report added before export, with optional Blender gates
```

## Phase 1: Evaluation Harness MVP

Goal: every mesh candidate gets scored before we debate aesthetics.

Deliverables:

```text
batch CSV manifest
JSON report output
geometry metrics: Chamfer, Hausdorff, normals, area/volume ratios
topology metrics: watertightness, boundaries, non-manifold counts, components
editability metrics: valence, poles, aspect ratios, face counts, subdivision smoke
production metrics: Blender import/export roundtrip
```

Exit criteria:

```text
50-100 fixed cases run end-to-end
TRELLIS.2 baseline report generated
reports are comparable across methods
failures are row-level, not batch-fatal
```

## Phase 2: Unified Surface Contract

Goal: make every asset go through one production pathway before expensive
refinement.

Deliverables:

```text
mesh passport report for every proxy/control mesh
surface normalization step after TRELLIS.2 or Easy3E
Poisson/SDF/voxel reconstruction on GPU workers when dependencies are available
cleanup fallback for local/dev workers
preview publish from the normalized control mesh
mesh-head policy defaulting to passport-gated refinement
generic feature/chart/seam retopology plan for every heavy control mesh
optional quad/cage sidecar with report and status asset URL
optional chart remesh sidecar with manifest and per-chart reports
optional chart stitch mesh with weld stats, quad ratio, and promotion hint
feature projection sidecar that preserves OBJ quad topology
production gate asset that reports whether the mesh is watertight/editable/promoted
```

Exit criteria:

```text
all vertex counts and component counts produce a bounded control mesh or clear failure
MeshRipple is skipped automatically for shredded conditioning surfaces
the product presents one visible ClearMesh flow
preview artifact exists before any slow autoregressive refinement
retopology plan exposes whether quads should be template, cross-field, or fallback
quad experiments never replace the final mesh unless explicitly promoted
Blender gates can be required in production without breaking local/dev workers
```

## Phase 3: TRELLIS.2 Baseline Dataset

Goal: freeze a reliable benchmark corpus.

Deliverables:

```text
50-100 curated input images/prompts
TRELLIS.2 raw outputs
proxy/reference meshes where possible
preview renders
case taxonomy: organic, hard surface, topology stress, production
manifest generation script
```

Exit criteria:

```text
all cases have stable IDs
all source inputs and TRELLIS.2 outputs are reproducible
baseline quality report exists
```

## Phase 4: Point-Cloud Bridge

Goal: decide if point-cloud conditioning is good enough for v1.

Deliverables:

```text
sample points from TRELLIS.2 meshes at 16k, 40k, 100k
script: scripts/data/sample_point_clouds.py
export PLY/NPZ formats compatible with mesh heads
normal/color/part-ID channels where available
bridge manifest linking point clouds to source assets
```

Exit criteria:

```text
same cases can be fed into each candidate head
thin-structure failure rate is measured
clear decision on point-cloud bridge vs O-Voxel/SLAT-native work
```

## Phase 5: Mesh-Head Bake-Off

Goal: pick one refinement operator for eligible control surfaces, not a universal
backbone that every job must wait for.

Candidates:

```text
MeshRipple first
Mesh Silksong second
DeepMesh third
TreeMeshGPT as baseline
MeshMosaic after part/component data exists
FastMesh only if inference is clean
```

Deliverables:

```text
per-method inference wrappers
per-method environment notes
standardized output folders
harness reports per method and point budget
side-by-side visual previews
```

Exit criteria:

```text
one primary mesh head chosen
one fallback baseline retained
known failure modes documented
no custom training unless inference-only bake-off fails hard
```

## Phase 6: OmniPart-Style Part Structure

Goal: introduce semantic editability without betting the whole system on parts immediately.

Deliverables:

```text
TRELLIS.2/O-Voxel to OmniPart-compatible experiment
part boxes, masks, IDs
part-level point clouds
part-level proxy meshes
part manifest schema
```

Exit criteria:

```text
parts improve editability metrics or user workflow
part outputs can feed the chosen mesh head
part labels survive export as metadata or object groups
```

## Phase 6: End-To-End Local Product MVP

Goal: one command creates a validated asset package.

Deliverables:

```text
single CLI job runner
input image/prompt
TRELLIS.2 proxy
chosen mesh head
repair/export
harness report
preview renders
zip package with GLB/OBJ/STL and metadata
optional autorigging flag
```

Exit criteria:

```text
10 representative assets complete without manual intervention
all artifacts follow stable storage layout
failures produce clear diagnostics
```

## Phase 7: API MVP

Goal: make generation and Easy3E editing available as an async service.

Deliverables:

```text
POST /v1/jobs
GET /v1/jobs/{id}
GET /v1/jobs/{id}/assets
POST /v1/uploads/sign
GET /v1/assets/{id}/download
POST /v1/webhooks/test
job modes: image_to_3d, text_to_3d, edit_image, edit_text
```

Core behavior:

```text
async jobs
signed uploads/downloads
idempotency keys
request validation
rate limits
structured errors
OpenAPI schema
```

Exit criteria:

```text
SDK or curl workflow can submit a job and download assets
API survives worker restarts
job status is accurate and auditable
```

## Phase 8: Auth, Teams, And Security

Goal: make the service safe for real users.

Deliverables:

```text
email/password or OAuth login
API keys
team/project membership
role-based access control
request signing or scoped tokens for API use
asset-level authorization
basic abuse prevention
```

Security checklist:

```text
secrets manager
encrypted object storage
signed URLs with expiration
audit logs
rate limiting per user/team/key
input file scanning
private-by-default assets
```

Exit criteria:

```text
users cannot access other users' assets
API keys can be rotated/revoked
admin can trace job and billing events
```

## Phase 9: Billing And Entitlements

Goal: connect GPU cost to product packaging.

Suggested model:

```text
free trial credits
usage-based generation credits
paid plans with monthly credit allotment
paid add-ons for high-res, batch, private deployments, autorigging
```

Deliverables:

```text
Stripe customer/subscription integration
checkout and billing portal
usage_events table
credit ledger
entitlement checks before job enqueue
cost accounting per job step
refund/retry policy
```

Exit criteria:

```text
job cannot exceed entitlement silently
failed infrastructure jobs do not burn customer credits
successful jobs record usage exactly once
billing state is visible in dashboard and API
```

## Phase 10: Web App

Goal: make the workflow pleasant for non-API users.

Pages:

```text
landing / waitlist
login/signup
project dashboard
new generation
job detail with live status
3D viewer and previews
asset downloads
billing/settings
API key management
```

Viewer features:

```text
GLB preview
part visibility toggles
quality report summary
download variants
rerun with new settings
optional rigging request
```

Exit criteria:

```text
a user can sign up, pay, generate, inspect, and download without support help
```

## Phase 11: Production Operations

Goal: make it boring to run.

Deliverables:

```text
containerized API and workers
GPU worker images pinned by model stack
infrastructure-as-code
staging and production environments
CI for API/unit tests
nightly benchmark runs
model/version registry
observability dashboards
on-call runbooks
```

Operational metrics:

```text
queue latency
GPU utilization
job success/failure rate
cost per successful asset
model quality regressions
API latency/error rate
billing event accuracy
```

Exit criteria:

```text
deploys are repeatable
rollbacks are documented
quality regressions are caught before users do
```

## Phase 12: Editing Productization

Goal: bring Easy3E into the product after the base generation pipeline is reliable.

Deliverables:

```text
edit_image and edit_text API flows
source asset lineage tracking
edit preview renders
edit masks and part-aware preservation controls
failed-edit fallback that preserves original asset
texture editing strategy after Ctrl-Adapter integration
```

Exit criteria:

```text
existing generated assets can be edited without regenerating from scratch
edit jobs use the same auth, billing, artifact, and quality-gate stack
Easy3E failures are isolated from static asset delivery
```

## Phase 13: Production V1 Launch

Launch scope:

```text
image-to-3D generation
fixed mesh-head backend
validated GLB/OBJ/STL exports
optional autorigging for eligible assets
API + web app
auth + teams
billing + credits
quality reports
```

Defer until after launch:

```text
full interactive editing
custom SLAT-native mesh head
native quad generation
B-Rep/STEP/CAD output
large-scale custom training
marketplace/community asset sharing
```

## What To Do Next

1. Run the full route on Thunder: TRELLIS.2 proxy, UltraShape or ManifoldPlus reference refinement, retopology plan, chart remesh, quad remesh, feature projection, MeshRipple/Silksong adapter, cleanup, production gate.
2. Run `retopology_planning` over the 50-100 TRELLIS.2 benchmark cases and cluster failure modes by chart count, boundary loops, feature graph complexity, and production-gate tier.
3. Compare `pyinstantmeshes`, Instant Meshes CLI, and `quadriflow_cli` on the same chart-level and whole-object cases after vertex welding and feature projection.
4. Replace the connected-component fallback with real OmniPart masks for the benchmark set, then measure per-part MeshRipple/Silksong against whole-object runs.
5. Harden seam stitching/promotion for chart remesh outputs. The first stitcher now concatenates/welds/scores chart outputs; it still needs seam-aware boundary alignment, UV/material handling, and stricter Blender promotion before becoming default final.
6. Build the first part-parametric quad templates as fast paths, not coverage assumptions: tube/handle, box/panel, sheet, limb, branch junction.
7. Wire the UI to show preview, final, quality report, production gate tier, retopology plan, chart remesh manifest, projected quad, and optional rig asset from `/v1/jobs/{job_id}/status`.
8. Add richer Blender gates: extraordinary vertex map, edge-loop continuity proxy, UV stretch proxy, rigging deformation smoke test, and visual screenshots for gate failures.

### Easy3E Worker Hook

`edit_image` and `edit_text` jobs now have a product worker seam via `scripts/product/run_easy3e_edit.py`. Public API callers cannot submit this command directly; trusted worker/job creation paths can set `metadata.easy3e_command` after resolving the source mesh and edit image into artifact-root-safe paths.

### Mesh Cleanup Stage

The production worker now includes a `mesh_cleanup` step after mesh-head generation. It preserves the raw `artist_mesh`, writes a separate `cleaned_mesh`, then evaluates and exports the cleaned mesh. This gives us artist-mesh bake-off truth while making customer-facing exports less cluttered.

Initial cleanup is conservative debris removal, not full repair. It reduces tiny components and non-manifold edges, but final quality still depends on better mesh-head settings and part-aware generation.
