# ClearMesh Product UI Spec

## Design Thesis

ClearMesh should feel as calm and direct as Claude, but with spatial output treated as the main conversation artifact. The chat is not the product by itself; the mesh, its variants, its quality report, and its edit history are the product.

## UI Principles

```text
light, warm, low-friction
conversation-first, but preview-dominant once output exists
few controls visible at once
advanced settings tucked behind progressive disclosure
quality reports summarized in human language first, raw metrics second
Easy3E editing feels like continuing the conversation, not opening a separate tool
```

## Primary Layout

```text
left rail: projects, recent assets, account
center: conversation / generation prompt / edit instructions
right: visual workspace with 3D preview, variants, quality, exports
bottom drawer: timeline, metrics, logs, advanced settings
```

On mobile:

```text
single-column conversation
preview card immediately after latest generated asset
quality/export actions collapse into tabs
left rail becomes command menu
```

## Core Screens

### New Generation

Fields:

```text
image upload or text prompt
style/use-case selector: game asset, product viz, miniature, riggable character
quality tier: draft, standard, high
optional toggles: semantic parts, autorigging, texture preservation
```

Primary action:

```text
Generate mesh
```

### Job Detail

Shows:

```text
large 3D preview
job stage timeline
latest assistant-style status message
quality summary
variants
exports
```

Primary progress states:

```text
queued: calm pending card
running: stage timeline, no scary logs
preview_ready: enable viewer/download preview mesh while high-res continues
final_ready: show editable mesh, exports, quality summary
failed: explain the failure and keep any usable preview artifact visible
```

The UI should consume `GET /v1/jobs/{job_id}/status`, especially:

```text
progress.phase
progress.message
progress.preview_ready
progress.final_ready
asset_urls.preview_asset_url
asset_urls.preview_mesh_asset_url
asset_urls.reference_mesh_asset_url
asset_urls.retopology_plan_asset_url
asset_urls.chart_remesh_manifest_asset_url
asset_urls.quad_mesh_asset_url
asset_urls.projected_quad_mesh_asset_url
asset_urls.quality_report_asset_url
asset_urls.production_gate_asset_url
asset_urls.final_asset_url
```

Quality summary example:

```text
Editable mesh: Passed
Topology: 2 small boundary loops repaired
Retopology plan: 18 charts, generic cross-field recommended
Production gate: Tier 1 watertight triangle, quad sidecar not promoted
Quad sidecar: Experimental, 96% quads, projected, awaiting seam/Blender promotion
Rigging: Eligible, not yet run
Exports: GLB and OBJ ready
```

### Easy3E Editing

Editing should be inline:

```text
"make the horns longer"
"turn this into a toy-ready version"
"split the armor into separate editable parts"
```

The UI should preserve lineage:

```text
v1 original
v2 longer horns
v3 toy-ready
v4 rigged character
```

### Export / Delivery

Download panel:

```text
GLB
OBJ + MTL
STL
FBX if rigged
quality report JSON
preview renders
```

## Competitive Edge

Most mesh tools expose either a black-box generate button or a dense DCC-like interface. ClearMesh can win by combining:

```text
Claude-like conversational calm
visual previews always visible
quality gates explained plainly
semantic parts as a product feature
Easy3E edits as a continuation of the same thread
optional autorigging as a one-click downstream step
```
