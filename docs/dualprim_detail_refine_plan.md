# DualPrim Detail Refinement Plan

## Goal

Keep DualPrim as the compact, editable, artist-like structural layer while adding a narrowly-scoped detail stage that improves local accuracy without blowing up the mesh.

The key product constraint is:

- DualPrim remains responsible for low-frequency structure, topology, and part layout.
- The refinement stage is only allowed to add controlled high-frequency detail near the DualPrim surface.
- We do not replace DualPrim with a dense neural field or a generic high-vertex reconstruction.

## Current status

Best current pure DualPrim checkpoint:

- `r6`
- `27 / 100` alive primitives
- `31,674` verts
- `63,280` faces
- `21` components

What we learned from the recent runs:

- `r4 -> r5 -> r6` proved the main unlock was late-stage alpha/prune scheduling.
- `r7` polish was a useful negative result: more generic polishing softened the mesh and increased clutter.
- `r8` paper-side gating/pruning refinement did not beat `r6`.
- `r9` full StableNormal supervision ran end-to-end but re-inflated the surface and also did not beat `r6`.
- `r10` is the safer follow-up: StableNormal blended toward analytic normals, late normal-weight ramp, and budgeted export cleanup.

So the current position is:

- `r6` remains the best pure DualPrim structural anchor.
- We still have some headroom inside DualPrim through safer supervision and export cleanup.
- The next architectural move after that should be primitive-local detail, not a global dense residual.

## Why this is needed

The current camera canary trajectory (`r2 -> r3 -> r4 -> r5 -> r6`) shows that better late-stage alpha/prune dynamics produce much more compact outputs. That said, the exported meshes still have two remaining issues:

1. Some boundaries are too fused or blobby.
2. Fine details around the lens and small protrusions are still underfit.

This matches what we would expect from DualPrim:

- Strengths:
  - compact structural decomposition
  - semantic-ish parts
  - negative-shape carving
  - editability
- Weaknesses:
  - very fine axial/ring details
  - local surface finish
  - crisp high-curvature detail on small subparts

## Learning from RNb-NeuS and FlexiCubes

### RNb-NeuS

RNb-NeuS shows that high-frequency reconstruction quality improves a lot when the optimization is driven by richer surface cues, especially normals and reflectance, not just coarse occupancy/silhouette agreement.

The practical learning for DualPrim is:

- do not ask the primitive scaffold to carry all detail by itself
- use a later stage with stronger local surface supervision

### FlexiCubes

FlexiCubes shows that a mesh extractor with extra local degrees of freedom can preserve much more detail than rigid extraction from the same underlying field.

The practical learning for DualPrim is:

- even if DualPrim has the right coarse field, a rigid export step can still lose detail
- a flexible detail stage can improve the final mesh without discarding the coarse representation

## The staged system

### Stage A: DualPrim coarse solve

Purpose:

- recover compact structure
- recover semantic-ish part layout
- recover holes and carve logic
- produce the editable primitive representation

Deliverable:

- `primitives.json`
- compact exported `refit.glb`

### Stage B: safer supervision and export cleanup

Purpose:

- keep the survivor set coherent
- sharpen boundaries without losing the compact basin
- improve final extraction quality without letting the mesh bloat

Concretely:

- safer StableNormal supervision:
  - blend StableNormal toward analytic normals
  - ramp `lambda_norm_reg` late instead of driving it hard from iter `0`
  - optionally boost StableNormal influence near boundaries only
- budgeted export cleanup:
  - remove tiny disconnected components
  - apply very light volume-preserving smoothing
  - keep hard budget limits so this remains polish, not remodelling

This is the `r8/r9/r10` family of experiments.

Operational handoff:

- training-side Stage B remains the `r8/r9/r10` canary family
- export-side Stage B is now a dedicated post-checkpoint pass via:
  - `scripts/dualprim/setup_stage_bc.py`
- this lets us polish the best checkpoint (`r6`) without re-entering a worse basin

### Stage C: primitive-local detail refinement

Purpose:

- add only the missing local detail
- operate in each surviving primitive's local frame
- preserve the coarse structure and compactness decisions made by DualPrim

This stage should be constrained by:

- a narrow spatial band around each primitive surface
- a maximum vertex growth budget
- a maximum component growth budget
- a maximum displacement budget

That way, we improve fidelity without defeating the purpose of DualPrim.

### Stage D: edge-aware structural refinement

Purpose:

- sharpen creases and part boundaries that remain too soft after Stage C
- improve "artist-made" intentionality without increasing top-level part count

Preferred mechanisms:

- crease-aware alignment loss between neighboring surviving primitives
- optional tiny dictionary extension for clearly structured shapes:
  - superquadric
  - cylinder / tapered cylinder

Guardrails:

- no free-form global field
- no unconstrained primitive proliferation
- all additions must preserve the compactness budget established by `r6`

### Stage E: hierarchical or narrow-band fallback

Purpose:

- provide one last controlled path if Stage C and Stage D still leave important local errors

Preferred order inside this fallback stage:

1. hierarchical parent-child primitives anchored to surviving coarse parts
2. only if necessary, a narrow-band residual around the combined DualPrim field

This stage exists as a fallback, not the default next move.

If we ever reach Stage E, the same project guardrails still apply:

- bounded component growth
- bounded vertex growth
- bounded local displacement
- no global remodelling that defeats DualPrim's compact/editable premise

## What we are explicitly not doing

We are not:

- replacing DualPrim with a dense mesh-first or field-first model
- letting the refinement stage freely change topology everywhere
- allowing uncontrolled vertex count growth
- using the detail stage as a second full reconstruction system

If a detail method cannot respect the compactness budget, it is the wrong detail method for this project.

## Recommended first implementation

The first refinement implementation should be a bounded, per-primitive local deformation model.

Conceptually:

- freeze or lightly tune the coarse DualPrim scaffold
- attach a tiny local residual module to each strong surviving primitive
- evaluate that module only near the primitive surface
- add its displacement in the primitive's local frame

This is preferred over a global residual because:

- detail stays attached to interpretable parts
- editability is preserved
- vertex/component growth is easier to budget
- the lens/body/grip can sharpen independently without turning the whole object into a dense field

The already-implemented preparation layer is still useful because it makes that local stage measurable and budgeted.

Specifically:

1. Compute a normalized coarse-vs-target comparison.
2. Sample target surface points and normals.
3. Project them onto the coarse DualPrim mesh.
4. Identify the narrow band where detail is missing.
5. Save:
   - a detail-refinement manifest
   - sampled supervision points
   - compactness budgets
6. Split those narrow-band samples by surviving primitive in the primitive's
   own local frame, with explicit per-primitive displacement budgets.

This gives us a disciplined handoff from DualPrim to a later primitive-local detail optimizer.

Current implementation:

- global narrow-band artifacts:
  - `detail_refine_manifest.json`
  - `detail_refine_samples.npz`
- primitive-local artifacts:
  - `primitive_local_manifest.json`
  - `primitive_XXX_samples.npz`

These are produced by:

- `prepare_detail_refine_artifacts(...)`
- `prepare_primitive_local_refine_artifacts(...)`
- `scripts/dualprim/setup_stage_bc.py`
- `train_primitive_local_refiners(...)`
- `scripts/dualprim/run_local_refine.py`

## Guardrails against overcomplication

The detail stage should ship with explicit budget fields:

- `max_vertex_growth`
- `max_face_growth`
- `max_component_growth`
- `max_displacement_frac`
- `band_radius_frac`

Recommended starting guardrails:

- vertex growth: at most `1.35x` the coarse mesh
- face growth: at most `1.35x` the coarse mesh
- component growth: at most `1.15x` the coarse mesh
- displacement magnitude: at most `3%` of normalized object extent
- refine only inside a narrow band around the coarse surface

These are deliberately conservative. The refinement stage is meant to polish, not to re-model the object from scratch.

## Why not swept/deformable superquadrics first?

Swept or deformable superquadrics are a good medium-term idea, especially for lens barrels, rods, handles, and other extruded/swept shapes.

But they are not the first thing to add, because:

- the current bottleneck is still largely boundary/detail refinement
- we are still getting real gains from the existing primitive family
- adding a richer primitive family also expands the optimization/search space

So the preferred order is:

1. finish the current DualPrim + safer supervision/export track
2. add primitive-local detail refinement
3. only then consider axial-profile / swept primitives if the remaining errors are still specifically extrusion-like

## Relation to the wider ClearMesh roadmap

DualPrim is not the only structured path in the repo.

Current wider ClearMesh status:

- the main generation pipeline already has an optional part decomposition stage in `clearmesh/pipeline.py`
- PartCrafter integration exists in `clearmesh/partcrafter/decompose.py`
- the current PartCrafter adapter is a real integration scaffold, but it is still optional and heavyweight:
  - external install required
  - very high VRAM
  - not yet the default or validated production path for this branch

How these efforts fit together:

- DualPrim track:
  - best for compact, editable, artist-like structure
  - current focus of this plan
- PartCrafter track:
  - parallel structured generative path already available in the broader ClearMesh pipeline
  - useful for semantic part decomposition and future comparative evaluation
- refinement track:
  - should start from the best DualPrim scaffold (`r6`) rather than replace it

In other words:

- PartCrafter is part of the bigger ClearMesh plan
- but the immediate next technical step on this branch is still DualPrim-based local refinement

## Success criteria for the detail stage

The detail stage is worth keeping only if it does all of the following:

1. visibly improves local detail accuracy
2. preserves the DualPrim coarse structure
3. stays inside the compactness budget
4. does not create lots of tiny floating junk components

If it improves detail but explodes the mesh, it fails the product goal.

## Current implementation status

Implemented in this branch:

- `clearmesh/dualprim/detail_refine.py`
  - detail refinement config
  - mesh sampling / narrow-band analysis
  - compactness budget generation
  - manifest + NPZ artifact export
- `scripts/dualprim/run_canary.py`
  - optional post-export detail-refinement artifact generation

This is phase 0 scaffolding:

- enough to prepare detail-refinement inputs in a disciplined way
- without yet committing to one specific residual-field or extraction backend

## Next likely implementation steps

1. Finish the safer supervision/export experiments from `r6` (`r10`-style track).
2. Generate manifests from the strongest DualPrim checkpoints.
3. Inspect which primitives and regions dominate the narrow-band error.
4. Add a bounded per-primitive local deformation stage.
5. Compare:
   - `r6` baseline
   - `r10`-style safer supervision/export
   - DualPrim + primitive-local refinement
6. If needed, add Stage D edge-aware refinement:
   - crease-aware alignment loss
   - optional tiny primitive dictionary
7. Only after that, consider Stage E fallback paths:
   - hierarchical parent-child primitives
   - narrow-band residual around the combined field

The refinement stage only graduates if the visual gain is real and the mesh remains artist-like and compact.
