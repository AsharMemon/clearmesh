# Light-SQ Integration Notes for ClearMesh

_Paper: [Light-SQ: Structure-aware Shape Abstraction with Superquadrics for Generated Meshes](https://arxiv.org/abs/2509.24986) — SIGGRAPH Asia 2025 (Wang et al., corresp. Chen Change Loy). [Project page](https://johann.wang/Light-SQ/) / [GitHub](https://github.com/johannwyh/Light-SQ)._

## 1. Paper summary

**What it is.** Light-SQ is a **pure optimization** (no neural network, no training data) framework for decomposing a mesh into a union of superquadrics. The "Light" in the name is a nod to the fact that, compared to Marching-Primitives and EMS, it is both faster and lower-footprint despite operating on the same class of inputs.

- **Input representation.** Mesh is voxelized into a **100³ Truncated Signed Distance Field** normalized to [-1,1]³. TSDF is the fitting target (not raw points or SDF zero-set), because it exposes both surface and interior information and makes SDF-carving (below) cheap.
- **Output representation.** A flat list of **superquadrics**, each with **11 parameters**: ε₁, ε₂ ∈ [0,2] (shape exponents), (aₓ,aᵧ,aᵤ) ∈ ℝ⁺³ (scales), (e₁,e₂,e₃) ∈ ℝ³ (Euler angles), (tₓ,tᵧ,tᵤ) ∈ ℝ³ (translation). Average output is **61–62 primitives per shape**. The paper reports this as a flat union; hierarchy/tree is _not_ part of the core output — downstream editability is achieved by selecting/editing individual primitives.
- **Core insight (the single most important one).** Three "structure-aware" tricks, layered on top of MLE superquadric optimization: (i) **SDF carving** — after each primitive is fit, the target TSDF is updated so the covered region no longer "owes" anything to future primitives, which drastically reduces overlap (Overlap Rate 1.015 vs. Marching-Primitives 4.201); (ii) **block-regrow-fill** — primitives are placed inside structure-aware volumetric blocks rather than greedy global residual; (iii) **adaptive residual pruning** — the SDF-update history is used to identify primitives that have stopped contributing, preventing runaway over-segmentation.
- **Training data.** None. It is an optimizer; "weights" don't exist.
- **Why it handles noisy generated meshes.** Existing optimizers (EMS, Marching-Primitives) assume clean, watertight scans. On TRELLIS/Hunyuan output they over-segment and produce massive primitive overlap because the surface contains fuzz, disconnected shells, and interior junk. Light-SQ's SDF-carving is explicitly a _residual-based_ loop that tolerates noisy residuals, and the evaluation is done on **3DGen-Prim**, a new benchmark the authors built from 510 Hunyuan3D-2.0 / TripoSG generations — i.e., the benchmark _is_ the TRELLIS-family regime, not ShapeNet.
- **Compute.** ~26 s/shape on a high-end GPU (spec'd 96 GB / 14,592 CUDA cores, i.e. H100 class). **10× faster than Marching-Primitives (~340 s)**, comparable to PrimitiveAnything (~29 s) but without any learned prior. VRAM ceiling is not explicitly stated; TSDF at 100³ + ~60 active primitives fits easily on any modern GPU — the 96 GB machine is the authors' workstation, not a requirement.

## 2. Integration surface with ClearMesh

ClearMesh's current final artifact is a `trimesh.Trimesh` of ~2 M faces after UltraShape-1.0 + Taubin + cumesh repair + quadric decimate. Light-SQ expects a TSDF, not a mesh.

- **Preprocess.** `mesh → TSDF(100³, bbox=[-1,1]³)`. Implement with `mesh-to-sdf` or `trimesh.voxel.creation.voxelize` + a narrow-band truncation (Light-SQ truncates at ~0.05 of the unit cube edge based on the paper). Point sampling (`mesh.sample(16_384)`) is **not** the right interface — Light-SQ specifically uses TSDF because point clouds lose interior/solid cues. Sampling normals is also unused.
- **Output consumption.** After optimization, we receive a Python list of (ε₁, ε₂, scale₃, euler₃, translate₃) tuples. Each superquadric has a **closed-form surface parameterization** (generalized sin/cos with ε exponents); instancing is `~3–5 ms` per primitive using the standard u,v-grid tessellation at e.g. 64×64. For ~60 primitives, that's <1 s of mesh generation.
- **Compile back to `trimesh.Trimesh`.** Three options, from cheapest to cleanest:
  1. **Concatenate and call it done.** `trimesh.util.concatenate([instance_i for i in primitives])`. Produces a non-manifold soup of intersecting shells. Fine for visualization, useless for CSG.
  2. **Boolean union.** Run `trimesh.boolean.union(...)` (Blender or Manifold3D backend). Manifold3D is ~100× faster and much more robust than Blender for this. Risk: ~60 superquadric meshes with shared overlap regions is a stress test even for Manifold3D. Expect 5–30 s.
  3. **Remesh via global SDF.** Evaluate `max_i SDF_i(x)` on a 256³ grid and marching-cubes. Guaranteed watertight. Loses sharp edges at primitive seams.

## 3. Open weights / code availability

- **GitHub:** [johannwyh/Light-SQ](https://github.com/johannwyh/Light-SQ). Repository exists but the README is currently a stub (paper title, author list). As of writing, **no install instructions, no license file, no inference scripts** have been pushed. The paper was published at SIGGRAPH Asia 2025 (Dec 2025) and the repo was created Sep 2025, so code release is plausibly imminent but not guaranteed.
- **License:** Unspecified as of the current repo state. **Assume restrictive until a LICENSE file appears.** Corresponding author is Chen Change Loy (MMLab NTU) — MMLab tends to ship S-Lab / CC-BY-NC on research code, which would disallow a production deployment inside ClearMesh as a commercial product but is fine for research prototyping.
- **Weights:** Not applicable (optimizer, no weights). Only deliverables needed are the Python implementation and the 3DGen-Prim benchmark harness.
- **Install pain forecast.** Almost certainly PyTorch + a CUDA-accelerated TSDF sampler. MMLab projects typically pin a specific torch/mmcv combination; at minimum expect 30–60 minutes of env wrangling. No custom CUDA kernel is implied by the paper, but a TSDF library that does not play nicely with torch 2.x (e.g. `mesh_to_sdf` uses `pyrender`) could slot in as the main headache. **Blocker risk: code release slipping past Q2 2026.**

## 4. Failure modes to expect on TRELLIS.2 output

- **Organic part over-primitive-ization.** A dragon's wing membrane is _not_ naturally a union of ellipsoids; Light-SQ will place 10–20 flattish superquadrics across it, and joins between them will be visibly seamed. Tolerable for mechanical scenes, bad for character/figurine output — hence this must be an opt-in branch, not default.
- **Under-fit on high-detail regions.** Gear teeth, knurling, threads — all features Light-SQ has no way to represent. The closest superquadric to a 24-tooth gear is a cylinder, and the teeth will live entirely in the residual that `adaptive residual pruning` is _designed to discard_. Expected outcome: gear teeth vanish.
- **Non-manifold boolean output.** 60+ superquadrics with sub-voxel overlap at their seams is a hard case for any boolean kernel. Expect Manifold3D failures on ~10–20 % of inputs without preconditioning.
- **Thin-shell failure.** TRELLIS.2 occasionally produces interior shells (floating particles inside a solid). TSDF voxelization at 100³ may register these as disconnected level sets and spend primitives modelling noise.
- **Axis-aligned bias.** Superquadric Euler angles in local frame can drift — the block-regrow-fill step assumes some dominant directions; for asymmetric organic shapes the "blocks" are less meaningful.

## 5. Proposed implementation for ClearMesh

**Sketch API** (target placement `clearmesh/refit/light_sq.py`):

```python
from clearmesh.refit.light_sq import LightSQRefiner, SuperQuadric
refiner   = LightSQRefiner(device="cuda", tsdf_res=100)
prims     = refiner.fit(mesh)                         # list[SuperQuadric]
refit_mesh = refiner.compile(prims, mode="manifold")  # trimesh.Trimesh
```

- **Module location.** `clearmesh/refit/light_sq.py` is appropriate. A sibling `clearmesh/refit/cad_recode.py` can live next to it. A shared `clearmesh/refit/base.py` can define `PrimitiveRefiner` and the `@dataclass SuperQuadric` record.
- **Pipeline insertion point.** Run **after R2 polish but before quadric decimate**. Reasoning: (a) Light-SQ wants a clean, closed surface — polish helps TSDF voxelization; (b) decimate is wasted compute if we're about to replace the geometry; (c) the Taubin smoothing artifacts are below Light-SQ's 100³ voxel resolution, so polish does not hurt the fit.
- **Parallel vs. sequential.** **Parallel branch, user-selected.** A CLI flag / prompt tag (`--mechanical` or detected from prompt keywords like "gear", "engine", "bracket") routes to the Light-SQ branch. For organic subjects, the default neural-mesh path wins. This matches the `primitive_refit_2026.md` recommendation to offer CAD-Recode as a parallel stage rather than a replacement. An automated router can be added later using a CLIP-style classifier on the input image.
- **Quality gate.** Always compute Chamfer distance between `mesh` and `refit_mesh` and fall back to the neural mesh if CD exceeds a threshold (e.g. 0.02 on the unit cube). This gives a free "is this actually mechanical?" signal.

## 6. Comparison vs. CAD-Recode

| Axis | Light-SQ | CAD-Recode |
|---|---|---|
| Output | 60 superquadrics (flat union) | CadQuery Python program |
| Sketch-extrude / fillets | No | Yes |
| Robustness to TRELLIS noise | **High** (benchmarked on 3DGen-Bench) | Medium (trained on DeepCAD) |
| Editability | Per-primitive handles | Full parametric code |
| Suited to "prototype-to-print" | Poor (no clean flat faces, no threads) | **Excellent** (real B-rep via OCCT) |
| Runtime | ~26 s, no weights | ~2–5 s, 1.5 B-param weights |
| Open code today | Pending | Yes |

For a "prototype-to-print" workflow, **CAD-Recode is more useful**: B-reps import cleanly into FreeCAD/Fusion/SOLIDWORKS, and the text program is editable. Light-SQ is more useful for **editing / art direction** (drag a primitive to deform a part) and as a **robust preprocessor**: its decomposition is more reliable on noisy TRELLIS meshes than CAD-Recode's direct point-to-program inference. The two compose naturally: **Light-SQ for gross primitive decomposition → LLM rewrites each primitive region as a CadQuery sketch-extrude using the primitive as a positional hint**. This would be a superior pipeline to either in isolation, and directly extends the "CADReasoner-style loop" called out in `primitive_refit_2026.md` §4.

## 7. Smallest viable prototype

Target mesh: `/workspace/demo_R2_easy3e/02_after_R2.glb` (the steampunk gearbox post-polish).

1. Wait for (or request) code release; if blocked, **port the paper's algorithm directly** — it's pure optimization with no training, and the SDF-carving + block-regrow-fill pseudocode should fit in ~400 LOC against a `mesh-to-sdf` + `torch.optim.Adam` core.
2. Load `02_after_R2.glb`, voxelize to TSDF-100³.
3. Run the optimizer for the paper's default schedule.
4. Export primitive list, instance-mesh each, concatenate (skip Boolean for v0).
5. Render side-by-side against original; eyeball structural preservation and gear-tooth loss.

Expected afternoon result: confirms (a) whether the SDF-carving stabilizes on our noisy meshes, (b) whether the 60-primitive count is too many/few for our typical prompts, (c) whether thin-shell failures are catastrophic. All three outcomes are actionable regardless.

## Go / no-go

**Go, but with a 2-week contingency.** The paper addresses exactly our distribution (generated meshes, not scans), requires no weights, has a published ~26 s runtime, and the algorithm is simple enough to reimplement from the paper if the authors' code is delayed. Blocking risk is license / code-availability only. Sequence Light-SQ **after** a CAD-Recode experiment (per `primitive_refit_2026.md` §4), since CAD-Recode is runnable today with zero blockers.
