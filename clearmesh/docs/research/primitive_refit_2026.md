# Primitive / Parametric Refit of Neural Meshes: 2026 SOTA Survey

_Scope: approaches that produce CSG trees, B-reps, sketch-extrude programs, or primitive assemblies that could post-process the organic output of TRELLIS.2 / UltraShape / Hunyuan3D into clean mechanical geometry._

## 1. Baseline (2021–2024), one-paragraph recap

The pre-2025 landscape clustered into three camps. **(a) Primitive fitting from point clouds**: SPFN, ParSeNet, CPFN, HPNet, and Point2CAD (2023) segment a cloud into plane/cylinder/sphere/cone/B-spline patches and fit each. Strong on isolated primitives, weak on topology and on output that would round-trip through a solid kernel. **(b) Program synthesis**: CSGNet, UCSG-Net, CSG-Stump, InverseCSG, SketchGen, DeepCAD (2021), SkexGen, Fusion360Gallery, and Point2Primitive emit an executable construction program (CSG tree or sketch-extrude sequence). Brittle, small vocabularies, limited to ~10 operations. **(c) Primitive diffusion / shape abstraction**: SPAGHETTI (Gaussian parts), Marching-Primitives (from SDF), and GALA abstract a shape into a coarse set of blobs — great for animation / editing handles, not for manufacturing. **BrepGen** (SIGGRAPH 2024, Autodesk) was the first diffusion model to sample an entire B-rep hierarchy (faces → edges → vertices) end-to-end, and **ComplexGen** (2022) pioneered joint edge/face/topology prediction. **MeshGPT** (2024) is adjacent but emits triangle soup, not parametric CAD.

## 2. 2026 SOTA — what's new and relevant

The last 18 months produced a genuine phase change: the field has moved from "fit one primitive" to "sample a full valid B-rep" and from "emit a fixed command grammar" to "emit Python / CadQuery code that runs under the OCCT kernel." Four sub-fields are worth tracking.

### 2.1 B-rep generative models (direct CAD sampling)

The post-BrepGen line has converged on two ideas: decouple topology from geometry, and push everything into one latent space so diffusion/AR sampling produces _valid_ solids out of the box.

- **DTGBrepGen** (CVPR 2025) — [arXiv](https://arxiv.org/html/2503.13110v1) / [site](https://jinli99.github.io/DTGBrepGen/) — splits the problem: a graph model first samples face/edge/vertex connectivity, then geometry is filled in. Consistently beats BrepGen on validity.
- **HoLa** (SIGGRAPH 2025) — [arXiv 2504.14257](https://arxiv.org/abs/2504.14257) / [site](https://vcc.tech/research/2025/HolaBrep) — single holistic latent for surfaces+curves; reduces curve/vertex redundancy. Reports **84% unconditional validity** vs. ~50% for BrepGen. Supports text, point cloud, multi-view image, and 2D sketch conditioning.
- **BrepDiff** (SIGGRAPH 2025) — single-stage diffusion, simpler pipeline, competitive with BrepGen on distributional metrics.
- **BrepGPT** (SIGGRAPH Asia 2025 / ToG) — [arXiv 2511.22171](https://arxiv.org/abs/2511.22171) / [GitHub](https://github.com/BunnySoCrazy/BrepGPT) — autoregressive B-rep via a **Voronoi Half-Patch** tokenization + decoder-only transformer. Supports conditioning on point clouds, images, text, and categories, plus autocompletion/interpolation — the most "GPT-like" B-rep model to date.
- **AutoBrep** (SIGGRAPH Asia 2025) — unified topology+geometry autoregression; sibling of BrepGPT.
- **Stitch-A-Shape** (SIGGRAPH 2025) — bottom-up B-rep assembly; builds topology incrementally rather than denoising a frozen grid.
- **BrepGiff** (CVPR 2025) — lightweight 3D-GAT diffusion over B-rep graphs.
- **GraphBRep** (JCDE Dec 2025) — explicit graph diffusion of B-rep topology.

### 2.2 Mesh / point cloud → parametric CAD (what we actually need)

- **CAD-Recode** (ICCV 2025) — [arXiv 2412.14042](https://arxiv.org/abs/2412.14042) / [GitHub](https://github.com/filaPro/cad-recode) / [site](https://cad-recode.github.io/). Point cloud → Qwen2-1.5B (plus a single linear projector) → executable **CadQuery Python**. Trained on 1 M procedural sequences. Reports **10× lower Chamfer distance** than prior SOTA on DeepCAD and Fusion360. Open-source, ~1.5 B params, runs on a single GPU. _This is the most tractable off-the-shelf piece for us._
- **CADReasoner** (2026 arXiv 2603.29847) — closed-loop editor: takes a point cloud + multi-view renders, outputs CadQuery, **re-renders the result, diffs it, and iteratively edits the program.** SOTA on DeepCAD / Fusion360 / MCB "scan-sim" track. This is the first method that actually mimics the way a human reverse-engineers.
- **CADDreamer** (CVPR 2025) — [arXiv 2502.20732](https://arxiv.org/html/2502.20732) / [site](https://lidan233.github.io/caddreamer/). Single image → primitive-aware multi-view diffusion → mesh with per-face primitive labels → Graph Cut patching → primitive-intersection reconstruction to a watertight B-rep. Explicitly handles the "organic mesh → clean primitives" direction, which is our exact use case.
- **Mesh2Brep** (2024) — robust primitive fitting + intersection-aware constraints; less capable than 2025 entries but a useful baseline.
- **CADCrafter** (CVPR 2025) — [arXiv 2504.04753](https://arxiv.org/abs/2504.04753). Unconstrained photo → CAD sequence; 88.7% validity. Transformer decoder with CLIP vision encoder.
- **Point2Primitive** (2025) — [arXiv 2505.02043](https://arxiv.org/html/2505.02043). Transformer decoder with explicit position queries, treats sketch-curve reconstruction as set prediction. Strong sketch-extrude baseline.
- **PICASSO** (WACV 2025) — feed-forward differentiable rendering for sketch inference.
- **Parametric Point Cloud Completion** (CVPR 2025) — reconstruct polygonal-surface CAD from partial clouds.

### 2.3 LLM / VLM agents emitting CAD code

This branch exploded in 2025 and is the most underrated lever for our project, because the LLM carries real geometric common sense about mechanical parts (threads, chamfers, bearings) that no supervised CAD model has seen.

- **CAD-Assistant** (ICCV 2025) — [arXiv 2412.13810](https://arxiv.org/abs/2412.13810) / [site](https://cadassistant.github.io/) / [GitHub](https://github.com/dimitrismallis/CAD-Assistant). Tool-augmented VLLM with a FreeCAD Python interpreter in the loop. Handles sketch→3D, scan-based reverse engineering, and visual design. The existence proof that **FreeCAD-as-a-tool for an LLM** works.
- **CAD-MLLM** (2024→2025) — [site](https://cad-mllm.github.io/) / [GitHub](https://github.com/CAD-MLLM/CAD-MLLM). Multimodal (text/image/point-cloud/mixed) → DeepCAD command sequence; Omni-CAD dataset of 450 K instances.
- **CADmium** (TMLR 2026) — [arXiv 2507.09792](https://arxiv.org/pdf/2507.09792). Fine-tuned Qwen-2.5 Coder that emits CAD history in minimal JSON. Purely text-to-text, trained on 176 K DeepCAD sequences annotated by GPT-4.1.
- **CAD-Coder / Text-to-CadQuery** (2025) — [arXiv 2505.06507](https://arxiv.org/pdf/2505.06507). VLM → CadQuery; feedback loop raises execution rate from 53% → 85%.
- **CAD-GPT** (AAAI 2025) — sketch construction sequences with spatial reasoning VLM.
- **EvoCAD** (2025 arXiv 2510.11631) — evolutionary optimization over LLM-generated CAD code.
- **CADSmith** (2026) — multi-agent CadQuery pipeline with nested execution + geometric-validation loops.
- **ShapeCraft** (NeurIPS 2025) — [arXiv 2510.17603](https://arxiv.org/abs/2510.17603). LLM multi-agent system (Parser / Coder / Evaluator) using a Graph-based Procedural Shape representation. Structured + textured + interactive assets.
- **FlexCAD** (ICLR 2025), **ReCAD** (2025, RL-enhanced), **GeoCAD** (NeurIPS 2025, local-controllable), **CQAsk** (open-source Copilot-for-CadQuery).

### 2.4 Primitive assembly / superquadric abstraction

Complementary to B-rep: decompose an organic mesh into a small number of parametric primitives you can then refit or replace.

- **PrimitiveAnything** (SIGGRAPH 2025) — [arXiv 2505.04622](https://arxiv.org/abs/2505.04622) / [GitHub](https://github.com/PrimitiveAnything/PrimitiveAnything). Shape-conditioned autoregressive transformer that emits a sequence of primitives (class, translation, rotation, scale). Open weights on HF. Designed for UGC/game assets, but the architecture is generic.
- **Light-SQ** (SIGGRAPH Asia 2025) — [arXiv 2509.24986](https://arxiv.org/abs/2509.24986). Superquadric abstraction **specifically tuned for generated meshes** (literally the TRELLIS/Hunyuan regime). SDF-carving + block-regrow-fill. This is the closest published work to our problem.
- **SuperDec** (ICCV 2025) — [site](https://super-dec.github.io/) / [GitHub](https://github.com/elisabettafedele/superdec). Instance-segmentation-driven superquadric decomposition of scenes.
- **DPA-Net** (2024, Amazon) — differentiable primitive assembly from sparse views.
- **HiT** (3DV 2026) — hierarchical transformers for unsupervised primitive abstraction.
- **ShapeLib** (2025) — LLMs author a library of procedural shape abstractions.

### 2.5 Industry signals

- **Autodesk Research** (AI Lab) shipped BrepGen (2024), HoLa (2025), and announced at **AU 2025** a neural foundation model called **"neural CAD"** — public details thin, but this is where the resources are. [Autodesk Research publications](https://www.research.autodesk.com/publications/brepgen/).
- **Adobe Research**: active on implicit fields, less on parametric CAD.
- **Shapr3D / Onshape / nTopology**: no public research releases yet; integrating LLM copilots at the UI layer.
- **SOLIDWORKS 2025** introduced AURA (generative assembly suggestions, fastener recognition) — product-level, not research-grade.

## 3. Practical assessment

| Method | Runnable open code? | Compute / inference | Failure mode |
|---|---|---|---|
| CAD-Recode | Yes (HF + GitHub) | 1 × GPU, seconds, Qwen2-1.5B | Limited to sketch+extrude; chokes on complex mechanical assemblies |
| CADReasoner | Paper only (Feb 2026) | Few iterations × VLM call | Latency; may not converge |
| CADDreamer | Partial | Heavier (multi-view diffusion) | Only 4–5 primitive classes (plane/cylinder/sphere/cone/BSpline); topology extraction fragile |
| BrepGen / HoLa / BrepGPT | Yes | ~diffusion/AR step budget | Trained on DeepCAD-ish distribution — fails on assemblies, threads, gears |
| PrimitiveAnything | Yes | Fast AR transformer | Cuboid/cylinder/sphere primitives; no topology guarantees |
| Light-SQ | Code pending | Optimization (minutes) | Superquadrics only — no planar faces / sharp edges |
| CAD-Assistant | Yes | VLM + FreeCAD per call | Expensive tokens; FreeCAD bugs; slow |
| CadQuery LLM agents | Trivial to prototype | 1 frontier API call per iter | Executable-but-wrong programs are common |

Typical failure modes across all classes: (i) validity collapse on non-watertight solids, (ii) "topology explosion" where face count balloons, (iii) missing features (fillets, chamfers, threads entirely absent), and (iv) distribution shift — every published mesh-to-CAD model is trained on DeepCAD/Fusion360, which is overwhelmingly simple prismatic parts. A steampunk engine with gears/fasteners is **out of distribution** for every single paper above.

## 4. Recommendation for ClearMesh

**Pick one to integrate first: CAD-Recode, as a _parallel_ stage to the neural pipeline, not a replacement.** It has the cleanest interface (point cloud in, CadQuery Python out), open weights, a sensible 1.5 B-param footprint, and the output is _human-readable_ — which means when it fails, we can still use the CadQuery code as a seed for an LLM repair loop. Sample TRELLIS.2's 512³-voxel SLAT, decode to a mesh, surface-sample 8 K points, feed to CAD-Recode, and keep the _best-of-N_ (measured by Chamfer distance against the neural mesh). For mechanical subjects where CAD-Recode wins, ship the CAD version; for organic subjects, keep the neural mesh. This gives a free "CAD-ness detector" as a byproduct.

**The more ambitious, higher-upside move is a hybrid "CADReasoner-style" loop** built on our own stack: TRELLIS.2 → mesh → **Light-SQ or PrimitiveAnything for a primitive skeleton** → Claude/GPT writes CadQuery that places primitives with matching dimensions → render → diff against the TRELLIS.2 mesh via rasterized IoU or Chamfer → feed the diff back in as a second prompt. Everything except the LLM call is already in our codebase. The insight is that frontier LLMs know _what a fastener looks like_ in a way that no DeepCAD-trained transformer ever will; CadQuery gives them a clean solid-kernel API; and the diff-loop makes their output grounded.

**Smallest viable afternoon prototype**: (1) clone `filaPro/cad-recode`, (2) run inference on one TRELLIS.2 output that's mechanical (e.g., a piston, a gear, a bracket) and on one that's organic (e.g., a chair, a character), (3) eyeball the CadQuery output + re-render, (4) compute Chamfer against the original neural mesh. This tells us in 2–3 hours whether the distribution gap between TRELLIS.2 output and CAD-Recode's training data is catastrophic or survivable. If survivable, stage two is the LLM repair loop (another afternoon) using Claude + CadQuery + a render-diff tool.

## 5. Top 5 things to look at first

1. **CAD-Recode** — [GitHub](https://github.com/filaPro/cad-recode) — runnable today; closest match to our needs.
2. **CADReasoner** (arXiv 2603.29847) — read the iterative-edit algorithm; this is the pattern we should copy.
3. **Light-SQ** ([arXiv 2509.24986](https://arxiv.org/abs/2509.24986)) — tuned for _generated_ meshes specifically; directly addresses our "TRELLIS output is noisy" problem.
4. **PrimitiveAnything** — [GitHub](https://github.com/PrimitiveAnything/PrimitiveAnything) — open weights, fast, composable.
5. **Awesome Neural CAD** — [bunnysocrazy.com](https://bunnysocrazy.com/) — the living index; check monthly.

Honorable mention: **CAD-Assistant** ([GitHub](https://github.com/dimitrismallis/CAD-Assistant)) as a reference implementation of "VLM + FreeCAD-in-a-loop."
