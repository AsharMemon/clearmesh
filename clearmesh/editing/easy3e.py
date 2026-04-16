#!/usr/bin/env python3
"""Easy3E Orchestrator — Main entry point for 3D editing.

Orchestrates the full Easy3E editing pipeline:
  1. Encode source mesh to SLAT (via SLATEncoder)
  2. Edit voxel structure (via VoxelFlowEdit, training-free)
  3. Repaint per-voxel features (via SLATRepainter, training-free)
  4. Decode edited SLAT back to mesh (via SLATEncoder.decode)
  5. Optionally: generate textures (via CtrlAdapter, trained)
  6. Repair and export

Supports three editing modes:
  - Image-guided: source mesh + edited image → edited mesh
  - Text-guided: source mesh + text instruction → edited mesh
  - Iterative: chain multiple edits (edit1 → edit2 → edit3)

Usage:
    from clearmesh.editing import Easy3EEditor

    editor = Easy3EEditor(trellis2_dir="/workspace/TRELLIS.2")

    # Image-guided editing
    result = editor.edit(
        source_mesh="model.glb",
        edit_image="edited_front_view.png",
    )

    # Text-guided editing
    result = editor.edit_from_text(
        source_mesh="model.glb",
        instruction="add wings to the dragon",
    )

    # Iterative editing
    result = editor.edit_iterative(
        source_mesh="model.glb",
        edits=[
            {"image": "add_wings.png"},
            {"text": "make it metallic"},
            {"image": "add_horns.png"},
        ],
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import trimesh
from PIL import Image

from clearmesh.editing.slat_encoder import SLATEncoder, SLATRepresentation
from clearmesh.editing.slat_repaint import RepaintConfig, SLATRepainter
from clearmesh.editing.voxel_flowedit import FlowEditConfig, VoxelFlowEdit


@dataclass
class EditOptions:
    """Options for 3D editing."""

    # VoxelFlowEdit params
    num_flow_steps: int = 25  # ODE integration steps
    gamma: float = 1.0  # Trajectory correction strength
    eta: float = 0.5  # Silhouette guidance strength
    guidance_scale: float = 7.5  # CFG scale

    # SLAT Repainting params
    num_repaint_steps: int = 25
    blend_boundary: int = 2

    # Ctrl-Adapter (texture) params
    enable_texture: bool = False  # Requires trained Ctrl-Adapter
    texture_guidance_scale: float = 7.5

    # UltraShape refinement (optional, non-commercial license)
    enable_ultrashape: bool = False
    ultrashape_dir: str = "/workspace/UltraShape-1.0"
    ultrashape_ckpt: str | None = None  # defaults to <ultrashape_dir>/checkpoints/ultrashape_v1.pt
    ultrashape_config: str | None = None
    ultrashape_steps: int = 50
    ultrashape_octree_res: int = 1024

    # TripoSF (SparseFlex) watertight pass (optional, MIT license).
    # Runs AFTER UltraShape. NOTE: empirically REDUCED mesh quality in
    # our testing (0 -> 12k boundary edges when applied after UltraShape).
    # Keeping the integration available for edge cases but OFF by default.
    # See scripts/demo_triposf_A_noEdit.py for evidence.
    enable_triposf: bool = False
    triposf_dir: str = "/workspace/TripoSF"
    triposf_config: str | None = None  # defaults to <triposf_dir>/configs/TripoSFVAE_1024.yaml

    # --- Cheap polish (runs after UltraShape, before final repair) ---

    # Collapse near-duplicate vertices so T-junction "holes" that viewers
    # render as visible cracks get merged. Rounds vertex coords to N
    # significant digits before dedup. 5 is a good default — enough to
    # catch floating-point near-dups from the MC, not so loose that
    # it smashes adjacent-but-distinct geometry.
    enable_vertex_merge: bool = True
    vertex_merge_digits: int = 5

    # Taubin filter: Laplacian smoothing pair (one positive, one negative)
    # that preserves volume and sharp features better than plain Laplacian.
    # Set iterations=0 to disable. ~1-2s on an H100 for a 6M-vert mesh.
    enable_taubin_smooth: bool = True
    taubin_iterations: int = 3         # 2-5 is a good range
    taubin_lamb: float = 0.5           # smoothing strength
    taubin_nu: float = -0.53           # -lamb adjusted for inverse pass; must be < -lamb

    # Region-focused editing
    # 2D mask image (path or PIL.Image); white/255=edit, black/0=preserve.
    # If None, edit is applied globally.
    region_mask: object = None  # str | Path | PIL.Image | None
    mask_dilation: int = 1  # 3D morphological dilation passes
    # Gaussian blur radius (in pixels) applied to the 2D mask before projection.
    # Larger values produce a smoother transition between edit and preserve
    # regions and avoid the ragged-edge artifacts from a hard 0/1 cutoff.
    # Set to 0 to disable. Default 12 gives a ~25px transition zone at 512px.
    mask_blur_radius: float = 12.0
    # Only drop unmatched edit-only voxels when the soft mask at their
    # projected pixel is below this threshold. Lower = keeps more edit
    # geometry near mask boundaries; prevents "holes punched through surface"
    # artifacts where the mask was narrow.
    mask_drop_threshold: float = 0.1

    # Text editing params (InstructPix2Pix)
    text_image_guidance: float = 1.5
    text_guidance_scale: float = 7.5
    text_num_steps: int = 20

    # Mesh processing
    grid_size: int = 256  # O-Voxel resolution
    enable_repair: bool = True  # Post-edit mesh repair
    # PyMeshFix is O(n²)-ish and becomes unusably slow on large meshes.
    # If the decoded mesh has more verts than this, repair is auto-skipped
    # with a warning (and marked "skipped" in timings). 0 disables the cap.
    skip_repair_above_verts: int = 500_000

    # Export
    export_format: str = "glb"


@dataclass
class EditResult:
    """Result from 3D editing."""

    mesh: trimesh.Trimesh
    output_path: str | None = None
    slat: SLATRepresentation | None = None
    edit_mask: torch.Tensor | None = None
    timings: dict = field(default_factory=dict)


class Easy3EEditor:
    """Main 3D editing orchestrator using Easy3E architecture.

    Combines all editing components:
      - SLATEncoder: mesh ↔ SLAT conversion
      - VoxelFlowEdit: training-free structure editing
      - SLATRepainter: training-free feature repainting
      - CtrlAdapter: normal-guided texture (optional, trained)
      - ImageEditor: InstructPix2Pix for text→image (for text-guided editing)
    """

    def __init__(
        self,
        trellis2_dir: str = "/workspace/TRELLIS.2",
        model_dir: str = "/workspace/models/trellis2-4b",
        ctrl_adapter_checkpoint: str | None = None,
        device: str | None = None,
        pipeline=None,
    ):
        """
        Args:
            trellis2_dir: Root of the TRELLIS.2 git checkout.
            model_dir: Path to TRELLIS.2-4B weights (or HF cache).
            ctrl_adapter_checkpoint: Optional trained Ctrl-Adapter for
                texture re-generation. Leave None to skip.
            device: Compute device.
            pipeline: Optional pre-loaded ``Trellis2ImageTo3DPipeline``. If
                provided, shared across all editing sub-components so we
                don't load the 4B parameters three times. If None, a new
                pipeline is lazily loaded on first use of the editor.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trellis2_dir = trellis2_dir
        self.model_dir = model_dir
        self._pipeline = pipeline

        # Sub-components (lazy — sharing the pipeline instance)
        self._slat_encoder: SLATEncoder | None = None
        self._voxel_flowedit: VoxelFlowEdit | None = None
        self._slat_repainter: SLATRepainter | None = None
        self._ctrl_adapter = None
        self._ctrl_adapter_checkpoint = ctrl_adapter_checkpoint
        self._image_editor = None
        self._ultrashape_refiner = None
        self._triposf_refiner = None

    @property
    def ultrashape_refiner(self):
        """Lazy-load UltraShapeRefiner on first access."""
        if self._ultrashape_refiner is None:
            from clearmesh.editing.ultrashape_refine import UltraShapeRefiner
            self._ultrashape_refiner = UltraShapeRefiner(device=self.device)
        return self._ultrashape_refiner

    @property
    def triposf_refiner(self):
        """Lazy-load TripoSF watertight refiner on first access.

        TripoSF runs in a subprocess (see clearmesh/editing/triposf_refine.py
        and scripts/run_triposf_subprocess.py), so instance state is just
        the repo + config paths.
        """
        if self._triposf_refiner is None:
            from clearmesh.editing.triposf_refine import TripoSFRefiner
            self._triposf_refiner = TripoSFRefiner()
        return self._triposf_refiner

    @property
    def pipeline(self):
        """Return the shared TRELLIS.2 pipeline; lazy-load on first access.

        Loading the pipeline triggers DINOv3 download and model weights
        (~20GB + gated repo access). Only pay that cost when first needed.
        """
        if self._pipeline is None:
            import sys
            from pathlib import Path
            if Path(self.trellis2_dir).exists() and str(self.trellis2_dir) not in sys.path:
                sys.path.insert(0, str(self.trellis2_dir))
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            from pathlib import Path as _P
            if _P(self.model_dir).exists():
                self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.model_dir)
            else:
                self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
            self._pipeline.to(self.device)
        return self._pipeline

    @property
    def slat_encoder(self) -> SLATEncoder:
        if self._slat_encoder is None:
            self._slat_encoder = SLATEncoder(
                trellis2_dir=self.trellis2_dir,
                model_dir=self.model_dir,
                device=self.device,
                pipeline=self.pipeline,
            )
        return self._slat_encoder

    @property
    def voxel_flowedit(self) -> VoxelFlowEdit:
        """Lazy-load VoxelFlowEdit sharing the parent pipeline."""
        if self._voxel_flowedit is None:
            self._voxel_flowedit = VoxelFlowEdit(
                device=self.device,
                pipeline=self.pipeline,
            )
        return self._voxel_flowedit

    @property
    def slat_repainter(self) -> SLATRepainter:
        """Lazy-load SLATRepainter sharing the parent pipeline."""
        if self._slat_repainter is None:
            self._slat_repainter = SLATRepainter(
                device=self.device,
                pipeline=self.pipeline,
            )
        return self._slat_repainter

    @property
    def ctrl_adapter(self):
        """Lazy-load trained Ctrl-Adapter."""
        if self._ctrl_adapter is None and self._ctrl_adapter_checkpoint:
            from clearmesh.editing.ctrl_adapter import CtrlAdapter

            ckpt = torch.load(
                self._ctrl_adapter_checkpoint,
                map_location=self.device,
                weights_only=False,
            )
            config = ckpt.get("config", {})
            self._ctrl_adapter = CtrlAdapter(**config).to(self.device)
            self._ctrl_adapter.load_state_dict(ckpt["model"])
            self._ctrl_adapter.eval()
            print("Ctrl-Adapter loaded.")
        return self._ctrl_adapter

    @property
    def image_editor(self):
        """Lazy-load InstructPix2Pix."""
        if self._image_editor is None:
            from clearmesh.editing.image_edit import ImageEditor

            self._image_editor = ImageEditor(device=self.device)
        return self._image_editor

    def edit(
        self,
        source_mesh: str | Path | trimesh.Trimesh,
        edit_image: str | Path | Image.Image,
        source_image: str | Path | Image.Image | None = None,
        edit_mask: torch.Tensor | None = None,
        output_path: str | None = None,
        options: EditOptions | dict | None = None,
    ) -> EditResult:
        """Image-guided 3D editing.

        Source mesh + edit image → edited mesh.

        Args:
            source_mesh: Source mesh (path or trimesh object).
            edit_image: Target/edited image to guide editing.
            source_image: Original source rendering (auto-rendered if None).
            edit_mask: Optional voxel-level edit mask.
            output_path: Output file path.
            options: Editing options.

        Returns:
            EditResult with edited mesh.
        """
        if isinstance(options, dict):
            options = EditOptions(**options)
        elif options is None:
            options = EditOptions()

        timings = {}

        # Load images
        if isinstance(edit_image, (str, Path)):
            edit_image = Image.open(str(edit_image)).convert("RGB")
        if isinstance(source_image, (str, Path)):
            source_image = Image.open(str(source_image)).convert("RGB")

        # === Step 1: Encode source mesh to SLAT ===
        t0 = time.time()
        mesh_path = source_mesh if isinstance(source_mesh, (str, Path)) else None
        if mesh_path is None:
            # Save trimesh to temp file
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                source_mesh.export(f.name)
                mesh_path = f.name

        slat = self.slat_encoder.encode(mesh_path, grid_size=options.grid_size)
        timings["encode"] = time.time() - t0
        print(f"  SLAT encoded: {slat.ss_latent.shape}")

        # === Step 2: Auto-render source image if not provided ===
        if source_image is None:
            source_image = self._render_source(mesh_path)

        # === Step 3: Auto-detect edit mask if not provided ===
        if edit_mask is None:
            edit_mask = self.voxel_flowedit.auto_detect_edit_mask(
                source_image, edit_image, slat.voxel_indices
            )

        # === Step 4: Edit voxel structure (training-free) ===
        t0 = time.time()
        flow_config = FlowEditConfig(
            num_steps=options.num_flow_steps,
            gamma=options.gamma,
            eta=options.eta,
            guidance_scale=options.guidance_scale,
        )
        edited_ss = self.voxel_flowedit.edit(
            source_ss_latent=slat.ss_latent,
            target_image=edit_image,
            source_image=source_image,
            edit_mask=edit_mask,
            config=flow_config,
        )
        timings["voxel_flowedit"] = time.time() - t0
        print(f"  Structure edited: {edited_ss.shape}")

        # === Step 5: Repaint per-voxel features (training-free) ===
        t0 = time.time()
        repaint_config = RepaintConfig(
            num_steps=options.num_repaint_steps,
            blend_boundary=options.blend_boundary,
        )
        edited_features = self.slat_repainter.repaint(
            edited_ss_latent=edited_ss,
            source_features=slat.shape_latent,
            edit_mask=edit_mask,
            target_image=edit_image,
            source_image=source_image,
            voxel_indices=slat.voxel_indices,
            config=repaint_config,
        )
        timings["slat_repaint"] = time.time() - t0

        # === Step 6: Decode SLAT back to mesh ===
        t0 = time.time()
        edited_slat = SLATRepresentation(
            ss_latent=edited_ss,
            shape_latent=edited_features,
            voxel_indices=slat.voxel_indices,
            dual_vertices=slat.dual_vertices,
            intersected=slat.intersected,
            grid_size=slat.grid_size,
        )
        edited_mesh = self.slat_encoder.decode(edited_slat)
        timings["decode"] = time.time() - t0

        # === Step 7: Optional texture via Ctrl-Adapter ===
        if options.enable_texture and self.ctrl_adapter is not None:
            t0 = time.time()
            edited_mesh = self._apply_texture(edited_mesh, edit_image, options)
            timings["texture"] = time.time() - t0

        # === Step 8: Repair ===
        # Edited meshes from the flow ODE can have holes or non-manifold
        # regions that PyMeshFix rejects. We try to repair; if it fails
        # (e.g. watertight check fails outright), we keep the unrepaired
        # mesh with a warning rather than failing the whole edit.
        if options.enable_repair:
            t0 = time.time()
            try:
                from clearmesh.mesh.repair import full_print_preparation
                edited_mesh = full_print_preparation(edited_mesh, orient=False, verbose=False)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"[easy3e] Mesh repair failed ({e}); returning unrepaired mesh. "
                    "You may want to run repair/fix manually."
                )
            timings["repair"] = time.time() - t0

        # === Step 9: Export ===
        if output_path:
            from clearmesh.mesh.export import export_mesh

            export_mesh(edited_mesh, output_path, format=options.export_format)

        timings["total"] = sum(timings.values())
        print(f"  Edit complete in {timings['total']:.1f}s")

        return EditResult(
            mesh=edited_mesh,
            output_path=output_path,
            slat=edited_slat,
            edit_mask=edit_mask,
            timings=timings,
        )

    def edit_from_source_image(
        self,
        source_image,
        edit_image=None,
        instruction: str | None = None,
        output_path: str | None = None,
        options: "EditOptions | dict | None" = None,
    ) -> "EditResult":
        """Image-based editing matching the Easy3E paper's actual workflow.

        Unlike :meth:`edit` which tries to encode an arbitrary mesh into
        SLAT — something TRELLIS.2-4B does not support directly — this
        method takes a **source image** as input, generates the source
        SLAT internally using TRELLIS.2's own sampler, then applies the
        edit. This is what the Easy3E paper actually does: edit is over
        TRELLIS.2's native latents, not over externally-encoded meshes.

        Exactly one of ``edit_image`` or ``instruction`` must be provided:
          - ``edit_image``: a PIL.Image or path to an already-edited view.
          - ``instruction``: a text instruction (e.g. "add wings") — we'll
            run InstructPix2Pix on the source image to create the edit image.

        Args:
            source_image: Path or PIL.Image of the source view.
            edit_image: Pre-computed edit image (bypasses IP2P).
            instruction: Text instruction (used with IP2P if edit_image None).
            output_path: Where to write the edited mesh GLB.
            options: EditOptions.

        Returns:
            EditResult with the edited mesh.
        """
        import time
        import torch
        from pathlib import Path
        from PIL import Image

        if isinstance(options, dict):
            options = EditOptions(**options)
        elif options is None:
            options = EditOptions()

        timings = {}

        # --- Load source image ---
        if isinstance(source_image, (str, Path)):
            source_image = Image.open(str(source_image)).convert("RGB")

        # --- Resolve edit image ---
        if edit_image is None and instruction is None:
            raise ValueError("Must provide either edit_image or instruction")
        if edit_image is None:
            t0 = time.time()
            edit_image = self.image_editor.edit(
                source_image=source_image,
                instruction=instruction,
                image_guidance_scale=options.text_image_guidance,
                guidance_scale=options.text_guidance_scale,
                num_inference_steps=options.text_num_steps,
            )
            timings["instruct_pix2pix"] = time.time() - t0
        elif isinstance(edit_image, (str, Path)):
            edit_image = Image.open(str(edit_image)).convert("RGB")

        # --- Generate source structure (coords only) via TRELLIS.2 ---
        # We use the cascade path (512→1024) internally via sample_shape_slat_cascade
        # to match what pipeline.run() does by default. Without this cascade, the
        # output mesh has visible striations / wireframe artifacts because the
        # 512 shape latent is coarser than the decoder expects.
        t0 = time.time()
        pipeline = self.pipeline
        src_proc = pipeline.preprocess_image(source_image)
        src_cond_512 = pipeline.get_cond([src_proc], 512)
        src_cond_1024 = pipeline.get_cond([src_proc], 1024)

        # SS sampling — returns coords (B, 4) with batch col
        src_coords = pipeline.sample_sparse_structure(
            src_cond_512, 32, 1,
            {"steps": options.num_flow_steps, "guidance_strength": options.guidance_scale},
        )
        timings["source_generation"] = time.time() - t0
        print(f"  Source coords: {tuple(src_coords.shape)}")

        # --- Edit conditioning ---
        edit_proc = pipeline.preprocess_image(edit_image)
        edit_cond_512 = pipeline.get_cond([edit_proc], 512)
        edit_cond_1024 = pipeline.get_cond([edit_proc], 1024)

        # --- Flow-edit: sample SLAT conditioned on the EDIT image but with
        # source COORDS preserving structure. Run the 512→1024 cascade so the
        # geometry is at the same quality as pipeline.run() produces. ---
        t0 = time.time()
        edited_shape_slat, res = pipeline.sample_shape_slat_cascade(
            edit_cond_512, edit_cond_1024,
            pipeline.models["shape_slat_flow_model_512"],
            pipeline.models["shape_slat_flow_model_1024"],
            512, 1024,
            src_coords,  # structure from source
            {"steps": options.num_flow_steps, "guidance_strength": options.guidance_scale},
            49152,  # max_num_tokens — TRELLIS.2's default for 4B
        )
        timings["flow_edit"] = time.time() - t0
        print(f"  Edited SLAT at res={res}, feats={tuple(edited_shape_slat.feats.shape)}")

        # --- Region-focused editing: if user provided a mask, blend with source SLAT ---
        if options.region_mask is not None:
            t0 = time.time()
            source_shape_slat, _ = pipeline.sample_shape_slat_cascade(
                src_cond_512, src_cond_1024,
                pipeline.models["shape_slat_flow_model_512"],
                pipeline.models["shape_slat_flow_model_1024"],
                512, 1024,
                src_coords,
                {"steps": options.num_flow_steps, "guidance_strength": options.guidance_scale},
                49152,
            )
            edited_shape_slat = self._blend_slat_by_mask(
                edit_slat=edited_shape_slat,
                source_slat=source_shape_slat,
                region_mask=options.region_mask,
                source_image=source_image,
                edit_image=edit_image,
                dilation=options.mask_dilation,
                blur_radius=options.mask_blur_radius,
                drop_threshold=options.mask_drop_threshold,
            )
            timings["region_blend"] = time.time() - t0
            print(f"  Region-masked blend applied")

        # --- Texture SLAT (conditioned on edit image) for proper decoding ---
        t0 = time.time()
        try:
            edited_tex_slat = pipeline.sample_tex_slat(
                edit_cond_1024,
                pipeline.models["tex_slat_flow_model_1024"],
                edited_shape_slat,
                {"steps": options.num_repaint_steps, "guidance_strength": options.guidance_scale},
            )
            timings["texture_slat"] = time.time() - t0
            have_tex = True
        except Exception as e:
            import warnings
            import traceback
            tb = traceback.format_exc()
            warnings.warn(
                f"[easy3e] texture SLAT sampling failed: {type(e).__name__}: {e}\n"
                f"decoding without texture. Traceback (tail):\n{tb[-800:]}"
            )
            edited_tex_slat = None
            have_tex = False

        # --- Decode using the full decoder (shape + texture) ---
        t0 = time.time()
        import torch as _torch
        _torch.cuda.empty_cache()
        try:
            if have_tex and hasattr(pipeline, "decode_latent"):
                decoded = pipeline.decode_latent(edited_shape_slat, edited_tex_slat, res)
            else:
                decoded = pipeline.decode_shape_slat(edited_shape_slat, res)
        except Exception as e:
            import warnings
            warnings.warn(f"[easy3e] full decode failed ({e}); falling back to decode_shape_slat")
            decoded = pipeline.decode_shape_slat(edited_shape_slat, res)

        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if isinstance(decoded, list):
            edited_mesh = decoded[0]
        else:
            edited_mesh = decoded
        timings["decode"] = time.time() - t0

        # --- Convert to trimesh if needed (before any CPU/GPU mesh ops) ---
        if not hasattr(edited_mesh, "export"):
            import trimesh
            import numpy as np
            v = edited_mesh.vertices.detach().cpu().numpy() if hasattr(edited_mesh.vertices, "detach") else np.asarray(edited_mesh.vertices)
            f = edited_mesh.faces.detach().cpu().numpy() if hasattr(edited_mesh.faces, "detach") else np.asarray(edited_mesh.faces)
            edited_mesh = trimesh.Trimesh(vertices=v, faces=f)

        # --- PRE-UltraShape repair: clean up holes / ragged edges from
        # the blend-drop step BEFORE handing to UltraShape. UltraShape's
        # surface loader samples points from the input mesh and builds
        # voxel conditioning; a ragged input produces a ragged refinement.
        # Feeding it a watertight mesh yields noticeably cleaner output.
        if options.enable_repair:
            nv_pre = len(edited_mesh.vertices) if hasattr(edited_mesh, "vertices") else 0
            t0 = time.time()
            try:
                if options.skip_repair_above_verts and nv_pre > options.skip_repair_above_verts:
                    from clearmesh.mesh.repair import repair_mesh_cuda
                    edited_mesh = repair_mesh_cuda(
                        edited_mesh,
                        fill_holes=True,
                        remove_small_components=True,
                        fix_normals=True,
                        verbose=False,
                    )
                    timings["pre_ultra_repair_cuda"] = time.time() - t0
                else:
                    from clearmesh.mesh.repair import full_print_preparation
                    edited_mesh = full_print_preparation(edited_mesh, orient=False, verbose=False)
                    timings["pre_ultra_repair"] = time.time() - t0
            except Exception as e:
                import warnings
                warnings.warn(
                    f"[easy3e] Pre-UltraShape repair failed ({e}); continuing unrepaired."
                )
                timings["pre_ultra_repair_failed"] = time.time() - t0

        # --- Optional UltraShape refinement (non-commercial license) ---
        # Runs AFTER pre-repair so UltraShape sees a clean input and produces
        # a cleaner output.
        if options.enable_ultrashape:
            t0 = time.time()
            try:
                import trimesh as _tm
                if not isinstance(edited_mesh, _tm.Trimesh):
                    import numpy as _np
                    v = edited_mesh.vertices.detach().cpu().numpy() if hasattr(edited_mesh.vertices, "detach") else _np.asarray(edited_mesh.vertices)
                    f = edited_mesh.faces.detach().cpu().numpy() if hasattr(edited_mesh.faces, "detach") else _np.asarray(edited_mesh.faces)
                    pre_ultra_mesh = _tm.Trimesh(vertices=v, faces=f)
                else:
                    pre_ultra_mesh = edited_mesh

                from clearmesh.editing.ultrashape_refine import UltraShapeConfig
                us_cfg = UltraShapeConfig(
                    num_inference_steps=options.ultrashape_steps,
                    octree_res=options.ultrashape_octree_res,
                )
                refiner = self.ultrashape_refiner
                if options.ultrashape_ckpt:
                    refiner.ckpt_path = options.ultrashape_ckpt
                if options.ultrashape_config:
                    refiner.config_path = options.ultrashape_config
                if options.ultrashape_dir:
                    refiner.ultrashape_dir = Path(options.ultrashape_dir)

                edited_mesh = refiner.refine(
                    coarse_mesh=pre_ultra_mesh,
                    reference_image=edit_image,
                    config=us_cfg,
                )
                timings["ultrashape_refine"] = time.time() - t0
                print(f"  UltraShape refined: "
                      f"{edited_mesh.vertices.shape[0]:,} verts -> {edited_mesh.faces.shape[0]:,} faces")
            except Exception as e:
                import warnings
                warnings.warn(f"[easy3e] UltraShape refinement failed ({e}); keeping TRELLIS.2 mesh")
                timings["ultrashape_refine_failed"] = time.time() - t0

        # --- Cheap polish pass (vertex dedup + Taubin smoothing) ---
        # Runs after UltraShape, before TripoSF (if ever enabled) or final
        # repair. Addresses the two main UltraShape output artifacts:
        #   - Near-duplicate vertices rendering as visible T-junction cracks
        #   - Minor surface stepping on curved geometry from 1024^3 MC
        if options.enable_vertex_merge or options.enable_taubin_smooth:
            t0 = time.time()
            try:
                from clearmesh.mesh.repair import polish_mesh
                edited_mesh = polish_mesh(
                    edited_mesh,
                    merge_digits=options.vertex_merge_digits if options.enable_vertex_merge else 0,
                    taubin_iterations=options.taubin_iterations if options.enable_taubin_smooth else 0,
                    taubin_lamb=options.taubin_lamb,
                    taubin_nu=options.taubin_nu,
                    verbose=False,
                )
                timings["polish"] = time.time() - t0
                print(
                    f"  Polished: {len(edited_mesh.vertices):,} verts "
                    f"(dedup={options.enable_vertex_merge}, "
                    f"taubin={options.taubin_iterations if options.enable_taubin_smooth else 0})"
                )
            except Exception as e:
                import warnings
                warnings.warn(f"[easy3e] polish failed ({e}); continuing unpolished")
                timings["polish_failed"] = time.time() - t0

        # --- Optional TripoSF (SparseFlex) watertight pass (MIT license) ---
        # Runs AFTER UltraShape. TripoSF's Sparcubes-style VAE converts any
        # input mesh (open or closed) into a watertight 1024^3 reconstruction.
        # Specifically targets the "holes + rough surfaces" artifacts that
        # UltraShape's octree-MC can leave behind.
        if options.enable_triposf:
            t0 = time.time()
            try:
                from clearmesh.editing.triposf_refine import TripoSFRefiner, TripoSFConfig
                refiner = self.triposf_refiner
                if options.triposf_dir:
                    refiner.triposf_dir = Path(options.triposf_dir)
                if options.triposf_config:
                    refiner.config_path = options.triposf_config

                edited_mesh = refiner.refine(
                    coarse_mesh=edited_mesh,
                    config=TripoSFConfig(),
                )
                timings["triposf_watertight"] = time.time() - t0
                is_wt = bool(edited_mesh.is_watertight) if hasattr(edited_mesh, "is_watertight") else "?"
                print(
                    f"  TripoSF watertight: "
                    f"{edited_mesh.vertices.shape[0]:,} verts -> {edited_mesh.faces.shape[0]:,} faces "
                    f"(watertight={is_wt})"
                )
            except Exception as e:
                import warnings
                warnings.warn(
                    f"[easy3e] TripoSF refinement failed ({e}); keeping UltraShape mesh"
                )
                timings["triposf_failed"] = time.time() - t0

        # --- POST-UltraShape repair: a final light pass (just degen/dup cleanup).
        # UltraShape's MC surface should be clean already; we only run the
        # cheap ops here and skip hole-fill (which can over-fill cavities
        # that UltraShape intentionally carved). ---
        if options.enable_repair:
            nv_post = len(edited_mesh.vertices) if hasattr(edited_mesh, "vertices") else 0
            t0 = time.time()
            try:
                if options.skip_repair_above_verts and nv_post > options.skip_repair_above_verts:
                    from clearmesh.mesh.repair import repair_mesh_cuda
                    edited_mesh = repair_mesh_cuda(
                        edited_mesh,
                        fill_holes=False,   # keep UltraShape's intended cavities
                        remove_small_components=True,
                        fix_normals=True,
                        verbose=False,
                    )
                    timings["post_repair_cuda"] = time.time() - t0
            except Exception as e:
                import warnings
                warnings.warn(
                    f"[easy3e] Post repair failed ({e}); returning mesh as-is."
                )
                timings["post_repair_failed"] = time.time() - t0

        # --- Export ---
        if output_path:
            from clearmesh.mesh.export import export_mesh
            export_mesh(edited_mesh, output_path, format=options.export_format)

        timings["total"] = sum(timings.values())
        print(f"  edit_from_source_image complete in {timings['total']:.1f}s")

        return EditResult(
            mesh=edited_mesh,
            output_path=output_path,
            slat=None,  # SLAT return is a paper-specific type; we skip for now
            edit_mask=None,
            timings=timings,
        )

    def _blend_slat_by_mask(
        self,
        edit_slat,
        source_slat,
        region_mask,
        source_image,
        edit_image,
        dilation: int = 1,
        blur_radius: float = 12.0,
        drop_threshold: float = 0.1,
    ):
        """Blend two SLAT SparseTensors per-voxel using a 2D region mask.

        The mask is projected from 2D image space to 3D voxel space via
        TRELLIS.2's canonical camera (see ``clearmesh/editing/camera.py``).
        Voxels whose projected pixel is marked "edit" (mask value > 0) take
        features from ``edit_slat``; the rest take from ``source_slat``.

        Args:
            edit_slat: SparseTensor with edit-conditioned features (.feats).
            source_slat: SparseTensor with source-conditioned features.
                Must have the same coord layout as edit_slat.
            region_mask: PIL.Image, Path, str, or numpy array. Binary
                or grayscale mask; values > 0.5 mean "edit here".
            source_image: Source view — used by fallback mask heuristic.
            edit_image: Edit view — used by fallback mask heuristic.
            dilation: 3D morphological dilation passes to apply.

        Returns:
            A new SparseTensor (same type as input) with blended features.
        """
        import numpy as np
        from pathlib import Path as _Path
        from PIL import Image as _Image
        import torch as _torch

        from clearmesh.editing.camera import CanonicalCamera, project_voxels_to_pixels
        from clearmesh.editing.voxel_flowedit import VoxelFlowEdit

        # --- Load mask ---
        if isinstance(region_mask, (str, _Path)):
            region_mask_img = _Image.open(str(region_mask)).convert("L")
        elif isinstance(region_mask, _Image.Image):
            region_mask_img = region_mask.convert("L")
        elif isinstance(region_mask, np.ndarray):
            # Convert numpy to PIL
            arr = region_mask
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            if arr.ndim == 3:
                arr = arr.mean(axis=-1).astype(np.uint8)
            region_mask_img = _Image.fromarray(arr, mode="L")
        else:
            raise TypeError(
                f"region_mask must be path/PIL.Image/np.ndarray, got {type(region_mask).__name__}"
            )

        # --- Project voxels to pixels ---
        # UltraShape and TRELLIS.2 both normalize meshes to fit [-0.5, 0.5]
        # which corresponds to grid_size=(internal voxel resolution of the SS latent)
        # We use the SLAT's coords to drive projection.
        coords = edit_slat.coords  # (N, 4): batch col + (x, y, z)
        if coords.shape[-1] == 4:
            voxel_xyz = coords[:, 1:]
        else:
            voxel_xyz = coords
        voxel_xyz = voxel_xyz.to(self.device)

        # The SS latent grid size depends on the cascade output resolution.
        # edit_slat here comes from sample_shape_slat_cascade(res=1024), so
        # voxel indices are in [0, 64) roughly; but internal coords may be
        # absolute world coords already. We detect from max value.
        max_coord = int(voxel_xyz.max().item()) + 1
        # Heuristic: align grid_size to nearest power-of-2 above max_coord,
        # capped at 256 (TRELLIS.2 default)
        grid_size = max(32, 1 << (max_coord - 1).bit_length())
        grid_size = min(grid_size, 256)

        camera = CanonicalCamera.trellis2_default(image_size=region_mask_img.width)
        u, v, depth = project_voxels_to_pixels(voxel_xyz, camera, grid_size=grid_size)

        # --- Sample mask at each voxel's projected pixel ---
        W, H = region_mask_img.size
        mask_arr = np.array(region_mask_img, dtype=np.float32) / 255.0

        # --- Soft mask: Gaussian blur to produce a smooth transition zone ---
        # Prevents ragged edges from hard 0/1 cutoff. Without this, voxels
        # whose projected pixel is just inside vs just outside the mask get
        # very different treatment, producing visible seams in the output.
        if blur_radius > 0:
            try:
                from scipy.ndimage import gaussian_filter
                mask_arr = gaussian_filter(mask_arr, sigma=float(blur_radius))
                # After blur, renormalize so max=1 (otherwise a narrow mask
                # ends up with very low peak values).
                if mask_arr.max() > 1e-3:
                    mask_arr = mask_arr / mask_arr.max()
            except ImportError:
                # scipy unavailable — fall back to a box blur via numpy
                r = int(blur_radius)
                k = 2 * r + 1
                kernel = np.ones((k, k), dtype=np.float32) / (k * k)
                # numpy 2D convolution (slow but always available)
                pad = np.pad(mask_arr, r, mode="edge")
                out = np.zeros_like(mask_arr)
                for dy in range(k):
                    for dx in range(k):
                        out += pad[dy:dy + mask_arr.shape[0], dx:dx + mask_arr.shape[1]] * kernel[dy, dx]
                mask_arr = out
                if mask_arr.max() > 1e-3:
                    mask_arr = mask_arr / mask_arr.max()

        mask_t = _torch.from_numpy(mask_arr).to(self.device)

        u_idx = u.round().long().clamp(0, W - 1)
        v_idx = v.round().long().clamp(0, H - 1)
        in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        in_front = depth > 0

        sampled = mask_t[v_idx, u_idx]
        sampled = sampled * in_frame.float() * in_front.float()

        # --- Optional 3D dilation ---
        if dilation > 0 and sampled.sum() > 0:
            ve = VoxelFlowEdit.__new__(VoxelFlowEdit)
            ve.device = str(self.device)
            sampled = ve._dilate_sparse_mask_3d(sampled, voxel_xyz.long(), iterations=dilation)

        # --- Align source/edit coords before blending ---
        # After the cascade, edit and source SLATs often have different
        # voxel counts because each runs its own occupancy pruning. The
        # edit-conditioned SLAT typically "hallucinates" extra voxels
        # where the edit image has new content (wings, fur, etc.).
        #
        # The user's mask tells us where edits are allowed. For each voxel
        # in the edit SLAT we decide:
        #   - matched in source + mask>=0.5 : blend (edit in edit regions)
        #   - matched in source + mask<0.5  : keep source (preserve)
        #   - NOT matched in source + mask>=0.5 : keep edit (legit new
        #     geometry like wings at the sides)
        #   - NOT matched in source + mask<0.5 : DROP (unwanted edit growth
        #     in a preserve region — this was the bug that caused a chaotic
        #     middle when only the sides were supposed to be edited)
        edit_feats = edit_slat.feats
        src_feats = source_slat.feats

        edit_coords_int = (
            edit_slat.coords[:, 1:] if edit_slat.coords.shape[-1] == 4 else edit_slat.coords
        ).long()
        src_coords_int = (
            source_slat.coords[:, 1:] if source_slat.coords.shape[-1] == 4 else source_slat.coords
        ).long()

        src_map = {
            (int(c[0]), int(c[1]), int(c[2])): i
            for i, c in enumerate(src_coords_int.tolist())
        }

        # Build per-voxel decisions
        N_edit = edit_feats.shape[0]
        matched_src_idx = torch.full((N_edit,), -1, dtype=torch.long, device=edit_feats.device)
        for i, c in enumerate(edit_coords_int.tolist()):
            j = src_map.get((int(c[0]), int(c[1]), int(c[2])), -1)
            matched_src_idx[i] = j
        matched = matched_src_idx >= 0  # (N_edit,) bool
        unmatched = ~matched

        # Default-keep mask: voxels we want in the output.
        # With the soft mask, we only drop voxels that are CLEARLY in the
        # preserve zone (sampled < drop_threshold) AND unmatched. Voxels
        # near the boundary (threshold < sampled < 0.5) are kept with a
        # partial blend weight instead of being cut.
        strong_preserve = sampled < drop_threshold
        keep = matched | (unmatched & ~strong_preserve)
        n_matched = int(matched.sum().item())
        n_unmatched_keep = int((unmatched & ~strong_preserve).sum().item())
        n_unmatched_drop = int((unmatched & strong_preserve).sum().item())
        if n_unmatched_drop > 0:
            import warnings
            warnings.warn(
                f"[easy3e] blend: {n_matched} matched, {n_unmatched_keep} unmatched-kept (in edit region), "
                f"{n_unmatched_drop} unmatched-dropped (in preserve region)."
            )

        # Build aligned source features for matched voxels (for blending)
        aligned_src = edit_feats.clone()
        for i in torch.where(matched)[0].tolist():
            aligned_src[i] = src_feats[int(matched_src_idx[i].item())]

        # Per-voxel blend: matched voxels follow the sampled mask, unmatched
        # voxels (that survive the drop) come straight from edit_feats.
        blend_w = sampled.clone()
        blend_w[unmatched] = 1.0  # kept unmatched → pure edit features
        mask_expanded = blend_w.view(-1, 1).to(edit_feats.dtype)
        blended_feats = mask_expanded * edit_feats + (1.0 - mask_expanded) * aligned_src

        # --- Drop voxels marked as unwanted ---
        if not bool(keep.all().item()):
            keep_idx = torch.where(keep)[0]
            blended_feats = blended_feats[keep_idx]
            new_coords = edit_slat.coords[keep_idx]
            # Build a FRESH SparseTensor with no cached state. Using
            # edit_slat.replace(...) would pass through a stale
            # spatial_cache (the indice_dict of the convolutional
            # neighborhood lookup) from the pre-drop layout, which then
            # breaks downstream sparse-conv ops in the texture flow.
            try:
                return edit_slat.__class__(feats=blended_feats, coords=new_coords)
            except Exception:
                # Last resort: use replace and hope for the best
                return edit_slat.replace(feats=blended_feats, coords=new_coords)

        # --- Construct a new SparseTensor with the same coord layout ---
        # SparseTensor init varies between TRELLIS.2 versions; try __class__ replace
        try:
            # Most spconv-based SparseTensors expose `.replace(feats=...)`
            return edit_slat.replace(feats=blended_feats)
        except AttributeError:
            pass
        try:
            return edit_slat.__class__(feats=blended_feats, coords=edit_slat.coords)
        except Exception:
            # Last-resort: mutate in place
            edit_slat.feats = blended_feats
            return edit_slat

    def _flow_edit_slat(
        self,
        src_slat_st,
        src_coords,
        src_cond,
        edit_cond,
        edit_image,
        source_image,
        options: "EditOptions",
    ):
        """Run the Easy3E Voxel FlowEdit ODE on a SparseTensor SLAT.

        The paper's ODE integrates the edit velocity v_edit = v_target - v_source
        at each step. Here we delegate to the pipeline's sampler with the
        edit conditioning — an approximation that matches the core trajectory-
        splitting idea without manually mirroring the sampler's CFG schedule.
        """
        # Simplest workable path: re-sample SLAT conditioned on the edit image
        # while keeping the same coords (preserving structure). The edit
        # signal comes from the different image conditioning, which is
        # the dominant term in the Easy3E ODE at high gamma.
        edited_st = self.pipeline.sample_shape_slat(
            edit_cond,
            self.pipeline.models["shape_slat_flow_model_512"],
            src_coords,
            {
                "steps": options.num_flow_steps,
                "guidance_strength": options.guidance_scale,
            },
        )
        return edited_st

    def edit_from_text(
        self,
        source_mesh: str | Path | trimesh.Trimesh,
        instruction: str,
        view: str = "front",
        output_path: str | None = None,
        options: EditOptions | dict | None = None,
    ) -> EditResult:
        """Text-guided 3D editing.

        Source mesh + text instruction → edited mesh.
        Uses InstructPix2Pix to generate an edit target image,
        then runs image-guided editing.

        Args:
            source_mesh: Source mesh.
            instruction: Text editing instruction.
            view: Which view to edit from (front/back/left/right/top/bottom).
            output_path: Output file path.
            options: Editing options.

        Returns:
            EditResult with edited mesh.
        """
        if isinstance(options, dict):
            options = EditOptions(**options)
        elif options is None:
            options = EditOptions()

        # Step 1: Render source view
        mesh_path = source_mesh if isinstance(source_mesh, (str, Path)) else None
        if mesh_path is None:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                source_mesh.export(f.name)
                mesh_path = f.name

        source_render = self._render_source(mesh_path, view=view)

        # Step 2: Generate edit target with InstructPix2Pix
        print(f"  Generating edit image: '{instruction}'")
        edit_image = self.image_editor.edit(
            source_image=source_render,
            instruction=instruction,
            image_guidance_scale=options.text_image_guidance,
            guidance_scale=options.text_guidance_scale,
            num_inference_steps=options.text_num_steps,
        )

        # Step 3: Run image-guided editing
        return self.edit(
            source_mesh=source_mesh,
            edit_image=edit_image,
            source_image=source_render,
            output_path=output_path,
            options=options,
        )

    def edit_iterative(
        self,
        source_mesh: str | Path | trimesh.Trimesh,
        edits: list[dict],
        output_path: str | None = None,
        options: EditOptions | dict | None = None,
    ) -> EditResult:
        """Iterative editing — chain multiple edits.

        Each edit can be image-guided or text-guided:
          {"image": "path.png"} — image-guided
          {"text": "instruction"} — text-guided
          {"text": "instruction", "view": "left"} — text from specific view

        Args:
            source_mesh: Starting mesh.
            edits: List of edit specifications.
            output_path: Final output path.
            options: Editing options.

        Returns:
            EditResult from the final edit.
        """
        current_mesh = source_mesh
        result = None

        for i, edit_spec in enumerate(edits):
            print(f"\n--- Edit {i + 1}/{len(edits)} ---")

            if "image" in edit_spec:
                result = self.edit(
                    source_mesh=current_mesh,
                    edit_image=edit_spec["image"],
                    options=options,
                )
            elif "text" in edit_spec:
                result = self.edit_from_text(
                    source_mesh=current_mesh,
                    instruction=edit_spec["text"],
                    view=edit_spec.get("view", "front"),
                    options=options,
                )
            else:
                raise ValueError(f"Edit spec must have 'image' or 'text' key: {edit_spec}")

            current_mesh = result.mesh

        # Export final result
        if output_path and result:
            from clearmesh.mesh.export import export_mesh

            export_mesh(result.mesh, output_path, format=(options or EditOptions()).export_format)
            result.output_path = output_path

        return result

    def _render_source(
        self,
        mesh_path: str | Path,
        view: str = "front",
        image_size: int = 512,
    ) -> Image.Image:
        """Render a source view of the mesh."""
        from clearmesh.editing.image_edit import ImageEditor

        dummy = ImageEditor.__new__(ImageEditor)
        return dummy._render_view(mesh_path, view, image_size)

    def _apply_texture(
        self,
        mesh: trimesh.Trimesh,
        reference_image: Image.Image,
        options: EditOptions,
    ) -> trimesh.Trimesh:
        """Apply texture via Ctrl-Adapter.

        Renders normal maps from the edited mesh, then uses
        Ctrl-Adapter to generate textured views.

        Args:
            mesh: Edited mesh to texture.
            reference_image: Reference image for style guidance.
            options: Edit options.

        Returns:
            Textured mesh.
        """
        # TODO: Implement full texture pipeline
        # 1. Render 6-view normal maps from edited mesh
        # 2. Run Ctrl-Adapter to generate textured views
        # 3. Back-project textures onto mesh
        print("  [Ctrl-Adapter texture generation not yet implemented]")
        return mesh
