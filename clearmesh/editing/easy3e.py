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

    # Text editing params (InstructPix2Pix)
    text_image_guidance: float = 1.5
    text_guidance_scale: float = 7.5
    text_num_steps: int = 20

    # Mesh processing
    grid_size: int = 256  # O-Voxel resolution
    enable_repair: bool = True  # Post-edit mesh repair

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
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-loaded components
        self._slat_encoder = SLATEncoder(
            trellis2_dir=trellis2_dir,
            model_dir=model_dir,
            device=self.device,
        )
        self._voxel_flowedit: VoxelFlowEdit | None = None
        self._slat_repainter: SLATRepainter | None = None
        self._ctrl_adapter = None
        self._ctrl_adapter_checkpoint = ctrl_adapter_checkpoint
        self._image_editor = None

    @property
    def voxel_flowedit(self) -> VoxelFlowEdit:
        """Lazy-load VoxelFlowEdit."""
        if self._voxel_flowedit is None:
            # TODO: Load TRELLIS.2's flow model and pass to VoxelFlowEdit
            self._voxel_flowedit = VoxelFlowEdit(
                flow_model=None,  # Will be loaded from TRELLIS.2
                device=self.device,
            )
        return self._voxel_flowedit

    @property
    def slat_repainter(self) -> SLATRepainter:
        """Lazy-load SLATRepainter."""
        if self._slat_repainter is None:
            self._slat_repainter = SLATRepainter(
                feature_flow_model=None,  # Will be loaded from TRELLIS.2
                device=self.device,
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

        slat = self._slat_encoder.encode(mesh_path, grid_size=options.grid_size)
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
        edited_mesh = self._slat_encoder.decode(edited_slat)
        timings["decode"] = time.time() - t0

        # === Step 7: Optional texture via Ctrl-Adapter ===
        if options.enable_texture and self.ctrl_adapter is not None:
            t0 = time.time()
            edited_mesh = self._apply_texture(edited_mesh, edit_image, options)
            timings["texture"] = time.time() - t0

        # === Step 8: Repair ===
        if options.enable_repair:
            t0 = time.time()
            from clearmesh.mesh.repair import full_print_preparation

            edited_mesh = full_print_preparation(edited_mesh, orient=False, verbose=False)
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
