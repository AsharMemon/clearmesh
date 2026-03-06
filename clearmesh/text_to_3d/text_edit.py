#!/usr/bin/env python3
"""Text-Guided 3D Editing — Text instruction → 3D edit.

Combines InstructPix2Pix (text→image edit) with Easy3E (image→3D edit):
  Source mesh + text instruction
    → render source view
    → InstructPix2Pix(view, instruction) → edit image
    → Easy3E(source mesh, edit image) → edited mesh

This module provides a convenience wrapper that chains the
image editing and 3D editing pipelines.

Usage:
    from clearmesh.text_to_3d.text_edit import TextGuidedEditor

    editor = TextGuidedEditor()

    # Single text edit
    result = editor.edit(
        source_mesh="dragon.glb",
        instruction="add metallic armor plating",
        output_path="dragon_armored.glb",
    )

    # Multi-view text edit (more consistent results)
    result = editor.edit_multiview(
        source_mesh="dragon.glb",
        instruction="make it look like a robot",
        views=["front", "left", "back"],
        output_path="dragon_robot.glb",
    )

    # Iterative text editing
    result = editor.edit_iterative(
        source_mesh="model.glb",
        instructions=[
            "add wings",
            "make it metallic",
            "add glowing eyes",
        ],
        output_path="model_edited.glb",
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import trimesh
from PIL import Image


@dataclass
class TextEditOptions:
    """Options for text-guided 3D editing."""

    # InstructPix2Pix params
    image_guidance_scale: float = 1.5  # Source image fidelity
    text_guidance_scale: float = 7.5  # Instruction adherence
    num_image_steps: int = 20  # InstructPix2Pix diffusion steps
    view: str = "front"  # Which view to edit from

    # Easy3E editing params (passed through)
    num_flow_steps: int = 25
    gamma: float = 1.0
    eta: float = 0.5
    enable_texture: bool = False
    enable_repair: bool = True

    # Export
    export_format: str = "glb"


@dataclass
class TextEditResult:
    """Result from text-guided editing."""

    instruction: str
    source_render: Image.Image
    edit_image: Image.Image
    mesh: trimesh.Trimesh
    output_path: str | None = None
    timings: dict = field(default_factory=dict)


class TextGuidedEditor:
    """Text-guided 3D editing via InstructPix2Pix + Easy3E.

    Convenience wrapper that chains:
      1. Mesh rendering (trimesh)
      2. Image editing (InstructPix2Pix)
      3. 3D editing (Easy3E)
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
        self._easy3e: object = None
        self._image_editor: object = None

        self._trellis2_dir = trellis2_dir
        self._model_dir = model_dir
        self._ctrl_adapter_checkpoint = ctrl_adapter_checkpoint

    @property
    def easy3e(self):
        """Lazy-load Easy3E editor."""
        if self._easy3e is None:
            from clearmesh.editing.easy3e import Easy3EEditor

            self._easy3e = Easy3EEditor(
                trellis2_dir=self._trellis2_dir,
                model_dir=self._model_dir,
                ctrl_adapter_checkpoint=self._ctrl_adapter_checkpoint,
                device=self.device,
            )
        return self._easy3e

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
        instruction: str,
        output_path: str | None = None,
        options: TextEditOptions | dict | None = None,
    ) -> TextEditResult:
        """Text-guided 3D editing.

        Args:
            source_mesh: Source mesh (path or trimesh object).
            instruction: Text editing instruction.
            output_path: Output mesh file path.
            options: Editing options.

        Returns:
            TextEditResult with edited mesh.
        """
        if isinstance(options, dict):
            options = TextEditOptions(**options)
        elif options is None:
            options = TextEditOptions()

        timings = {}

        # Resolve mesh path
        mesh_path = source_mesh if isinstance(source_mesh, (str, Path)) else None
        if mesh_path is None:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                source_mesh.export(f.name)
                mesh_path = f.name

        # Step 1: Render source view
        t0 = time.time()
        source_render = self.image_editor._render_view(
            mesh_path, options.view, image_size=512
        )
        timings["render"] = time.time() - t0

        # Step 2: Generate edit image with InstructPix2Pix
        t0 = time.time()
        print(f"  Text edit: '{instruction}'")
        edit_image = self.image_editor.edit(
            source_image=source_render,
            instruction=instruction,
            image_guidance_scale=options.image_guidance_scale,
            guidance_scale=options.text_guidance_scale,
            num_inference_steps=options.num_image_steps,
        )
        timings["instruct_pix2pix"] = time.time() - t0

        # Step 3: Run Easy3E with the edit image
        t0 = time.time()
        from clearmesh.editing.easy3e import EditOptions

        edit_options = EditOptions(
            num_flow_steps=options.num_flow_steps,
            gamma=options.gamma,
            eta=options.eta,
            enable_texture=options.enable_texture,
            enable_repair=options.enable_repair,
            export_format=options.export_format,
        )

        easy3e_result = self.easy3e.edit(
            source_mesh=source_mesh,
            edit_image=edit_image,
            source_image=source_render,
            output_path=output_path,
            options=edit_options,
        )
        timings["easy3e"] = time.time() - t0
        timings["total"] = sum(timings.values())

        return TextEditResult(
            instruction=instruction,
            source_render=source_render,
            edit_image=edit_image,
            mesh=easy3e_result.mesh,
            output_path=output_path,
            timings=timings,
        )

    def edit_multiview(
        self,
        source_mesh: str | Path | trimesh.Trimesh,
        instruction: str,
        views: list[str] | None = None,
        output_path: str | None = None,
        options: TextEditOptions | dict | None = None,
    ) -> TextEditResult:
        """Multi-view text-guided editing for more consistent results.

        Generates edit images from multiple views and uses the best
        one (highest confidence) for 3D editing.

        Args:
            source_mesh: Source mesh.
            instruction: Text editing instruction.
            views: Views to edit from (default: front, left, back).
            output_path: Output path.
            options: Options.

        Returns:
            TextEditResult from the best view.
        """
        views = views or ["front", "left", "back"]

        if isinstance(options, dict):
            options = TextEditOptions(**options)
        elif options is None:
            options = TextEditOptions()

        # Generate edit images for each view
        best_result = None
        best_diff = 0

        for view in views:
            view_options = TextEditOptions(**{**vars(options), "view": view})
            result = self.edit(
                source_mesh=source_mesh,
                instruction=instruction,
                options=view_options,
            )

            # Simple heuristic: view with most change is "best"
            import numpy as np

            src = np.array(result.source_render).astype(float)
            tgt = np.array(result.edit_image).astype(float)
            diff = np.abs(src - tgt).mean()

            if diff > best_diff:
                best_diff = diff
                best_result = result

        # Re-run with best view's edit image for final output
        if output_path and best_result:
            best_result.output_path = output_path
            from clearmesh.mesh.export import export_mesh

            export_mesh(
                best_result.mesh,
                output_path,
                format=options.export_format,
            )

        return best_result

    def edit_iterative(
        self,
        source_mesh: str | Path | trimesh.Trimesh,
        instructions: list[str],
        output_path: str | None = None,
        options: TextEditOptions | dict | None = None,
    ) -> TextEditResult:
        """Chain multiple text edits iteratively.

        Args:
            source_mesh: Starting mesh.
            instructions: List of text instructions to apply sequentially.
            output_path: Final output path.
            options: Editing options.

        Returns:
            TextEditResult from the final edit.
        """
        current_mesh = source_mesh
        result = None

        for i, instruction in enumerate(instructions):
            print(f"\n--- Text Edit {i + 1}/{len(instructions)}: '{instruction}' ---")
            result = self.edit(
                source_mesh=current_mesh,
                instruction=instruction,
                options=options,
            )
            current_mesh = result.mesh

        if output_path and result:
            from clearmesh.mesh.export import export_mesh

            export_mesh(
                result.mesh,
                output_path,
                format=(options or TextEditOptions()).export_format,
            )
            result.output_path = output_path

        return result
