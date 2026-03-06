#!/usr/bin/env python3
"""Image Edit — InstructPix2Pix wrapper for text-guided image editing.

Used in the text-guided 3D editing pipeline:
  Text instruction + source mesh render → InstructPix2Pix → edited image → Easy3E

InstructPix2Pix (Brooks et al., 2023) edits images based on text
instructions while preserving the overall structure — ideal for
generating "edit target" images from source mesh renders.

Usage:
    editor = ImageEditor()
    edited = editor.edit(
        source_image="source_render.png",
        instruction="make it look like a robot",
    )
    edited.save("edited_view.png")

    # Or with a PIL Image directly:
    edited = editor.edit(
        source_image=pil_image,
        instruction="add armor plating",
        image_guidance_scale=1.5,
    )
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


class ImageEditor:
    """InstructPix2Pix-based image editor.

    Wraps the InstructPix2Pix diffusion pipeline for text-guided
    image editing. Used to generate "edit target" images from
    source mesh renders for the Easy3E 3D editing pipeline.
    """

    def __init__(
        self,
        model_id: str = "timbrooks/instruct-pix2pix",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the image editor.

        Args:
            model_id: HuggingFace model ID for InstructPix2Pix.
            device: Compute device.
            dtype: Model dtype (float16 for efficiency).
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy-load the InstructPix2Pix pipeline."""
        if self._pipeline is None:
            print(f"Loading InstructPix2Pix from {self.model_id}...")
            from diffusers import StableDiffusionInstructPix2PixPipeline

            self._pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
            )
            self._pipeline.to(self.device)
            # Enable memory optimizations
            if hasattr(self._pipeline, "enable_model_cpu_offload"):
                # Only use if VRAM is tight
                pass
            print("InstructPix2Pix loaded.")
        return self._pipeline

    def edit(
        self,
        source_image: str | Path | Image.Image,
        instruction: str,
        num_inference_steps: int = 20,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """Edit an image based on a text instruction.

        Args:
            source_image: Source image (path or PIL Image).
            instruction: Text editing instruction (e.g., "make it red").
            num_inference_steps: Diffusion steps (more = higher quality).
            image_guidance_scale: How much to preserve source structure.
                Higher values = more faithful to source, less change.
            guidance_scale: Text guidance scale (CFG).
                Higher values = stronger adherence to instruction.
            seed: Random seed for reproducibility.
            output_size: Resize output to (W, H). None = match input size.

        Returns:
            Edited PIL Image.
        """
        # Load image if path
        if isinstance(source_image, (str, Path)):
            source_image = Image.open(str(source_image)).convert("RGB")

        original_size = source_image.size

        # Resize to model's expected resolution (multiples of 8)
        w, h = source_image.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        if (w, h) != source_image.size:
            source_image = source_image.resize((w, h), Image.LANCZOS)

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run InstructPix2Pix
        result = self.pipeline(
            prompt=instruction,
            image=source_image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        edited = result.images[0]

        # Resize to output size or original size
        target_size = output_size or original_size
        if edited.size != target_size:
            edited = edited.resize(target_size, Image.LANCZOS)

        return edited

    def batch_edit(
        self,
        source_images: list[Image.Image],
        instruction: str,
        **kwargs,
    ) -> list[Image.Image]:
        """Edit multiple images with the same instruction.

        Useful for editing multiple views of a 3D model consistently.

        Args:
            source_images: List of source PIL Images.
            instruction: Text editing instruction.
            **kwargs: Additional arguments passed to edit().

        Returns:
            List of edited PIL Images.
        """
        return [
            self.edit(img, instruction, **kwargs)
            for img in source_images
        ]

    def render_and_edit(
        self,
        mesh_path: str | Path,
        instruction: str,
        view: str = "front",
        image_size: int = 512,
        **edit_kwargs,
    ) -> Image.Image:
        """Render a mesh view and edit it in one step.

        Convenience method for the text-guided editing pipeline.

        Args:
            mesh_path: Path to mesh file.
            instruction: Text editing instruction.
            view: View name (front/back/left/right/top/bottom).
            image_size: Render resolution.
            **edit_kwargs: Additional arguments for edit().

        Returns:
            Edited PIL Image.
        """
        # Render the source view
        source_render = self._render_view(mesh_path, view, image_size)

        # Edit with InstructPix2Pix
        return self.edit(source_render, instruction, **edit_kwargs)

    def _render_view(
        self,
        mesh_path: str | Path,
        view: str,
        image_size: int,
    ) -> Image.Image:
        """Render a single view of a mesh.

        Args:
            mesh_path: Path to mesh file.
            view: View name.
            image_size: Output resolution.

        Returns:
            Rendered PIL Image.
        """
        from io import BytesIO

        import numpy as np
        import trimesh

        # View definitions matching render_ctrl_adapter_data.py
        VIEW_CAMERAS = {
            "front": {"eye": (0, 0, 2), "up": (0, 1, 0)},
            "back": {"eye": (0, 0, -2), "up": (0, 1, 0)},
            "left": {"eye": (-2, 0, 0), "up": (0, 1, 0)},
            "right": {"eye": (2, 0, 0), "up": (0, 1, 0)},
            "top": {"eye": (0, 2, 0), "up": (0, 0, -1)},
            "bottom": {"eye": (0, -2, 0), "up": (0, 0, 1)},
        }

        mesh = trimesh.load(str(mesh_path), force="mesh")
        mesh.vertices -= mesh.centroid
        scale = mesh.extents.max()
        if scale > 0:
            mesh.vertices /= scale

        cam = VIEW_CAMERAS.get(view, VIEW_CAMERAS["front"])
        scene = trimesh.Scene(mesh)
        camera_transform = trimesh.transformations.look_at(
            np.array(cam["eye"]),
            np.array([0, 0, 0]),
            np.array(cam["up"]),
        )
        scene.camera_transform = camera_transform

        try:
            png = scene.save_image(resolution=(image_size, image_size))
            return Image.open(BytesIO(png)).convert("RGB")
        except Exception:
            return Image.new("RGB", (image_size, image_size), (128, 128, 128))
