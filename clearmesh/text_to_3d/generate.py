#!/usr/bin/env python3
"""Text-to-3D Generation — Text → Image → 3D Pipeline.

Two-step approach:
  1. Text → Image: FLUX.1-schnell (Black Forest Labs), 4-step generation
  2. Image → 3D: ClearMesh pipeline (TRELLIS.2 + Stage 2 + repair + export)

FLUX.1-schnell is chosen for:
  - Speed: 4 inference steps (vs 20-50 for SDXL)
  - Quality: state-of-the-art text-to-image
  - License: Apache 2.0

Usage:
    from clearmesh.text_to_3d import TextTo3D

    gen = TextTo3D()

    # Basic text-to-3D
    result = gen.generate("a medieval castle", output_path="castle.glb")

    # With custom image generation params
    result = gen.generate(
        "a detailed robot warrior",
        image_steps=8,
        image_guidance_scale=3.5,
        image_size=(1024, 1024),
        output_path="robot.stl",
        mesh_options={"target_scale": "32mm"},
    )

    # Just generate the image (no 3D)
    image = gen.text_to_image("a cute cat figurine")
    image.save("cat_reference.png")

CLI:
    python -m clearmesh.text_to_3d.generate \
        --prompt "a fierce dragon" \
        --output dragon.glb \
        --format glb \
        --scale 32mm
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image


@dataclass
class TextTo3DResult:
    """Result from text-to-3D generation."""

    prompt: str
    reference_image: Image.Image
    mesh: object  # trimesh.Trimesh (lazy import)
    output_path: str | None = None
    generation_result: object = None  # GenerationResult from pipeline
    timings: dict = None


class TextTo3D:
    """Text-to-3D generation via FLUX.1-schnell → ClearMesh pipeline.

    Generates a reference image from text, then runs the full
    ClearMesh image-to-3D pipeline to produce a print-ready mesh.
    """

    def __init__(
        self,
        flux_model_id: str = "black-forest-labs/FLUX.1-schnell",
        stage2_checkpoint: str | None = None,
        model_dir: str = "/workspace/models",
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize TextTo3D.

        Args:
            flux_model_id: HuggingFace model ID for FLUX.1-schnell.
            stage2_checkpoint: Path to Stage 2 RefinementDiT checkpoint.
            model_dir: Directory with model weights.
            device: Compute device.
            dtype: Model dtype (bfloat16 for FLUX).
        """
        self.flux_model_id = flux_model_id
        self.stage2_checkpoint = stage2_checkpoint
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Lazy-loaded components
        self._flux_pipeline = None
        self._clearmesh_pipeline = None

    @property
    def flux_pipeline(self):
        """Lazy-load FLUX.1-schnell pipeline."""
        if self._flux_pipeline is None:
            print(f"Loading FLUX.1-schnell from {self.flux_model_id}...")
            from diffusers import FluxPipeline

            self._flux_pipeline = FluxPipeline.from_pretrained(
                self.flux_model_id,
                torch_dtype=self.dtype,
            )
            self._flux_pipeline.to(self.device)

            # Enable memory optimizations
            try:
                self._flux_pipeline.enable_model_cpu_offload()
            except Exception:
                pass  # Not all setups support this

            print("FLUX.1-schnell loaded.")
        return self._flux_pipeline

    @property
    def clearmesh_pipeline(self):
        """Lazy-load ClearMesh pipeline."""
        if self._clearmesh_pipeline is None:
            from clearmesh.pipeline import ClearMeshPipeline

            self._clearmesh_pipeline = ClearMeshPipeline(
                stage2_checkpoint=self.stage2_checkpoint,
                model_dir=self.model_dir,
                device=self.device,
            )
        return self._clearmesh_pipeline

    def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate an image from a text prompt using FLUX.1-schnell.

        Args:
            prompt: Text description of the desired image.
            negative_prompt: What to avoid in the image.
            num_inference_steps: Diffusion steps (4 for schnell, more for quality).
            guidance_scale: CFG scale (0.0 for schnell — it's distilled).
            width: Image width.
            height: Image height.
            seed: Random seed for reproducibility.

        Returns:
            Generated PIL Image.
        """
        # Enhance prompt for 3D-friendly generation
        enhanced_prompt = self._enhance_prompt(prompt)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.flux_pipeline(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )

        return result.images[0]

    def generate(
        self,
        prompt: str,
        output_path: str | None = None,
        image_steps: int = 4,
        image_guidance_scale: float = 0.0,
        image_size: tuple[int, int] = (1024, 1024),
        seed: int | None = None,
        reference_image: Image.Image | None = None,
        mesh_options: dict | None = None,
    ) -> TextTo3DResult:
        """Full text-to-3D generation.

        Args:
            prompt: Text description.
            output_path: Output mesh file path.
            image_steps: FLUX inference steps.
            image_guidance_scale: FLUX CFG scale.
            image_size: Generated image size (W, H).
            seed: Random seed.
            reference_image: Skip image generation, use this image instead.
            mesh_options: Options dict passed to ClearMesh pipeline.

        Returns:
            TextTo3DResult with mesh and reference image.
        """
        timings = {}

        # Step 1: Generate reference image
        if reference_image is None:
            t0 = time.time()
            print(f"Generating reference image: '{prompt}'")
            reference_image = self.text_to_image(
                prompt=prompt,
                num_inference_steps=image_steps,
                guidance_scale=image_guidance_scale,
                width=image_size[0],
                height=image_size[1],
                seed=seed,
            )
            timings["text_to_image"] = time.time() - t0
            print(f"  Image generated in {timings['text_to_image']:.1f}s")
        else:
            print("Using provided reference image")

        # Step 2: Save reference image temporarily
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            reference_image.save(f.name)
            temp_image_path = f.name

        # Step 3: Run ClearMesh Image-to-3D pipeline
        t0 = time.time()
        print("Running ClearMesh pipeline...")

        from clearmesh.pipeline import GenerationOptions

        if mesh_options:
            options = GenerationOptions(**mesh_options)
        else:
            options = GenerationOptions()

        # Skip background removal for generated images (already clean)
        options.skip_background_removal = False  # FLUX may include backgrounds

        gen_result = self.clearmesh_pipeline.generate(
            temp_image_path,
            output_path=output_path,
            options=options,
        )
        timings["image_to_3d"] = time.time() - t0
        timings["total"] = sum(timings.values())

        # Cleanup temp file
        import os

        os.unlink(temp_image_path)

        print(f"\nText-to-3D complete in {timings['total']:.1f}s")
        print(f"  Prompt: '{prompt}'")
        if output_path:
            print(f"  Output: {output_path}")

        return TextTo3DResult(
            prompt=prompt,
            reference_image=reference_image,
            mesh=gen_result.mesh,
            output_path=output_path or gen_result.output_path,
            generation_result=gen_result,
            timings=timings,
        )

    def _enhance_prompt(self, prompt: str) -> str:
        """Enhance a prompt for better 3D-friendly image generation.

        Adds modifiers that help generate images more suitable for
        3D reconstruction (centered object, clean background, etc.).

        Args:
            prompt: Original text prompt.

        Returns:
            Enhanced prompt.
        """
        # Add 3D-friendly modifiers if not already present
        modifiers = []
        prompt_lower = prompt.lower()

        if "3d" not in prompt_lower and "render" not in prompt_lower:
            modifiers.append("3D render")
        if "white background" not in prompt_lower and "background" not in prompt_lower:
            modifiers.append("white background")
        if "centered" not in prompt_lower:
            modifiers.append("centered")
        if "high detail" not in prompt_lower and "detailed" not in prompt_lower:
            modifiers.append("highly detailed")

        if modifiers:
            return f"{prompt}, {', '.join(modifiers)}"
        return prompt


def main():
    parser = argparse.ArgumentParser(description="ClearMesh Text-to-3D")
    parser.add_argument("--prompt", type=str, required=True, help="Text description")
    parser.add_argument("--output", type=str, default=None, help="Output mesh path")
    parser.add_argument(
        "--format", type=str, default="glb", choices=["stl", "glb", "obj", "fbx"]
    )
    parser.add_argument("--scale", type=str, default=None, choices=["28mm", "32mm", "54mm", "75mm"])
    parser.add_argument("--image-steps", type=int, default=4, help="FLUX inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--stage2-checkpoint", type=str, default=None)
    parser.add_argument("--save-image", type=str, default=None, help="Save reference image")
    args = parser.parse_args()

    gen = TextTo3D(stage2_checkpoint=args.stage2_checkpoint)

    output = args.output or f"text_to_3d_output.{args.format}"

    mesh_options = {"export_format": args.format}
    if args.scale:
        mesh_options["target_scale"] = args.scale

    result = gen.generate(
        prompt=args.prompt,
        output_path=output,
        image_steps=args.image_steps,
        seed=args.seed,
        mesh_options=mesh_options,
    )

    if args.save_image:
        result.reference_image.save(args.save_image)
        print(f"Reference image saved: {args.save_image}")


if __name__ == "__main__":
    main()
