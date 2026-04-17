#!/usr/bin/env python3
"""Text-to-3D Generation — Text → Image → 3D Pipeline.

Two-step approach:
  1. Text → Image: Qwen-Image (Alibaba, 2025) — default
  2. Image → 3D: ClearMesh pipeline (TRELLIS.2 + Stage 2 + repair + export)

Qwen-Image is the default because:
  - Strong single-object framing prior (unlike SDXL, which interprets
    mechanical prompts like "steampunk gearbox" as 2D art collages and
    produces floating gear sprites — see `docs/research/` for why)
  - Apache 2.0, not gated (unlike FLUX.1-schnell)
  - Strong prompt adherence for compositional prompts
  - Diffusers support via QwenImagePipeline

Other supported models (via ``--model-id`` / ``flux_model_id`` arg):
  - black-forest-labs/FLUX.1-schnell       Apache 2.0, gated (needs HF_TOKEN)
  - stabilityai/stable-diffusion-3.5-large SAI community licence
  - stabilityai/stable-diffusion-xl-base-1.0  legacy, ungated but 2D-biased
  - PixArt-alpha/PixArt-Sigma-XL-2-1024-MS    Apache 2.0 DiT

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
        model_id: str = "Qwen/Qwen-Image",
        stage2_checkpoint: str | None = None,
        model_dir: str = "/workspace/models",
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        flux_model_id: str | None = None,
    ):
        """Initialize TextTo3D.

        Args:
            model_id: HuggingFace model ID for the text-to-image model.
                Default ``Qwen/Qwen-Image`` (Apache 2.0, strong single-
                object prior). Accepts FLUX, SDXL, SD3.5, PixArt-Σ too
                — auto-detected from the id substring.
            stage2_checkpoint: Path to Stage 2 RefinementDiT checkpoint.
            model_dir: Directory with model weights.
            device: Compute device.
            dtype: Model dtype (bfloat16 recommended).
            flux_model_id: DEPRECATED alias for ``model_id``. Preserved
                for backward compat with earlier callers.
        """
        # Back-compat: older callers pass flux_model_id=
        if flux_model_id is not None:
            model_id = flux_model_id
        self.model_id = model_id
        # Keep the old attribute name for any external code that reads it
        self.flux_model_id = model_id
        self.stage2_checkpoint = stage2_checkpoint
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Lazy-loaded components
        self._t2i_pipeline = None
        self._flux_pipeline = None  # retained alias (same object)
        self._clearmesh_pipeline = None

    @property
    def t2i_pipeline(self):
        """Lazy-load the text-to-image pipeline.

        Selects the right diffusers pipeline class based on ``model_id``:
          - ``qwen`` in id   -> QwenImagePipeline
          - ``flux`` in id   -> FluxPipeline (requires HF_TOKEN)
          - otherwise        -> AutoPipelineForText2Image (SDXL, SD3, PixArt, etc)
        """
        if self._t2i_pipeline is None:
            mid = self.model_id.lower()
            print(f"Loading text-to-image model: {self.model_id}...")
            if "qwen" in mid:
                try:
                    from diffusers import QwenImagePipeline
                    pipe = QwenImagePipeline.from_pretrained(
                        self.model_id, torch_dtype=self.dtype,
                    )
                except ImportError:
                    from diffusers import AutoPipelineForText2Image
                    pipe = AutoPipelineForText2Image.from_pretrained(
                        self.model_id, torch_dtype=self.dtype, use_safetensors=True,
                    )
            elif "flux" in mid:
                from diffusers import FluxPipeline
                pipe = FluxPipeline.from_pretrained(
                    self.model_id, torch_dtype=self.dtype,
                )
            else:
                from diffusers import AutoPipelineForText2Image
                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model_id, torch_dtype=self.dtype, use_safetensors=True,
                )

            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe.to(self.device)

            pipe._is_qwen = "qwen" in mid
            pipe._is_flux = "flux" in mid
            pipe._is_sd3 = "stable-diffusion-3" in mid
            self._t2i_pipeline = pipe
            self._flux_pipeline = pipe  # backward-compat alias
            print(f"Loaded {self.model_id}.")
        return self._t2i_pipeline

    @property
    def flux_pipeline(self):
        """Deprecated alias for ``t2i_pipeline``. Kept for backward compat."""
        return self.t2i_pipeline

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
        negative_prompt: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Picks sensible steps/CFG per model family if not supplied:
          - Qwen-Image      : 50 steps, true_cfg_scale=4.0
          - FLUX.1-schnell  : 4 steps, guidance_scale=0.0 (distilled)
          - SD3.5           : 28 steps, guidance_scale=7.0
          - SDXL / PixArt   : 25 steps, guidance_scale=7.0

        Args:
            prompt: Text description of the desired image.
            negative_prompt: What to avoid. Defaults to a 3D-friendly negative.
                FLUX ignores this.
            num_inference_steps: Override per-family default.
            guidance_scale: Override per-family default.
            width / height: Image dimensions.
            seed: Random seed for reproducibility.

        Returns:
            Generated PIL Image.
        """
        pipe = self.t2i_pipeline
        is_qwen = getattr(pipe, "_is_qwen", False)
        is_flux = getattr(pipe, "_is_flux", False)
        is_sd3 = getattr(pipe, "_is_sd3", False)

        enhanced_prompt = self._enhance_prompt(prompt)
        if negative_prompt is None:
            negative_prompt = (
                "collage, grid, multiple objects, duplicates, floating parts, "
                "montage, side by side, diptych, triptych, text, watermark, blur"
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Family-specific defaults
        if is_qwen:
            steps = num_inference_steps or 50
            kwargs = dict(true_cfg_scale=guidance_scale or 4.0, num_inference_steps=steps)
            neg_kwargs = dict(negative_prompt=negative_prompt)
        elif is_flux:
            steps = num_inference_steps or 4
            kwargs = dict(guidance_scale=guidance_scale or 0.0, num_inference_steps=steps)
            neg_kwargs = {}  # FLUX schnell ignores negative prompts
        elif is_sd3:
            steps = num_inference_steps or 28
            kwargs = dict(guidance_scale=guidance_scale or 7.0, num_inference_steps=steps)
            neg_kwargs = dict(negative_prompt=negative_prompt)
        else:
            steps = num_inference_steps or 25
            kwargs = dict(guidance_scale=guidance_scale or 7.0, num_inference_steps=steps)
            neg_kwargs = dict(negative_prompt=negative_prompt)

        try:
            result = pipe(
                prompt=enhanced_prompt,
                width=width, height=height,
                generator=generator,
                **neg_kwargs,
                **kwargs,
            )
        except TypeError:
            # Older pipeline that doesn't accept negative_prompt / true_cfg_scale
            result = pipe(
                prompt=enhanced_prompt,
                width=width, height=height,
                generator=generator,
                **kwargs,
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
            # Let family defaults take over if the call site still passes
            # the old FLUX-schnell defaults of (image_steps=4, cfg=0.0);
            # those would cripple Qwen/SD3.
            _steps = None if (image_steps == 4 and image_guidance_scale == 0.0) else image_steps
            _cfg = None if (image_steps == 4 and image_guidance_scale == 0.0) else image_guidance_scale
            reference_image = self.text_to_image(
                prompt=prompt,
                num_inference_steps=_steps,
                guidance_scale=_cfg,
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

        Adds modifiers that force single-object framing — critical for
        TRELLIS.2 reconstruction. Without "single object / full body /
        isolated subject" in the prompt, some T2I models (especially
        SDXL) interpret mechanical prompts like "steampunk gearbox"
        as 2D art-style collages: floating gear sprites on a page,
        not one coherent 3D object. The collage then gets reconstructed
        as fragmented half-watertight mesh, which is unusable downstream.

        Args:
            prompt: Original text prompt.

        Returns:
            Enhanced prompt.
        """
        prompt_lower = prompt.lower()
        modifiers = []

        if "single" not in prompt_lower and "one " not in prompt_lower:
            modifiers.append("single object")
        if "full body" not in prompt_lower and "whole" not in prompt_lower:
            modifiers.append("full body")
        if "center" not in prompt_lower:
            modifiers.append("centered composition")
        if "product" not in prompt_lower and "3d" not in prompt_lower:
            modifiers.append("studio product photography")
        if "background" not in prompt_lower:
            modifiers.append("plain white background")
        if "isolated" not in prompt_lower:
            modifiers.append("one isolated subject only")
        if "detail" not in prompt_lower:
            modifiers.append("sharp focus, highly detailed")

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
