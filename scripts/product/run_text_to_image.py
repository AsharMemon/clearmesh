#!/usr/bin/env python3
"""Generate a reference image from text for the ClearMesh pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="stabilityai/sdxl-turbo")
    parser.add_argument("--backend", default=None, choices=["diffusers", "hidream"])
    parser.add_argument("--hidream-dir", default=None)
    parser.add_argument("--model-type", default=None, choices=["full", "dev"])
    parser.add_argument("--negative-prompt", default="blurry, low quality, malformed, text, watermark")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def looks_like_hidream(model: str) -> bool:
    return "hidream" in str(model or "").lower()


def run_hidream(args: argparse.Namespace) -> int:
    hidream_dir = Path(args.hidream_dir or os.getenv("CLEARMESH_HIDREAM_DIR", "/ephemeral/HiDream-O1-Image")).expanduser()
    inference = hidream_dir / "inference.py"
    if not inference.exists():
        raise FileNotFoundError(f"HiDream inference.py not found: {inference}")
    model_type = args.model_type or os.getenv("CLEARMESH_HIDREAM_MODEL_TYPE", "dev")
    command = [
        sys.executable,
        str(inference),
        "--model_path",
        args.model,
        "--prompt",
        args.prompt,
        "--output_image",
        str(args.output),
        "--height",
        str(int(args.height)),
        "--width",
        str(int(args.width)),
        "--seed",
        str(int(args.seed)),
        "--model_type",
        model_type,
    ]
    proc = subprocess.run(command, cwd=str(hidream_dir), text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"HiDream text-to-image failed with exit={proc.returncode}")
    if not args.output.exists():
        raise FileNotFoundError(f"HiDream did not write output image: {args.output}")
    print(args.output)
    return 0


def run_diffusers(args: argparse.Namespace) -> int:
    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model,
        torch_dtype=dtype,
        variant="fp16" if dtype is torch.float16 else None,
        use_safetensors=True,
    )
    pipe = pipe.to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(int(args.seed)) if args.seed >= 0 else None
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        num_inference_steps=max(1, int(args.steps)),
        guidance_scale=float(args.guidance_scale),
        width=int(args.width),
        height=int(args.height),
        generator=generator,
    ).images[0]
    image.save(args.output)
    print(args.output)
    return 0


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend = args.backend or os.getenv("CLEARMESH_TEXT_TO_IMAGE_BACKEND") or ("hidream" if looks_like_hidream(args.model) else "diffusers")
    if backend == "hidream":
        try:
            return run_hidream(args)
        except FileNotFoundError as exc:
            if os.getenv("CLEARMESH_ALLOW_TEXT_TO_IMAGE_FALLBACK", "1") == "0":
                raise
            fallback_model = os.getenv("CLEARMESH_TEXT_TO_IMAGE_FALLBACK_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
            args.model = fallback_model
            args.steps = int(os.getenv("CLEARMESH_TEXT_TO_IMAGE_FALLBACK_STEPS", "28"))
            args.guidance_scale = float(os.getenv("CLEARMESH_TEXT_TO_IMAGE_FALLBACK_GUIDANCE_SCALE", "7.0"))
            print(f"HiDream unavailable ({exc}); falling back to diffusers model={fallback_model}", file=sys.stderr)
    return run_diffusers(args)


if __name__ == "__main__":
    raise SystemExit(main())
