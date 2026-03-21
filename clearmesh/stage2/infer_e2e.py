#!/usr/bin/env python3
"""End-to-end image -> mesh inference for the residual SLAT refiner.

This compatibility CLI replaces the old diffusion/SDF e2e path with the current
residual workflow: TRELLIS.2 coarse SLAT -> residual refinement -> TRELLIS.2
shape decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
import yaml

from clearmesh.stage2.infer_slat import (
    load_image,
    load_stage2_model,
    load_trellis2_pipeline,
    process_single_image,
    setup_trellis2,
)


def gather_images(args) -> list[tuple[str, str]]:
    images: list[tuple[str, str]] = []

    if args.image:
        images.append((Path(args.image).stem, args.image))
    elif args.image_url:
        name = args.image_url.split("/")[-1].split("?")[0].split(".")[0] or "url_image"
        images.append((name, args.image_url))
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for image_path in sorted(image_dir.glob(ext)):
                images.append((image_path.stem, str(image_path)))
    elif args.image_list:
        with open(args.image_list) as f:
            items = json.load(f)
        for item in items:
            path_or_url = item if isinstance(item, str) else item["path"]
            name = item.get("name") if isinstance(item, dict) else None
            if not name:
                name = (
                    Path(path_or_url).stem
                    if not path_or_url.startswith(("http://", "https://"))
                    else "url_image"
                )
            images.append((name, path_or_url))
    else:
        print("Error: specify --image, --image_url, --image_dir, or --image_list")
        sys.exit(1)

    return images


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end image to mesh inference for residual Stage 2"
    )
    parser.add_argument("--config", required=True, help="Stage 2 training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Stage 2 checkpoint .pt file")

    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--image_url", default=None, help="Single image URL")
    parser.add_argument("--image_dir", default=None, help="Directory of images")
    parser.add_argument("--image_list", default=None, help="JSON list of paths or URLs")

    parser.add_argument("--trellis2_dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--model_dir", default="/workspace/models/trellis2-4b")
    parser.add_argument(
        "--pipeline_type",
        default="512",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        help="Residual Stage 2 is trained on TRELLIS.2 512 coarse SLAT. Other values are rejected.",
    )

    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Deprecated for residual mode; accepted for CLI compatibility.")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Deprecated for residual mode; accepted for CLI compatibility.")
    parser.add_argument("--save_baselines", action="store_true")

    parser.add_argument("--output_dir", default="e2e_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.pipeline_type != "512":
        print("Error: residual Stage 2 expects TRELLIS.2 512 coarse SLAT. Use --pipeline_type 512.")
        sys.exit(1)

    if args.num_steps != 50:
        print("Note: --num_steps is ignored in residual mode (single-pass refinement).")
    if args.resolution != 128:
        print("Note: --resolution is ignored in residual mode (TRELLIS decoder handles mesh extraction).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"\n{'=' * 60}")
    print("ClearMesh End-to-End Residual Inference")
    print(f"{'=' * 60}")

    print("\nLoading TRELLIS.2 pipeline...")
    setup_trellis2(args.trellis2_dir)
    trellis_pipeline = load_trellis2_pipeline(args.model_dir, device=device)

    print("\nLoading ClearMesh Stage 2...")
    stage2_model, step = load_stage2_model(config, args.checkpoint, device=device)
    print(f"  Stage 2 loaded (step {step})")

    images = gather_images(args)
    print(f"\nProcessing {len(images)} image(s)...")
    print(f"  Mode:         Direct residual prediction")
    print(f"  Delta scale:  {args.delta_scale}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Output:       {args.output_dir}")

    results = []
    for i, (name, path_or_url) in enumerate(images, start=1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(images)}] {name}")
        print(f"{'=' * 60}")
        try:
            image = load_image(path_or_url)
            summary = process_single_image(
                trellis_pipeline,
                stage2_model,
                image,
                name,
                output_dir=args.output_dir,
                max_tokens=args.max_tokens,
                delta_scale=args.delta_scale,
                seed=args.seed,
                save_baselines=args.save_baselines,
                device=device,
            )
            results.append(summary)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            results.append({"name": name, "error": str(exc)})

    results_path = Path(args.output_dir) / "all_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
