#!/usr/bin/env python3
"""Run UltraShape refinement behind ClearMesh's reference-refinement hook."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.stage2.ultrashape_refiner import (  # noqa: E402
    ULTRASHAPE_PAPER_CHUNK_SIZE,
    ULTRASHAPE_PAPER_NORMALIZE_SCALE,
    ULTRASHAPE_PAPER_NUM_LATENTS,
    ULTRASHAPE_PAPER_NUM_STEPS,
    ULTRASHAPE_PAPER_OCTREE_RESOLUTION,
    ULTRASHAPE_PAPER_SEED,
    UltraShapeRefiner,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", default="ultrashape_reference.glb")
    parser.add_argument("--ultrashape-dir", default="/workspace/UltraShape-1.0")
    parser.add_argument("--checkpoint", default="/workspace/checkpoints/ultrashape_v1.pt")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--remove-bg", action="store_true")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Accepted for subprocess callers; this script already runs UltraShape directly inside its own process.",
    )
    parser.add_argument("--num-steps", type=int, default=ULTRASHAPE_PAPER_NUM_STEPS)
    parser.add_argument("--octree-resolution", type=int, default=ULTRASHAPE_PAPER_OCTREE_RESOLUTION)
    parser.add_argument("--num-latents", type=int, default=ULTRASHAPE_PAPER_NUM_LATENTS)
    parser.add_argument("--chunk-size", type=int, default=ULTRASHAPE_PAPER_CHUNK_SIZE)
    parser.add_argument("--scale", type=float, default=ULTRASHAPE_PAPER_NORMALIZE_SCALE)
    parser.add_argument("--seed", type=int, default=ULTRASHAPE_PAPER_SEED)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    refiner = UltraShapeRefiner(
        ultrashape_dir=str(args.ultrashape_dir),
        checkpoint=str(args.checkpoint),
        config_path=args.config_path,
        device=args.device,
        low_vram=args.low_vram,
        isolated_process=False,
        remove_background=args.remove_bg,
    )
    refined = refiner.refine(
        coarse_mesh=args.mesh,
        reference_image=args.image,
        num_steps=args.num_steps,
        octree_resolution=args.octree_resolution,
        num_latents=args.num_latents,
        chunk_size=args.chunk_size,
        scale=args.scale,
        seed=args.seed,
    )
    refined.export(output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
