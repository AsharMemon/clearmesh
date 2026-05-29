#!/usr/bin/env python3
"""Generate a TRELLIS.2 proxy GLB from an input image.

This wrapper keeps the official TRELLIS.2 example behind a stable ClearMesh CLI
so workers can invoke it through metadata.trellis_command.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input image path or file:// URI")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", default="trellis_proxy.glb")
    parser.add_argument(
        "--face-proxy-output-name",
        default=None,
        help="Optional second GLB export decimated for FACE-Q input.",
    )
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--face-proxy-decimation-target", type=int, default=4096)
    parser.add_argument("--texture-size", type=int, default=4096)
    remesh_group = parser.add_mutually_exclusive_group()
    remesh_group.add_argument("--remesh", dest="remesh", action="store_true", default=True)
    remesh_group.add_argument("--no-remesh", dest="remesh", action="store_false")
    parser.add_argument("--remesh-band", type=int, default=1)
    parser.add_argument("--remesh-project", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_input(value: str) -> Path:
    if value.startswith("file://"):
        value = value.removeprefix("file://")
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"input image not found: {path}")
    return path


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    input_path = resolve_input(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    face_proxy_path = args.output_dir / args.face_proxy_output_name if args.face_proxy_output_name else None
    trellis2_dir = os.getenv("CLEARMESH_TRELLIS2_DIR")
    if trellis2_dir and Path(trellis2_dir).exists():
        sys.path.insert(0, str(Path(trellis2_dir).resolve()))

    from PIL import Image
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    image = Image.open(input_path).convert("RGBA")
    mesh = pipeline.run(image)[0]
    simplify_target = max(
        int(args.decimation_target),
        int(args.face_proxy_decimation_target if face_proxy_path is not None else 0),
    )
    if simplify_target > 0:
        mesh.simplify(simplify_target)

    def export_glb(path: Path, decimation_target: int) -> None:
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(decimation_target),
            texture_size=args.texture_size,
            remesh=args.remesh,
            remesh_band=args.remesh_band,
            remesh_project=args.remesh_project,
            verbose=True,
        )
        glb.export(path, extension_webp=True)

    export_glb(output_path, args.decimation_target)
    if face_proxy_path is not None:
        export_glb(face_proxy_path, args.face_proxy_decimation_target)
    print(output_path)
    if face_proxy_path is not None:
        print(face_proxy_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
