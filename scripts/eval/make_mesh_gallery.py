#!/usr/bin/env python3
"""Render a directory of meshes into a lightweight visual gallery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh.preview import render_wire_preview_file

MESH_SUFFIXES = {".glb", ".gltf", ".obj", ".ply", ".stl"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--title", default="Mesh gallery")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cell-size", type=int, default=320)
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()

    mesh_paths = sorted(path for path in args.mesh_dir.rglob("*") if path.suffix.lower() in MESH_SUFFIXES)
    if args.limit > 0:
        mesh_paths = mesh_paths[: args.limit]
    if not mesh_paths:
        raise SystemExit(f"no mesh files found in {args.mesh_dir}")

    preview_dir = args.preview_dir or (args.output.parent / "previews")
    preview_dir.mkdir(parents=True, exist_ok=True)
    previews = []
    for index, path in enumerate(mesh_paths):
        preview = render_wire_preview_file(path, preview_dir / f"{index:04d}_{path.stem}.png", size=args.cell_size)
        previews.append((path, preview))

    sheet = _compose(previews, title=args.title, cell_size=args.cell_size, columns=max(1, int(args.columns)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(json.dumps({"gallery": str(args.output), "meshes": len(mesh_paths)}, indent=2, sort_keys=True))
    return 0


def _compose(previews: list[tuple[Path, Path]], *, title: str, cell_size: int, columns: int) -> Image.Image:
    gutter = 22
    title_h = 58
    label_h = 44
    rows = (len(previews) + columns - 1) // columns
    width = gutter + columns * (cell_size + gutter)
    height = title_h + rows * (cell_size + label_h + gutter) + gutter
    image = Image.new("RGB", (width, height), (246, 244, 238))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((gutter, 22), title, fill=(30, 32, 34), font=font)
    for index, (mesh_path, preview_path) in enumerate(previews):
        row = index // columns
        col = index % columns
        x = gutter + col * (cell_size + gutter)
        y = title_h + row * (cell_size + label_h + gutter)
        image.paste(Image.open(preview_path).resize((cell_size, cell_size)), (x, y))
        label = mesh_path.stem
        if len(label) > 48:
            label = label[:45] + "..."
        draw.text((x, y + cell_size + 10), label, fill=(42, 45, 48), font=font)
    return image


if __name__ == "__main__":
    raise SystemExit(main())
