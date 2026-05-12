#!/usr/bin/env python3
"""Render teacher/generated mesh pairs into a lightweight contact sheet."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cell-size", type=int, default=420)
    parser.add_argument("--title", default="FACE reconstruction gate")
    args = parser.parse_args()

    generated = sorted(args.mesh_dir.glob("*_generated.glb"))
    if args.limit > 0:
        generated = generated[: args.limit]
    if not generated:
        raise SystemExit(f"no *_generated.glb files found in {args.mesh_dir}")

    preview_dir = args.preview_dir or (args.output.parent / "previews")
    preview_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_index = _load_metrics(args.report)

    rows: list[tuple[str, Path | None, Path | None, dict[str, object], str | None]] = []
    for index, generated_path in enumerate(generated):
        base = generated_path.name[: -len("_generated.glb")]
        teacher_path = args.mesh_dir / f"{base}_teacher.glb"
        teacher_preview = None
        render_error = None
        if teacher_path.exists():
            teacher_preview, teacher_error = _safe_render_preview_file(
                teacher_path,
                preview_dir / f"{index:04d}_teacher.png",
                size=args.cell_size,
            )
            render_error = teacher_error
        generated_preview, generated_error = _safe_render_preview_file(
            generated_path,
            preview_dir / f"{index:04d}_generated.png",
            size=args.cell_size,
        )
        render_error = generated_error or render_error
        rows.append((base, teacher_preview, generated_preview, metrics_by_index.get(index, {}), render_error))

    sheet = _compose(rows, title=args.title, cell_size=args.cell_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(json.dumps({"contact_sheet": str(args.output), "rows": len(rows)}, indent=2, sort_keys=True))
    return 0


def _safe_render_preview_file(input_mesh: Path, output: Path, *, size: int) -> tuple[Path | None, str | None]:
    try:
        return render_wire_preview_file(input_mesh, output, size=size), None
    except Exception as exc:  # pragma: no cover - defensive for bad generated meshes.
        return None, str(exc)


def _load_metrics(report: Path | None) -> dict[int, dict[str, object]]:
    if report is None or not report.exists():
        return {}
    data = json.loads(report.read_text())
    out: dict[int, dict[str, object]] = {}
    for index, result in enumerate(data.get("results", [])):
        out[index] = result
    return out


def _compose(
    rows: list[tuple[str, Path | None, Path | None, dict[str, object], str | None]],
    *,
    title: str,
    cell_size: int,
) -> Image.Image:
    gutter = 24
    label_h = 78
    title_h = 60
    width = gutter * 3 + cell_size * 2
    height = title_h + len(rows) * (cell_size + label_h + gutter) + gutter
    image = Image.new("RGB", (width, height), (246, 244, 238))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((gutter, 22), title, fill=(30, 32, 34), font=font)
    draw.text((gutter, title_h - 20), "teacher", fill=(70, 74, 77), font=font)
    draw.text((gutter * 2 + cell_size, title_h - 20), "generated", fill=(70, 74, 77), font=font)

    y = title_h
    for index, (name, teacher_preview, generated_preview, metrics, render_error) in enumerate(rows):
        x_left = gutter
        x_right = gutter * 2 + cell_size
        if teacher_preview is not None:
            image.paste(Image.open(teacher_preview).resize((cell_size, cell_size)), (x_left, y))
        else:
            draw.rectangle((x_left, y, x_left + cell_size, y + cell_size), fill=(230, 226, 218))
            draw.text((x_left + 18, y + 18), "no teacher mesh", fill=(90, 94, 98), font=font)
        if generated_preview is not None:
            image.paste(Image.open(generated_preview).resize((cell_size, cell_size)), (x_right, y))
        else:
            draw.rectangle((x_right, y, x_right + cell_size, y + cell_size), fill=(230, 226, 218))
            draw.text((x_right + 18, y + 18), "no generated mesh", fill=(90, 94, 98), font=font)

        label_y = y + cell_size + 10
        summary = _metric_line(metrics)
        draw.text((x_left, label_y), f"{index:02d} {name}", fill=(30, 32, 34), font=font)
        if summary:
            draw.text((x_left, label_y + 18), summary, fill=(74, 78, 82), font=font)
        if render_error:
            draw.text((x_right, label_y + 18), f"render: {render_error[:86]}", fill=(142, 64, 56), font=font)
        y += cell_size + label_h + gutter
    return image


def _metric_line(metrics: dict[str, object]) -> str:
    if not metrics:
        return ""
    pieces = []
    for key, label in [
        ("teacher_forced_accuracy", "tf_acc"),
        ("chamfer_l2", "chamfer"),
        ("boundary_edges", "boundary"),
        ("watertight", "watertight"),
    ]:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            pieces.append(f"{label}={value:.4g}")
        else:
            pieces.append(f"{label}={value}")
    return " | ".join(pieces)


if __name__ == "__main__":
    raise SystemExit(main())
