#!/usr/bin/env python3
"""Optional texture/UV post-process for ClearMesh product inference.

This wrapper keeps texture/UV work behind a stable CLI. In production it can
call a dedicated AI texture model; during product tests it can still publish a
clear textured fallback from the Trellis GLB instead of pretending FACE-Q has
texturing wired.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def copy_mesh(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)


def run_command_template(command: str, *, input_mesh: Path, reference_mesh: Path | None, reference_image: Path | None, prompt: str, output: Path, output_dir: Path) -> None:
    values = {
        "input": str(input_mesh),
        "input_mesh": str(input_mesh),
        "reference": str(reference_mesh or ""),
        "reference_mesh": str(reference_mesh or ""),
        "reference_image": str(reference_image or ""),
        "prompt": prompt,
        "output": str(output),
        "output_dir": str(output_dir),
    }
    subprocess.run([part.format(**values) for part in shlex.split(command)], check=True)


def blender_uv_script(*, input_mesh: Path, reference_image: Path | None, output: Path) -> str:
    image_line = f"image_path = {str(reference_image)!r}" if reference_image else "image_path = ''"
    return f"""
import bpy
from pathlib import Path

input_path = {str(input_mesh)!r}
output_path = {str(output)!r}
{image_line}

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.import_scene.gltf(filepath=input_path)
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
for obj in meshes:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if not obj.data.uv_layers:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
    if image_path and Path(image_path).exists():
        material = bpy.data.materials.new(name='Clearmesh texture')
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get('Principled BSDF')
        texture = material.node_tree.nodes.new('ShaderNodeTexImage')
        texture.image = bpy.data.images.load(image_path)
        material.node_tree.links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])
        obj.data.materials.clear()
        obj.data.materials.append(material)
    obj.select_set(False)

bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', export_texcoords=True, export_materials='EXPORT')
"""


def run_blender_uv(*, blender_bin: str, input_mesh: Path, reference_image: Path | None, output: Path, output_dir: Path) -> None:
    script_path = output_dir / "blender_uv_project.py"
    script_path.write_text(blender_uv_script(input_mesh=input_mesh, reference_image=reference_image, output=output), encoding="utf-8")
    subprocess.run([blender_bin, "-b", "--python", str(script_path)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mesh", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", default="textured_mesh.glb")
    parser.add_argument("--reference-mesh", type=Path)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--mode", default=os.getenv("CLEARMESH_TEXTURE_UV_MODE", "auto"), choices=["auto", "command", "blender", "copy-reference", "copy-input", "disabled"])
    parser.add_argument("--command", default=os.getenv("CLEARMESH_TEXTURE_UV_COMMAND", ""))
    parser.add_argument("--blender-bin", default=os.getenv("BLENDER_BIN", "blender"))
    args = parser.parse_args()

    started = time.time()
    input_mesh = args.input_mesh.expanduser().resolve()
    if not input_mesh.exists():
        raise SystemExit(f"input mesh not found: {input_mesh}")
    reference_mesh = args.reference_mesh.expanduser().resolve() if args.reference_mesh else None
    if reference_mesh and not reference_mesh.exists():
        reference_mesh = None
    reference_image = args.reference_image.expanduser().resolve() if args.reference_image else None
    if reference_image and not reference_image.exists():
        reference_image = None

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / args.output_name
    report_path = output_dir / "texture_uv_report.json"

    mode = args.mode
    if mode == "auto":
        if args.command:
            mode = "command"
        elif os.getenv("CLEARMESH_TEXTURE_UV_USE_BLENDER", "0") == "1":
            mode = "blender"
        elif reference_mesh is not None:
            mode = "copy-reference"
        else:
            mode = "copy-input"

    report = {
        "adapter": "texture_uv",
        "mode": mode,
        "input_mesh": str(input_mesh),
        "reference_mesh": str(reference_mesh) if reference_mesh else None,
        "reference_image": str(reference_image) if reference_image else None,
        "output_path": str(output),
        "uv_ready": mode in {"command", "blender", "copy-reference"},
        "ai_textured": mode in {"command", "copy-reference"},
        "fallback": mode in {"copy-reference", "copy-input"},
    }

    try:
        if mode == "disabled":
            report["skipped"] = True
            write_report(report_path, report)
            return 0
        if mode == "command":
            if not args.command:
                raise RuntimeError("command mode requires --command or CLEARMESH_TEXTURE_UV_COMMAND")
            run_command_template(
                args.command,
                input_mesh=input_mesh,
                reference_mesh=reference_mesh,
                reference_image=reference_image,
                prompt=args.prompt,
                output=output,
                output_dir=output_dir,
            )
        elif mode == "blender":
            run_blender_uv(
                blender_bin=args.blender_bin,
                input_mesh=input_mesh,
                reference_image=reference_image,
                output=output,
                output_dir=output_dir,
            )
        elif mode == "copy-reference":
            if reference_mesh is None:
                raise RuntimeError("copy-reference mode requires --reference-mesh")
            copy_mesh(reference_mesh, output)
            report["texture_source"] = "trellis_pbr"
            report["note"] = "Published the Trellis textured/UV GLB as the texture fallback; FACE-Q geometry remains available separately."
        elif mode == "copy-input":
            copy_mesh(input_mesh, output)
            report["note"] = "No texture model/reference mesh was available; copied input mesh as UV/texture placeholder."
        else:
            raise RuntimeError(f"unknown mode: {mode}")
    except Exception as exc:  # noqa: BLE001 - report failures for the API to preserve.
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["elapsed_seconds"] = time.time() - started
        write_report(report_path, report)
        print(json.dumps(report, sort_keys=True), file=sys.stderr)
        return 1

    report["ok"] = output.exists()
    report["elapsed_seconds"] = time.time() - started
    if not output.exists():
        report["error"] = f"output missing: {output}"
        write_report(report_path, report)
        print(json.dumps(report, sort_keys=True), file=sys.stderr)
        return 1
    write_report(report_path, report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
