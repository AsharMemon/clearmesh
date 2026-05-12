#!/usr/bin/env python3
"""Render a mesh preview image with Blender.

Run with Blender, for example:
  blender --background --python scripts/eval/render_mesh_preview_blender.py -- mesh.obj preview.png
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        import bpy
    except ImportError as exc:  # pragma: no cover - only available inside Blender.
        raise SystemExit("This script must be run by Blender's Python") from exc

    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    if len(args) < 2:
        raise SystemExit("usage: blender --background --python render_mesh_preview_blender.py -- INPUT_MESH OUTPUT_PNG")
    mesh_path = Path(args[0]).resolve()
    output_path = Path(args[1]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    lower = mesh_path.suffix.lower()
    if lower == ".obj":
        bpy.ops.wm.obj_import(filepath=str(mesh_path))
    elif lower in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    elif lower == ".stl":
        bpy.ops.wm.stl_import(filepath=str(mesh_path))
    else:
        raise SystemExit(f"unsupported mesh format: {mesh_path.suffix}")

    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise SystemExit("no mesh objects imported")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]

    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.ops.view3d.camera_to_view_selected()

    # Basic studio setup.
    bpy.ops.object.light_add(type="AREA", location=(3, -4, 5))
    light = bpy.context.object
    light.data.energy = 450
    light.data.size = 5
    bpy.ops.object.camera_add(location=(2.6, -3.2, 2.1), rotation=(1.1, 0.0, 0.68))
    bpy.context.scene.camera = bpy.context.object

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 48
    bpy.context.scene.render.resolution_x = 1200
    bpy.context.scene.render.resolution_y = 900
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    print(output_path)


if __name__ == "__main__":
    main()
