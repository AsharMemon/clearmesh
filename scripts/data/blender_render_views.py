#!/usr/bin/env python3
"""Render pre-exported GLB views with Blender Cycles.

Usage:
  blender --background --factory-startup --python blender_render_views.py -- \
    --mesh /tmp/model.glb --output_dir /tmp/renders --views_json '[[0,20],[45,15]]'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--views_json", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--fov_deg", type=float, default=40.0)
    parser.add_argument("--fill_fraction", type=float, default=0.78)
    parser.add_argument("--samples", type=int, default=16)
    return parser.parse_args(argv)


def reset_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.use_persistent_data = False
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    return scene


def configure_cycles(scene: bpy.types.Scene, samples: int) -> None:
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 2
    scene.cycles.transmission_bounces = 0
    scene.cycles.device = "GPU"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            enabled = False
            for device in prefs.devices:
                if device.type != "CPU":
                    device.use = True
                    enabled = True
                else:
                    device.use = False
            if enabled:
                return
        except Exception:
            continue
    scene.cycles.device = "CPU"


def import_mesh(filepath: str) -> bpy.types.Object:
    bpy.ops.import_scene.gltf(filepath=filepath)
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {filepath}")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    if len(mesh_objects) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def normalize_object(obj: bpy.types.Object) -> float:
    mesh = obj.data
    if not mesh.vertices:
        raise RuntimeError("Imported mesh has no vertices")

    verts = [obj.matrix_world @ v.co for v in mesh.vertices]
    center = sum(verts, Vector((0.0, 0.0, 0.0))) / len(verts)
    for v in mesh.vertices:
        v.co -= center
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=False)

    radius = max(v.co.length for v in mesh.vertices)
    if radius <= 1e-6:
        raise RuntimeError("Imported mesh is degenerate")

    scale = 1.0 / radius
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return 1.0


def assign_material(obj: bpy.types.Object) -> None:
    """Create a Cycles material for the mesh.

    If the mesh has vertex colors (from normal-based shading in
    prepare_mesh_asset.py), the material uses a Color Attribute node so
    Blender renders geometric cues that DINOv2 can extract meaningful
    features from.  Falls back to flat gray only when no vertex colors
    exist.
    """
    mat = bpy.data.materials.new(name="ClearmeshRender")
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes.get("Principled BSDF")
    if bsdf is None:
        # Shouldn't happen, but guard against unusual Blender versions
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        return

    # Check for vertex colors (Blender 3.x: color_attributes; 2.x: vertex_colors)
    mesh_data = obj.data
    has_vcol = False
    vcol_name = None
    if hasattr(mesh_data, "color_attributes") and len(mesh_data.color_attributes) > 0:
        has_vcol = True
        vcol_name = mesh_data.color_attributes[0].name
    elif hasattr(mesh_data, "vertex_colors") and len(mesh_data.vertex_colors) > 0:
        has_vcol = True
        vcol_name = mesh_data.vertex_colors[0].name

    if has_vcol and vcol_name:
        # Wire vertex colors → Base Color via Color Attribute node
        attr_node = tree.nodes.new("ShaderNodeVertexColor")
        attr_node.layer_name = vcol_name
        tree.links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        # No vertex colors — use flat gray (last resort)
        bsdf.inputs["Base Color"].default_value = (0.82, 0.82, 0.82, 1.0)

    bsdf.inputs["Roughness"].default_value = 0.7
    spec_input = (
        bsdf.inputs.get("Specular IOR Level")
        or bsdf.inputs.get("Specular")
    )
    if spec_input is not None:
        spec_input.default_value = 0.2

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def setup_lights() -> None:
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = 0.3

    def add_area(name: str, location: tuple[float, float, float], energy: float, size: float):
        light_data = bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light)
        light.location = location
        direction = Vector((0.0, 0.0, 0.0)) - light.location
        light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    add_area("KeyLight", (3.0, -3.0, 2.8), 1800.0, 3.0)
    add_area("FillLight", (-2.5, -1.5, 1.6), 900.0, 4.0)
    add_area("RimLight", (0.0, 3.5, 2.2), 700.0, 3.5)


def setup_camera(fov_deg: float, fill_fraction: float) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.angle = math.radians(fov_deg)
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    distance = 1.0 / (fill_fraction * math.tan(cam_data.angle / 2.0))
    cam["distance"] = distance
    return cam


def set_camera_view(cam: bpy.types.Object, yaw_deg: float, pitch_deg: float) -> None:
    distance = float(cam.get("distance", 3.0))
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    x = distance * math.cos(pitch) * math.sin(yaw)
    y = -distance * math.cos(pitch) * math.cos(yaw)
    z = distance * math.sin(pitch)
    cam.location = (x, y, z)
    direction = Vector((0.0, 0.0, 0.0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = json.loads(args.views_json)

    scene = reset_scene()
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    configure_cycles(scene, args.samples)

    obj = import_mesh(args.mesh)
    normalize_object(obj)
    assign_material(obj)
    setup_lights()
    cam = setup_camera(args.fov_deg, args.fill_fraction)

    for idx, (yaw_deg, pitch_deg) in enumerate(views):
        set_camera_view(cam, float(yaw_deg), float(pitch_deg))
        scene.render.filepath = str(output_dir / f"view_{idx:02d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
