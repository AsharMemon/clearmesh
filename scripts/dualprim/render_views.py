"""Render a mesh to the paper's 26-view set (§5.1).

Output per view:
  - ``{i:02d}_rgb.png``      256x256 RGB
  - ``{i:02d}_mask.png``     256x256 binary mask
  - ``{i:02d}_normal.png``   256x256 world-space normals (for "analytic" mode)
  - ``views.json``           camera intrinsics + per-view extrinsics

The analytic-normal output is used when config.normal_source="analytic"
(the "mesh_rendered_views" supervision mode that skips StableNormal
noise). For "paper" mode the normals are overwritten by a later
StableNormal pass on the RGB images.

Camera convention: OpenGL-style (−Z forward, +Y up). Extrinsics are
world→camera.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Need pyrender headless
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Ensure the clearmesh package is importable whether run on the pod or
# laptop
_CANDIDATE_ROOTS = [
    "/workspace/clearmesh",
    str(Path(__file__).resolve().parents[2]),
]
for _root in _CANDIDATE_ROOTS:
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)

import numpy as np
import pyrender
import trimesh
from PIL import Image


# ---------------------------------------------------------------------
# View set — 24 evenly-spaced on the unit sphere + top + bottom = 26
# ---------------------------------------------------------------------

def fibonacci_sphere(n: int) -> list[tuple[float, float, float]]:
    """n roughly-evenly-spaced unit vectors via Fibonacci lattice."""
    points = []
    phi_golden = math.pi * (math.sqrt(5.0) - 1.0)
    for i in range(n):
        y = 1.0 - (i / float(n - 1)) * 2.0          # [-1, 1]
        r = math.sqrt(1.0 - y * y)
        theta = phi_golden * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append((x, y, z))
    return points


def paper_view_directions(n_sphere: int = 24) -> list[tuple[float, float, float]]:
    """Paper §5.1: 24 sphere + top + bottom."""
    sphere = fibonacci_sphere(n_sphere)
    return sphere + [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]


def camera_pose_from_direction(direction, distance=2.0):
    """Camera world-space pose looking toward origin from `direction`."""
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / np.linalg.norm(direction)
    eye = direction * distance
    forward = -direction
    up_world = np.array([0.0, 1.0, 0.0])
    if abs(forward @ up_world) > 0.99:
        up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward   # pyrender: -Z is forward
    pose[:3, 3] = eye
    return pose


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def _normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fit mesh into [-0.5, 0.5]^3 (so 2-unit camera distance sees
    the whole object at yfov≈40°)."""
    mesh = mesh.copy()
    mesh.vertices -= mesh.centroid
    s = mesh.extents.max()
    if s > 0:
        mesh.vertices /= s
    return mesh


def render_views(
    mesh_path: str,
    out_dir: str,
    resolution: int = 256,
    distance: float = 2.0,
    yfov_deg: float = 40.0,
    n_sphere_views: int = 24,
    render_normals: bool = True,
    bg_color: tuple = (255, 255, 255, 0),
):
    """Write rgb/mask/normal PNGs for all 26 views + views.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh = _normalize_mesh(mesh)

    directions = paper_view_directions(n_sphere_views)
    yfov = math.radians(yfov_deg)

    # --- Single scene for RGB+mask; separate for normals ---
    scene_rgb = pyrender.Scene(
        ambient_light=(0.35, 0.35, 0.35),
        bg_color=bg_color,
    )
    rmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene_rgb.add(rmesh)

    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    cam_node = scene_rgb.add(cam, pose=np.eye(4))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
    light_node = scene_rgb.add(light, pose=np.eye(4))

    renderer = pyrender.OffscreenRenderer(resolution, resolution)

    views = []
    for i, d in enumerate(directions):
        pose = camera_pose_from_direction(d, distance=distance)
        scene_rgb.set_pose(cam_node, pose)
        scene_rgb.set_pose(light_node, pose)

        color, depth = renderer.render(scene_rgb)
        rgb = color[..., :3]
        mask = (depth > 0).astype(np.uint8) * 255

        Image.fromarray(rgb).save(out / f"{i:02d}_rgb.png")
        Image.fromarray(mask).save(out / f"{i:02d}_mask.png")

        # --- Normals: render a normal-shaded version of the mesh ---
        if render_normals:
            normal_map = _render_world_normals(mesh, pose, resolution, yfov)
            Image.fromarray(normal_map).save(out / f"{i:02d}_normal.png")

        views.append({
            "index": i,
            "direction": list(d),
            "pose_world_from_camera": pose.tolist(),
            "resolution": resolution,
            "yfov_rad": yfov,
        })

    renderer.delete()

    with open(out / "views.json", "w") as f:
        json.dump({
            "views": views,
            "camera": {"yfov_rad": yfov, "distance": distance, "resolution": resolution},
            "n_views": len(directions),
        }, f, indent=2)

    print(f"[render_views] wrote {len(directions)} views to {out}")


def _render_world_normals(mesh, pose, resolution, yfov):
    """Render world-space normals by baking per-vertex RGB = (n+1)/2
    then alpha-compositing. Clean for analytic supervision."""
    # Compute vertex normals
    if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
        mesh = mesh.copy()
        mesh.fix_normals()
    n = np.asarray(mesh.vertex_normals, dtype=np.float32)
    # Encode [-1,1] → [0,1]
    rgb = ((n + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    # Attach as vertex colors
    colored = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces, process=False,
        vertex_colors=np.concatenate([rgb, np.full((len(rgb), 1), 255, dtype=np.uint8)], axis=-1),
    )

    scene = pyrender.Scene(
        ambient_light=(1.0, 1.0, 1.0),  # no lighting shading on normals
        bg_color=(0, 0, 0, 0),
    )
    rm = pyrender.Mesh.from_trimesh(colored, smooth=True)
    scene.add(rm)
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    scene.add(cam, pose=pose)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    color, _ = renderer.render(
        scene, flags=pyrender.constants.RenderFlags.FLAT,
    )
    renderer.delete()
    return color[..., :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to GLB/OBJ/PLY")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--distance", type=float, default=2.0)
    ap.add_argument("--yfov-deg", type=float, default=40.0)
    ap.add_argument("--n-sphere-views", type=int, default=24)
    ap.add_argument("--no-normals", action="store_true")
    args = ap.parse_args()

    render_views(
        args.input, args.out,
        resolution=args.resolution, distance=args.distance,
        yfov_deg=args.yfov_deg, n_sphere_views=args.n_sphere_views,
        render_normals=not args.no_normals,
    )


if __name__ == "__main__":
    main()
