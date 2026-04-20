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
import trimesh
from PIL import Image

# pyrender is imported at function scope so the geometry helpers
# (fibonacci_sphere / paper_view_directions) can be imported on
# machines without GL / EGL (e.g. a headless laptop for unit tests).


# ---------------------------------------------------------------------
# View set — 24 evenly-spaced on the unit sphere + top + bottom = 26
# ---------------------------------------------------------------------

def fibonacci_sphere(n: int) -> list[tuple[float, float, float]]:
    """n roughly-evenly-spaced unit vectors via Fibonacci lattice.

    Uses the (i + 0.5) / n offset so the sampled points do NOT coincide
    with the exact poles — that lets callers append separate top/bottom
    views without duplicating them, which matches the paper's
    "24 sphere + top + bottom = 26" setup.
    """
    points = []
    phi_golden = math.pi * (math.sqrt(5.0) - 1.0)
    for i in range(n):
        y = 1.0 - ((i + 0.5) / float(n)) * 2.0      # (-1, 1), excludes poles
        r = math.sqrt(max(1.0 - y * y, 0.0))
        theta = phi_golden * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append((x, y, z))
    return points


def paper_view_directions(n_sphere: int = 24) -> list[tuple[float, float, float]]:
    """Paper §5.1: 24 sphere + top + bottom = 26 unique directions.

    The fibonacci_sphere() above excludes the poles, so appending
    (0, 1, 0) and (0, -1, 0) here does not duplicate any sphere view.
    """
    sphere = fibonacci_sphere(n_sphere)
    return sphere + [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]


def hole_axis_ring_directions(
    hole_axis: int, n_ring: int = 12, tilt_deg: float = 15.0,
) -> list[tuple[float, float, float]]:
    """Directions clustered around a hole axis.

    Same shape as in hole_metric.py: 2*n_ring directions on small cones
    (±axis, tilt=15°). All see through the hole. Used to augment the
    paper's 26-view training set when we know a specific axis is
    topologically important.
    """
    import math as _m
    tilt = _m.radians(tilt_deg)
    dirs = []
    for sign in (1.0, -1.0):
        for i in range(n_ring):
            phi = 2.0 * _m.pi * i / n_ring
            d = [0.0, 0.0, 0.0]
            d[hole_axis] = sign * _m.cos(tilt)
            perp_axes = [a for a in range(3) if a != hole_axis]
            d[perp_axes[0]] = _m.sin(tilt) * _m.cos(phi)
            d[perp_axes[1]] = _m.sin(tilt) * _m.sin(phi)
            dirs.append(tuple(d))
    return dirs


def detect_hole_axis(mesh: trimesh.Trimesh, probe_res: int = 256) -> int:
    """Return 0/1/2 for X/Y/Z — the axis exposing the most through-hole.

    Cheap: renders the mesh silhouette from each axis, counts "hole
    pixels" = (binary_fill_holes(mask) AND NOT mask), picks the max.
    """
    import pyrender
    from scipy.ndimage import binary_fill_holes
    yfov = math.radians(40.0)
    m = mesh.copy()
    m.vertices -= m.centroid
    s = m.extents.max()
    if s > 0: m.vertices /= s
    best_axis, best_count = 0, -1
    for axis in range(3):
        d = [0.0, 0.0, 0.0]; d[axis] = 1.0
        pose = camera_pose_from_direction(d, distance=2.0)
        sc = pyrender.Scene(ambient_light=(0, 0, 0), bg_color=(0, 0, 0, 0))
        sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        sc.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0), pose=pose)
        rr = pyrender.OffscreenRenderer(probe_res, probe_res)
        _, depth = rr.render(sc); rr.delete()
        mask = (depth > 0)
        filled = binary_fill_holes(mask)
        hole_pixels = int((filled & ~mask).sum())
        if hole_pixels > best_count:
            best_count, best_axis = hole_pixels, axis
    return best_axis


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
    add_hole_axis_views: bool = False,
    n_hole_ring: int = 12,
    hole_tilt_deg: float = 15.0,
):
    """Write rgb/mask/normal PNGs for all 26 views + views.json.

    Uses the same "fresh scene per view" pattern as
    ``scripts/demo_end_to_end.render_mesh`` — set_pose on a persistent
    camera/light node would be cheaper but on some Vast.ai images it
    triggers ``EGL_BAD_SURFACE`` during the second-onwards render.
    Recreating the scene each call sidesteps the problem at ~50 ms/view
    cost.
    """
    import pyrender
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh = _normalize_mesh(mesh)
    mesh.fix_normals()   # ensure vertex_normals are present for _render_world_normals

    directions = paper_view_directions(n_sphere_views)
    if add_hole_axis_views:
        hole_axis = detect_hole_axis(mesh)
        print(f"[render_views] auto-detected hole axis: {'XYZ'[hole_axis]}, "
              f"adding {2 * n_hole_ring} ring views at tilt={hole_tilt_deg}°")
        directions = directions + hole_axis_ring_directions(
            hole_axis, n_ring=n_hole_ring, tilt_deg=hole_tilt_deg,
        )
    yfov = math.radians(yfov_deg)

    views = []
    for i, d in enumerate(directions):
        pose = camera_pose_from_direction(d, distance=distance)

        # --- Fresh scene per view for RGB+mask (EGL-stable) ---
        scene_rgb = pyrender.Scene(
            ambient_light=(0.35, 0.35, 0.35),
            bg_color=bg_color,
        )
        scene_rgb.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
        scene_rgb.add(cam, pose=pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
        scene_rgb.add(light, pose=pose)

        renderer = pyrender.OffscreenRenderer(resolution, resolution)
        color, depth = renderer.render(scene_rgb)
        renderer.delete()

        rgb = color[..., :3]
        mask = (depth > 0).astype(np.uint8) * 255

        Image.fromarray(rgb).save(out / f"{i:02d}_rgb.png")
        Image.fromarray(mask).save(out / f"{i:02d}_mask.png")

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
    import pyrender
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
    ap.add_argument("--add-hole-axis-views", action="store_true",
                    help="Auto-detect the hole axis and append 2*12 "
                         "extra training views clustered around it "
                         "(cone tilt 15°). Gives the optimizer much "
                         "stronger signal for carving through-holes.")
    ap.add_argument("--n-hole-ring", type=int, default=12)
    ap.add_argument("--hole-tilt-deg", type=float, default=15.0)
    args = ap.parse_args()

    render_views(
        args.input, args.out,
        resolution=args.resolution, distance=args.distance,
        yfov_deg=args.yfov_deg, n_sphere_views=args.n_sphere_views,
        render_normals=not args.no_normals,
        add_hole_axis_views=args.add_hole_axis_views,
        n_hole_ring=args.n_hole_ring,
        hole_tilt_deg=args.hole_tilt_deg,
    )


if __name__ == "__main__":
    main()
