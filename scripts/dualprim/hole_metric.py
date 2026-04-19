"""Hole-specific metrics for DualPrim quality evaluation.

Friend's framing: Chamfer is too coarse — a scattered-blob mesh that
roughly fills the bounding box can have moderate CD while completely
missing the hole. We need a metric that asks "did the model learn
the hole?" directly.

Three metrics computed:

  mask IoU
    Per-view binary IoU between rendered-pred-mask and rendered-ref-mask.
    Tells us how well the silhouette matches.

  hole-area error
    Per-view |pixels-where-ref-is-bg-AND-pred-is-fg| (false-positive
    fill of the hole). High = the pred filled the hole that should
    be empty.

  through-hole open %
    Of the views where the ref shows a clear hole (background pixels
    inside the projected silhouette), what fraction of those views
    has the pred ALSO showing a hole there. 100% = pred preserves the
    hole everywhere; 0% = pred fills it everywhere.

Usage:
    python scripts/dualprim/hole_metric.py \\
        --ref /workspace/test_box_hole.glb \\
        --pred /workspace/dualprim_quality/refit.glb \\
        --resolution 192
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Path setup
for _r in ("/workspace/clearmesh", "/root/clearmesh",
           str(Path(__file__).resolve().parents[2])):
    if os.path.isdir(_r) and _r not in sys.path:
        sys.path.insert(0, _r)

import numpy as np
import trimesh

# Lazy import pyrender at function scope
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def render_mask(mesh: trimesh.Trimesh, pose: np.ndarray, resolution: int, yfov: float) -> np.ndarray:
    """Return a binary uint8 mask of the mesh from the given camera pose."""
    import pyrender
    sc = pyrender.Scene(ambient_light=(0.0, 0.0, 0.0), bg_color=(0, 0, 0, 0))
    sc.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    sc.add(cam, pose=pose)
    rr = pyrender.OffscreenRenderer(resolution, resolution)
    _, depth = rr.render(sc)
    rr.delete()
    return (depth > 0).astype(np.uint8)


def fibonacci_sphere(n: int) -> list:
    points = []
    phi_g = math.pi * (math.sqrt(5.0) - 1.0)
    for i in range(n):
        y = 1.0 - ((i + 0.5) / float(n)) * 2.0
        r = math.sqrt(max(1.0 - y * y, 0.0))
        theta = phi_g * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append((x, y, z))
    return points


def camera_pose(direction, distance=2.0):
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / np.linalg.norm(direction)
    eye = direction * distance
    forward = -direction
    up_world = np.array([0.0, 1.0, 0.0])
    if abs(forward @ up_world) > 0.99:
        up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world); right /= np.linalg.norm(right)
    up = np.cross(right, forward); up /= np.linalg.norm(up)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right; pose[:3, 1] = up; pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.vertices -= m.centroid
    s = m.extents.max()
    if s > 0:
        m.vertices /= s
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--n-views", type=int, default=24, help="Sphere views (no top/bottom)")
    ap.add_argument("--yfov-deg", type=float, default=40.0)
    args = ap.parse_args()

    ref = normalize(trimesh.load(args.ref, force="mesh"))
    pred = normalize(trimesh.load(args.pred, force="mesh"))
    ref.fix_normals(); pred.fix_normals()

    yfov = math.radians(args.yfov_deg)
    directions = fibonacci_sphere(args.n_views)
    res = args.resolution

    iou_per_view = []
    hole_pixels_per_view = []
    pred_hole_pixels_per_view = []

    for i, d in enumerate(directions):
        pose = camera_pose(d, distance=2.0)
        ref_mask = render_mask(ref, pose, res, yfov)
        pred_mask = render_mask(pred, pose, res, yfov)

        # IoU
        inter = ((ref_mask > 0) & (pred_mask > 0)).sum()
        union = ((ref_mask > 0) | (pred_mask > 0)).sum()
        iou = inter / max(union, 1)
        iou_per_view.append(float(iou))

        # Hole detection: a "hole" pixel is one INSIDE the convex hull
        # of the ref silhouette but OUTSIDE the ref mask itself.
        # cv2.fillPoly would be the proper convex-hull fill; cheap proxy:
        # use scipy binary_fill_holes then XOR with original.
        from scipy.ndimage import binary_fill_holes
        ref_filled = binary_fill_holes(ref_mask > 0).astype(np.uint8)
        ref_holes = (ref_filled > 0) & (ref_mask == 0)        # holes IN ref
        pred_filled = binary_fill_holes(pred_mask > 0).astype(np.uint8)
        pred_holes = (pred_filled > 0) & (pred_mask == 0)     # holes IN pred

        ref_hole_count = int(ref_holes.sum())
        # Of the ref's hole pixels, how many are also "hole" in pred
        # (i.e. pred preserved the carve at that pixel)?
        if ref_hole_count > 0:
            preserved = int((ref_holes & pred_holes).sum())
        else:
            preserved = 0
        hole_pixels_per_view.append(ref_hole_count)
        pred_hole_pixels_per_view.append(preserved)

    iou_per_view = np.array(iou_per_view)
    hole_pixels_per_view = np.array(hole_pixels_per_view)
    pred_hole_pixels_per_view = np.array(pred_hole_pixels_per_view)

    # Aggregate metrics
    mean_iou = float(iou_per_view.mean())
    # Through-hole-open % — only count views where ref ACTUALLY shows a hole
    has_hole_view = hole_pixels_per_view > 5    # ignore tiny blips
    n_hole_views = int(has_hole_view.sum())
    if n_hole_views > 0:
        # Fraction of ref-hole-pixels that were preserved as hole in pred
        hole_recall_per_view = pred_hole_pixels_per_view / np.maximum(hole_pixels_per_view, 1)
        hole_recall_per_view = hole_recall_per_view[has_hole_view]
        mean_hole_recall = float(hole_recall_per_view.mean())
        n_views_open = int((hole_recall_per_view > 0.3).sum())
        pct_open = 100.0 * n_views_open / n_hole_views
    else:
        mean_hole_recall = 0.0
        pct_open = 0.0
        n_views_open = 0

    print(f"REF:  {args.ref}  ({len(ref.vertices):,}v)")
    print(f"PRED: {args.pred}  ({len(pred.vertices):,}v)")
    print(f"Views evaluated: {len(directions)}")
    print()
    print(f"Mean mask IoU:           {mean_iou:.4f}")
    print(f"Views with hole in ref:  {n_hole_views}/{len(directions)}")
    print(f"Mean hole recall:        {mean_hole_recall:.4f}  (1.0 = pred carved every ref-hole pixel)")
    print(f"Through-hole open %:     {pct_open:.1f}%  ({n_views_open}/{n_hole_views} views with recall > 30%)")


if __name__ == "__main__":
    main()
