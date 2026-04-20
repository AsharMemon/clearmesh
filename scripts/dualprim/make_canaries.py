"""Generate DualPrim canary meshes via trimesh boolean operations.

Regenerates the 5 canary test meshes used by the autonomous_runner
(test_box_hole, test_stool, test_dumbbell, test_camera,
test_window_box) since they're not checked into git — the originals
were created on the vast.ai pod which is now exited.

Geometries reproduce the hand-authored meshes measured at:
  - test_box_hole.glb: 104v, extents [1.0, 0.6, 0.6], hole axis Y,
    hole radius ~0.15, box volume 0.318 (88% fill)

For the other canaries I use sensible defaults matching their
EXPERIMENT_LIBRARY descriptions. If exact-match matters later we
can calibrate from mesh-stats measurements.

Usage:
    python scripts/dualprim/make_canaries.py --out-dir /workspace
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def make_box_with_hole(
    box_extents: tuple = (1.0, 0.6, 0.6),
    hole_radius: float = 0.15,
    hole_axis: int = 1,  # 0=X, 1=Y, 2=Z
) -> trimesh.Trimesh:
    """Box with a cylindrical hole through it along the specified axis.

    Default: 1x0.6x0.6 box with Y-axis hole of radius 0.15 — matches
    the original test_box_hole.glb from the vast.ai pod.
    """
    box = trimesh.creation.box(extents=box_extents)
    # Cylinder long enough to punch through the box with margin
    cyl_height = box_extents[hole_axis] * 1.5
    cyl = trimesh.creation.cylinder(
        radius=hole_radius, height=cyl_height, sections=48,
    )
    # By default cylinder is along Z. Rotate if needed.
    if hole_axis != 2:
        if hole_axis == 0:
            rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        elif hole_axis == 1:
            rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        cyl.apply_transform(rot)

    result = box.difference(cyl, engine="manifold")
    return result


def make_window_box(
    box_extents: tuple = (1.0, 0.6, 0.6),
    window_extents: tuple = (0.3, 0.3, 1.2),
    window_axis: int = 0,  # 0=X-axis cut through two 0.6x0.6 faces
) -> trimesh.Trimesh:
    """Box with a rectangular window cut through it."""
    box = trimesh.creation.box(extents=box_extents)
    win = trimesh.creation.box(extents=window_extents)
    return box.difference(win, engine="manifold")


def make_stool(
    seat_radius: float = 0.35,
    seat_thickness: float = 0.06,
    seat_y: float = 0.35,
    leg_radius: float = 0.04,
    leg_length: float = 0.7,
    leg_spread: float = 0.28,
) -> trimesh.Trimesh:
    """Cylindrical seat + 4 vertical legs at the corners of a square."""
    seat = trimesh.creation.cylinder(
        radius=seat_radius, height=seat_thickness, sections=32,
    )
    # Rotate seat to lie flat (cylinder default along Z → along Y)
    seat.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    seat.apply_translation([0, seat_y, 0])

    meshes = [seat]
    for dx in (-leg_spread, leg_spread):
        for dz in (-leg_spread, leg_spread):
            leg = trimesh.creation.cylinder(
                radius=leg_radius, height=leg_length, sections=16,
            )
            leg.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]),
            )
            leg.apply_translation([dx, seat_y - seat_thickness / 2 - leg_length / 2, dz])
            meshes.append(leg)

    return trimesh.util.concatenate(meshes)


def make_dumbbell(
    ball_radius: float = 0.25,
    ball_offset: float = 0.5,  # distance of each ball center from origin along X
    rod_radius: float = 0.06,
    rod_length: float = 1.0,
) -> trimesh.Trimesh:
    """Two spheres connected by a cylindrical rod."""
    ball1 = trimesh.creation.icosphere(subdivisions=3, radius=ball_radius)
    ball1.apply_translation([-ball_offset, 0, 0])
    ball2 = trimesh.creation.icosphere(subdivisions=3, radius=ball_radius)
    ball2.apply_translation([ball_offset, 0, 0])
    rod = trimesh.creation.cylinder(
        radius=rod_radius, height=rod_length, sections=32,
    )
    # Cylinder default along Z → along X
    rod.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    return trimesh.util.concatenate([ball1, ball2, rod])


def make_camera(
    body_extents: tuple = (0.8, 0.5, 0.4),
    lens_radius: float = 0.15,
    lens_length: float = 0.25,
    lens_offset_x: float = 0.2,  # offset from body center along X
) -> trimesh.Trimesh:
    """Box body + cylindrical lens protruding from the front."""
    body = trimesh.creation.box(extents=body_extents)
    lens = trimesh.creation.cylinder(
        radius=lens_radius, height=lens_length, sections=32,
    )
    # Cylinder default along Z → protruding forward (+Z)
    lens.apply_translation([lens_offset_x, 0, body_extents[2] / 2 + lens_length / 2 - 0.02])
    return trimesh.util.concatenate([body, lens])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace",
                    help="Directory to write test_*.glb files")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meshes = {
        "test_box_hole.glb":   make_box_with_hole(),
        "test_window_box.glb": make_window_box(),
        "test_stool.glb":      make_stool(),
        "test_dumbbell.glb":   make_dumbbell(),
        "test_camera.glb":     make_camera(),
    }
    for name, m in meshes.items():
        path = out / name
        m.export(path)
        print(f"[make_canaries] wrote {path} ({len(m.vertices):,}v / {len(m.faces):,}f, "
              f"extents={m.extents}, watertight={m.is_watertight})")


if __name__ == "__main__":
    main()
