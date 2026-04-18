"""Sanity checks for camera / ray / mesh2sdf coordinate conventions.

Designed to run on the LAPTOP (no GPU, no pod credits) before kicking
off any pod training. Catches the runtime risks the friend flagged:

  1. Camera pose convention: render_views builds (world ← camera)
     poses in OpenGL style (−Z forward, +Y up). The ray sampler in
     run_canary._build_views_sampler must agree.

  2. mesh2sdf coordinate frame: returns positive-inside; we flip to
     negative-inside. Must also agree with the [-1, 1]^3 grid normalization
     that scene init uses.

  3. SQ implicit sign convention: f(p, Q) = ... − 1, so negative ⇒
     inside, positive ⇒ outside. P_E gates on this convention.

Each check prints PASS/FAIL and a diagnostic value. Non-zero exit on
any FAIL.

Run:
    python scripts/dualprim/check_conventions.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

_CANDIDATE_ROOTS = [
    "/workspace/clearmesh",
    str(Path(__file__).resolve().parents[2]),
]
for _r in _CANDIDATE_ROOTS:
    if os.path.isdir(_r) and _r not in sys.path:
        sys.path.insert(0, _r)

import numpy as np
import torch
import trimesh


FAILURES: list[str] = []


def _check(name: str, cond: bool, detail: str = ""):
    tag = "PASS" if cond else "FAIL"
    line = f"  [{tag}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------
# 1. SQ implicit sign convention
# ---------------------------------------------------------------------

def check_sq_sign():
    print("\n[1] SQ implicit sign convention")
    from clearmesh.dualprim.superquadric import sq_implicit

    # Unit sphere at origin with ε=1
    t = torch.tensor([[0.0, 0.0, 0.0]])
    r = torch.tensor([[0.0, 0.0, 0.0]])
    s = torch.tensor([[0.5, 0.5, 0.5]])
    sh = torch.tensor([[1.0, 1.0]])

    # Inside point
    p_in = torch.tensor([[0.0, 0.0, 0.0]])
    f_in = sq_implicit(p_in, t, r, s, sh).item()
    _check("inside point has f < 0", f_in < 0, f"f={f_in:.3f}")

    # Surface point (at scale 0.5 along +X)
    p_surf = torch.tensor([[0.5, 0.0, 0.0]])
    f_surf = sq_implicit(p_surf, t, r, s, sh).item()
    _check("surface point has f ≈ 0", abs(f_surf) < 0.02, f"f={f_surf:.3f}")

    # Outside point
    p_out = torch.tensor([[1.0, 0.0, 0.0]])
    f_out = sq_implicit(p_out, t, r, s, sh).item()
    _check("outside point has f > 0", f_out > 0, f"f={f_out:.3f}")


# ---------------------------------------------------------------------
# 2. P_E gate on hand-placed PSQ + NSQ
# ---------------------------------------------------------------------

def check_pe_gate():
    print("\n[2] P_E gate behavior")
    from clearmesh.dualprim.superquadric import (
        sq_implicit, effectiveness_probability, combined_field,
    )

    psq_t = torch.tensor([[0.0, 0.0, 0.0]])
    psq_s = torch.tensor([[0.5, 0.5, 0.5]])
    nsq_s = torch.tensor([[0.3, 0.3, 0.3]])
    sh = torch.tensor([[1.0, 1.0]])
    rot = torch.tensor([[0.0, 0.0, 0.0]])
    theta = torch.tensor([0.5])

    # Inside both — P_E should be high, combined field flips sign
    p0 = torch.tensor([[0.0, 0.0, 0.0]])
    f_p = sq_implicit(p0, psq_t, rot, psq_s, sh)
    f_n = sq_implicit(p0, psq_t, rot, nsq_s, sh)
    p_e = effectiveness_probability(f_p, f_n, theta).item()
    f_c = combined_field(f_p, f_n, effectiveness_probability(f_p, f_n, theta)).item()
    _check("deep-inside-both: P_E > 0.5", p_e > 0.5, f"P_E={p_e:.3f}")
    _check("deep-inside-both: f_combined > 0 (carved)", f_c > 0, f"f_c={f_c:.3f}")

    # Inside PSQ only — P_E ≈ 0
    p1 = torch.tensor([[0.4, 0.0, 0.0]])
    f_p = sq_implicit(p1, psq_t, rot, psq_s, sh)
    f_n = sq_implicit(p1, psq_t, rot, nsq_s, sh)
    p_e = effectiveness_probability(f_p, f_n, theta).item()
    _check("inside-PSQ-only: P_E < 0.3", p_e < 0.3, f"P_E={p_e:.3f}")


# ---------------------------------------------------------------------
# 3. Camera pose ↔ ray sampler agreement
# ---------------------------------------------------------------------

def check_camera_ray_convention():
    print("\n[3] Camera pose ↔ ray sampler convention")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dualprim.render_views import camera_pose_from_direction

    # Place camera looking at origin from +Z — a ray through the image
    # center should go from eye TOWARD origin, i.e. in the −Z direction
    # in world coordinates.
    pose = camera_pose_from_direction((0.0, 0.0, 1.0), distance=2.0)
    eye = pose[:3, 3]
    _check(
        "camera eye is at +Z when looking from +Z",
        abs(eye[0]) < 1e-4 and abs(eye[1]) < 1e-4 and abs(eye[2] - 2.0) < 1e-4,
        f"eye={eye.tolist()}",
    )

    # Replicate the ray construction in run_canary._build_views_sampler:
    # cam_dirs[H/2, W/2] should be roughly (0, 0, −1).
    H = W = 32
    yfov = math.radians(40.0)
    fx = fy = 0.5 * H / math.tan(yfov / 2.0)
    ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    cam_dirs = np.stack([
        (xs - W / 2.0) / fx,
        -(ys - H / 2.0) / fy,
        -np.ones_like(xs),
    ], axis=-1)
    cam_dirs /= np.linalg.norm(cam_dirs, axis=-1, keepdims=True)
    center_dir = cam_dirs[H // 2, W // 2]
    _check(
        "image-center cam-space dir points toward -Z",
        center_dir[2] < -0.9 and abs(center_dir[0]) < 0.1 and abs(center_dir[1]) < 0.1,
        f"cam_dir={center_dir.tolist()}",
    )

    # Apply the pose rotation to that cam-dir and confirm it points
    # toward origin in world space (from the +Z eye).
    R = pose[:3, :3]
    world_dir = R @ center_dir
    _check(
        "world-space ray from +Z eye through image-center points to −Z",
        world_dir[2] < -0.9,
        f"world_dir={world_dir.tolist()}",
    )

    # The ray from eye + t * world_dir should pass through origin near
    # t = 2.0 (the camera distance)
    t_hit = -eye[2] / world_dir[2]
    hit = eye + t_hit * world_dir
    _check(
        "ray hits origin (within 1e-3)",
        np.linalg.norm(hit) < 1e-3,
        f"hit={hit.tolist()}, t={t_hit:.3f}",
    )


# ---------------------------------------------------------------------
# 4. mesh2sdf coordinate frame + sign
# ---------------------------------------------------------------------

def check_mesh2sdf_frame():
    print("\n[4] mesh2sdf coordinate frame (if installed)")
    try:
        import mesh2sdf
    except ImportError:
        print("  [SKIP] mesh2sdf not installed; will only matter on pod")
        return

    # Build a unit sphere mesh, normalize to [-0.98, 0.98]^3
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    sphere.vertices -= sphere.centroid
    s = sphere.extents.max()
    sphere.vertices *= (2.0 / s) * 0.98

    # Run mesh2sdf at res=32
    res = 32
    verts = np.asarray(sphere.vertices, dtype=np.float32)
    faces = np.asarray(sphere.faces, dtype=np.int32)
    sdf_pos_in = mesh2sdf.compute(
        verts, faces, size=res, fix=False, level=2.0 / res, return_mesh=False,
    )
    # We flip: paper convention is negative-inside
    sdf = -sdf_pos_in

    # Centre voxel of the grid should be deeply inside the sphere
    mid = res // 2
    _check(
        "centre of [-1,1]^3 grid is inside sphere (SDF < 0)",
        sdf[mid, mid, mid] < -0.5,
        f"sdf_mid={sdf[mid, mid, mid]:.3f}",
    )
    # Corner should be outside
    _check(
        "corner (+1, +1, +1) is outside sphere (SDF > 0)",
        sdf[-1, -1, -1] > 0.0,
        f"sdf_corner={sdf[-1, -1, -1]:.3f}",
    )


# ---------------------------------------------------------------------
# 5. Fibonacci sphere + paper_view_directions distinctness
# ---------------------------------------------------------------------

def check_view_directions():
    print("\n[5] 26-view set has no duplicates")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dualprim.render_views import paper_view_directions

    views = paper_view_directions()
    _check("total 26 views", len(views) == 26, f"got {len(views)}")

    # Check min pairwise angular distance
    import itertools
    V = np.asarray(views, dtype=np.float64)
    V /= np.linalg.norm(V, axis=-1, keepdims=True)
    min_cos = -1.0
    for i, j in itertools.combinations(range(len(V)), 2):
        c = V[i] @ V[j]
        min_cos = max(min_cos, c)
    min_deg = math.degrees(math.acos(min(max(min_cos, -1.0), 1.0)))
    _check(
        "all view pairs are >5° apart",
        min_deg > 5.0,
        f"closest pair = {min_deg:.2f}°",
    )


# ---------------------------------------------------------------------
# 6. End-to-end loss gradient sanity
# ---------------------------------------------------------------------

def check_end_to_end_gradients():
    print("\n[6] End-to-end gradient flow (tiny CPU)")
    from clearmesh.dualprim import DualPrimConfig, init_scene, RaySampleBatch
    from clearmesh.dualprim.renderer import render_rays
    from clearmesh.dualprim.losses import total_loss, loss_tsdf

    config = DualPrimConfig(
        num_primitives_init=3, num_samples_per_ray=8,
        num_iterations=1, pruning_interval=1000,
    )
    scene = init_scene(config, device="cpu")

    # Render loss path
    origins = torch.zeros(4, 3)
    dirs = torch.nn.functional.normalize(
        torch.tensor([
            [0.0, 0.0, -1.0], [0.1, 0.0, -1.0],
            [0.0, 0.1, -1.0], [0.1, 0.1, -1.0],
        ]), dim=-1,
    )
    origins[:, 2] = 2.0   # camera at z=2 pointing toward origin
    out = render_rays(
        scene, origins, dirs,
        num_samples=config.num_samples_per_ray,
        near=config.near_plane, far=config.far_plane,
    )
    rgb_gt = torch.full_like(out.rgb, 0.5)
    mask_gt = torch.ones(4)
    normals_gt = torch.full_like(out.normals, 0.0)
    loss, parts = total_loss(scene, out, rgb_gt, mask_gt, normals_gt)
    loss.backward()
    gnorm = scene.params.grad.norm().item()
    _check(
        "total_loss produces nonzero grad (render path)",
        gnorm > 1e-8,
        f"grad_norm={gnorm:.4f}, loss={loss.item():.4f}",
    )

    # TSDF loss path
    scene.params.grad = None
    qp = torch.randn(64, 3) * 0.4
    tg = torch.linalg.norm(qp, dim=-1) - 0.3
    l = loss_tsdf(scene, qp, tg)
    l.backward()
    gnorm2 = scene.params.grad.norm().item()
    _check(
        "loss_tsdf produces nonzero grad (mesh_fit path)",
        gnorm2 > 1e-8,
        f"grad_norm={gnorm2:.4f}, loss={l.item():.4f}",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("DualPrim convention sanity checks\n" + "=" * 50)
    check_sq_sign()
    check_pe_gate()
    check_camera_ray_convention()
    check_mesh2sdf_frame()
    check_view_directions()
    check_end_to_end_gradients()

    print()
    print("=" * 50)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
