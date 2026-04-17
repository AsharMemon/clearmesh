"""Demo: RANSAC primitive fitting + Liu 2023 watertight BLP selection.

The Selection Module (Liu 2023, §3.3) is source-agnostic — we can feed
it any pool of candidate faces. For this demo we use a cheap RANSAC
plane/sphere/cylinder detector via ``pyransac3d`` to produce the pool,
then run the BLP to pick a watertight subset.

This isn't meant to match the paper's published numbers (they use HPNet
for much better segmentation); it's a smallest-viable demo that
validates the selection code end-to-end.

Usage on pod:
    pip install pyransac3d pulp
    # Optional: pip install gurobipy (commercial / free academic licence)
    cd /workspace/clearmesh
    python scripts/demo_watertight_select.py \\
        --input /workspace/demo_text_to_3d_qwen/04_final.glb \\
        --out /workspace/demo_watertight_select \\
        --max-primitives 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

for p in ("/workspace/clearmesh",):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import trimesh


# ---------------------------------------------------------------------
# RANSAC-based candidate face generation
# ---------------------------------------------------------------------

def _plane_mesh(a: float, b: float, c: float, d: float, inlier_pts: np.ndarray) -> trimesh.Trimesh:
    """Build a small rectangular patch lying on the plane n.x + d = 0 that
    covers the AABB of the inliers, projected onto the plane.
    """
    n = np.array([a, b, c], dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    # Project inliers onto the plane
    proj = inlier_pts - (inlier_pts @ n - d)[:, None] * n
    # Two orthonormal in-plane axes
    ax = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, ax)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    uv = np.stack([proj @ u, proj @ v], axis=-1)
    umin, umax = uv[:, 0].min(), uv[:, 0].max()
    vmin, vmax = uv[:, 1].min(), uv[:, 1].max()
    # 4 corners of the bounding rectangle
    centre = proj.mean(axis=0)
    corners = np.stack([
        centre + u * (umin - centre @ u) + v * (vmin - centre @ v),
        centre + u * (umax - centre @ u) + v * (vmin - centre @ v),
        centre + u * (umax - centre @ u) + v * (vmax - centre @ v),
        centre + u * (umin - centre @ u) + v * (vmax - centre @ v),
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    m = trimesh.Trimesh(vertices=corners, faces=faces, process=False)
    return m


def _cylinder_mesh(
    centre: np.ndarray, axis: np.ndarray, radius: float, inlier_pts: np.ndarray,
) -> trimesh.Trimesh:
    """Build a capped cylinder from RANSAC parameters, length covering
    the inliers projected onto ``axis``.
    """
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    proj = (inlier_pts - centre) @ axis
    lo, hi = proj.min(), proj.max()
    length = max(hi - lo, 1e-3)
    c_new = centre + axis * ((lo + hi) / 2.0)
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
    # Orient along axis — align Z to axis
    z = np.array([0, 0, 1.0])
    v = np.cross(z, axis)
    s = np.linalg.norm(v)
    if s > 1e-9:
        c = z @ axis
        V = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        R = np.eye(3) + V + V @ V * ((1 - c) / max(s * s, 1e-12))
    else:
        R = np.eye(3)
    cyl.apply_transform(np.block([[R, c_new.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]]))
    return cyl


def _sphere_mesh(centre: np.ndarray, radius: float) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    m.apply_translation(centre)
    return m


def generate_candidates_ransac(
    points: np.ndarray,
    max_primitives: int = 25,
    plane_thresh: float = 0.01,
    sphere_thresh: float = 0.01,
    cyl_thresh: float = 0.01,
    min_inliers: int = 100,
    verbose: bool = True,
) -> list:
    """Greedy multi-primitive RANSAC using pyransac3d, then carve inliers
    from the working point set. Tries plane -> cylinder -> sphere each
    round until we hit max_primitives or no primitive gets above
    min_inliers.
    """
    try:
        import pyransac3d as pyrsc
    except ImportError:
        raise RuntimeError(
            "pyransac3d not installed. pip install pyransac3d"
        )

    remaining = points.copy()
    out = []
    for i in range(max_primitives):
        if len(remaining) < min_inliers:
            break

        best = None  # (n_inliers, prim_name, prim_mesh, inlier_indices)
        # Try plane
        try:
            p = pyrsc.Plane()
            eq, inl = p.fit(remaining, thresh=plane_thresh)
            if len(inl) > 0:
                a, b, c, d = eq
                if len(inl) > (best[0] if best else 0):
                    mesh = _plane_mesh(a, b, c, d, remaining[inl])
                    best = (len(inl), "plane", mesh, inl)
        except Exception as e:
            if verbose:
                print(f"  plane fail: {e}")

        # Try cylinder
        try:
            cy = pyrsc.Cylinder()
            centre, axis, radius, inl = cy.fit(remaining, thresh=cyl_thresh)
            if len(inl) > (best[0] if best else 0):
                mesh = _cylinder_mesh(np.asarray(centre), np.asarray(axis), float(radius), remaining[inl])
                best = (len(inl), "cylinder", mesh, inl)
        except Exception as e:
            if verbose:
                print(f"  cyl fail: {e}")

        # Try sphere
        try:
            sp = pyrsc.Sphere()
            centre, radius, inl = sp.fit(remaining, thresh=sphere_thresh)
            if len(inl) > (best[0] if best else 0):
                mesh = _sphere_mesh(np.asarray(centre), float(radius))
                best = (len(inl), "sphere", mesh, inl)
        except Exception as e:
            if verbose:
                print(f"  sphere fail: {e}")

        if best is None or best[0] < min_inliers:
            break
        n_in, name, mesh, inl = best
        out.append((name, mesh, n_in))
        if verbose:
            print(f"[ransac] #{i+1} {name:8s} inliers={n_in:,}, rem={len(remaining)-n_in:,}")
        # Carve inliers so next round finds a different primitive
        mask = np.ones(len(remaining), dtype=bool)
        mask[inl] = False
        remaining = remaining[mask]

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="/workspace/demo_watertight_select")
    ap.add_argument("--max-primitives", type=int, default=25)
    ap.add_argument("--n-points", type=int, default=40_000)
    ap.add_argument("--decimate-input", type=int, default=200_000)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--lambda-f", type=float, default=1.0)
    ap.add_argument("--lambda-ss", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from clearmesh.refit.watertight_select import select_watertight

    print(f"[demo] loading {args.input}")
    mesh = trimesh.load(args.input, force="mesh")
    if len(mesh.faces) > args.decimate_input:
        print(f"[demo] decimating to {args.decimate_input:,} faces")
        from clearmesh.mesh.repair import quadric_decimate
        mesh = quadric_decimate(mesh, target_faces=args.decimate_input)

    # Normalise to [-1, 1]^3
    c = mesh.bounding_box.centroid.copy()
    mesh.apply_translation(-c)
    ext = max(mesh.extents)
    if ext > 0:
        mesh.apply_scale(2.0 / ext)

    print(f"[demo] sampling {args.n_points:,} surface points")
    pts, _ = trimesh.sample.sample_surface(mesh, args.n_points)
    pts = np.asarray(pts, dtype=np.float32)

    print(f"[demo] RANSAC generating up to {args.max_primitives} candidates")
    t0 = time.time()
    cand_rs = generate_candidates_ransac(
        pts, max_primitives=args.max_primitives,
        plane_thresh=0.02, sphere_thresh=0.02, cyl_thresh=0.02,
        min_inliers=200,
    )
    print(f"[demo] RANSAC in {time.time()-t0:.1f}s, {len(cand_rs)} primitives")
    candidates = [c[1] for c in cand_rs]

    # Liu 2023 §3.2 — pairwise triangle splitting so candidate faces
    # share vertex coordinates at intersection lines. Without this the
    # BLP's watertightness constraint has nothing to operate on.
    print(f"[demo] splitting candidates at intersection planes")
    t0 = time.time()
    from clearmesh.refit import split_candidates
    candidates = split_candidates(candidates, verbose=True)
    print(f"[demo] split in {time.time()-t0:.1f}s")

    print(f"[demo] running BLP selection")
    t0 = time.time()
    result = select_watertight(
        candidates,
        pts,
        lambda_f=args.lambda_f,
        lambda_ss=args.lambda_ss,
        epsilon=args.epsilon,
        verbose=True,
    )
    print(f"[demo] select in {time.time()-t0:.1f}s")

    # Save
    for i, m in enumerate(candidates):
        m.export(os.path.join(args.out, f"cand_{i:02d}.glb"))
    result.selected_mesh.export(os.path.join(args.out, "selected.glb"))
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({
            "primitives": [c[0] for c in cand_rs],
            "inliers": [c[2] for c in cand_rs],
            "selected": result.selected_indices,
            "energies": {
                "total": result.energy_total,
                "fit": result.energy_fit,
                "sim": result.energy_similarity,
            },
            "solver": result.solver,
            "n_edges": result.n_edges_constrained,
            "timings": result.timings,
        }, f, indent=2)

    print("=" * 60)
    print(f"DONE")
    print(f"  candidates:  {result.n_candidates}")
    print(f"  selected:    {len(result.selected_indices)}")
    print(f"  edges:       {result.n_edges_constrained}")
    print(f"  solver:      {result.solver}")
    print(f"  E_f:         {result.energy_fit:.4f}")
    print(f"  E_ss:        {result.energy_similarity:.4f}")
    print(f"  output:      {os.path.join(args.out, 'selected.glb')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
