"""Render a simple shaded mesh comparison with matched viewpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

_CANDIDATE_ROOTS = [
    "/workspace/clearmesh",
    str(Path(__file__).resolve().parents[2]),
]
for _root in _CANDIDATE_ROOTS:
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)


def _load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            return trimesh.Trimesh()
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _prepare_mesh(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    out = mesh.copy()
    if len(out.faces) > max_faces:
        step = max(1, len(out.faces) // max_faces)
        out = trimesh.Trimesh(
            vertices=out.vertices.copy(),
            faces=out.faces[::step].copy(),
            process=False,
        )
        out.remove_unreferenced_vertices()
    return out


def _equal_axes(ax, points: np.ndarray):
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins)) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _add_mesh(ax, mesh: trimesh.Trimesh, base_color: str):
    if len(mesh.faces) == 0:
        ax.set_axis_off()
        return
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    tris = verts[faces]
    normals = mesh.face_normals
    light_dir = np.array([0.45, 0.35, 0.82], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    shade = np.clip(normals @ light_dir, 0.0, 1.0)
    base_rgb = np.array(matplotlib.colors.to_rgb(base_color), dtype=np.float32)
    colors = np.clip((0.35 + 0.65 * shade)[:, None] * base_rgb[None, :], 0.0, 1.0)
    poly = Poly3DCollection(
        tris,
        facecolors=colors,
        linewidths=0.05,
        edgecolors=(0.15, 0.15, 0.15, 0.08),
        alpha=1.0,
    )
    ax.add_collection3d(poly)
    ax.set_axis_off()


def render_mesh_compare(
    target_mesh_path: str | Path,
    coarse_mesh_path: str | Path,
    refined_mesh_path: str | Path,
    out_path: str | Path,
    *,
    max_faces: int = 12000,
) -> None:
    target = _prepare_mesh(_load_mesh(target_mesh_path), max_faces=max_faces)
    coarse = _prepare_mesh(_load_mesh(coarse_mesh_path), max_faces=max_faces)
    refined = _prepare_mesh(_load_mesh(refined_mesh_path), max_faces=max_faces)

    all_points = [m.vertices for m in (target, coarse, refined) if len(m.vertices)]
    combined = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3), dtype=np.float32)

    fig = plt.figure(figsize=(13, 7))
    views = [(20, 35), (12, -60)]
    titles = ["target", "r6 / Stage B mesh", "Stage C preview"]
    meshes = [target, coarse, refined]
    colors = ["#596272", "#7396c1", "#cf7a57"]

    for row, (elev, azim) in enumerate(views):
        for col, (mesh, title, color) in enumerate(zip(meshes, titles, colors)):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            _add_mesh(ax, mesh, color)
            _equal_axes(ax, combined)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, fontsize=10)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mesh", required=True)
    ap.add_argument("--coarse-mesh", required=True)
    ap.add_argument("--refined-mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-faces", type=int, default=12000)
    args = ap.parse_args()
    render_mesh_compare(
        args.target_mesh,
        args.coarse_mesh,
        args.refined_mesh,
        args.out,
        max_faces=args.max_faces,
    )
    print(f"[mesh-compare] out={args.out}")


if __name__ == "__main__":
    main()
