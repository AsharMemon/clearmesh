"""Lightweight mesh preview rendering for workers without Blender."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh


def load_scene_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"no mesh geometry found in {path}")
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise ValueError(f"unsupported mesh type: {type(loaded).__name__}")


def render_wire_preview(
    mesh: trimesh.Trimesh,
    output: str | Path,
    *,
    size: int = 1024,
    elev: float = 28,
    azim: float = -38,
) -> Path:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0:
        raise ValueError("mesh has no vertices")
    vertices = vertices - vertices.mean(axis=0)
    scale = np.max(np.linalg.norm(vertices[:, :3], axis=1)) or 1.0
    vertices = (vertices @ _rotation_matrix(elev, azim).T) / scale
    xy = vertices[:, :2]
    xy[:, 1] *= -1
    xy = xy * (size * 0.38) + (size / 2)

    image = Image.new("RGB", (size, size), (248, 246, 240))
    draw = ImageDraw.Draw(image)
    faces = np.asarray(mesh.faces, dtype=int)
    if len(faces):
        z = vertices[faces].mean(axis=1)[:, 2]
        order = np.argsort(z)
        for face in faces[order]:
            pts = [tuple(xy[index]) for index in face]
            shade = int(208 + max(-0.5, min(0.5, vertices[face].mean(axis=0)[2])) * 36)
            draw.polygon(pts, fill=(shade, shade + 4, min(255, shade + 10)))
        for face in faces[order]:
            pts = [tuple(xy[index]) for index in face]
            draw.line([pts[0], pts[1], pts[2], pts[0]], fill=(64, 70, 72), width=1)
    else:
        for x, y in xy:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(64, 70, 72))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def render_wire_preview_file(input_mesh: str | Path, output: str | Path, *, size: int = 1024) -> Path:
    return render_wire_preview(load_scene_mesh(input_mesh), output, size=size)


def _rotation_matrix(elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(elev), -np.sin(elev)], [0, np.sin(elev), np.cos(elev)]])
    rz = np.array([[np.cos(azim), -np.sin(azim), 0], [np.sin(azim), np.cos(azim), 0], [0, 0, 1]])
    return rz @ rx

