"""Point-cloud bridge utilities for mesh-head bake-offs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class PointCloudSample:
    points: np.ndarray
    normals: np.ndarray
    face_indices: np.ndarray
    source_mesh: str
    sample_count: int


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file and collapse scenes into a single mesh."""
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"{path} contains no geometry")
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(loaded).__name__}")
    return loaded


def sample_mesh_surface(
    mesh_path: str | Path,
    count: int,
    seed: int = 0,
) -> PointCloudSample:
    """Sample a mesh surface with normals for mesh-head conditioning."""
    mesh_path = Path(mesh_path)
    mesh = load_mesh(mesh_path)
    if not len(mesh.faces):
        raise ValueError(f"{mesh_path} has no faces")
    points, face_indices = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = mesh.face_normals[face_indices]
    return PointCloudSample(
        points=np.asarray(points, dtype=np.float32),
        normals=np.asarray(normals, dtype=np.float32),
        face_indices=np.asarray(face_indices, dtype=np.int64),
        source_mesh=str(mesh_path),
        sample_count=count,
    )


def write_npz(sample: PointCloudSample, output_path: str | Path) -> Path:
    """Write point cloud as compressed NPZ."""
    output_path = Path(output_path).with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=sample.points,
        normals=sample.normals,
        face_indices=sample.face_indices,
        source_mesh=np.array(sample.source_mesh),
        sample_count=np.array(sample.sample_count, dtype=np.int64),
    )
    return output_path


def write_ply(sample: PointCloudSample, output_path: str | Path) -> Path:
    """Write point cloud as ASCII PLY with normals."""
    output_path = Path(output_path).with_suffix(".ply")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(sample.points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property float nx\nproperty float ny\nproperty float nz\n")
        handle.write("end_header\n")
        for point, normal in zip(sample.points, sample.normals):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{normal[0]:.8f} {normal[1]:.8f} {normal[2]:.8f}\n"
            )
    return output_path


def sample_to_files(
    mesh_path: str | Path,
    output_stem: str | Path,
    count: int,
    seed: int = 0,
    formats: tuple[str, ...] = ("npz", "ply"),
) -> dict[str, str]:
    """Sample a mesh and write one or more conditioning file formats."""
    sample = sample_mesh_surface(mesh_path, count=count, seed=seed)
    output_stem = Path(output_stem)
    paths: dict[str, str] = {}
    if "npz" in formats:
        paths["npz"] = str(write_npz(sample, output_stem))
    if "ply" in formats:
        paths["ply"] = str(write_ply(sample, output_stem))
    return paths


def sample_metadata(sample_paths: dict[str, str], **extra: Any) -> dict[str, Any]:
    """Build stable metadata for manifests and job artifacts."""
    return {"point_clouds": sample_paths, **extra}
