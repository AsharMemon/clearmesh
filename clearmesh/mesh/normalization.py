"""Normalize arbitrary generator meshes into a stable control surface.

This is the practical bridge between visual 3D generators and topology-aware
mesh heads. When Open3D is available, we reconstruct a coherent surface from
sampled oriented points. Otherwise we fall back to conservative cleanup so the
pipeline still runs in CPU-only development environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from clearmesh.eval.mesh_quality import load_mesh
from clearmesh.mesh.cleanup import CleanupOptions, cleanup_mesh
from clearmesh.product.mesh_passport import fast_passport_metrics


@dataclass(frozen=True)
class SurfaceNormalizationOptions:
    target_faces: int = 50_000
    sample_points: int = 120_000
    poisson_depth: int = 8
    density_quantile: float = 0.02
    min_component_faces: int = 32
    keep_largest_components: int = 32
    engine: str = "auto"
    orient_normals: bool = False


@dataclass
class SurfaceNormalizationReport:
    input_path: str
    output_path: str
    engine: str
    target_faces: int
    sample_points: int
    input_metrics: dict[str, Any]
    output_metrics: dict[str, Any]
    notes: list[str]


def normalize_surface_file(
    input_path: str | Path,
    output_path: str | Path,
    options: SurfaceNormalizationOptions | None = None,
) -> SurfaceNormalizationReport:
    options = options or SurfaceNormalizationOptions()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_metrics = fast_passport_metrics(input_path)
    mesh = load_mesh(input_path)
    notes: list[str] = []

    engine = options.engine
    normalized: trimesh.Trimesh
    if engine in {"auto", "poisson"}:
        try:
            normalized = _poisson_reconstruct(mesh, options)
            engine = "poisson"
        except Exception as exc:  # noqa: BLE001 - fallback keeps worker robust.
            if options.engine == "poisson":
                raise
            notes.append(f"poisson unavailable or failed: {type(exc).__name__}: {exc}")
            normalized = _cleanup_fallback(mesh, options)
            engine = "cleanup"
    elif engine == "cleanup":
        normalized = _cleanup_fallback(mesh, options)
    else:
        raise ValueError(f"unsupported surface normalization engine: {engine}")

    normalized.export(output_path)
    output_metrics = fast_passport_metrics(output_path)
    return SurfaceNormalizationReport(
        input_path=str(input_path),
        output_path=str(output_path),
        engine=engine,
        target_faces=options.target_faces,
        sample_points=options.sample_points,
        input_metrics=input_metrics,
        output_metrics=output_metrics,
        notes=notes,
    )


def normalization_options_from_metadata(metadata: dict[str, Any]) -> SurfaceNormalizationOptions:
    return SurfaceNormalizationOptions(
        target_faces=int(metadata.get("surface_target_faces", 50_000)),
        sample_points=int(metadata.get("surface_sample_points", 120_000)),
        poisson_depth=int(metadata.get("surface_poisson_depth", 8)),
        density_quantile=float(metadata.get("surface_density_quantile", 0.02)),
        min_component_faces=int(metadata.get("surface_min_component_faces", 32)),
        keep_largest_components=int(metadata.get("surface_keep_largest_components", 32)),
        engine=str(metadata.get("surface_engine", "auto")),
        orient_normals=_bool(metadata.get("surface_orient_normals", False)),
    )


def _cleanup_fallback(mesh: trimesh.Trimesh, options: SurfaceNormalizationOptions) -> trimesh.Trimesh:
    cleaned, _ = cleanup_mesh(
        mesh,
        CleanupOptions(
            min_component_faces=options.min_component_faces,
            keep_largest_components=options.keep_largest_components,
            fill_holes=True,
            fix_normals=True,
            merge_vertices=True,
        ),
    )
    if len(cleaned.faces) > options.target_faces:
        cleaned = _simplify_trimesh(cleaned, options.target_faces)
    return cleaned


def _poisson_reconstruct(mesh: trimesh.Trimesh, options: SurfaceNormalizationOptions) -> trimesh.Trimesh:
    import open3d as o3d  # type: ignore

    mesh = mesh.copy()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")

    count = max(1_000, int(options.sample_points))
    points, face_indices = trimesh.sample.sample_surface(mesh, count)
    normals = mesh.face_normals[face_indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    if options.orient_normals:
        try:
            pcd.orient_normals_consistent_tangent_plane(32)
        except Exception:
            # Face normals from the source mesh are usually good enough for generated
            # assets; Open3D orientation can fail on very noisy point sets.
            pass

    reconstructed, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(options.poisson_depth),
    )
    densities_np = np.asarray(densities)
    if densities_np.size:
        threshold = np.quantile(densities_np, options.density_quantile)
        reconstructed.remove_vertices_by_mask(densities_np < threshold)

    bbox = pcd.get_axis_aligned_bounding_box()
    reconstructed = reconstructed.crop(bbox)
    vertices = np.asarray(reconstructed.vertices)
    faces = np.asarray(reconstructed.triangles)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError("poisson reconstruction returned an empty mesh")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if len(mesh.faces) > options.target_faces:
        mesh = _simplify_trimesh(mesh, options.target_faces)
    return mesh


def _simplify_trimesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=int(target_faces))
        if simplified is not None and len(simplified.faces) > 0:
            return simplified
    except TypeError:
        try:
            simplified = mesh.simplify_quadric_decimation(int(target_faces))
            if simplified is not None and len(simplified.faces) > 0:
                return simplified
        except Exception:
            pass
    except Exception:
        pass
    return mesh


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def report_to_dict(report: SurfaceNormalizationReport) -> dict[str, Any]:
    return asdict(report)
