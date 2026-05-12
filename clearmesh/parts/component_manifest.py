"""MeshMosaic-style component fallback for part manifests."""

from __future__ import annotations

from pathlib import Path

import trimesh

from clearmesh.eval.mesh_quality import load_mesh
from clearmesh.parts.manifest import PartRecord, write_parts_manifest
from clearmesh.pointcloud import sample_to_files


def write_component_parts_manifest(
    mesh_path: str | Path,
    output_dir: str | Path,
    *,
    max_parts: int = 8,
    min_faces: int = 32,
    point_count: int = 4096,
) -> Path:
    """Split connected components into a stable part manifest.

    This is not semantic OmniPart, but it gives MeshMosaic/part-aware downstream
    stages a component-aware contract whenever the official part model is absent.
    """

    mesh_path = Path(mesh_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = load_mesh(mesh_path)
    components = [component for component in mesh.split(only_watertight=False) if len(component.faces) >= min_faces]
    if not components:
        components = [mesh]
    components = sorted(components, key=lambda component: len(component.faces), reverse=True)[: max(1, max_parts)]
    parts: list[PartRecord] = []
    for index, component in enumerate(components):
        part_id = f"component_{index:03d}"
        part_dir = output_dir / part_id
        part_dir.mkdir(parents=True, exist_ok=True)
        part_mesh_path = part_dir / f"{part_id}.obj"
        component.export(part_mesh_path)
        sample_count = max(64, min(int(point_count), max(len(component.faces) * 8, 64)))
        point_paths = sample_to_files(part_mesh_path, part_dir / part_id, count=sample_count)
        bounds = component.bounds.tolist() if isinstance(component, trimesh.Trimesh) and len(component.vertices) else None
        bbox = [float(value) for row in bounds for value in row] if bounds else None
        parts.append(
            PartRecord(
                id=part_id,
                label=f"Component {index + 1}",
                point_cloud_path=point_paths.get("ply"),
                proxy_mesh_path=str(part_mesh_path),
                bbox=bbox,
                metadata={"source": "connected_component", "face_count": len(component.faces)},
            )
        )
    return write_parts_manifest(output_dir / "parts_manifest.json", parts, source=str(mesh_path))
