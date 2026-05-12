"""Mesh evaluation harness for ClearMesh mesh-head bake-offs.

The goal is not just visual similarity. These metrics make topology,
editability, and DCC roundtrip problems visible before we commit to a mesh head.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass
class MeshMetrics:
    path: str
    ok: bool
    error: str | None
    vertex_count: int | None = None
    face_count: int | None = None
    connected_components: int | None = None
    tiny_component_count: int | None = None
    watertight: bool | None = None
    winding_consistent: bool | None = None
    euler_number: int | None = None
    genus_estimate: float | None = None
    boundary_edge_count: int | None = None
    boundary_loop_count: int | None = None
    nonmanifold_edge_count: int | None = None
    nonmanifold_vertex_count: int | None = None
    self_intersections: int | None = None
    self_intersections_note: str | None = None
    surface_area: float | None = None
    volume: float | None = None
    mean_aspect_ratio: float | None = None
    p95_aspect_ratio: float | None = None
    max_aspect_ratio: float | None = None
    min_triangle_area: float | None = None
    degenerate_face_count: int | None = None
    valence_histogram: dict[str, int] | None = None
    pole_vertex_count: int | None = None
    quad_ratio: float | None = None
    subdivision_ok: bool | None = None
    blender_roundtrip_ok: bool | None = None
    blender_roundtrip_error: str | None = None


@dataclass
class PairMetrics:
    chamfer_l2: float | None
    hausdorff_l2: float | None
    chamfer_l2_normalized: float | None
    hausdorff_l2_normalized: float | None
    reference_bbox_diagonal: float | None
    normal_consistency: float | None
    surface_area_ratio: float | None
    volume_ratio: float | None


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh and collapse scenes to one mesh when needed."""
    loaded = trimesh.load(path, force="scene", skip_materials=True)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("scene contains no geometry")
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(loaded).__name__}")
    return loaded


def evaluate_mesh(path: str | Path, blender: str | None = None) -> dict[str, Any]:
    """Evaluate one generated mesh for production-relevant failure modes."""
    path = Path(path)
    try:
        mesh = load_mesh(path)
        metrics = _evaluate_loaded_mesh(mesh, path)
        if blender:
            ok, error = _check_blender_roundtrip(path, blender)
            metrics.blender_roundtrip_ok = ok
            metrics.blender_roundtrip_error = error
        return asdict(metrics)
    except Exception as exc:  # noqa: BLE001 - harness should report failures, not crash batches.
        return asdict(MeshMetrics(path=str(path), ok=False, error=f"{type(exc).__name__}: {exc}"))


def evaluate_mesh_pair(
    generated_path: str | Path,
    reference_path: str | Path,
    samples: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate generated-vs-reference geometry metrics."""
    generated = load_mesh(generated_path)
    reference = load_mesh(reference_path)
    return asdict(_evaluate_pair(generated, reference, samples=samples, seed=seed))


def _evaluate_loaded_mesh(mesh: trimesh.Trimesh, path: Path) -> MeshMetrics:
    mesh = mesh.copy()
    if not mesh.faces.size:
        raise ValueError("mesh has no faces")

    edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    boundary_edges = mesh.edges_unique[edge_counts == 1]
    nonmanifold_edges = mesh.edges_unique[edge_counts != 2]
    components = mesh.split(only_watertight=False)
    face_total = max(int(len(mesh.faces)), 1)
    tiny_components = sum(1 for component in components if len(component.faces) / face_total < 0.01)

    valences = _vertex_valences(mesh)
    aspect = _triangle_aspect_ratios(mesh)
    genus = None
    if mesh.is_watertight:
        # For orientable closed components: chi = 2C - 2G.
        genus = (2 * max(len(components), 1) - mesh.euler_number) / 2

    subdivision_ok = True
    try:
        _ = mesh.subdivide()
    except Exception:  # noqa: BLE001 - any subdivision failure is useful signal.
        subdivision_ok = False

    return MeshMetrics(
        path=str(path),
        ok=True,
        error=None,
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        connected_components=int(len(components)),
        tiny_component_count=int(tiny_components),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        euler_number=int(mesh.euler_number),
        genus_estimate=None if genus is None else float(genus),
        boundary_edge_count=int(len(boundary_edges)),
        boundary_loop_count=int(_boundary_loop_count(boundary_edges)),
        nonmanifold_edge_count=int(len(nonmanifold_edges)),
        nonmanifold_vertex_count=int(count_nonmanifold_vertices(mesh)),
        self_intersections=None,
        self_intersections_note="not computed locally; use Blender or libigl-backed extension for exact counts",
        surface_area=float(mesh.area),
        volume=float(mesh.volume) if mesh.is_watertight else None,
        mean_aspect_ratio=float(np.mean(aspect)),
        p95_aspect_ratio=float(np.percentile(aspect, 95)),
        max_aspect_ratio=float(np.max(aspect)),
        min_triangle_area=float(np.min(mesh.area_faces)),
        degenerate_face_count=int(np.sum(mesh.area_faces <= 1e-12)),
        valence_histogram={str(k): int(v) for k, v in _histogram(valences).items()},
        pole_vertex_count=int(np.sum((valences != 4) & (valences != 6))),
        quad_ratio=0.0,
        subdivision_ok=subdivision_ok,
    )


def _evaluate_pair(
    generated: trimesh.Trimesh,
    reference: trimesh.Trimesh,
    samples: int,
    seed: int,
) -> PairMetrics:
    gen_points, gen_face_idx = trimesh.sample.sample_surface(generated, samples, seed=seed)
    ref_points, ref_face_idx = trimesh.sample.sample_surface(reference, samples, seed=seed + 1)

    gen_to_ref_tree = cKDTree(ref_points)
    ref_to_gen_tree = cKDTree(gen_points)
    gen_to_ref_dist, gen_nearest_ref = gen_to_ref_tree.query(gen_points, k=1)
    ref_to_gen_dist, _ = ref_to_gen_tree.query(ref_points, k=1)

    gen_normals = generated.face_normals[gen_face_idx]
    ref_normals = reference.face_normals[ref_face_idx][gen_nearest_ref]
    normal_consistency = np.abs(np.einsum("ij,ij->i", gen_normals, ref_normals))

    chamfer_l2 = float(np.mean(gen_to_ref_dist**2) + np.mean(ref_to_gen_dist**2))
    hausdorff_l2 = float(max(np.max(gen_to_ref_dist), np.max(ref_to_gen_dist)))
    reference_diag = float(np.linalg.norm(reference.bounds[1] - reference.bounds[0]))
    if reference_diag > 1e-12:
        chamfer_l2_normalized = float(chamfer_l2 / (reference_diag**2))
        hausdorff_l2_normalized = float(hausdorff_l2 / reference_diag)
    else:
        chamfer_l2_normalized = None
        hausdorff_l2_normalized = None

    return PairMetrics(
        chamfer_l2=chamfer_l2,
        hausdorff_l2=hausdorff_l2,
        chamfer_l2_normalized=chamfer_l2_normalized,
        hausdorff_l2_normalized=hausdorff_l2_normalized,
        reference_bbox_diagonal=reference_diag,
        normal_consistency=float(np.mean(normal_consistency)),
        surface_area_ratio=_safe_ratio(generated.area, reference.area),
        volume_ratio=_safe_ratio(abs(generated.volume), abs(reference.volume))
        if generated.is_watertight and reference.is_watertight
        else None,
    )


def _vertex_valences(mesh: trimesh.Trimesh) -> np.ndarray:
    valences = np.zeros(len(mesh.vertices), dtype=np.int64)
    edges = mesh.edges_unique
    np.add.at(valences, edges[:, 0], 1)
    np.add.at(valences, edges[:, 1], 1)
    return valences


def count_nonmanifold_vertices(mesh: trimesh.Trimesh) -> int:
    """Count vertices whose one-ring link is not a single path or cycle.

    Bad edge counts catch open and over-used edges, but they miss "pinch"
    vertices where two closed sheets touch at one point. Those are painful in
    DCC tools, so the production harness checks the vertex link directly.
    """

    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_count = int(len(mesh.vertices))
    if vertex_count == 0 or len(faces) == 0:
        return 0
    links: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
    for a, b, c in faces.reshape(-1, 3):
        a_i, b_i, c_i = int(a), int(b), int(c)
        links[a_i].append((b_i, c_i))
        links[b_i].append((c_i, a_i))
        links[c_i].append((a_i, b_i))
    return int(sum(1 for link_edges in links if link_edges and not _is_manifold_vertex_link(link_edges)))


def _is_manifold_vertex_link(link_edges: list[tuple[int, int]]) -> bool:
    if not link_edges:
        return True
    degrees: dict[int, int] = {}
    adjacency: dict[int, set[int]] = {}
    for raw_a, raw_b in link_edges:
        a, b = int(raw_a), int(raw_b)
        if a == b:
            return False
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    start = next(iter(adjacency))
    stack = [start]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(neighbor for neighbor in adjacency.get(current, set()) if neighbor not in visited)
    if len(visited) != len(adjacency):
        return False
    degree_values = list(degrees.values())
    if any(degree not in {1, 2} for degree in degree_values):
        return False
    degree_one_count = sum(1 for degree in degree_values if degree == 1)
    return degree_one_count in {0, 2}


def _triangle_aspect_ratios(mesh: trimesh.Trimesh) -> np.ndarray:
    triangles = mesh.vertices[mesh.faces]
    lengths = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ],
        axis=1,
    )
    shortest = np.maximum(np.min(lengths, axis=1), 1e-12)
    return np.max(lengths, axis=1) / shortest


def _boundary_loop_count(boundary_edges: np.ndarray) -> int:
    if len(boundary_edges) == 0:
        return 0
    adjacency: dict[int, set[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))

    seen: set[int] = set()
    loops = 0
    for start in adjacency:
        if start in seen:
            continue
        loops += 1
        stack = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return loops


def _histogram(values: np.ndarray) -> dict[int, int]:
    keys, counts = np.unique(values, return_counts=True)
    return {int(k): int(v) for k, v in zip(keys, counts)}


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return None
    return float(numerator / denominator)


def _check_blender_roundtrip(path: Path, blender: str) -> tuple[bool, str | None]:
    blender_path = shutil.which(blender) or blender
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "roundtrip.py"
        out_path = Path(tmpdir) / "roundtrip.glb"
        script.write_text(
            "import bpy, sys\n"
            "src, dst = sys.argv[-2], sys.argv[-1]\n"
            "bpy.ops.object.select_all(action='SELECT')\n"
            "bpy.ops.object.delete()\n"
            "if src.lower().endswith('.obj'):\n"
            "    bpy.ops.wm.obj_import(filepath=src)\n"
            "elif src.lower().endswith(('.glb', '.gltf')):\n"
            "    bpy.ops.import_scene.gltf(filepath=src)\n"
            "elif src.lower().endswith('.stl'):\n"
            "    bpy.ops.wm.stl_import(filepath=src)\n"
            "else:\n"
            "    raise RuntimeError('unsupported Blender import format')\n"
            "bpy.ops.export_scene.gltf(filepath=dst, export_format='GLB')\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [blender_path, "--background", "--python", str(script), "--", str(path), str(out_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode == 0:
        return True, None
    return False, (result.stderr or result.stdout).strip()[-1000:]


def dumps_report(report: dict[str, Any]) -> str:
    """Serialize reports with stable formatting for diffs and dashboards."""
    return json.dumps(report, indent=2, sort_keys=True)
