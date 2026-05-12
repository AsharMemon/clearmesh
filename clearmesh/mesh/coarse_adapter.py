"""Coarse proxy adapter for TRELLIS-style generator meshes.

UltraShape's published pipeline expects a coherent coarse mesh from Hunyuan.
TRELLIS/TRELLIS.2 can instead produce beautiful but fragmented triangle soup.
This adapter turns that proxy into a bounded, mostly watertight conditioning
surface before manifoldization/reference refinement sees it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from clearmesh.eval.mesh_quality import load_mesh
from clearmesh.mesh.cleanup import CleanupOptions, cleanup_mesh
from clearmesh.product.mesh_passport import fast_passport_metrics


@dataclass(frozen=True)
class CoarseAdapterOptions:
    engine: str = "auto"
    target_faces: int = 150_000
    max_target_face_ratio: float = 1.25
    sample_points: int = 180_000
    min_component_faces: int = 64
    min_component_face_ratio: float = 0.0001
    keep_largest_components: int = 128
    max_output_components: int = 1
    max_boundary_loops: int = 0
    max_nonmanifold_edges: int = 0
    require_watertight: bool = True
    voxel_resolution: int = 192
    voxel_dilate: int = 2
    voxel_close: int = 1
    voxel_pad_ratio: float = 0.08
    mesh_voxel_max_faces: int = 75_000
    hull_max_points: int = 20_000
    poisson_depth: int = 8
    poisson_density_quantile: float = 0.01
    orient_normals: bool = False
    fallback: str = "convex_hull"


@dataclass
class CoarseAdapterReport:
    input_path: str
    output_path: str
    engine: str
    accepted: bool
    input_metrics: dict[str, Any]
    preclean_metrics: dict[str, Any]
    output_metrics: dict[str, Any]
    options: dict[str, Any]
    notes: list[str]
    attempts: list[dict[str, Any]]


def adapt_coarse_mesh_file(
    input_path: str | Path,
    output_path: str | Path,
    options: CoarseAdapterOptions | None = None,
) -> CoarseAdapterReport:
    """Write a coherent coarse proxy and return an audit report."""

    options = options or CoarseAdapterOptions()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_metrics = fast_passport_metrics(input_path)
    source_mesh = load_mesh(input_path)
    notes: list[str] = []
    attempts: list[dict[str, Any]] = []

    precleaned = _preclean_source_mesh(source_mesh, options)
    preclean_metrics = _metrics_for_mesh(precleaned)
    notes.append(f"preclean kept {preclean_metrics['connected_components']} components and {preclean_metrics['face_count']} faces")

    engine = str(options.engine).lower()
    engines = _engine_order(engine, options)
    selected_mesh: trimesh.Trimesh | None = None
    selected_engine = ""
    selected_metrics: dict[str, Any] = {}

    for candidate_engine in engines:
        try:
            candidate = _build_candidate(precleaned, options, candidate_engine)
            candidate = _postprocess_candidate(candidate, options)
            metrics = _metrics_for_mesh(candidate)
            accepted = _accept_metrics(metrics, options)
            attempts.append(
                {
                    "engine": candidate_engine,
                    "accepted": accepted,
                    "metrics": metrics,
                }
            )
            selected_mesh = candidate
            selected_engine = candidate_engine
            selected_metrics = metrics
            if accepted:
                break
        except Exception as exc:  # noqa: BLE001 - report failed candidates and keep trying.
            attempts.append(
                {
                    "engine": candidate_engine,
                    "accepted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            notes.append(f"{candidate_engine} failed: {type(exc).__name__}: {exc}")

    if selected_mesh is None:
        raise RuntimeError("coarse adapter could not produce any candidate mesh")

    selected_mesh.export(output_path)
    output_metrics = fast_passport_metrics(output_path)
    accepted = _accept_metrics(output_metrics, options)
    if not accepted:
        notes.append("adapter output did not meet the configured acceptance contract")

    return CoarseAdapterReport(
        input_path=str(input_path),
        output_path=str(output_path),
        engine=selected_engine,
        accepted=accepted,
        input_metrics=input_metrics,
        preclean_metrics=preclean_metrics,
        output_metrics=output_metrics,
        options=asdict(options),
        notes=notes,
        attempts=attempts,
    )


def coarse_adapter_options_from_metadata(metadata: dict[str, Any]) -> CoarseAdapterOptions:
    return CoarseAdapterOptions(
        engine=str(metadata.get("coarse_adapter_engine", "auto")),
        target_faces=int(metadata.get("coarse_adapter_target_faces", 150_000)),
        max_target_face_ratio=float(metadata.get("coarse_adapter_max_target_face_ratio", 1.25)),
        sample_points=int(metadata.get("coarse_adapter_sample_points", 180_000)),
        min_component_faces=int(metadata.get("coarse_adapter_min_component_faces", 64)),
        min_component_face_ratio=float(metadata.get("coarse_adapter_min_component_face_ratio", 0.0001)),
        keep_largest_components=int(metadata.get("coarse_adapter_keep_largest_components", 128)),
        max_output_components=int(metadata.get("coarse_adapter_max_output_components", 1)),
        max_boundary_loops=int(metadata.get("coarse_adapter_max_boundary_loops", 0)),
        max_nonmanifold_edges=int(metadata.get("coarse_adapter_max_nonmanifold_edges", 0)),
        require_watertight=_bool(metadata.get("coarse_adapter_require_watertight", True)),
        voxel_resolution=int(metadata.get("coarse_adapter_voxel_resolution", 192)),
        voxel_dilate=int(metadata.get("coarse_adapter_voxel_dilate", 2)),
        voxel_close=int(metadata.get("coarse_adapter_voxel_close", 1)),
        voxel_pad_ratio=float(metadata.get("coarse_adapter_voxel_pad_ratio", 0.08)),
        mesh_voxel_max_faces=int(metadata.get("coarse_adapter_mesh_voxel_max_faces", 75_000)),
        hull_max_points=int(metadata.get("coarse_adapter_hull_max_points", 20_000)),
        poisson_depth=int(metadata.get("coarse_adapter_poisson_depth", 8)),
        poisson_density_quantile=float(metadata.get("coarse_adapter_poisson_density_quantile", 0.01)),
        orient_normals=_bool(metadata.get("coarse_adapter_orient_normals", False)),
        fallback=str(metadata.get("coarse_adapter_fallback", "convex_hull")),
    )


def report_to_dict(report: CoarseAdapterReport) -> dict[str, Any]:
    return asdict(report)


def _engine_order(engine: str, options: CoarseAdapterOptions) -> list[str]:
    if engine == "auto":
        order = ["voxel_shell", "poisson"]
        if options.fallback:
            order.append(str(options.fallback).lower())
        return _dedupe(order)
    if engine in {"voxel", "voxel_shell", "poisson", "cleanup", "convex_hull"}:
        order = [engine]
        if engine != options.fallback and options.fallback:
            order.append(str(options.fallback).lower())
        return _dedupe(order)
    raise ValueError(f"unsupported coarse adapter engine: {engine}")


def _build_candidate(mesh: trimesh.Trimesh, options: CoarseAdapterOptions, engine: str) -> trimesh.Trimesh:
    if engine in {"voxel", "voxel_shell"}:
        return _voxel_shell(mesh, options)
    if engine == "poisson":
        return _poisson_shell(mesh, options)
    if engine == "cleanup":
        return mesh.copy()
    if engine == "convex_hull":
        points = np.asarray(mesh.vertices, dtype=np.float64)
        max_points = max(128, int(options.hull_max_points))
        if len(points) > max_points:
            rng = np.random.default_rng(0)
            indices = rng.choice(len(points), size=max_points, replace=False)
            points = points[indices]
            points = np.concatenate([points, _bounds_corners(mesh.bounds)], axis=0)
        hull = trimesh.points.PointCloud(points).convex_hull
        if hull is None or len(hull.faces) == 0:
            raise ValueError("convex hull returned an empty mesh")
        return hull
    raise ValueError(f"unsupported coarse adapter engine: {engine}")


def _postprocess_candidate(mesh: trimesh.Trimesh, options: CoarseAdapterOptions) -> trimesh.Trimesh:
    cleaned, _ = cleanup_mesh(
        mesh,
        CleanupOptions(
            min_component_faces=1,
            min_component_face_ratio=0.0,
            keep_largest_components=max(1, int(options.max_output_components)),
            fill_holes=True,
            fix_normals=True,
            merge_vertices=True,
        ),
    )
    if int(options.target_faces) > 0 and len(cleaned.faces) > int(options.target_faces):
        cleaned = _simplify_trimesh(cleaned, int(options.target_faces))
    cleaned.remove_unreferenced_vertices()
    cleaned.fix_normals()
    return cleaned


def _voxel_shell(mesh: trimesh.Trimesh, options: CoarseAdapterOptions) -> trimesh.Trimesh:
    try:
        return _trimesh_voxel_shell(mesh, options)
    except Exception:
        return _point_voxel_shell(mesh, options)


def _trimesh_voxel_shell(mesh: trimesh.Trimesh, options: CoarseAdapterOptions) -> trimesh.Trimesh:
    if len(mesh.faces) > int(options.mesh_voxel_max_faces):
        raise ValueError(
            f"mesh voxelization skipped for {len(mesh.faces)} faces "
            f"(limit {options.mesh_voxel_max_faces})"
        )
    resolution = max(16, int(options.voxel_resolution))
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))
    if not np.isfinite(max_extent) or max_extent <= 0:
        raise ValueError("mesh bounds are degenerate")
    pitch = max_extent / float(resolution - 1)
    voxel_grid = mesh.voxelized(pitch).fill()
    candidate = voxel_grid.marching_cubes
    if candidate is None or len(candidate.faces) == 0:
        raise ValueError("trimesh voxelization returned an empty mesh")
    candidate.apply_transform(voxel_grid.transform)
    return candidate


def _point_voxel_shell(mesh: trimesh.Trimesh, options: CoarseAdapterOptions) -> trimesh.Trimesh:
    from scipy import ndimage
    from skimage import measure

    points = _sample_points(mesh, int(options.sample_points))
    if len(points) == 0:
        raise ValueError("no points sampled from mesh")

    resolution = max(16, int(options.voxel_resolution))
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))
    if not np.isfinite(max_extent) or max_extent <= 0:
        raise ValueError("mesh bounds are degenerate")
    pad = max_extent * max(0.0, float(options.voxel_pad_ratio))
    lower = bounds[0] - pad
    upper = bounds[1] + pad
    span = np.maximum(upper - lower, max_extent / resolution)

    coords = np.floor((points - lower) / span * (resolution - 1)).astype(np.int64)
    coords = np.clip(coords, 0, resolution - 1)
    occupancy = np.zeros((resolution, resolution, resolution), dtype=bool)
    occupancy[coords[:, 0], coords[:, 1], coords[:, 2]] = True

    dilate = max(0, int(options.voxel_dilate))
    if dilate:
        occupancy = ndimage.binary_dilation(occupancy, iterations=dilate)
    close = max(0, int(options.voxel_close))
    if close:
        occupancy = ndimage.binary_closing(occupancy, iterations=close)
    occupancy = ndimage.binary_fill_holes(occupancy)

    if int(np.count_nonzero(occupancy)) < 8:
        raise ValueError("voxel occupancy is too sparse")

    vertices, faces, _, _ = measure.marching_cubes(
        occupancy.astype(np.float32),
        level=0.5,
        spacing=tuple((span / (resolution - 1)).tolist()),
    )
    vertices = vertices + lower
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("marching cubes returned an empty mesh")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _poisson_shell(mesh: trimesh.Trimesh, options: CoarseAdapterOptions) -> trimesh.Trimesh:
    import open3d as o3d  # type: ignore

    points, face_indices = trimesh.sample.sample_surface(mesh, max(1_000, int(options.sample_points)))
    normals = mesh.face_normals[face_indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    if options.orient_normals:
        try:
            pcd.orient_normals_consistent_tangent_plane(32)
        except Exception:
            pass

    reconstructed, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=max(4, int(options.poisson_depth)),
    )
    densities_np = np.asarray(densities)
    if densities_np.size:
        threshold = np.quantile(densities_np, float(options.poisson_density_quantile))
        reconstructed.remove_vertices_by_mask(densities_np < threshold)
    if len(reconstructed.triangles) > int(options.target_faces):
        reconstructed = reconstructed.simplify_quadric_decimation(
            target_number_of_triangles=max(4, int(options.target_faces))
        )

    vertices = np.asarray(reconstructed.vertices)
    faces = np.asarray(reconstructed.triangles)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError("poisson reconstruction returned an empty mesh")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _sample_points(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    mesh = mesh.copy()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")
    count = max(1_000, int(count))
    points, _ = trimesh.sample.sample_surface(mesh, count)
    if len(mesh.vertices):
        points = np.concatenate([points, np.asarray(mesh.vertices, dtype=np.float64)], axis=0)
    return points


def _preclean_source_mesh(mesh: trimesh.Trimesh, options: CoarseAdapterOptions) -> trimesh.Trimesh:
    mesh = mesh.copy()
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    component_count, labels = _face_connected_components(mesh)
    if component_count <= 1:
        mesh.fix_normals()
        return mesh

    component_faces = np.bincount(labels, minlength=component_count) if len(labels) else np.array([], dtype=np.int64)
    face_total = max(int(len(mesh.faces)), 1)
    keep_labels: list[int] = []
    for label, face_count in enumerate(component_faces):
        if int(face_count) < int(options.min_component_faces):
            continue
        if float(face_count) / face_total < float(options.min_component_face_ratio):
            continue
        keep_labels.append(int(label))

    if not keep_labels and component_faces.size:
        keep_labels = [int(np.argmax(component_faces))]

    keep_labels = sorted(keep_labels, key=lambda label: int(component_faces[label]), reverse=True)[
        : max(1, int(options.keep_largest_components))
    ]
    face_mask = np.isin(labels, np.asarray(keep_labels, dtype=np.int64))
    if not np.any(face_mask):
        face_mask = np.ones(len(mesh.faces), dtype=bool)

    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
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


def _metrics_for_mesh(mesh: trimesh.Trimesh) -> dict[str, Any]:
    edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    boundary_edges = mesh.edges_unique[edge_counts == 1]
    nonmanifold_edges = mesh.edges_unique[edge_counts != 2]
    component_count, labels = _face_connected_components(mesh)
    face_total = max(int(len(mesh.faces)), 1)
    component_faces = np.bincount(labels, minlength=component_count) if len(labels) else np.array([], dtype=np.int64)
    tiny_components = int(np.sum(component_faces / face_total < 0.01)) if component_faces.size else 0
    return {
        "ok": True,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "connected_components": int(component_count),
        "tiny_component_count": int(tiny_components),
        "watertight": bool(len(edge_counts) > 0 and np.all(edge_counts == 2)),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edge_count": int(len(boundary_edges)),
        "boundary_loop_count": int(_boundary_loop_count(boundary_edges)),
        "nonmanifold_edge_count": int(len(nonmanifold_edges)),
        "surface_area": float(mesh.area),
        "volume": float(mesh.volume) if len(edge_counts) > 0 and bool(np.all(edge_counts == 2)) else None,
    }


def _face_connected_components(mesh: trimesh.Trimesh) -> tuple[int, np.ndarray]:
    face_count = int(len(mesh.faces))
    if face_count == 0:
        return 0, np.array([], dtype=np.int64)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if adjacency.size == 0:
        return face_count, np.arange(face_count, dtype=np.int64)
    rows = np.concatenate([adjacency[:, 0], adjacency[:, 1]])
    cols = np.concatenate([adjacency[:, 1], adjacency[:, 0]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(face_count, face_count))
    count, labels = connected_components(graph, directed=False, return_labels=True)
    return int(count), labels.astype(np.int64, copy=False)


def _bounds_corners(bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=np.float64)
    lower, upper = bounds[0], bounds[1]
    return np.asarray(
        [
            [lower[0], lower[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], upper[1], upper[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], lower[2]],
            [upper[0], upper[1], upper[2]],
        ],
        dtype=np.float64,
    )


def _accept_metrics(metrics: dict[str, Any], options: CoarseAdapterOptions) -> bool:
    if not metrics.get("ok", True):
        return False
    if options.require_watertight and not bool(metrics.get("watertight")):
        return False
    if int(metrics.get("connected_components") or 0) > int(options.max_output_components):
        return False
    if int(metrics.get("boundary_loop_count") or 0) > int(options.max_boundary_loops):
        return False
    if int(metrics.get("nonmanifold_edge_count") or 0) > int(options.max_nonmanifold_edges):
        return False
    target_faces = int(options.target_faces)
    if target_faces > 0:
        max_faces = int(np.ceil(float(target_faces) * max(1.0, float(options.max_target_face_ratio))))
        if int(metrics.get("face_count") or 0) > max_faces:
            return False
    return True


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


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out
