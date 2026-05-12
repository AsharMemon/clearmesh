"""Generic feature and chart analysis for retopology planning.

This is the wide-ranging layer above templates. It detects feature edges,
decomposes arbitrary meshes into smooth-ish charts, builds a chart seam graph,
and recommends whether each chart should use a known template, a generic
cross-field/quadrangulation baseline, or stay in the triangle control fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from clearmesh.eval.mesh_quality import load_mesh


@dataclass(frozen=True)
class RetopologyPlanningOptions:
    crease_angle_degrees: float = 45.0
    target_quads: int = 5_000
    min_chart_faces: int = 16
    max_report_charts: int = 256
    merge_small_charts: bool = True
    merge_min_faces: int = 0
    merge_max_passes: int = 4
    include_chart_face_indices: bool = True
    max_chart_face_indices: int = 250_000
    elongated_ratio: float = 2.75
    sheet_thickness_ratio: float = 0.08
    hard_feature_density: float = 0.03


@dataclass
class FeatureCurveRecord:
    id: str
    edge_count: int
    vertex_count: int
    length: float
    closed: bool


@dataclass
class ChartRecord:
    id: str
    face_count: int
    vertex_count: int
    area: float
    bbox_min: list[float]
    bbox_max: list[float]
    extents: list[float]
    mean_normal: list[float]
    normal_coherence: float
    boundary_edge_count: int
    feature_edge_count: int
    adjacent_chart_count: int
    adjacent_chart_ids: list[str]
    chart_type: str
    recommended_operator: str
    target_quads: int
    sample_faces: list[int]
    face_indices: list[int]
    face_indices_truncated: bool


@dataclass
class RetopologyPlan:
    input_path: str
    options: dict[str, Any]
    summary: dict[str, Any]
    charts: list[ChartRecord]
    feature_curves: list[FeatureCurveRecord]
    seam_graph: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def retopology_planning_options_from_metadata(metadata: dict[str, Any]) -> RetopologyPlanningOptions:
    return RetopologyPlanningOptions(
        crease_angle_degrees=float(metadata.get("retopo_crease_angle_degrees", 45.0)),
        target_quads=int(metadata.get("retopo_target_quads", metadata.get("quad_target_faces", 5_000))),
        min_chart_faces=int(metadata.get("retopo_min_chart_faces", 16)),
        max_report_charts=int(metadata.get("retopo_max_report_charts", 256)),
        merge_small_charts=_bool(metadata.get("retopo_merge_small_charts", True)),
        merge_min_faces=int(metadata.get("retopo_merge_min_faces", 0)),
        merge_max_passes=int(metadata.get("retopo_merge_max_passes", 4)),
        include_chart_face_indices=_bool(metadata.get("retopo_include_chart_face_indices", True)),
        max_chart_face_indices=int(metadata.get("retopo_max_chart_face_indices", 250_000)),
        elongated_ratio=float(metadata.get("retopo_elongated_ratio", 2.75)),
        sheet_thickness_ratio=float(metadata.get("retopo_sheet_thickness_ratio", 0.08)),
        hard_feature_density=float(metadata.get("retopo_hard_feature_density", 0.03)),
    )


def analyze_retopology_file(
    input_path: str | Path,
    options: RetopologyPlanningOptions | None = None,
) -> RetopologyPlan:
    options = options or RetopologyPlanningOptions()
    input_path = Path(input_path)
    mesh = load_mesh(input_path)
    if len(mesh.faces) == 0:
        raise ValueError(f"{input_path} has no faces")

    face_count = int(len(mesh.faces))
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_areas = np.asarray(mesh.area_faces, dtype=float)
    total_area = float(np.sum(face_areas))
    total_area_safe = max(total_area, 1e-12)

    adjacency = np.asarray(getattr(mesh, "face_adjacency", np.empty((0, 2), dtype=np.int64)), dtype=np.int64)
    adjacency_edges = np.asarray(getattr(mesh, "face_adjacency_edges", np.empty((0, 2), dtype=np.int64)), dtype=np.int64)
    adjacency_angles = _face_adjacency_angles(mesh, adjacency)
    feature_pair_mask = adjacency_angles >= np.deg2rad(options.crease_angle_degrees)

    chart_roots = _smooth_chart_roots(face_count, adjacency, feature_pair_mask)
    chart_ids, chart_face_lists = _renumber_chart_roots(chart_roots)
    initial_chart_count = len(chart_face_lists)
    merge_summary: dict[str, Any] = {
        "enabled": bool(options.merge_small_charts),
        "initial_chart_count": int(initial_chart_count),
        "merged_chart_count": int(initial_chart_count),
        "small_charts_merged": 0,
        "passes": 0,
        "merge_min_faces": int(options.merge_min_faces or options.min_chart_faces),
    }
    if options.merge_small_charts:
        chart_ids, chart_face_lists, merge_summary = _merge_small_charts(
            chart_ids,
            adjacency,
            min_faces=max(1, int(options.merge_min_faces or options.min_chart_faces)),
            max_passes=max(0, int(options.merge_max_passes)),
        )
    face_to_chart = _face_to_chart(face_count, chart_face_lists)

    _, boundary_edge_ids, boundary_edges = _edge_counts(mesh)
    boundary_counts_by_chart = _boundary_counts_by_chart(mesh, boundary_edge_ids, face_to_chart)
    feature_counts_by_chart = _feature_counts_by_chart(adjacency, adjacency_edges, feature_pair_mask, face_to_chart)
    seam_counts = _seam_counts(adjacency, feature_pair_mask, face_to_chart)
    adjacency_by_chart = _adjacency_by_chart(seam_counts)
    feature_edges = _combined_feature_edges(adjacency_edges, feature_pair_mask, boundary_edges)
    feature_curves = _feature_curves(vertices, feature_edges)

    charts: list[ChartRecord] = []
    operator_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    reported_chart_count = min(len(chart_face_lists), max(0, options.max_report_charts))
    for chart_id, face_indices in enumerate(chart_face_lists[:reported_chart_count]):
        chart = _build_chart_record(
            chart_id=chart_id,
            face_indices=face_indices,
            vertices=vertices,
            faces=faces,
            face_areas=face_areas,
            total_area=total_area_safe,
            boundary_edge_count=boundary_counts_by_chart.get(chart_id, 0),
            feature_edge_count=feature_counts_by_chart.get(chart_id, 0),
            adjacent_chart_ids=sorted(adjacency_by_chart.get(chart_id, set())),
            options=options,
        )
        charts.append(chart)
        operator_counts[chart.recommended_operator] = operator_counts.get(chart.recommended_operator, 0) + 1
        type_counts[chart.chart_type] = type_counts.get(chart.chart_type, 0) + 1

    # Include omitted charts in summary counts without bloating JSON reports.
    for chart_id, face_indices in enumerate(chart_face_lists[reported_chart_count:], start=reported_chart_count):
        chart_type, operator = _classify_chart(
            vertices[faces[face_indices].reshape(-1)],
            face_areas[face_indices],
            boundary_counts_by_chart.get(chart_id, 0),
            feature_counts_by_chart.get(chart_id, 0),
            len(face_indices),
            options,
        )
        operator_counts[operator] = operator_counts.get(operator, 0) + 1
        type_counts[chart_type] = type_counts.get(chart_type, 0) + 1

    seam_graph = [
        {"a": f"chart_{a:04d}", "b": f"chart_{b:04d}", "edge_count": int(count)}
        for (a, b), count in sorted(seam_counts.items(), key=lambda item: (-item[1], item[0]))[: options.max_report_charts]
    ]
    summary = {
        "face_count": face_count,
        "vertex_count": int(len(vertices)),
        "initial_chart_count": int(initial_chart_count),
        "chart_count": int(len(chart_face_lists)),
        "reported_chart_count": int(reported_chart_count),
        "chart_merge": merge_summary,
        "feature_edge_count": int(len(feature_edges)),
        "boundary_edge_count": int(len(boundary_edges)),
        "feature_curve_count": int(len(feature_curves)),
        "closed_feature_curve_count": int(sum(1 for curve in feature_curves if curve.closed)),
        "open_feature_curve_count": int(sum(1 for curve in feature_curves if not curve.closed)),
        "chart_type_counts": type_counts,
        "operator_counts": operator_counts,
        "risk_level": _risk_level(len(chart_face_lists), len(boundary_edges), len(feature_edges), face_count),
        "notes": _plan_notes(len(chart_face_lists), len(boundary_edges), len(feature_edges), face_count),
    }
    return RetopologyPlan(
        input_path=str(input_path),
        options=asdict(options),
        summary=summary,
        charts=charts,
        feature_curves=feature_curves[: options.max_report_charts],
        seam_graph=seam_graph,
    )


def write_retopology_plan(path: str | Path, plan: RetopologyPlan) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _face_adjacency_angles(mesh, adjacency: np.ndarray) -> np.ndarray:
    if len(adjacency) == 0:
        return np.empty(0, dtype=float)
    if hasattr(mesh, "face_adjacency_angles"):
        angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
        if len(angles) == len(adjacency):
            return angles
    normals = np.asarray(mesh.face_normals, dtype=float)
    dot = np.einsum("ij,ij->i", normals[adjacency[:, 0]], normals[adjacency[:, 1]])
    return np.arccos(np.clip(dot, -1.0, 1.0))


def _smooth_chart_roots(face_count: int, adjacency: np.ndarray, feature_pair_mask: np.ndarray) -> np.ndarray:
    parent = np.arange(face_count, dtype=np.int64)
    rank = np.zeros(face_count, dtype=np.int8)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for (a, b), is_feature in zip(adjacency, feature_pair_mask):
        if not is_feature:
            union(int(a), int(b))
    return np.asarray([find(face) for face in range(face_count)], dtype=np.int64)


def _renumber_chart_roots(roots: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique, inverse = np.unique(roots, return_inverse=True)
    order = sorted(range(len(unique)), key=lambda chart_id: int(np.sum(inverse == chart_id)), reverse=True)
    remap = {old: new for new, old in enumerate(order)}
    chart_ids = np.asarray([remap[int(value)] for value in inverse], dtype=np.int64)
    chart_face_lists = [np.where(chart_ids == chart_id)[0] for chart_id in range(len(unique))]
    return chart_ids, chart_face_lists


def _renumber_chart_ids(chart_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique, inverse = np.unique(chart_ids, return_inverse=True)
    order = sorted(range(len(unique)), key=lambda chart_id: int(np.sum(inverse == chart_id)), reverse=True)
    remap = {old: new for new, old in enumerate(order)}
    new_chart_ids = np.asarray([remap[int(value)] for value in inverse], dtype=np.int64)
    chart_face_lists = [np.where(new_chart_ids == chart_id)[0] for chart_id in range(len(unique))]
    return new_chart_ids, chart_face_lists


def _face_to_chart(face_count: int, chart_face_lists: list[np.ndarray]) -> np.ndarray:
    face_to_chart = np.empty(face_count, dtype=np.int64)
    for chart_id, face_indices in enumerate(chart_face_lists):
        face_to_chart[face_indices] = chart_id
    return face_to_chart


def _merge_small_charts(
    chart_ids: np.ndarray,
    adjacency: np.ndarray,
    *,
    min_faces: int,
    max_passes: int,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Absorb chart confetti into the largest shared-neighbor seam.

    This is deliberately conservative: it only merges charts below the repair
    threshold and only across actual face adjacency. Hard-feature seams remain
    recorded in the feature graph; the merge just stops one-triangle islands
    from becoming separate remesh jobs.
    """

    if max_passes <= 0 or min_faces <= 1 or len(adjacency) == 0:
        chart_ids, chart_face_lists = _renumber_chart_ids(chart_ids)
        return (
            chart_ids,
            chart_face_lists,
            {
                "enabled": True,
                "initial_chart_count": int(len(chart_face_lists)),
                "merged_chart_count": int(len(chart_face_lists)),
                "small_charts_merged": 0,
                "passes": 0,
                "merge_min_faces": int(min_faces),
            },
        )

    initial_count = int(np.max(chart_ids) + 1) if len(chart_ids) else 0
    total_merged = 0
    passes = 0
    current_ids = np.asarray(chart_ids, dtype=np.int64).copy()
    for _ in range(max_passes):
        chart_count = int(np.max(current_ids) + 1) if len(current_ids) else 0
        if chart_count <= 1:
            break
        face_counts = np.bincount(current_ids, minlength=chart_count)
        small_charts = {int(index) for index, count in enumerate(face_counts) if 0 < count < min_faces}
        if not small_charts:
            break

        contacts: dict[int, dict[int, int]] = {}
        for face_a, face_b in adjacency:
            chart_a = int(current_ids[int(face_a)])
            chart_b = int(current_ids[int(face_b)])
            if chart_a == chart_b:
                continue
            if chart_a in small_charts:
                contacts.setdefault(chart_a, {})[chart_b] = contacts.setdefault(chart_a, {}).get(chart_b, 0) + 1
            if chart_b in small_charts:
                contacts.setdefault(chart_b, {})[chart_a] = contacts.setdefault(chart_b, {}).get(chart_a, 0) + 1

        remap = np.arange(chart_count, dtype=np.int64)
        changed = 0
        for chart_id in sorted(small_charts):
            neighbors = contacts.get(chart_id, {})
            if not neighbors:
                continue
            best_neighbor = max(neighbors, key=lambda neighbor: (neighbors[neighbor], face_counts[neighbor], -neighbor))
            remap[chart_id] = int(best_neighbor)
            changed += 1
        if changed == 0:
            break

        current_ids = remap[current_ids]
        current_ids, _ = _renumber_chart_ids(current_ids)
        total_merged += changed
        passes += 1

    current_ids, chart_face_lists = _renumber_chart_ids(current_ids)
    return (
        current_ids,
        chart_face_lists,
        {
            "enabled": True,
            "initial_chart_count": initial_count,
            "merged_chart_count": int(len(chart_face_lists)),
            "small_charts_merged": int(total_merged),
            "passes": int(passes),
            "merge_min_faces": int(min_faces),
        },
    )


def _edge_counts(mesh) -> tuple[np.ndarray, set[int], np.ndarray]:
    edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    boundary_edge_ids = {int(index) for index in np.where(edge_counts == 1)[0]}
    boundary_edges = np.asarray(mesh.edges_unique[list(boundary_edge_ids)], dtype=np.int64) if boundary_edge_ids else np.empty((0, 2), dtype=np.int64)
    return edge_counts, boundary_edge_ids, boundary_edges


def _boundary_counts_by_chart(mesh, boundary_edge_ids: set[int], face_to_chart: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    if not boundary_edge_ids:
        return counts
    face_edges = np.asarray(mesh.edges_unique_inverse, dtype=np.int64).reshape((-1, 3))
    for face_id, edge_ids in enumerate(face_edges):
        chart_id = int(face_to_chart[face_id])
        counts[chart_id] = counts.get(chart_id, 0) + sum(1 for edge_id in edge_ids if int(edge_id) in boundary_edge_ids)
    return counts


def _feature_counts_by_chart(adjacency: np.ndarray, adjacency_edges: np.ndarray, feature_pair_mask: np.ndarray, face_to_chart: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    if len(adjacency_edges) == 0:
        return counts
    for (face_a, face_b), is_feature in zip(adjacency, feature_pair_mask):
        if not is_feature:
            continue
        chart_a = int(face_to_chart[int(face_a)])
        chart_b = int(face_to_chart[int(face_b)])
        counts[chart_a] = counts.get(chart_a, 0) + 1
        if chart_b != chart_a:
            counts[chart_b] = counts.get(chart_b, 0) + 1
    return counts


def _seam_counts(adjacency: np.ndarray, feature_pair_mask: np.ndarray, face_to_chart: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for (face_a, face_b), is_feature in zip(adjacency, feature_pair_mask):
        chart_a = int(face_to_chart[int(face_a)])
        chart_b = int(face_to_chart[int(face_b)])
        if chart_a == chart_b or not is_feature:
            continue
        key = (min(chart_a, chart_b), max(chart_a, chart_b))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _adjacency_by_chart(seam_counts: dict[tuple[int, int], int]) -> dict[int, set[str]]:
    adjacency: dict[int, set[str]] = {}
    for chart_a, chart_b in seam_counts:
        adjacency.setdefault(chart_a, set()).add(f"chart_{chart_b:04d}")
        adjacency.setdefault(chart_b, set()).add(f"chart_{chart_a:04d}")
    return adjacency


def _combined_feature_edges(adjacency_edges: np.ndarray, feature_pair_mask: np.ndarray, boundary_edges: np.ndarray) -> np.ndarray:
    pieces: list[np.ndarray] = []
    if len(adjacency_edges):
        pieces.append(adjacency_edges[feature_pair_mask])
    if len(boundary_edges):
        pieces.append(boundary_edges)
    if not pieces:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack(pieces)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _feature_curves(vertices: np.ndarray, feature_edges: np.ndarray) -> list[FeatureCurveRecord]:
    if len(feature_edges) == 0:
        return []
    parent: dict[int, int] = {}
    degree: dict[int, int] = {}
    length_by_root: dict[int, float] = {}
    edge_count_by_root: dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for a, b in feature_edges:
        union(int(a), int(b))
        degree[int(a)] = degree.get(int(a), 0) + 1
        degree[int(b)] = degree.get(int(b), 0) + 1

    for a, b in feature_edges:
        root = find(int(a))
        length_by_root[root] = length_by_root.get(root, 0.0) + float(np.linalg.norm(vertices[int(a)] - vertices[int(b)]))
        edge_count_by_root[root] = edge_count_by_root.get(root, 0) + 1

    vertices_by_root: dict[int, list[int]] = {}
    for vertex_id in degree:
        vertices_by_root.setdefault(find(vertex_id), []).append(vertex_id)

    curves: list[FeatureCurveRecord] = []
    for index, (root, vertex_ids) in enumerate(sorted(vertices_by_root.items(), key=lambda item: (-len(item[1]), item[0]))):
        curves.append(
            FeatureCurveRecord(
                id=f"curve_{index:04d}",
                edge_count=int(edge_count_by_root.get(root, 0)),
                vertex_count=int(len(vertex_ids)),
                length=float(length_by_root.get(root, 0.0)),
                closed=all(degree[vertex_id] == 2 for vertex_id in vertex_ids),
            )
        )
    return curves


def _build_chart_record(
    *,
    chart_id: int,
    face_indices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_areas: np.ndarray,
    total_area: float,
    boundary_edge_count: int,
    feature_edge_count: int,
    adjacent_chart_ids: list[str],
    options: RetopologyPlanningOptions,
) -> ChartRecord:
    chart_vertices = vertices[faces[face_indices].reshape(-1)]
    chart_area = float(np.sum(face_areas[face_indices]))
    bbox_min = np.min(chart_vertices, axis=0)
    bbox_max = np.max(chart_vertices, axis=0)
    extents = np.maximum(bbox_max - bbox_min, 0.0)
    chart_type, operator = _classify_chart(
        chart_vertices,
        face_areas[face_indices],
        boundary_edge_count,
        feature_edge_count,
        len(face_indices),
        options,
    )
    normals = _face_normals(vertices, faces[face_indices])
    weighted = normals * face_areas[face_indices, None]
    normal_sum = np.sum(weighted, axis=0)
    area_sum = max(chart_area, 1e-12)
    mean_normal = normal_sum / max(np.linalg.norm(normal_sum), 1e-12)
    normal_coherence = float(np.linalg.norm(normal_sum) / area_sum)
    target_quads = max(4, int(round(options.target_quads * (chart_area / total_area))))
    max_face_indices = max(0, int(options.max_chart_face_indices))
    include_face_indices = bool(options.include_chart_face_indices and max_face_indices > 0)
    face_indices_list = [int(value) for value in face_indices[:max_face_indices]] if include_face_indices else []
    return ChartRecord(
        id=f"chart_{chart_id:04d}",
        face_count=int(len(face_indices)),
        vertex_count=int(len(np.unique(faces[face_indices].reshape(-1)))),
        area=chart_area,
        bbox_min=[float(value) for value in bbox_min],
        bbox_max=[float(value) for value in bbox_max],
        extents=[float(value) for value in extents],
        mean_normal=[float(value) for value in mean_normal],
        normal_coherence=normal_coherence,
        boundary_edge_count=int(boundary_edge_count),
        feature_edge_count=int(feature_edge_count),
        adjacent_chart_count=int(len(adjacent_chart_ids)),
        adjacent_chart_ids=adjacent_chart_ids[:32],
        chart_type=chart_type,
        recommended_operator=operator,
        target_quads=target_quads,
        sample_faces=[int(value) for value in face_indices[:16]],
        face_indices=face_indices_list,
        face_indices_truncated=bool(include_face_indices and len(face_indices) > max_face_indices),
    )


def _classify_chart(
    chart_vertices: np.ndarray,
    chart_face_areas: np.ndarray,
    boundary_edge_count: int,
    feature_edge_count: int,
    face_count: int,
    options: RetopologyPlanningOptions,
) -> tuple[str, str]:
    if face_count < options.min_chart_faces:
        return "small_fragment", "cleanup_or_merge"
    extents = np.sort(np.ptp(chart_vertices, axis=0))[::-1]
    longest = max(float(extents[0]), 1e-12)
    middle = max(float(extents[1]), 1e-12)
    shortest = max(float(extents[2]), 0.0)
    feature_density = (boundary_edge_count + feature_edge_count) / max(face_count, 1)
    if shortest / longest <= options.sheet_thickness_ratio and middle / longest >= 0.25:
        return "sheet_like", "template_or_cross_field_chart"
    if longest / middle >= options.elongated_ratio:
        return "elongated_or_tube_like", "template_or_cross_field_chart"
    if feature_density >= options.hard_feature_density:
        return "feature_rich_freeform", "cross_field_with_feature_constraints"
    return "smooth_freeform", "cross_field_chart"


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] /= lengths[valid][:, None]
    return normals


def _risk_level(chart_count: int, boundary_edge_count: int, feature_edge_count: int, face_count: int) -> str:
    chart_ratio = chart_count / max(face_count, 1)
    boundary_ratio = boundary_edge_count / max(face_count, 1)
    ratio_applies = face_count >= 1_000
    if chart_count > 512 or (ratio_applies and chart_ratio > 0.05) or boundary_ratio > 0.25:
        return "high"
    if chart_count > 128 or (ratio_applies and chart_ratio > 0.015) or boundary_ratio > 0.05:
        return "medium"
    return "low"


def _plan_notes(chart_count: int, boundary_edge_count: int, feature_edge_count: int, face_count: int) -> list[str]:
    notes: list[str] = []
    if chart_count > 128:
        notes.append("many charts detected; prefer part-level or chart-level processing over whole-object remeshing")
    if boundary_edge_count / max(face_count, 1) > 0.05:
        notes.append("many boundary edges detected; require weld/repair before promoting quad output")
    if feature_edge_count == 0:
        notes.append("no strong feature edges detected; generic smooth cross-field baseline is appropriate")
    return notes


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
