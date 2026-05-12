"""Stitch per-chart remesh sidecars into one promotable quad candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from clearmesh.eval.mesh_quality import evaluate_mesh
from clearmesh.retopology.quad_remesh import quad_mesh_stats, weld_obj_vertices


@dataclass(frozen=True)
class ChartStitchOptions:
    weld_tolerance: float = 1e-6
    min_quad_ratio: float = 0.85
    max_components: int = 16
    max_boundary_loops: int = 128
    require_watertight: bool = False


@dataclass
class ChartStitchReport:
    manifest_path: str
    output_path: str
    status: str
    input_chart_count: int
    stitched_chart_count: int
    skipped_chart_count: int
    postprocess: dict[str, Any]
    quad_stats: dict[str, Any]
    mesh_metrics: dict[str, Any]
    promotion: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chart_stitch_options_from_metadata(metadata: dict[str, Any]) -> ChartStitchOptions:
    return ChartStitchOptions(
        weld_tolerance=float(metadata.get("chart_stitch_weld_tolerance", metadata.get("quad_weld_tolerance", 1e-6))),
        min_quad_ratio=float(metadata.get("chart_stitch_min_quad_ratio", metadata.get("production_min_quad_ratio", 0.85))),
        max_components=int(metadata.get("chart_stitch_max_components", 16)),
        max_boundary_loops=int(metadata.get("chart_stitch_max_boundary_loops", 128)),
        require_watertight=_bool(metadata.get("chart_stitch_require_watertight", False)),
    )


def stitch_chart_remesh_outputs(
    manifest_path: str | Path,
    output_path: str | Path,
    options: ChartStitchOptions | None = None,
) -> ChartStitchReport:
    options = options or ChartStitchOptions()
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    stitched = 0
    skipped = 0
    notes: list[str] = []

    for entry in entries:
        if not isinstance(entry, dict) or entry.get("status") != "succeeded":
            skipped += 1
            continue
        mesh_path = entry.get("output_mesh_path")
        if not mesh_path or not Path(str(mesh_path)).exists():
            skipped += 1
            continue
        chart_vertices, chart_faces = _read_obj_vertices_faces(Path(str(mesh_path)))
        if len(chart_vertices) == 0 or not chart_faces:
            skipped += 1
            continue
        offset = len(vertices)
        vertices.extend(tuple(float(value) for value in vertex) for vertex in chart_vertices)
        faces.extend([[index + offset for index in face] for face in chart_faces])
        stitched += 1

    if not vertices or not faces:
        output_path.write_text("# ClearMesh empty chart stitch\n", encoding="utf-8")
        metrics = evaluate_mesh(output_path)
        return ChartStitchReport(
            manifest_path=str(manifest_path),
            output_path=str(output_path),
            status="empty",
            input_chart_count=len(entries),
            stitched_chart_count=0,
            skipped_chart_count=skipped,
            postprocess={"enabled": False, "reason": "no_chart_meshes"},
            quad_stats=quad_mesh_stats(output_path),
            mesh_metrics=metrics,
            promotion=_promotion_decision(metrics, {}, options),
            notes=["no successful chart meshes were available to stitch"],
        )

    _write_obj(output_path, np.asarray(vertices, dtype=float), faces)
    postprocess = weld_obj_vertices(output_path, tolerance=options.weld_tolerance)
    metrics = evaluate_mesh(output_path)
    stats = quad_mesh_stats(output_path)
    promotion = _promotion_decision(metrics, stats, options)
    if stitched < len(entries):
        notes.append("some chart entries were skipped; stitched mesh may be partial")
    if promotion.get("promoted"):
        notes.append("stitched chart mesh is eligible for feature projection and production gates")
    else:
        notes.append("stitched chart mesh remains a sidecar until promotion criteria pass")
    return ChartStitchReport(
        manifest_path=str(manifest_path),
        output_path=str(output_path),
        status="succeeded",
        input_chart_count=len(entries),
        stitched_chart_count=stitched,
        skipped_chart_count=skipped,
        postprocess=postprocess,
        quad_stats=stats,
        mesh_metrics=metrics,
        promotion=promotion,
        notes=notes,
    )


def write_chart_stitch_report(path: str | Path, report: ChartStitchReport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _promotion_decision(metrics: dict[str, Any], stats: dict[str, Any], options: ChartStitchOptions) -> dict[str, Any]:
    metrics_ok = bool(metrics.get("ok"))
    quad_ratio = float(stats.get("quad_ratio") or 0.0)
    component_count = int(metrics.get("connected_components") or 0)
    boundary_loops = int(metrics.get("boundary_loop_count") or 0)
    watertight = bool(metrics.get("watertight"))
    promoted = bool(
        metrics_ok
        and quad_ratio >= options.min_quad_ratio
        and component_count <= options.max_components
        and boundary_loops <= options.max_boundary_loops
        and (watertight or not options.require_watertight)
    )
    reasons: list[str] = []
    if not metrics_ok:
        reasons.append(str(metrics.get("error", "mesh_metrics_failed")))
    if quad_ratio < options.min_quad_ratio:
        reasons.append("quad_ratio_below_threshold")
    if component_count > options.max_components:
        reasons.append("too_many_components")
    if boundary_loops > options.max_boundary_loops:
        reasons.append("too_many_boundary_loops")
    if options.require_watertight and not watertight:
        reasons.append("not_watertight")
    return {
        "promoted": promoted,
        "eligible_for_feature_projection": bool(metrics_ok and quad_ratio >= options.min_quad_ratio),
        "quad_ratio": quad_ratio,
        "component_count": component_count,
        "boundary_loop_count": boundary_loops,
        "watertight": watertight,
        "reasons": reasons,
    }


def _read_obj_vertices_faces(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                face: list[int] = []
                for token in line.strip().split()[1:]:
                    raw = token.split("/", 1)[0]
                    if not raw:
                        continue
                    index = int(raw)
                    if index < 0:
                        index = len(vertices) + index + 1
                    face.append(index - 1)
                if len(face) >= 3:
                    faces.append(face)
    return np.asarray(vertices, dtype=float), faces


def _write_obj(path: Path, vertices: np.ndarray, faces: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# ClearMesh chart stitch output\n")
        for vertex in np.asarray(vertices, dtype=float):
            handle.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in faces:
            handle.write("f " + " ".join(str(index + 1) for index in face) + "\n")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
