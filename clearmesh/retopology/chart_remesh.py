"""Per-chart quad remesh sidecar execution.

This module turns the retopology plan into bounded, inspectable remesh jobs.
It does not stitch charts into a final production quad mesh yet; that seam
stitching/promotion step should only happen after the Blender gates agree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from clearmesh.eval.mesh_quality import load_mesh
from clearmesh.retopology.quad_remesh import QuadRemeshOptions, quad_remesh_file


@dataclass(frozen=True)
class ChartRemeshOptions:
    engine: str = "auto"
    max_charts: int = 8
    min_chart_faces: int = 16
    allow_cleanup_charts: bool = False
    target_quads_total: int = 5_000
    min_target_quads: int = 16
    max_target_quads_per_chart: int = 2_048
    weld_tolerance: float = 1e-8
    measure_weld_metrics: bool = True
    instant_meshes_path: str | None = None
    quadriflow_path: str | None = None
    quadriflow_sharp: bool = True
    quadriflow_mcf: bool = False


@dataclass
class ChartRemeshEntry:
    chart_id: str
    status: str
    source_face_count: int
    source_chart_type: str
    recommended_operator: str
    target_quads: int
    source_submesh_path: str | None = None
    output_mesh_path: str | None = None
    report_path: str | None = None
    reason: str | None = None


@dataclass
class ChartRemeshManifest:
    input_path: str
    plan_path: str
    output_dir: str
    options: dict[str, Any]
    entries: list[ChartRemeshEntry]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chart_remesh_options_from_metadata(metadata: dict[str, Any]) -> ChartRemeshOptions:
    return ChartRemeshOptions(
        engine=str(metadata.get("chart_remesh_engine", metadata.get("quad_remesh_engine", "auto"))),
        max_charts=int(metadata.get("chart_remesh_max_charts", 8)),
        min_chart_faces=int(metadata.get("chart_remesh_min_chart_faces", metadata.get("retopo_min_chart_faces", 16))),
        allow_cleanup_charts=_bool(metadata.get("chart_remesh_allow_cleanup_charts", False)),
        target_quads_total=int(metadata.get("chart_remesh_target_quads", metadata.get("quad_target_faces", 5_000))),
        min_target_quads=int(metadata.get("chart_remesh_min_target_quads", 16)),
        max_target_quads_per_chart=int(metadata.get("chart_remesh_max_target_quads_per_chart", 2_048)),
        weld_tolerance=float(metadata.get("chart_remesh_weld_tolerance", metadata.get("quad_weld_tolerance", 1e-8))),
        measure_weld_metrics=_bool(metadata.get("chart_remesh_measure_weld_metrics", True)),
        instant_meshes_path=_optional_str(metadata.get("instant_meshes_path")),
        quadriflow_path=_optional_str(metadata.get("quadriflow_path")),
        quadriflow_sharp=_bool(metadata.get("chart_remesh_quadriflow_sharp", metadata.get("quadriflow_sharp", True))),
        quadriflow_mcf=_bool(metadata.get("chart_remesh_quadriflow_mcf", metadata.get("quadriflow_mcf", False))),
    )


def remesh_plan_charts(
    input_path: str | Path,
    plan_path: str | Path,
    output_dir: str | Path,
    options: ChartRemeshOptions | None = None,
) -> ChartRemeshManifest:
    options = options or ChartRemeshOptions()
    input_path = Path(input_path)
    plan_path = Path(plan_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    mesh = load_mesh(input_path)
    charts = sorted(plan.get("charts", []), key=lambda chart: int(chart.get("face_count", 0)), reverse=True)
    entries: list[ChartRemeshEntry] = []
    skipped = 0
    failed = 0
    succeeded = 0

    for chart in charts:
        if len([entry for entry in entries if entry.status == "succeeded"]) >= max(0, options.max_charts):
            break
        chart_id = str(chart.get("id", f"chart_{len(entries):04d}"))
        face_count = int(chart.get("face_count", 0))
        operator = str(chart.get("recommended_operator", ""))
        chart_type = str(chart.get("chart_type", ""))
        face_indices = [int(value) for value in chart.get("face_indices", [])]
        reason = _skip_reason(chart, face_indices, options)
        if reason:
            skipped += 1
            continue

        chart_dir = output_dir / chart_id
        chart_dir.mkdir(parents=True, exist_ok=True)
        source_path = chart_dir / "source_chart.obj"
        output_path = chart_dir / "quad_chart.obj"
        report_path = chart_dir / "quad_report.json"
        target_quads = _chart_target_quads(chart, options)
        try:
            submesh = extract_chart_submesh(mesh, face_indices)
            submesh.export(source_path)
            report = quad_remesh_file(
                source_path,
                output_path,
                QuadRemeshOptions(
                    engine=options.engine,
                    target_faces=target_quads,
                    pure_quad=True,
                    instant_meshes_path=options.instant_meshes_path,
                    quadriflow_path=options.quadriflow_path,
                    quadriflow_sharp=options.quadriflow_sharp,
                    quadriflow_mcf=options.quadriflow_mcf,
                    weld_tolerance=options.weld_tolerance,
                    measure_weld_metrics=options.measure_weld_metrics,
                ),
            )
            report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
            entries.append(
                ChartRemeshEntry(
                    chart_id=chart_id,
                    status="succeeded",
                    source_face_count=face_count,
                    source_chart_type=chart_type,
                    recommended_operator=operator,
                    target_quads=target_quads,
                    source_submesh_path=str(source_path),
                    output_mesh_path=str(output_path),
                    report_path=str(report_path),
                )
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001 - one bad chart should not kill the manifest.
            failed += 1
            entries.append(
                ChartRemeshEntry(
                    chart_id=chart_id,
                    status="failed",
                    source_face_count=face_count,
                    source_chart_type=chart_type,
                    recommended_operator=operator,
                    target_quads=target_quads,
                    source_submesh_path=str(source_path),
                    output_mesh_path=str(output_path),
                    report_path=str(report_path),
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )

    summary = {
        "reported_chart_count": int(len(charts)),
        "attempted_chart_count": int(len(entries)),
        "succeeded_chart_count": int(succeeded),
        "failed_chart_count": int(failed),
        "skipped_chart_count": int(skipped),
        "note": "per-chart outputs are sidecars until seam stitching and Blender gates pass",
    }
    return ChartRemeshManifest(
        input_path=str(input_path),
        plan_path=str(plan_path),
        output_dir=str(output_dir),
        options=asdict(options),
        entries=entries,
        summary=summary,
    )


def write_chart_remesh_manifest(path: str | Path, manifest: ChartRemeshManifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def extract_chart_submesh(mesh: trimesh.Trimesh, face_indices: list[int]) -> trimesh.Trimesh:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    valid_faces = np.asarray([index for index in face_indices if 0 <= index < len(faces)], dtype=np.int64)
    if len(valid_faces) == 0:
        raise ValueError("chart has no valid face indices")
    chart_faces = faces[valid_faces]
    used_vertices = np.unique(chart_faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    sub_faces = remap[chart_faces]
    return trimesh.Trimesh(vertices=vertices[used_vertices], faces=sub_faces, process=False)


def _skip_reason(chart: dict[str, Any], face_indices: list[int], options: ChartRemeshOptions) -> str | None:
    if int(chart.get("face_count", 0)) < options.min_chart_faces:
        return "below_min_chart_faces"
    if chart.get("face_indices_truncated"):
        return "face_indices_truncated"
    if not face_indices:
        return "missing_face_indices"
    operator = str(chart.get("recommended_operator", ""))
    if operator == "cleanup_or_merge" and not options.allow_cleanup_charts:
        return "cleanup_chart_requires_merge"
    return None


def _chart_target_quads(chart: dict[str, Any], options: ChartRemeshOptions) -> int:
    requested = int(chart.get("target_quads", 0) or 0)
    if requested <= 0:
        requested = options.target_quads_total // max(1, options.max_charts)
    requested = max(options.min_target_quads, requested)
    return min(requested, max(options.min_target_quads, options.max_target_quads_per_chart))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
