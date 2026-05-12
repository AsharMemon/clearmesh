from __future__ import annotations

from pathlib import Path

import trimesh

from clearmesh.retopology.feature_graph import (
    RetopologyPlanningOptions,
    analyze_retopology_file,
    write_retopology_plan,
)
from clearmesh.retopology.chart_remesh import ChartRemeshOptions, remesh_plan_charts
from clearmesh.retopology.chart_stitch import ChartStitchOptions, stitch_chart_remesh_outputs


def test_retopology_plan_finds_hard_surface_box_charts(tmp_path: Path):
    mesh_path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(1, 2, 3)).export(mesh_path)

    plan = analyze_retopology_file(
        mesh_path,
        RetopologyPlanningOptions(crease_angle_degrees=30, min_chart_faces=1, target_quads=120),
    )

    assert plan.summary["chart_count"] == 6
    assert plan.summary["feature_edge_count"] >= 12
    assert plan.summary["feature_curve_count"] >= 1
    assert plan.summary["operator_counts"]
    assert all(chart.recommended_operator != "" for chart in plan.charts)
    assert sum(chart.target_quads for chart in plan.charts) >= 24


def test_retopology_plan_keeps_smooth_sphere_generic(tmp_path: Path):
    mesh_path = tmp_path / "sphere.obj"
    trimesh.creation.icosphere(subdivisions=2, radius=1.0).export(mesh_path)

    plan = analyze_retopology_file(
        mesh_path,
        RetopologyPlanningOptions(crease_angle_degrees=75, min_chart_faces=1, target_quads=500),
    )

    assert plan.summary["chart_count"] == 1
    assert plan.charts[0].chart_type == "smooth_freeform"
    assert plan.charts[0].recommended_operator == "cross_field_chart"


def test_retopology_plan_writes_json_report(tmp_path: Path):
    mesh_path = tmp_path / "box.obj"
    report_path = tmp_path / "retopology_plan.json"
    trimesh.creation.box(extents=(1, 1, 1)).export(mesh_path)

    plan = analyze_retopology_file(mesh_path, RetopologyPlanningOptions(min_chart_faces=1))
    write_retopology_plan(report_path, plan)

    assert report_path.exists()
    assert "chart_count" in report_path.read_text(encoding="utf-8")


def test_retopology_plan_merges_tiny_charts_when_requested(tmp_path: Path):
    mesh_path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(1, 1, 1)).export(mesh_path)

    plan = analyze_retopology_file(
        mesh_path,
        RetopologyPlanningOptions(
            crease_angle_degrees=30,
            min_chart_faces=1,
            merge_small_charts=True,
            merge_min_faces=3,
        ),
    )

    assert plan.summary["initial_chart_count"] == 6
    assert plan.summary["chart_count"] < plan.summary["initial_chart_count"]
    assert plan.summary["chart_merge"]["small_charts_merged"] > 0


def test_chart_remesh_manifest_runs_bounded_template_cages(tmp_path: Path):
    mesh_path = tmp_path / "box.obj"
    report_path = tmp_path / "retopology_plan.json"
    trimesh.creation.box(extents=(1, 2, 3)).export(mesh_path)
    plan = analyze_retopology_file(
        mesh_path,
        RetopologyPlanningOptions(crease_angle_degrees=30, min_chart_faces=1, target_quads=120),
    )
    write_retopology_plan(report_path, plan)

    manifest = remesh_plan_charts(
        mesh_path,
        report_path,
        tmp_path / "charts",
        ChartRemeshOptions(engine="template_cage", max_charts=2, min_chart_faces=1, target_quads_total=120),
    )

    assert manifest.summary["succeeded_chart_count"] == 2
    assert all(entry.output_mesh_path and Path(entry.output_mesh_path).exists() for entry in manifest.entries)


def test_chart_stitch_welds_adjacent_quad_outputs(tmp_path: Path):
    left = tmp_path / "left.obj"
    right = tmp_path / "right.obj"
    left.write_text(
        "\n".join(["v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0", "f 1 2 3 4", ""]),
        encoding="utf-8",
    )
    right.write_text(
        "\n".join(["v 1 0 0", "v 2 0 0", "v 2 1 0", "v 1 1 0", "f 1 2 3 4", ""]),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
{
  "entries": [
    {"status": "succeeded", "output_mesh_path": "%s"},
    {"status": "succeeded", "output_mesh_path": "%s"}
  ]
}
"""
        % (left, right),
        encoding="utf-8",
    )

    report = stitch_chart_remesh_outputs(
        manifest,
        tmp_path / "stitched.obj",
        ChartStitchOptions(weld_tolerance=1e-8, max_components=1, max_boundary_loops=1),
    )

    assert report.stitched_chart_count == 2
    assert report.quad_stats["quad_ratio"] == 1.0
    assert report.mesh_metrics["connected_components"] == 1
    assert report.promotion["promoted"]
