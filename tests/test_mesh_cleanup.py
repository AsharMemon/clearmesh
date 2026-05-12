from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import trimesh

from clearmesh.eval.mesh_quality import evaluate_mesh, evaluate_mesh_pair
from clearmesh.eval.blender_gates import BlenderGateOptions, run_blender_mesh_gates
from clearmesh.mesh.cleanup import CleanupOptions, cleanup_mesh_file, cleanup_options_from_metadata
from clearmesh.mesh.coarse_adapter import CoarseAdapterOptions, adapt_coarse_mesh_file
from clearmesh.mesh.normalization import SurfaceNormalizationOptions, normalize_surface_file
from clearmesh.mesh.shrinkwrap import ShrinkwrapOptions, shrinkwrap_file
from clearmesh.mesh.shrinkwrap import shrinkwrap_obj_vertices_file
from clearmesh.product.mesh_passport import create_mesh_passport
from clearmesh.parts.component_manifest import write_component_parts_manifest
from clearmesh.parts.manifest import load_parts_manifest
from clearmesh.retopology.quad_remesh import QuadRemeshOptions, quad_mesh_stats, quad_remesh_file, weld_obj_vertices


def box_at(offset: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    mesh.apply_translation((offset, 0, 0))
    return mesh


def single_triangle(offset: float) -> trimesh.Trimesh:
    vertices = np.array([[offset, 0, 0], [offset + 0.1, 0, 0], [offset, 0.1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def pinched_tetrahedra() -> trimesh.Trimesh:
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        ],
        dtype=float,
    )
    faces = np.array(
        [
            (0, 2, 1),
            (0, 1, 3),
            (1, 2, 3),
            (2, 0, 3),
            (0, 4, 5),
            (0, 6, 4),
            (4, 6, 5),
            (5, 6, 0),
        ],
        dtype=int,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_cleanup_removes_tiny_components(tmp_path: Path):
    mesh = trimesh.util.concatenate([box_at(0), box_at(3), single_triangle(6), single_triangle(7)])
    input_path = tmp_path / "dirty.obj"
    output_path = tmp_path / "clean.obj"
    mesh.export(input_path)

    before = evaluate_mesh(input_path)
    report = cleanup_mesh_file(input_path, output_path, CleanupOptions(min_component_faces=8))
    after = evaluate_mesh(output_path)

    assert output_path.exists()
    assert report.removed_components >= 2
    assert after["connected_components"] < before["connected_components"]
    assert after["face_count"] < before["face_count"]


def test_cleanup_can_unweld_pinched_nonmanifold_vertices(tmp_path: Path):
    input_path = tmp_path / "pinched.glb"
    output_path = tmp_path / "unpinched.glb"
    pinched_tetrahedra().export(input_path)

    before = evaluate_mesh(input_path)
    report = cleanup_mesh_file(
        input_path,
        output_path,
        CleanupOptions(min_component_faces=1, split_nonmanifold_vertices=True, merge_vertices=False),
    )
    after = evaluate_mesh(output_path)

    assert before["watertight"]
    assert before["nonmanifold_vertex_count"] == 1
    assert after["watertight"]
    assert after["nonmanifold_vertex_count"] == 0
    assert report.split_vertices_added == 1


def test_cleanup_metadata_splits_nonmanifold_vertices_by_default():
    options = cleanup_options_from_metadata({})

    assert options.split_nonmanifold_vertices


def test_cleanup_keeps_dominant_component_when_tiny_bubbles_are_allowed(tmp_path: Path):
    main = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    debris = [single_triangle(3 + i * 0.25) for i in range(20)]
    mesh = trimesh.util.concatenate([main, *debris])
    input_path = tmp_path / "ultrashape_with_bubbles.obj"
    output_path = tmp_path / "dominant.obj"
    mesh.export(input_path)

    report = cleanup_mesh_file(
        input_path,
        output_path,
        CleanupOptions(
            min_component_faces=1,
            min_component_face_ratio=0.0,
            dominant_component_face_ratio=0.8,
            fill_holes=True,
        ),
    )
    after = evaluate_mesh(output_path)

    assert output_path.exists()
    assert report.removed_components >= 20
    assert after["connected_components"] == 1
    assert after["tiny_component_count"] == 0
    assert after["watertight"]


def test_mesh_passport_flags_fragmented_mesh(tmp_path: Path):
    mesh = trimesh.util.concatenate([single_triangle(i) for i in range(30)])
    input_path = tmp_path / "fragments.obj"
    mesh.export(input_path)

    passport = create_mesh_passport(input_path)

    assert passport.normalized_surface_required
    assert not passport.high_resolution_ready
    assert passport.risk_level in {"high", "medium"}


def test_pair_metrics_include_scale_normalized_distances(tmp_path: Path):
    mesh = trimesh.creation.box(extents=(10, 2, 1))
    left = tmp_path / "left.glb"
    right = tmp_path / "right.glb"
    mesh.export(left)
    mesh.export(right)

    metrics = evaluate_mesh_pair(left, right, samples=500)

    assert metrics["reference_bbox_diagonal"] > 0
    assert np.isclose(metrics["chamfer_l2_normalized"], metrics["chamfer_l2"] / (metrics["reference_bbox_diagonal"] ** 2))
    assert np.isclose(metrics["hausdorff_l2_normalized"], metrics["hausdorff_l2"] / metrics["reference_bbox_diagonal"])


def test_surface_normalization_cleanup_fallback_exports_control_mesh(tmp_path: Path):
    mesh = trimesh.util.concatenate([box_at(0), box_at(3), single_triangle(6)])
    input_path = tmp_path / "dirty.obj"
    output_path = tmp_path / "control.obj"
    mesh.export(input_path)

    report = normalize_surface_file(
        input_path,
        output_path,
        SurfaceNormalizationOptions(engine="cleanup", min_component_faces=8, target_faces=100),
    )

    assert output_path.exists()
    assert report.engine == "cleanup"
    assert report.output_metrics["ok"]
    assert report.output_metrics["connected_components"] < report.input_metrics["connected_components"]


def test_coarse_adapter_exports_accepted_single_proxy(tmp_path: Path):
    mesh = trimesh.util.concatenate([box_at(0), single_triangle(4), single_triangle(5)])
    input_path = tmp_path / "trellis_fragments.obj"
    output_path = tmp_path / "coarse.glb"
    mesh.export(input_path)

    report = adapt_coarse_mesh_file(
        input_path,
        output_path,
        CoarseAdapterOptions(
            engine="convex_hull",
            min_component_faces=8,
            keep_largest_components=1,
            target_faces=128,
        ),
    )

    assert output_path.exists()
    assert report.accepted
    assert report.engine == "convex_hull"
    assert report.output_metrics["watertight"]
    assert report.output_metrics["connected_components"] == 1


def test_coarse_adapter_target_faces_zero_disables_simplification(tmp_path: Path):
    input_path = tmp_path / "sphere.obj"
    output_path = tmp_path / "voxel_shell.glb"
    trimesh.creation.icosphere(subdivisions=2, radius=1.0).export(input_path)

    report = adapt_coarse_mesh_file(
        input_path,
        output_path,
        CoarseAdapterOptions(
            engine="voxel_shell",
            fallback="",
            target_faces=0,
            voxel_resolution=16,
            voxel_dilate=1,
            voxel_close=1,
            sample_points=5000,
            mesh_voxel_max_faces=5000,
            require_watertight=True,
        ),
    )

    assert output_path.exists()
    assert report.accepted
    assert report.output_metrics["watertight"]
    assert report.output_metrics["nonmanifold_edge_count"] == 0


def test_shrinkwrap_projects_control_mesh_toward_target(tmp_path: Path):
    source = trimesh.creation.icosphere(subdivisions=1, radius=0.75)
    target = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    source_path = tmp_path / "source.obj"
    target_path = tmp_path / "target.obj"
    output_path = tmp_path / "wrapped.obj"
    source.export(source_path)
    target.export(target_path)

    before = source.vertices.copy()
    report = shrinkwrap_file(
        source_path,
        target_path,
        output_path,
        ShrinkwrapOptions(sample_points=2000, iterations=3, attraction=0.8, smoothing=0.02, max_step_ratio=0.2),
    )
    wrapped = trimesh.load(output_path, force="mesh")

    assert output_path.exists()
    assert report.output_metrics["ok"]
    assert len(wrapped.faces) == len(source.faces)
    assert np.mean(np.linalg.norm(wrapped.vertices, axis=1)) > np.mean(np.linalg.norm(before, axis=1))


def test_shrinkwrap_obj_projection_preserves_quad_faces(tmp_path: Path):
    source_path = tmp_path / "quad.obj"
    target_path = tmp_path / "target.obj"
    output_path = tmp_path / "projected.obj"
    source_path.write_text(
        "\n".join(
            [
                "v -0.5 -0.5 0",
                "v 0.5 -0.5 0",
                "v 0.5 0.5 0",
                "v -0.5 0.5 0",
                "f 1 2 3 4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    trimesh.creation.icosphere(subdivisions=1, radius=1.0).export(target_path)

    report = shrinkwrap_obj_vertices_file(
        source_path,
        target_path,
        output_path,
        ShrinkwrapOptions(iterations=1, sample_points=1000, attraction=0.2, smoothing=0.0),
    )

    assert output_path.exists()
    assert report.output_metrics["ok"]
    assert quad_mesh_stats(output_path)["quad_ratio"] == 1.0


def test_quad_remesh_template_cage_outputs_scored_quads(tmp_path: Path):
    input_path = tmp_path / "source.obj"
    output_path = tmp_path / "quad.obj"
    trimesh.creation.box(extents=(1, 2, 3)).export(input_path)

    report = quad_remesh_file(
        input_path,
        output_path,
        QuadRemeshOptions(engine="template_cage", cage_subdivisions=2, target_faces=24),
    )

    assert output_path.exists()
    assert report.engine == "template_cage"
    assert report.quad_stats["pure_quad"]
    assert report.quad_stats["quad_ratio"] == 1.0
    assert report.quad_stats["face_count"] == 24
    assert report.pre_weld_metrics["ok"]
    assert report.mesh_metrics["ok"]
    assert report.weld_effect["ok"]


def test_quad_remesh_pyinstantmeshes_uses_array_api_for_glb(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "source.glb"
    output_path = tmp_path / "quad.obj"
    trimesh.creation.box(extents=(1, 1, 1)).export(input_path)

    fake = types.SimpleNamespace()

    def remesh_file(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("file API should not receive GLB")

    def remesh(vertices, faces, **kwargs):  # noqa: ARG001
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3
        return (
            np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
            np.asarray([[0, 1, 2, 3]], dtype=int),
        )

    fake.remesh_file = remesh_file
    fake.remesh = remesh
    monkeypatch.setitem(sys.modules, "pyinstantmeshes", fake)

    report = quad_remesh_file(input_path, output_path, QuadRemeshOptions(engine="pyinstantmeshes"))

    assert output_path.exists()
    assert report.engine == "pyinstantmeshes"
    assert report.notes == []
    assert report.quad_stats["pure_quad"]


def test_quad_obj_weld_merges_duplicate_remesher_vertices(tmp_path: Path):
    path = tmp_path / "duplicated_quads.obj"
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "v 1 0 0",
                "v 2 0 0",
                "v 2 1 0",
                "v 1 1 0",
                "f 1 2 3 4",
                "f 5 6 7 8",
                "",
            ]
        ),
        encoding="utf-8",
    )

    postprocess = weld_obj_vertices(path, tolerance=1e-8)
    metrics = evaluate_mesh(path)

    assert postprocess["vertices_before"] == 8
    assert postprocess["vertices_after"] == 6
    assert quad_mesh_stats(path)["pure_quad"]
    assert metrics["connected_components"] == 1


def test_component_part_manifest_exports_mesh_and_point_clouds(tmp_path: Path):
    mesh = trimesh.util.concatenate([box_at(0), box_at(3)])
    mesh_path = tmp_path / "components.obj"
    mesh.export(mesh_path)

    manifest_path = write_component_parts_manifest(mesh_path, tmp_path / "parts", max_parts=2, min_faces=1, point_count=128)
    parts = load_parts_manifest(manifest_path)

    assert len(parts) == 2
    assert all(part.proxy_mesh_path and Path(part.proxy_mesh_path).exists() for part in parts)
    assert all(part.point_cloud_path and Path(part.point_cloud_path).exists() for part in parts)


def test_blender_gates_skip_when_binary_missing(tmp_path: Path):
    mesh_path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(1, 1, 1)).export(mesh_path)

    report = run_blender_mesh_gates(
        mesh_path,
        tmp_path / "blender",
        BlenderGateOptions(blender="/definitely/not/a/blender", timeout_seconds=1),
    )

    assert report.skipped
    assert report.reason == "blender_not_found"
