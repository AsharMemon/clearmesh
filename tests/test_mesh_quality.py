from __future__ import annotations

import trimesh

from clearmesh.eval.mesh_quality import evaluate_mesh


def test_vertex_link_nonmanifold_detects_watertight_pinch(tmp_path):
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    ]
    faces = [
        (0, 2, 1),
        (0, 1, 3),
        (1, 2, 3),
        (2, 0, 3),
        (0, 4, 5),
        (0, 6, 4),
        (4, 6, 5),
        (5, 6, 0),
    ]
    path = tmp_path / "pinched_tetrahedra.glb"
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)

    metrics = evaluate_mesh(path)

    assert metrics["watertight"]
    assert metrics["nonmanifold_edge_count"] == 0
    assert metrics["nonmanifold_vertex_count"] == 1


def test_vertex_link_nonmanifold_accepts_regular_closed_mesh(tmp_path):
    path = tmp_path / "box.glb"
    trimesh.creation.box().export(path)

    metrics = evaluate_mesh(path)

    assert metrics["watertight"]
    assert metrics["nonmanifold_vertex_count"] == 0
