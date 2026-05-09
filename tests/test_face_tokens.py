import numpy as np
import pytest
import trimesh

from clearmesh.mesh_heads.face_tokens import (
    decode_face_tokens_to_mesh,
    decode_paper_face_tokens_to_mesh,
    encode_mesh_to_face_tokens,
    encode_mesh_to_paper_face_tokens,
    face_token_stats,
)
from clearmesh.mesh_heads.face_topology import (
    face_token_topology_report,
    repair_face_tokens,
    topology_coordinate_weights,
    topology_event_labels,
)


def test_face_token_roundtrip_preserves_cube_watertightness():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    sequence = encode_mesh_to_face_tokens(mesh, num_bins=128)
    decoded = decode_face_tokens_to_mesh(sequence)

    assert sequence.face_count == len(mesh.faces)
    assert decoded.is_watertight
    assert len(decoded.faces) == len(mesh.faces)
    np.testing.assert_allclose(decoded.extents, mesh.extents, atol=0.04)


def test_paper_face_token_roundtrip_uses_zyx_coordinates():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    sequence = encode_mesh_to_paper_face_tokens(mesh, num_bins=128)
    decoded = decode_paper_face_tokens_to_mesh(sequence)

    assert sequence.tokens.shape == (len(mesh.faces), 9)
    assert decoded.is_watertight
    np.testing.assert_allclose(decoded.extents, mesh.extents, atol=0.04)


def test_paper_face_token_order_sorts_by_minimum_zyx_vertex():
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    sequence = encode_mesh_to_paper_face_tokens(mesh, num_bins=128)
    faces = sequence.tokens.reshape(-1, 3, 3)
    min_vertices = np.asarray([face[np.lexsort((face[:, 2], face[:, 1], face[:, 0]))[0]] for face in faces])
    order = np.lexsort((min_vertices[:, 2], min_vertices[:, 1], min_vertices[:, 0]))

    np.testing.assert_array_equal(order, np.arange(len(order)))


def test_paper_face_token_within_face_order_variants():
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=float,
        ),
        faces=np.asarray([[1, 2, 0], [3, 1, 0]], dtype=np.int64),
        process=False,
    )

    preserved = encode_mesh_to_paper_face_tokens(mesh, num_bins=128, within_face_order="preserve")
    rotated = encode_mesh_to_paper_face_tokens(mesh, num_bins=128, within_face_order="rotate_min_zyx")
    sorted_seq = encode_mesh_to_paper_face_tokens(mesh, num_bins=128, within_face_order="sort_zyx")

    assert preserved.tokens.shape == rotated.tokens.shape == sorted_seq.tokens.shape
    assert not np.array_equal(preserved.tokens, rotated.tokens)
    first_face = rotated.tokens.reshape(-1, 3, 3)[0]
    min_offset = np.lexsort((first_face[:, 2], first_face[:, 1], first_face[:, 0]))[0]
    assert int(min_offset) == 0


def test_paper_face_token_encoding_drops_colinear_quantized_faces():
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )

    sequence = encode_mesh_to_paper_face_tokens(mesh, num_bins=128)
    decoded = decode_paper_face_tokens_to_mesh(sequence)

    assert sequence.face_count == 1
    assert len(decoded.faces) == 1
    assert float(decoded.area) > 0.0


def test_face_token_encoding_is_deterministic_under_vertex_permutation():
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    rng = np.random.default_rng(9)
    order = rng.permutation(len(mesh.vertices))
    old_to_new = np.empty_like(order)
    old_to_new[order] = np.arange(len(order))
    shuffled = trimesh.Trimesh(vertices=mesh.vertices[order], faces=old_to_new[mesh.faces], process=False)

    original = encode_mesh_to_face_tokens(mesh, num_bins=128)
    permuted = encode_mesh_to_face_tokens(shuffled, num_bins=128)

    np.testing.assert_array_equal(original.tokens, permuted.tokens)


def test_face_token_stats_report_vertex_welding_ratio():
    mesh = trimesh.creation.box()
    sequence = encode_mesh_to_face_tokens(mesh, num_bins=128)
    stats = face_token_stats(sequence)

    assert stats["faces"] == len(mesh.faces)
    assert stats["coordinate_tokens"] == len(mesh.faces) * 9
    assert stats["unique_quantized_vertices"] == len(mesh.vertices)
    assert stats["duplicate_coordinate_ratio"] > 0.0


def test_face_token_max_faces_guard():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    with pytest.raises(ValueError, match="above max_faces"):
        encode_mesh_to_face_tokens(mesh, num_bins=128, max_faces=12)


def test_face_token_topology_reports_cube_as_closed():
    sequence = encode_mesh_to_face_tokens(trimesh.creation.box(), num_bins=128)
    report = face_token_topology_report(sequence.tokens)

    assert report.face_count == len(sequence.tokens)
    assert report.watertight_edge_graph
    assert report.boundary_edge_count == 0
    assert report.nonmanifold_edge_count == 0
    assert report.edge_pairing_ratio == 1.0


def test_face_token_topology_reports_missing_face_boundary():
    sequence = encode_mesh_to_face_tokens(trimesh.creation.box(), num_bins=128)
    report = face_token_topology_report(sequence.tokens[:-1])

    assert not report.watertight_edge_graph
    assert report.boundary_edge_count > 0
    assert report.edge_pairing_ratio < 1.0


def test_face_token_repair_drops_duplicate_and_fills_triangle_hole():
    sequence = encode_mesh_to_face_tokens(trimesh.creation.box(), num_bins=128)
    damaged = np.concatenate([sequence.tokens[:-1], sequence.tokens[:1]], axis=0)

    repaired, repair_report = repair_face_tokens(damaged, mode="dedupe")
    report = face_token_topology_report(repaired)

    assert repair_report.dropped_duplicate_faces == 1
    assert repair_report.filled_triangle_holes >= 1
    assert report.watertight_edge_graph


def test_topology_coordinate_weights_emphasize_reused_vertices():
    sequence = encode_mesh_to_face_tokens(trimesh.creation.box(), num_bins=128)
    weights = topology_coordinate_weights(
        sequence.tokens,
        reuse_vertex_weight=0.5,
        edge_closure_weight=1.0,
    )

    assert weights.shape == sequence.tokens.shape
    assert float(weights.max()) > 1.0
    assert float(weights.mean()) > 1.0


def test_topology_event_labels_report_reuse_and_edge_closure():
    sequence = encode_mesh_to_face_tokens(trimesh.creation.box(), num_bins=128)
    labels = topology_event_labels(sequence.tokens)

    assert labels["reuse_vertex"].shape == (sequence.face_count, 3)
    assert labels["edge_closure_count"].shape == (sequence.face_count,)
    assert int(labels["reuse_vertex"].sum()) > 0
    assert int(labels["edge_closure_count"].sum()) > 0
