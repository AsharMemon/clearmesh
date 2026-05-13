from pathlib import Path

import numpy as np
import trimesh

from clearmesh.mesh_heads.face_indexed import (
    FaceIndexedSequence,
    IndexedDecodeState,
    canonical_indexed_face_orientations,
    coordinate_tokens_to_indexed,
    decode_indexed_face_tokens_to_mesh,
    drop_geometric_degenerate_indexed_faces,
    encode_mesh_to_indexed_face_tokens,
    fill_indexed_boundary_loops,
    indexed_face_closure_counts,
    indexed_face_stats,
    indexed_to_coordinate_tokens,
    order_indexed_faces_boundary_growth,
    score_indexed_face_candidate,
    select_boundary_edge_action_face,
    select_constrained_indexed_face,
    select_topology_fallback_indexed_face,
    _boundary_loops,
    _candidate_preserves_vertex_links,
)
from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_indexed_face_decoder
from clearmesh.mesh_heads.face_tokens import FaceTokenTransform, encode_mesh_to_face_tokens
from clearmesh.mesh_heads.face_topology import face_token_topology_report
from scripts.research.train_face_indexed_conditioned_tiny import (
    _closure_count_targets,
    _edge_action_targets,
    _edge_choice_targets,
    _load_dataset,
)
from scripts.research.eval_face_indexed_conditioned_tiny import _teacher_identity_faces


def test_indexed_face_roundtrip_preserves_cube_watertightness():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    sequence = encode_mesh_to_indexed_face_tokens(mesh, num_bins=128)
    decoded = decode_indexed_face_tokens_to_mesh(sequence)
    tokens = indexed_to_coordinate_tokens(sequence)
    report = face_token_topology_report(tokens)

    assert sequence.face_count == len(mesh.faces)
    assert sequence.vertex_count == len(mesh.vertices)
    assert decoded.is_watertight
    assert report.watertight_edge_graph
    np.testing.assert_allclose(decoded.extents, mesh.extents, atol=0.04)


def test_teacher_identity_faces_are_exact_copied_targets():
    teacher = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    generated, stats = _teacher_identity_faces(teacher)
    generated[0, 0] = 99

    assert stats == {"token_accuracy": 1.0, "face_exact_ratio": 1.0}
    assert teacher[0, 0] == 0


def test_canonical_indexed_face_orientations_match_training_face_rotation():
    orientations = canonical_indexed_face_orientations(7, 3, 5)

    assert orientations == ((3, 5, 7), (3, 7, 5))


def test_indexed_face_encoding_is_deterministic_under_vertex_permutation():
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    rng = np.random.default_rng(13)
    order = rng.permutation(len(mesh.vertices))
    old_to_new = np.empty_like(order)
    old_to_new[order] = np.arange(len(order))
    shuffled = trimesh.Trimesh(vertices=mesh.vertices[order], faces=old_to_new[mesh.faces], process=False)

    original = encode_mesh_to_indexed_face_tokens(mesh, num_bins=128)
    permuted = encode_mesh_to_indexed_face_tokens(shuffled, num_bins=128)

    np.testing.assert_array_equal(original.vertices, permuted.vertices)
    np.testing.assert_array_equal(original.faces, permuted.faces)


def test_boundary_growth_order_reduces_connected_sequence_jumps():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    lex_sequence = encode_mesh_to_indexed_face_tokens(mesh, num_bins=128, face_order="lex")
    growth_sequence = encode_mesh_to_indexed_face_tokens(mesh, num_bins=128, face_order="boundary_growth")

    lex_closures = indexed_face_closure_counts(lex_sequence.faces)
    growth_closures = indexed_face_closure_counts(growth_sequence.faces)

    assert int(np.sum(growth_closures[1:] == 0)) == 0
    assert int(np.sum(growth_closures[1:] == 0)) <= int(np.sum(lex_closures[1:] == 0))
    assert {tuple(sorted(face)) for face in growth_sequence.faces.tolist()} == {
        tuple(sorted(face)) for face in lex_sequence.faces.tolist()
    }


def test_boundary_growth_order_starts_new_frontier_only_when_disconnected():
    faces = np.asarray(
        [
            [0, 1, 2],
            [10, 11, 12],
            [0, 1, 3],
            [10, 11, 13],
        ],
        dtype=np.int64,
    )

    ordered = order_indexed_faces_boundary_growth(faces)
    closures = indexed_face_closure_counts(ordered)

    assert int(np.sum(closures == 0)) == 2
    assert sorted(tuple(sorted(face)) for face in ordered.tolist()) == sorted(tuple(sorted(face)) for face in faces.tolist())


def test_indexed_boundary_loop_fill_closes_quad_hole():
    sequence = encode_mesh_to_indexed_face_tokens(trimesh.creation.box(), num_bins=128, face_order="boundary_growth")
    damaged = type(sequence)(
        vertices=sequence.vertices,
        faces=sequence.faces[:-2],
        num_bins=sequence.num_bins,
        transform=sequence.transform,
    )

    repaired, report = fill_indexed_boundary_loops(damaged)
    decoded = decode_indexed_face_tokens_to_mesh(repaired)

    assert report.input_boundary_edges > 0
    assert report.filled_faces > 0
    assert report.output_boundary_edges == 0
    assert decoded.is_watertight


def test_centroid_boundary_fill_avoids_degenerate_anchor_caps():
    sequence = FaceIndexedSequence(
        vertices=np.asarray(
            [
                [0, 0, 0],
                [10, 0, 0],
                [20, 0, 0],
                [10, 10, 0],
                [10, 3, 10],
            ],
            dtype=np.int64,
        ),
        faces=np.asarray(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ],
            dtype=np.int64,
        ),
        num_bins=128,
        transform=FaceTokenTransform(center=(0.0, 0.0, 0.0), scale=1.0),
    )

    fan_repaired, fan_report = fill_indexed_boundary_loops(sequence, strategy="fan")
    centroid_repaired, centroid_report = fill_indexed_boundary_loops(sequence, strategy="centroid")
    fan_decoded = decode_indexed_face_tokens_to_mesh(fan_repaired)
    centroid_decoded = decode_indexed_face_tokens_to_mesh(centroid_repaired)

    assert fan_report.skipped_loops == 1
    assert fan_report.output_boundary_edges == 4
    assert not fan_decoded.is_watertight
    assert centroid_report.output_boundary_edges == 0
    assert centroid_report.filled_faces == 4
    assert centroid_repaired.vertex_count == sequence.vertex_count + 1
    assert centroid_decoded.is_watertight


def test_drop_degenerate_indexed_faces_exposes_repairable_boundary():
    sequence = FaceIndexedSequence(
        vertices=np.asarray(
            [
                [0, 0, 0],
                [10, 0, 0],
                [20, 0, 0],
                [10, 10, 0],
                [10, 3, 10],
            ],
            dtype=np.int64,
        ),
        faces=np.asarray(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
                [0, 1, 2],
                [0, 2, 3],
            ],
            dtype=np.int64,
        ),
        num_bins=128,
        transform=FaceTokenTransform(center=(0.0, 0.0, 0.0), scale=1.0),
    )

    assert face_token_topology_report(indexed_to_coordinate_tokens(sequence)).watertight_edge_graph
    assert not decode_indexed_face_tokens_to_mesh(sequence).is_watertight

    filtered, dropped = drop_geometric_degenerate_indexed_faces(sequence)
    repaired, report = fill_indexed_boundary_loops(filtered, strategy="centroid")

    assert dropped == 1
    assert report.output_boundary_edges == 0
    assert decode_indexed_face_tokens_to_mesh(repaired).is_watertight


def test_boundary_loop_finder_splits_pinched_figure_eight_cycles():
    loops = _boundary_loops(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (0, 3),
            (3, 4),
            (4, 0),
        ]
    )

    assert sorted(tuple(loop) for loop in loops) == [(0, 1, 2), (0, 3, 4)]


def test_coordinate_tokens_convert_to_indexed_equivalent_topology():
    coord_sequence = encode_mesh_to_face_tokens(trimesh.creation.box(), num_bins=128)
    indexed = coordinate_tokens_to_indexed(
        coord_sequence.tokens,
        num_bins=coord_sequence.num_bins,
        transform=coord_sequence.transform,
    )
    expanded = indexed_to_coordinate_tokens(indexed)

    assert indexed.vertex_count == 8
    assert indexed.face_count == 12
    assert face_token_topology_report(expanded).watertight_edge_graph


def test_indexed_face_stats_report_graph_tokens():
    sequence = encode_mesh_to_indexed_face_tokens(trimesh.creation.box(), num_bins=128)
    stats = indexed_face_stats(sequence)

    assert stats["vertices"] == 8
    assert stats["faces"] == 12
    assert stats["index_tokens"] == 36
    assert stats["coordinate_tokens"] == 24
    assert stats["compression_vs_xyz_face_tokens"] < 1.0


def test_indexed_dataset_loader_reads_builder_shards(tmp_path: Path):
    mesh = trimesh.creation.box()
    sequence = encode_mesh_to_indexed_face_tokens(mesh, num_bins=128)
    shard = tmp_path / "sample.npz"
    np.savez_compressed(
        shard,
        indexed_vertices=sequence.vertices.astype(np.int16),
        indexed_faces=sequence.faces.astype(np.int32),
        center=np.asarray(sequence.transform.center, dtype=np.float32),
        scale=np.asarray([sequence.transform.scale], dtype=np.float32),
        num_bins=np.asarray([sequence.num_bins], dtype=np.int32),
        surface_points=np.zeros((8, 3), dtype=np.float32),
        surface_normals=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (8, 1)),
    )

    samples, num_bins = _load_dataset(tmp_path)

    assert num_bins == 128
    assert len(samples) == 1
    assert samples[0].vertices.shape == (8, 3)
    assert samples[0].faces.shape == (12, 3)


def test_constrained_indexed_selector_avoids_nonmanifold_edge():
    state = IndexedDecodeState.from_faces(np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64))
    logits = np.full((3, 6), -10.0, dtype=np.float64)
    logits[0, 0] = 10.0
    logits[1, 1] = 10.0
    logits[2, 4] = 10.0
    logits[0, 2] = 9.0
    logits[1, 3] = 9.0
    logits[2, 4] = 10.0

    selected = select_constrained_indexed_face(logits, state, vertex_count=6, top_k=4)

    assert set(selected.tolist()) != {0, 1, 4}
    assert not ({0, 1}.issubset(set(selected.tolist())))


def test_constrained_indexed_selector_prefers_boundary_closure():
    state = IndexedDecodeState.from_faces(np.asarray([[0, 1, 2]], dtype=np.int64))
    logits = np.zeros((3, 5), dtype=np.float64)
    logits[0, 0] = 1.0
    logits[1, 1] = 1.0
    logits[2, 3] = 1.0
    logits[0, 0] = 0.9
    logits[1, 2] = 0.9
    logits[2, 3] = 0.9

    selected = select_constrained_indexed_face(
        logits,
        state,
        vertex_count=5,
        top_k=4,
        closure_bonus=4.0,
        require_boundary_closure_after=1,
    )

    assert len(set(selected.tolist())) == 3
    assert any(edge.issubset(set(selected.tolist())) for edge in [{0, 1}, {1, 2}, {0, 2}])


def test_constrained_indexed_selector_rejects_pinched_vertex_link():
    state = IndexedDecodeState.from_faces(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64))
    logits = np.full((3, 6), -10.0, dtype=np.float64)
    logits[0, 0] = 10.0
    logits[1, 4] = 10.0
    logits[2, 5] = 10.0
    logits[1, 3] = 9.0
    logits[2, 4] = 9.0

    selected = select_constrained_indexed_face(
        logits,
        state,
        vertex_count=6,
        top_k=5,
        closure_bonus=0.0,
        new_edge_penalty=0.0,
        enforce_vertex_link_manifold=True,
    )

    assert set(selected.tolist()) != {0, 4, 5}
    assert {0, 3}.issubset(set(selected.tolist()))


def test_vertex_link_candidate_cache_reuses_and_invalidates():
    state = IndexedDecodeState.from_faces(np.asarray([[0, 1, 2]], dtype=np.int64))

    assert _candidate_preserves_vertex_links(state, (0, 2, 3))
    assert state.vertex_link_candidate_cache
    cached = dict(state.vertex_link_candidate_cache)

    assert _candidate_preserves_vertex_links(state, (0, 2, 3))
    assert state.vertex_link_candidate_cache == cached

    state.add_face((0, 2, 3))

    assert state.vertex_link_candidate_cache == {}


def test_constrained_selector_rejects_impossible_boundary_budget():
    state = IndexedDecodeState.from_faces(
        np.asarray(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
            ],
            dtype=np.int64,
        )
    )
    logits = np.full((3, 5), -10.0, dtype=np.float64)
    logits[0, 1] = 8.0
    logits[1, 2] = 8.0
    logits[2, 4] = 10.0
    logits[2, 3] = 8.0

    selected = select_constrained_indexed_face(
        logits,
        state,
        vertex_count=5,
        top_k=5,
        closure_bonus=0.0,
        new_edge_penalty=0.0,
        target_face_count=4,
    )

    assert set(selected.tolist()) == {1, 2, 3}


def test_boundary_budget_makes_edge_capacity_hard_even_when_relaxed():
    state = IndexedDecodeState.from_faces(np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64))

    score = score_indexed_face_candidate(
        (0, 1, 4),
        100.0,
        state,
        target_face_count=4,
        strict_manifold=False,
    )

    assert score is None


def test_topology_fallback_never_reuses_full_edges():
    state = IndexedDecodeState.from_faces(np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64))

    selected = select_topology_fallback_indexed_face(
        state,
        vertex_count=6,
        target_face_count=4,
    )

    assert selected is not None
    edges = (
        tuple(sorted((int(selected[0]), int(selected[1])))),
        tuple(sorted((int(selected[1]), int(selected[2])))),
        tuple(sorted((int(selected[2]), int(selected[0])))),
    )
    assert max(state.edge_counts[edge] for edge in edges) < 2


def test_boundary_edge_action_selector_forces_open_edge_completion():
    state = IndexedDecodeState.from_faces(np.asarray([[0, 1, 2]], dtype=np.int64))
    logits = np.zeros((3, 6), dtype=np.float64)
    logits[0, 3] = 10.0
    logits[1, 4] = 10.0
    logits[2, 5] = 10.0
    logits[0, 0] = 7.0
    logits[1, 1] = 7.0
    logits[2, 3] = 7.0

    selected = select_boundary_edge_action_face(
        logits,
        state,
        vertex_count=6,
        top_k=6,
        closure_bonus=0.0,
        new_edge_penalty=0.0,
    )

    selected_set = set(selected.tolist())
    assert selected_set != {3, 4, 5}
    assert any(edge.issubset(selected_set) for edge in [{0, 1}, {1, 2}, {0, 2}])


def test_boundary_edge_action_selector_falls_back_without_boundary_edges():
    state = IndexedDecodeState.empty()
    logits = np.full((3, 5), -10.0, dtype=np.float64)
    logits[0, 0] = 4.0
    logits[1, 1] = 4.0
    logits[2, 2] = 4.0

    selected = select_boundary_edge_action_face(logits, state, vertex_count=5, top_k=4)

    assert selected.tolist() == [0, 1, 2]


def test_constrained_selector_can_use_learned_closure_target_scores():
    state = IndexedDecodeState.from_faces(np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64))
    logits = np.zeros((3, 5), dtype=np.float64)

    selected = select_constrained_indexed_face(
        logits,
        state,
        vertex_count=5,
        top_k=5,
        closure_bonus=0.0,
        new_edge_penalty=0.0,
        require_boundary_closure_after=1,
        closure_target_scores=np.asarray([-5.0, -5.0, 5.0, -5.0], dtype=np.float64),
        closure_target_bonus=1.0,
    )

    assert set(selected.tolist()) in ({0, 2, 3}, {1, 2, 3})


def test_indexed_closure_count_targets_follow_teacher_edge_rhythm():
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )

    labels = _closure_count_targets(faces)

    np.testing.assert_array_equal(labels, np.asarray([0, 1, 2, 3], dtype=np.int64))


def test_indexed_edge_action_targets_follow_boundary_completion():
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )

    edges, thirds = _edge_action_targets(faces)

    np.testing.assert_array_equal(edges[0], np.asarray([-1, -1], dtype=np.int64))
    assert thirds[0] == -100
    assert tuple(edges[1]) == (0, 1)
    assert thirds[1] == 3
    assert thirds[2] in {2, 3}
    assert thirds[3] in {1, 2, 3}


def test_indexed_edge_choice_targets_put_teacher_edge_first():
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )

    candidates, labels = _edge_choice_targets(faces, max_choices=4)

    assert labels[0] == -100
    assert labels[1] == 0
    np.testing.assert_array_equal(candidates[1, 0], np.asarray([0, 1], dtype=np.int64))
    assert labels[2] == 0
    assert tuple(candidates[2, 0]) in {(0, 2), (0, 3), (2, 3)}


def test_seed_face_head_outputs_three_corner_logits():
    import torch

    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=16,
        max_vertices=5,
        max_faces=4,
        point_feature_dim=6,
        hidden_size=16,
        layers=1,
        heads=4,
        condition_tokens=2,
    )
    point_features = torch.zeros((2, 8, 6), dtype=torch.float32)
    vertex_table = torch.tensor(
        [
            [[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -1, -1]],
            [[1, 1, 1], [2, 3, 4], [5, 6, 7], [-1, -1, -1], [-1, -1, -1]],
        ],
        dtype=torch.long,
    )

    logits = model.seed_face_logits(point_features, vertex_table)

    assert logits.shape == (2, 3, 5)


def test_geometry_edge_heads_use_vertex_table_coordinates():
    import torch

    torch.manual_seed(7)
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=32,
        max_vertices=5,
        max_faces=4,
        point_feature_dim=6,
        hidden_size=16,
        layers=1,
        heads=4,
        condition_tokens=2,
        edge_head_mode="geometry",
    )
    point_features = torch.zeros((1, 8, 6), dtype=torch.float32)
    input_faces = torch.full((1, 2, 3), -1, dtype=torch.long)
    edge_indices = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)
    vertex_table = torch.tensor(
        [[[0, 0, 0], [8, 0, 0], [0, 8, 0], [0, 0, 8], [-1, -1, -1]]],
        dtype=torch.long,
    )
    moved_vertex_table = vertex_table.clone()
    moved_vertex_table[0, 2] = torch.tensor([31, 31, 31], dtype=torch.long)

    logits = model.forward_edge_action(point_features, vertex_table, input_faces, edge_indices)
    moved_logits = model.forward_edge_action(point_features, moved_vertex_table, input_faces, edge_indices)
    choice_logits = model.forward_edge_choice(point_features, vertex_table, input_faces, edge_indices)

    assert logits.shape == (1, 2, 5)
    assert choice_logits.shape == (1, 2)
    assert torch.isneginf(logits[0, 0, 0]) or logits[0, 0, 0] < -1e8
    assert not torch.allclose(logits, moved_logits)
