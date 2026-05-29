from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_indexed_face_decoder
from clearmesh.mesh_heads.face_indexed import IndexedDecodeState
from scripts.research.eval_face_indexed_conditioned_tiny import _select_corner_causal_boundary_face
from scripts.research.train_face_indexed_conditioned_tiny import _corner_closure_presence_loss


class _FakeBoundaryActionModel:
    def __init__(self, vertex_count: int = 8) -> None:
        self.vertex_count = int(vertex_count)

    def _corner_causal_logits_from_hidden(self, hidden, corner_prefix, vertex_table=None):  # type: ignore[no-untyped-def]
        logits = torch.zeros(hidden.shape[0], 1, 3, self.vertex_count, device=hidden.device)
        logits[:, :, 0, 2] = 10.0
        logits[:, :, 0, 1] = 9.0
        return logits

    def _edge_action_logits_from_hidden(self, hidden, edge_indices, vertex_table=None):  # type: ignore[no-untyped-def]
        logits = torch.zeros(hidden.shape[0], 1, self.vertex_count, device=hidden.device)
        logits[:, :, 1] = 5.0
        return logits


def test_boundary_action_bonus_scores_original_third_after_canonical_rotation() -> None:
    state = IndexedDecodeState.empty()
    state.edge_counts[(5, 6)] = 1
    state.accepted_faces = 1
    model = _FakeBoundaryActionModel()
    hidden = torch.zeros(1, 1, 4)
    vertex_table = torch.zeros(1, 8, 3, dtype=torch.long)
    point_features = torch.zeros(1, 4, 6)
    input_faces = torch.full((1, 1, 3), -1, dtype=torch.long)
    logits0 = torch.zeros(8).numpy()
    logits0[2] = 10.0
    logits0[1] = 9.0

    face = _select_corner_causal_boundary_face(
        model,
        hidden,
        vertex_table,
        point_features,
        input_faces,
        logits0,
        torch.device("cpu"),
        state,
        vertex_count=8,
        vertices=None,
        top_k=4,
        local_candidate_neighbors=0,
        closure_bonus=0.0,
        new_edge_penalty=0.0,
        edge_length_penalty=0.0,
        aspect_penalty=0.0,
        edge_action_bonus=1.0,
        edge_action_candidate_top_k=2,
        edge_choice_bonus=0.0,
        edge_choice_candidate_top_k=0,
        require_boundary_closure_after=1,
        closure_target_scores=None,
        closure_target_bonus=0.0,
        enforce_vertex_link_manifold=False,
        target_face_count=None,
    )

    assert face is not None
    assert set(face.tolist()) == {1, 5, 6}


def test_corner_closure_presence_loss_rewards_boundary_vertices() -> None:
    target_edge_actions = torch.tensor([[[1, 2], [-1, -1]]])
    target_edge_thirds = torch.tensor([[3, -100]])
    bad_logits = torch.zeros(1, 2, 3, 6, requires_grad=True)
    good_logits = torch.full((1, 2, 3, 6), -5.0, requires_grad=True)
    good_logits.data[0, 0, 0, 1] = 5.0
    good_logits.data[0, 0, 1, 2] = 5.0
    good_logits.data[0, 0, 2, 3] = 5.0

    bad_loss = _corner_closure_presence_loss(torch, bad_logits, target_edge_actions, target_edge_thirds)
    good_loss = _corner_closure_presence_loss(torch, good_logits, target_edge_actions, target_edge_thirds)

    assert good_loss.item() < bad_loss.item() * 0.1
    good_loss.backward()
    assert good_logits.grad is not None


def test_corner_closure_presence_loss_ignores_absent_targets() -> None:
    logits = torch.randn(1, 2, 3, 6, requires_grad=True)
    target_edge_actions = torch.full((1, 2, 2), -1)
    target_edge_thirds = torch.full((1, 2), -100)

    loss = _corner_closure_presence_loss(torch, logits, target_edge_actions, target_edge_thirds)

    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0


def test_voxset_spatial_cross_attention_forward_and_count_head() -> None:
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=32,
        max_vertices=8,
        max_faces=5,
        point_feature_dim=6,
        hidden_size=32,
        layers=1,
        heads=4,
        condition_tokens=6,
        condition_backend="voxset",
        decoder_backend="spatial_cross_attn",
        encoder_layers=1,
        latent_dim=16,
        face_output_mode="geometry",
        voxset_resolution=4,
        spatial_gate_sigma=0.5,
        spatial_gate_top_k=3,
    )
    point_features = torch.randn(2, 24, 6)
    point_features[..., :3] = point_features[..., :3].tanh()
    vertex_table = torch.randint(0, 32, (2, 8, 3))
    input_faces = torch.full((2, 5, 3), -1, dtype=torch.long)
    input_faces[:, 1:] = torch.randint(0, 8, (2, 4, 3))

    logits = model(point_features, vertex_table, input_faces)
    assert logits.shape == (2, 5, 3, 8)

    count_logits = model.predict_face_count_logits(point_features, vertex_table)
    assert count_logits.shape == (2, 6)


def test_voxset_spatial_modulated_cross_attention_forward() -> None:
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=32,
        max_vertices=8,
        max_faces=5,
        point_feature_dim=6,
        hidden_size=32,
        layers=1,
        heads=4,
        condition_tokens=6,
        condition_backend="voxset",
        decoder_backend="spatial_modulated_cross_attn",
        encoder_layers=1,
        latent_dim=16,
        face_output_mode="geometry",
        voxset_resolution=4,
        spatial_gate_sigma=0.5,
        spatial_gate_top_k=3,
    )
    point_features = torch.randn(2, 24, 6)
    point_features[..., :3] = point_features[..., :3].tanh()
    vertex_table = torch.randint(0, 32, (2, 8, 3))
    input_faces = torch.full((2, 5, 3), -1, dtype=torch.long)
    input_faces[:, 1:] = torch.randint(0, 8, (2, 4, 3))

    outputs = model(point_features, vertex_table, input_faces, return_hidden=True)

    assert outputs["face_logits"].shape == (2, 5, 3, 8)
    assert outputs["hidden"].shape == (2, 5, 32)


def test_voxset_spatial_topology_modulated_forward() -> None:
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=32,
        max_vertices=8,
        max_faces=5,
        point_feature_dim=6,
        hidden_size=32,
        layers=1,
        heads=4,
        condition_tokens=6,
        condition_backend="voxset",
        decoder_backend="spatial_topology_modulated_cross_attn",
        encoder_layers=1,
        latent_dim=16,
        face_output_mode="geometry",
        voxset_resolution=4,
        spatial_gate_sigma=0.5,
        spatial_gate_top_k=3,
    )
    point_features = torch.randn(2, 24, 6)
    point_features[..., :3] = point_features[..., :3].tanh()
    vertex_table = torch.randint(0, 32, (2, 8, 3))
    input_faces = torch.full((2, 5, 3), -1, dtype=torch.long)
    input_faces[:, 1:] = torch.randint(0, 8, (2, 4, 3))
    edge_actions = torch.randint(0, 8, (2, 5, 2))
    edge_choices = torch.randint(0, 8, (2, 5, 4, 2))

    outputs = model(
        point_features,
        vertex_table,
        input_faces,
        return_topology=True,
        edge_action_indices=edge_actions,
        edge_choice_candidates=edge_choices,
    )

    assert outputs["face_logits"].shape == (2, 5, 3, 8)
    assert outputs["closure_logits"].shape == (2, 5, 4)
    assert outputs["edge_action_logits"].shape == (2, 5, 8)
    assert outputs["edge_choice_logits"].shape == (2, 5, 4)


def test_spatial_cross_attention_requires_voxset_conditioning() -> None:
    with pytest.raises(ValueError, match="requires condition_backend='voxset'"):
        build_tiny_point_conditioned_indexed_face_decoder(
            num_bins=32,
            max_vertices=8,
            max_faces=5,
            condition_backend="vecset",
            decoder_backend="spatial_modulated_cross_attn",
        )


def test_voxset_spatial_gate_is_causal_with_future_faces() -> None:
    torch.manual_seed(7)
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=32,
        max_vertices=8,
        max_faces=5,
        point_feature_dim=6,
        hidden_size=32,
        layers=1,
        heads=4,
        condition_tokens=6,
        condition_backend="voxset",
        decoder_backend="spatial_cross_attn",
        encoder_layers=1,
        latent_dim=16,
        face_output_mode="geometry",
        voxset_resolution=4,
        spatial_gate_sigma=0.35,
        spatial_gate_top_k=3,
    )
    model.eval()
    point_features = torch.randn(1, 24, 6)
    point_features[..., :3] = point_features[..., :3].tanh()
    vertex_table = torch.randint(0, 32, (1, 8, 3))
    input_faces = torch.full((1, 5, 3), -1, dtype=torch.long)
    input_faces[:, 1:] = torch.randint(0, 8, (1, 4, 3))
    changed = input_faces.clone()
    changed[:, 4] = torch.tensor([[7, 6, 5]])

    with torch.no_grad():
        original_logits = model(point_features, vertex_table, input_faces)
        changed_logits = model(point_features, vertex_table, changed)

    torch.testing.assert_close(original_logits[:, :4], changed_logits[:, :4])


def test_voxset_spatial_modulated_gate_is_causal_with_future_faces() -> None:
    torch.manual_seed(11)
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=32,
        max_vertices=8,
        max_faces=5,
        point_feature_dim=6,
        hidden_size=32,
        layers=1,
        heads=4,
        condition_tokens=6,
        condition_backend="voxset",
        decoder_backend="spatial_modulated_cross_attn",
        encoder_layers=1,
        latent_dim=16,
        face_output_mode="geometry",
        voxset_resolution=4,
        spatial_gate_sigma=0.35,
        spatial_gate_top_k=3,
    )
    model.eval()
    point_features = torch.randn(1, 24, 6)
    point_features[..., :3] = point_features[..., :3].tanh()
    vertex_table = torch.randint(0, 32, (1, 8, 3))
    input_faces = torch.full((1, 5, 3), -1, dtype=torch.long)
    input_faces[:, 1:] = torch.randint(0, 8, (1, 4, 3))
    changed = input_faces.clone()
    changed[:, 4] = torch.tensor([[7, 6, 5]])

    with torch.no_grad():
        original_logits = model(point_features, vertex_table, input_faces)
        changed_logits = model(point_features, vertex_table, changed)

    torch.testing.assert_close(original_logits[:, :4], changed_logits[:, :4])


def test_voxset_spatial_topology_modulated_gate_is_causal_with_future_faces() -> None:
    torch.manual_seed(13)
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=32,
        max_vertices=8,
        max_faces=5,
        point_feature_dim=6,
        hidden_size=32,
        layers=1,
        heads=4,
        condition_tokens=6,
        condition_backend="voxset",
        decoder_backend="spatial_topology_modulated_cross_attn",
        encoder_layers=1,
        latent_dim=16,
        face_output_mode="geometry",
        voxset_resolution=4,
        spatial_gate_sigma=0.35,
        spatial_gate_top_k=3,
    )
    model.eval()
    point_features = torch.randn(1, 24, 6)
    point_features[..., :3] = point_features[..., :3].tanh()
    vertex_table = torch.randint(0, 32, (1, 8, 3))
    input_faces = torch.full((1, 5, 3), -1, dtype=torch.long)
    input_faces[:, 1:] = torch.randint(0, 8, (1, 4, 3))
    changed = input_faces.clone()
    changed[:, 4] = torch.tensor([[7, 6, 5]])

    with torch.no_grad():
        original = model(point_features, vertex_table, input_faces, return_topology=True)
        updated = model(point_features, vertex_table, changed, return_topology=True)

    torch.testing.assert_close(original["face_logits"][:, :4], updated["face_logits"][:, :4])
    torch.testing.assert_close(original["closure_logits"][:, :4], updated["closure_logits"][:, :4])


def test_spatial_topology_edge_next_helpers_use_local_head() -> None:
    torch.manual_seed(17)
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=32,
        max_vertices=8,
        max_faces=5,
        point_feature_dim=6,
        hidden_size=32,
        layers=1,
        heads=4,
        condition_tokens=6,
        condition_backend="voxset",
        decoder_backend="spatial_topology_modulated_cross_attn",
        encoder_layers=1,
        latent_dim=16,
        face_output_mode="geometry",
        voxset_resolution=4,
        spatial_gate_sigma=0.35,
        spatial_gate_top_k=3,
    )
    model.eval()
    point_features = torch.randn(1, 24, 6)
    point_features[..., :3] = point_features[..., :3].tanh()
    vertex_table = torch.randint(0, 32, (1, 8, 3))
    input_faces = torch.full((1, 5, 3), -1, dtype=torch.long)
    input_faces[:, 1:] = torch.randint(0, 8, (1, 4, 3))
    edge_action = torch.tensor([[1, 2]])
    edge_choices = torch.tensor([[[1, 2], [2, 3], [3, 4]]])

    with torch.no_grad():
        hidden = model._hidden(point_features, vertex_table, input_faces)[:, -1:, :]
        expected_action = model.edge_action_logits_from_hidden(
            hidden,
            edge_action.reshape(1, 1, 2),
            point_features=point_features,
            vertex_table=vertex_table,
            input_faces=input_faces,
        )[:, 0, :]
        expected_choice = model.edge_choice_logits_from_hidden(
            hidden,
            edge_choices.reshape(1, 1, 3, 2),
            point_features=point_features,
            vertex_table=vertex_table,
            input_faces=input_faces,
        )[:, 0, :]
        action = model.edge_action_next_logits(point_features, vertex_table, input_faces, edge_action)
        choice = model.edge_choice_next_logits(point_features, vertex_table, input_faces, edge_choices)

    torch.testing.assert_close(action, expected_action)
    torch.testing.assert_close(choice, expected_choice)
