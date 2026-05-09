import pytest

torch = pytest.importorskip("torch")

from clearmesh.mesh_heads.face_paper import build_paper_face_arae


FACE_EMBEDDING_VARIANTS = ["token_concat_project", "continuous_mlp", "discrete_sum"]
CAUSAL_MLP_VARIANTS = ["legacy_concat", "paper_chain"]


@pytest.mark.parametrize("causal_mlp_variant", CAUSAL_MLP_VARIANTS)
@pytest.mark.parametrize("face_embedding_variant", FACE_EMBEDDING_VARIANTS)
def test_incremental_hidden_matches_full_causal_hidden(face_embedding_variant, causal_mlp_variant):
    torch.manual_seed(7)
    model = build_paper_face_arae(
        num_bins=16,
        max_faces=5,
        point_feature_dim=6,
        hidden_size=32,
        encoder_hidden_size=32,
        encoder_layers=1,
        decoder_layers=2,
        heads=4,
        vecset_tokens=8,
        latent_dim=16,
        causal_mlp_variant=causal_mlp_variant,
        face_embedding_variant=face_embedding_variant,
    )
    model.eval()
    point_features = torch.randn(1, 24, 6)
    target_faces = torch.randint(0, 16, (1, 5, 9))
    input_faces = torch.full((1, 5, 9), -1, dtype=torch.long)
    input_faces[:, 1:] = target_faces[:, :-1]

    with torch.no_grad():
        full_hidden = model.hidden(point_features, input_faces)
        cache = model.init_incremental_cache(point_features)
        incremental_hidden = []
        for position in range(input_faces.shape[1]):
            incremental_hidden.append(model.incremental_hidden_step(input_faces[:, position], position, cache))
        incremental_hidden = torch.cat(incremental_hidden, dim=1)

    torch.testing.assert_close(incremental_hidden, full_hidden, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("causal_mlp_variant", CAUSAL_MLP_VARIANTS)
@pytest.mark.parametrize("face_embedding_variant", FACE_EMBEDDING_VARIANTS)
def test_greedy_face_decode_matches_slotwise_logits(face_embedding_variant, causal_mlp_variant):
    torch.manual_seed(11)
    model = build_paper_face_arae(
        num_bins=12,
        max_faces=3,
        point_feature_dim=6,
        hidden_size=24,
        encoder_hidden_size=24,
        encoder_layers=1,
        decoder_layers=1,
        heads=4,
        vecset_tokens=4,
        latent_dim=12,
        causal_mlp_variant=causal_mlp_variant,
        face_embedding_variant=face_embedding_variant,
    )
    model.eval()
    hidden = torch.randn(2, 1, 24)

    greedy = model.greedy_face_from_hidden(hidden, limit_bins=12)
    slotwise = torch.full((2, 9), -1, dtype=torch.long)
    with torch.no_grad():
        for coord in range(9):
            logits = model._causal_logits_from_hidden(hidden, slotwise.reshape(2, 1, 9))[:, 0, :, :]
            slotwise[:, coord] = torch.argmax(logits[:, coord, :12], dim=-1)

    torch.testing.assert_close(greedy, slotwise)


@pytest.mark.parametrize("causal_mlp_variant", CAUSAL_MLP_VARIANTS)
@pytest.mark.parametrize("face_embedding_variant", FACE_EMBEDDING_VARIANTS)
def test_incremental_free_run_matches_full_recompute(face_embedding_variant, causal_mlp_variant):
    torch.manual_seed(13)
    model = build_paper_face_arae(
        num_bins=10,
        max_faces=6,
        point_feature_dim=6,
        hidden_size=32,
        encoder_hidden_size=32,
        encoder_layers=1,
        decoder_layers=2,
        heads=4,
        vecset_tokens=8,
        latent_dim=16,
        causal_mlp_variant=causal_mlp_variant,
        face_embedding_variant=face_embedding_variant,
    )
    model.eval()
    point_features = torch.randn(1, 28, 6)
    point_features[..., 3:6] = torch.nn.functional.normalize(point_features[..., 3:6], dim=-1)

    with torch.no_grad():
        cache = model.init_incremental_cache(point_features)
        previous = torch.full((1, 9), -1, dtype=torch.long)
        incremental_faces = []
        for position in range(6):
            hidden_step = model.incremental_hidden_step(previous, position, cache)
            previous = model.greedy_face_from_hidden(hidden_step, limit_bins=10)
            incremental_faces.append(previous)
        incremental = torch.stack(incremental_faces, dim=1).squeeze(2)

        input_faces = torch.full((1, 1, 9), -1, dtype=torch.long)
        recomputed_faces = []
        for _ in range(6):
            hidden_step = model.hidden(point_features, input_faces)[:, -1:, :]
            next_face = model.greedy_face_from_hidden(hidden_step, limit_bins=10)
            recomputed_faces.append(next_face)
            input_faces = torch.cat([input_faces, next_face.unsqueeze(1)], dim=1)
        recomputed = torch.stack(recomputed_faces, dim=1).squeeze(2)

    torch.testing.assert_close(incremental, recomputed)


@pytest.mark.parametrize("causal_mlp_variant", CAUSAL_MLP_VARIANTS)
@pytest.mark.parametrize("face_embedding_variant", FACE_EMBEDDING_VARIANTS)
def test_face_embedding_variants_support_backward(face_embedding_variant, causal_mlp_variant):
    torch.manual_seed(17)
    model = build_paper_face_arae(
        num_bins=11,
        max_faces=4,
        point_feature_dim=6,
        hidden_size=32,
        encoder_hidden_size=32,
        encoder_layers=1,
        decoder_layers=1,
        heads=4,
        vecset_tokens=8,
        latent_dim=16,
        causal_mlp_variant=causal_mlp_variant,
        face_embedding_variant=face_embedding_variant,
    )
    point_features = torch.randn(2, 24, 6)
    point_features[..., 3:6] = torch.nn.functional.normalize(point_features[..., 3:6], dim=-1)
    target_faces = torch.randint(0, 11, (2, 4, 9))
    input_faces = torch.full((2, 4, 9), -1, dtype=torch.long)
    input_faces[:, 1:] = target_faces[:, :-1]

    logits = model.forward_causal(point_features, input_faces, target_faces)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), target_faces.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss)


def test_paper_chain_prefix_is_order_sensitive():
    torch.manual_seed(23)
    model = build_paper_face_arae(
        num_bins=13,
        max_faces=2,
        point_feature_dim=6,
        hidden_size=36,
        encoder_hidden_size=36,
        encoder_layers=1,
        decoder_layers=1,
        heads=4,
        vecset_tokens=4,
        latent_dim=12,
        causal_mlp_variant="paper_chain",
        face_embedding_variant="token_concat_project",
    )
    hidden = torch.randn(1, 1, 36)
    prefix_ab = torch.tensor([[[1, 2, -1, -1, -1, -1, -1, -1, -1]]])
    prefix_ba = torch.tensor([[[2, 1, -1, -1, -1, -1, -1, -1, -1]]])

    logits_ab = model._causal_logits_from_hidden(hidden, prefix_ab)[:, 0, 2]
    logits_ba = model._causal_logits_from_hidden(hidden, prefix_ba)[:, 0, 2]

    assert not torch.allclose(logits_ab, logits_ba)
