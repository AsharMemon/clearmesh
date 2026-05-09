import pytest

torch = pytest.importorskip("torch")

from clearmesh.mesh_heads.face_paper import build_paper_face_arae


@pytest.mark.parametrize("causal_mlp_variant", ["legacy_concat", "paper_chain"])
@pytest.mark.parametrize("face_embedding_variant", ["token_concat_project", "continuous_mlp", "discrete_sum"])
def test_incremental_hidden_matches_full_causal_hidden(face_embedding_variant, causal_mlp_variant):
    torch.manual_seed(7)
    model = build_paper_face_arae(
        num_bins=32,
        max_faces=12,
        point_feature_dim=6,
        hidden_size=32,
        encoder_hidden_size=32,
        encoder_layers=1,
        decoder_layers=2,
        heads=4,
        vecset_tokens=8,
        latent_dim=16,
        encoder_backend="shape2vecset",
        causal_mlp_variant=causal_mlp_variant,
        face_embedding_variant=face_embedding_variant,
    )
    model.eval()
    point_features = torch.randn(1, 32, 6)
    point_features[..., 3:6] = torch.nn.functional.normalize(point_features[..., 3:6], dim=-1)
    tokens = torch.randint(0, 32, (1, 12, 9))
    input_faces = torch.full((1, 12, 9), -1, dtype=torch.long)
    input_faces[:, 1:] = tokens[:, :-1]

    with torch.no_grad():
        full_hidden = model.hidden(point_features, input_faces)
        cache = model.init_incremental_cache(point_features)
        incremental = []
        for position in range(input_faces.shape[1]):
            hidden_step = model.incremental_hidden_step(input_faces[:, position], position, cache)
            incremental.append(hidden_step)
        incremental_hidden = torch.cat(incremental, dim=1)

    torch.testing.assert_close(incremental_hidden, full_hidden, rtol=1e-4, atol=1e-5)
