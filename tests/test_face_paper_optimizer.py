import pytest

torch = pytest.importorskip("torch")

from clearmesh.mesh_heads.face_paper import build_paper_face_arae
from scripts.research.train_face_paper_faithful import _build_optimizer


class _Args:
    optimizer = "muon"
    lr = 6e-4
    weight_decay = 0.1


@pytest.mark.parametrize("causal_mlp_variant", ["legacy_concat", "paper_chain"])
def test_muon_optimizer_routes_embeddings_to_adamw(causal_mlp_variant):
    model = build_paper_face_arae(
        num_bins=32,
        max_faces=4,
        point_feature_dim=6,
        hidden_size=32,
        encoder_hidden_size=32,
        encoder_layers=1,
        decoder_layers=1,
        heads=4,
        vecset_tokens=4,
        latent_dim=16,
        causal_mlp_variant=causal_mlp_variant,
    )

    optimizer = _build_optimizer(torch, model, _Args())
    optimizers = getattr(optimizer, "optimizers", [optimizer])
    adamw_params = {
        id(param)
        for inner in optimizers
        if isinstance(inner, torch.optim.AdamW)
        for group in inner.param_groups
        for param in group["params"]
    }
    muon_params = {
        id(param)
        for inner in optimizers
        if not isinstance(inner, torch.optim.AdamW)
        for group in inner.param_groups
        for param in group["params"]
    }
    named = dict(model.named_parameters())

    for name, param in named.items():
        if "embedding" in name or "embed" in name or name == "bos_face":
            assert id(param) in adamw_params
            assert id(param) not in muon_params

    assert id(named["decoder_blocks.0.ff.0.weight"]) in muon_params
