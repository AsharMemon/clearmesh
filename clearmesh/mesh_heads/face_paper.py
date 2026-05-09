"""Paper-faithful FACE ARAE components.

This module intentionally avoids the ClearMesh indexed-topology deviations. It
implements the FACE paper's core reconstruction path:

point cloud + normals -> FPS/downsampled-query VecSet encoder -> causal
face-token decoder with layerwise VecSet cross-attention -> CausalMLP over nine
quantized coordinate tokens.
"""

from __future__ import annotations


def build_paper_face_arae(
    *,
    num_bins: int,
    max_faces: int,
    point_feature_dim: int = 6,
    hidden_size: int = 1024,
    encoder_hidden_size: int = 768,
    encoder_layers: int = 8,
    decoder_layers: int = 24,
    heads: int = 16,
    vecset_tokens: int = 2048,
    latent_dim: int = 64,
    causal_mlp_hidden: int | None = None,
    encoder_backend: str = "native",
    causal_mlp_variant: str = "legacy_concat",
    face_embedding_variant: str = "token_concat_project",
    enable_eos_head: bool = True,
):
    """Build a FACE-style autoregressive autoencoder.

    New paper-profile training should use ``encoder_backend='shape2vecset'``.
    ``legacy_concat`` is the closest public-code match to the CausalMLP cited
    by FACE (TreeMeshGPT-style staged coordinate heads). ``paper_chain`` remains
    available as an experimental ablation. ``continuous_mlp`` and
    ``discrete_sum`` preserve old checkpoints; new paper-profile training uses
    ``token_concat_project`` for previous-face embedding.
    """

    try:
        import math

        import torch
        from torch import einsum, nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - optional ML env
        raise RuntimeError("PyTorch is required for paper-faithful FACE") from exc

    causal_mlp_hidden = int(causal_mlp_hidden or hidden_size)
    encoder_backend = encoder_backend.strip().lower()
    causal_mlp_variant = causal_mlp_variant.strip().lower()
    face_embedding_variant = face_embedding_variant.strip().lower()
    if encoder_backend not in {"native", "shape2vecset"}:
        raise ValueError(f"unknown FACE encoder backend: {encoder_backend}")
    if causal_mlp_variant not in {"legacy_concat", "paper_chain"}:
        raise ValueError(f"unknown FACE causal MLP variant: {causal_mlp_variant}")
    if face_embedding_variant not in {"continuous_mlp", "discrete_sum", "token_concat_project"}:
        raise ValueError(f"unknown FACE face embedding variant: {face_embedding_variant}")

    class GEGLU(nn.Module):
        def forward(self, x):  # type: ignore[no-untyped-def]
            x, gates = x.chunk(2, dim=-1)
            return x * F.gelu(gates)

    class FeedForward(nn.Module):
        def __init__(self, dim: int, mult: int = 4) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim * mult * 2),
                GEGLU(),
                nn.Linear(dim * mult, dim),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

    class PreNorm(nn.Module):
        def __init__(self, dim: int, fn, context_dim: int | None = None) -> None:  # type: ignore[no-untyped-def]
            super().__init__()
            self.fn = fn
            self.norm = nn.LayerNorm(dim)
            self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None

        def forward(self, x, **kwargs):  # type: ignore[no-untyped-def]
            x = self.norm(x)
            if self.norm_context is not None:
                kwargs["context"] = self.norm_context(kwargs["context"])
            return self.fn(x, **kwargs)

    class Attention(nn.Module):
        """3DShape2VecSet-style attention: explicit q plus shared kv projection."""

        def __init__(self, query_dim: int, context_dim: int | None = None, heads_: int = 8, dim_head: int = 64) -> None:
            super().__init__()
            inner_dim = int(dim_head) * int(heads_)
            context_dim = int(context_dim or query_dim)
            self.scale = float(dim_head) ** -0.5
            self.heads = int(heads_)
            self.dim_head = int(dim_head)
            self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
            self.to_out = nn.Linear(inner_dim, query_dim)

        def forward(self, x, context=None, mask=None):  # type: ignore[no-untyped-def]
            batch, query_count, _ = x.shape
            context = x if context is None else context
            key_count = context.shape[1]
            q = self.to_q(x)
            k, v = self.to_kv(context).chunk(2, dim=-1)
            q = q.view(batch, query_count, self.heads, self.dim_head).transpose(1, 2)
            k = k.view(batch, key_count, self.heads, self.dim_head).transpose(1, 2)
            v = v.view(batch, key_count, self.heads, self.dim_head).transpose(1, 2)
            attn_mask = None
            if mask is not None:
                attn_mask = mask.reshape(batch, 1, 1, key_count)
            if hasattr(F, "scaled_dot_product_attention"):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
            else:  # pragma: no cover - older PyTorch compatibility.
                q_flat = q.reshape(batch * self.heads, query_count, self.dim_head)
                k_flat = k.reshape(batch * self.heads, key_count, self.dim_head)
                v_flat = v.reshape(batch * self.heads, key_count, self.dim_head)
                sim = einsum("b i d, b j d -> b i j", q_flat, k_flat) * self.scale
                if attn_mask is not None:
                    mask_flat = mask.reshape(batch, -1).unsqueeze(1).repeat_interleave(self.heads, dim=0)
                    sim.masked_fill_(~mask_flat, -torch.finfo(sim.dtype).max)
                attn = sim.softmax(dim=-1)
                out = einsum("b i j, b j d -> b i d", attn, v_flat)
                out = out.view(batch, self.heads, query_count, self.dim_head)
            out = out.view(batch, self.heads, query_count, self.dim_head).transpose(1, 2).reshape(batch, query_count, self.heads * self.dim_head)
            return self.to_out(out)

    class PointEmbed(nn.Module):
        """Sinusoidal 3D point embedding used by 3DShape2VecSet."""

        def __init__(self, dim: int, hidden_dim: int = 48) -> None:
            super().__init__()
            if hidden_dim % 6 != 0:
                raise ValueError("point embedding hidden_dim must be divisible by 6")
            basis_count = hidden_dim // 6
            e = torch.pow(2, torch.arange(basis_count).float()) * math.pi
            basis = torch.stack(
                [
                    torch.cat([e, torch.zeros(basis_count), torch.zeros(basis_count)]),
                    torch.cat([torch.zeros(basis_count), e, torch.zeros(basis_count)]),
                    torch.cat([torch.zeros(basis_count), torch.zeros(basis_count), e]),
                ]
            )
            self.register_buffer("basis", basis)
            self.mlp = nn.Linear(hidden_dim + 3, dim)

        def forward(self, points):  # type: ignore[no-untyped-def]
            projections = torch.einsum("bnd,de->bne", points, self.basis)
            embeddings = torch.cat([projections.sin(), projections.cos(), points], dim=-1)
            return self.mlp(embeddings)

    class NativeVecSetEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vecset_tokens = int(vecset_tokens)
            self.query_projection = nn.Linear(point_feature_dim, encoder_hidden_size)
            self.point_projection = nn.Linear(point_feature_dim, encoder_hidden_size)
            self.cross_attention = nn.MultiheadAttention(encoder_hidden_size, heads, batch_first=True)
            layer = nn.TransformerEncoderLayer(
                d_model=encoder_hidden_size,
                nhead=heads,
                dim_feedforward=encoder_hidden_size * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers)
            self.norm = nn.LayerNorm(encoder_hidden_size)
            self.bottleneck = nn.Linear(encoder_hidden_size, latent_dim)

        def forward(self, point_features, query_indices=None):  # type: ignore[no-untyped-def]
            query_indices = (
                _farthest_point_indices(point_features[..., :3], self.vecset_tokens)
                if query_indices is None
                else query_indices.to(device=point_features.device, dtype=torch.long)
            )
            gather = query_indices.unsqueeze(-1).expand(-1, -1, point_features.shape[-1])
            query_points = torch.gather(point_features, 1, gather)
            queries = self.query_projection(query_points)
            point_tokens = self.point_projection(point_features)
            vecset, _ = self.cross_attention(queries, point_tokens, point_tokens, need_weights=False)
            return self.bottleneck(self.norm(self.encoder(vecset)))

    class Shape2VecSetEncoder(nn.Module):
        """Dependency-light 3DShape2VecSet encoder adapted to XYZ+normal inputs.

        FACE specifies 8192 points with normals. The public 3DShape2VecSet code
        embeds XYZ only, so normals are injected as an additive learned feature.
        The attention/FFN topology follows the official implementation.
        """

        def __init__(self) -> None:
            super().__init__()
            self.vecset_tokens = int(vecset_tokens)
            dim_head = max(1, encoder_hidden_size // max(1, heads))
            self.point_embed = PointEmbed(dim=encoder_hidden_size)
            self.normal_embed = nn.Linear(3, encoder_hidden_size) if point_feature_dim >= 6 else None
            self.cross_attend = PreNorm(
                encoder_hidden_size,
                Attention(encoder_hidden_size, encoder_hidden_size, heads_=1, dim_head=encoder_hidden_size),
                context_dim=encoder_hidden_size,
            )
            self.cross_ff = PreNorm(encoder_hidden_size, FeedForward(encoder_hidden_size))
            self.layers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            PreNorm(encoder_hidden_size, Attention(encoder_hidden_size, heads_=heads, dim_head=dim_head)),
                            PreNorm(encoder_hidden_size, FeedForward(encoder_hidden_size)),
                        ]
                    )
                    for _ in range(encoder_layers)
                ]
            )
            self.norm = nn.LayerNorm(encoder_hidden_size)
            self.bottleneck = nn.Linear(encoder_hidden_size, latent_dim)

        def _embed_features(self, point_features):  # type: ignore[no-untyped-def]
            embedded = self.point_embed(point_features[..., :3])
            if self.normal_embed is not None and point_features.shape[-1] >= 6:
                embedded = embedded + self.normal_embed(point_features[..., 3:6])
            return embedded

        def forward(self, point_features, query_indices=None):  # type: ignore[no-untyped-def]
            query_indices = (
                _farthest_point_indices(point_features[..., :3], self.vecset_tokens)
                if query_indices is None
                else query_indices.to(device=point_features.device, dtype=torch.long)
            )
            gather = query_indices.unsqueeze(-1).expand(-1, -1, point_features.shape[-1])
            sampled_features = torch.gather(point_features, 1, gather)
            sampled_embeddings = self._embed_features(sampled_features)
            point_embeddings = self._embed_features(point_features)
            x = self.cross_attend(sampled_embeddings, context=point_embeddings) + sampled_embeddings
            x = self.cross_ff(x) + x
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x
            return self.bottleneck(self.norm(x))

    class FaceDecoderBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(
                hidden_size,
                heads,
                batch_first=True,
                kdim=latent_dim,
                vdim=latent_dim,
            )
            self.norm_self = nn.LayerNorm(hidden_size)
            self.norm_cross = nn.LayerNorm(hidden_size)
            self.norm_ff = nn.LayerNorm(hidden_size)
            self.ff = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )

        def forward(self, x, vecset, causal_mask):  # type: ignore[no-untyped-def]
            self_norm = self.norm_self(x)
            self_out, _ = self.self_attn(
                self_norm,
                self_norm,
                self_norm,
                attn_mask=causal_mask,
                need_weights=False,
            )
            x = x + self_out
            cross_out, _ = self.cross_attn(self.norm_cross(x), vecset, vecset, need_weights=False)
            x = x + cross_out
            x = x + self.ff(self.norm_ff(x))
            return x

    class LegacyConcatCausalCoordinateMLP(nn.Module):
        """TreeMeshGPT-style causal coordinate heads generalized to nine slots."""

        def __init__(self) -> None:
            super().__init__()
            self.previous_embeddings = nn.ModuleList([nn.Embedding(num_bins, hidden_size) for _ in range(8)])
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_size * (slot + 1), causal_mlp_hidden),
                        nn.ReLU(),
                        nn.Linear(causal_mlp_hidden, causal_mlp_hidden),
                        nn.ReLU(),
                        nn.Linear(causal_mlp_hidden, num_bins),
                    )
                    for slot in range(9)
                ]
            )

        def forward(self, hidden, target_faces):  # type: ignore[no-untyped-def]
            clamped = target_faces.clamp(min=0, max=num_bins - 1)
            logits = []
            for slot, head in enumerate(self.heads):
                parts = [hidden]
                for prev in range(slot):
                    parts.append(self.previous_embeddings[prev](clamped[:, :, prev]))
                logits.append(head(torch.cat(parts, dim=-1)))
            return torch.stack(logits, dim=2)

        def greedy_decode(self, hidden, limit_bins: int | None = None):  # type: ignore[no-untyped-def]
            if hidden.ndim != 3 or hidden.shape[1] != 1:
                raise ValueError(f"hidden must have shape (B, 1, H), got {tuple(hidden.shape)}")
            limit_bins = int(limit_bins or num_bins)
            hidden_step = hidden[:, 0, :]
            decoded = torch.full((hidden.shape[0], 9), -1, dtype=torch.long, device=hidden.device)
            for slot, head in enumerate(self.heads):
                parts = [hidden_step]
                for prev in range(slot):
                    parts.append(self.previous_embeddings[prev](decoded[:, prev]))
                logits = head(torch.cat(parts, dim=-1))
                decoded[:, slot] = torch.argmax(logits[:, :limit_bins], dim=-1)
            return decoded

    class PaperChainCausalCoordinateMLP(nn.Module):
        """FACE-style causal coordinate MLP with shared slot-aware prefix state.

        The paper does not publish code for CausalMLP. This variant keeps the
        specified causal factorization while preserving the order of previous
        coordinate tokens inside the face. Earlier experiments used a pooled
        prefix; that made teacher-forced loss look better than free-running
        topology because z/y/x slot order was partially erased.
        """

        def __init__(self) -> None:
            super().__init__()
            self.prefix_token_dim = max(1, hidden_size // 9)
            self.coord_embedding = nn.Embedding(num_bins, self.prefix_token_dim)
            self.prefix_slot_embedding = nn.Embedding(9, self.prefix_token_dim)
            self.slot_embedding = nn.Embedding(9, hidden_size)
            self.prefix_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.prefix_token_dim * slot, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.GELU(),
                        nn.Linear(hidden_size, hidden_size),
                    )
                    for slot in range(1, 9)
                ]
            )
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size * 3, causal_mlp_hidden * 2),
                GEGLU(),
                nn.Linear(causal_mlp_hidden, causal_mlp_hidden * 2),
                GEGLU(),
                nn.Linear(causal_mlp_hidden, num_bins),
            )

        def _slot_logits(self, hidden_step, decoded_or_teacher, slot: int):  # type: ignore[no-untyped-def]
            batch_shape = hidden_step.shape[:-1]
            slot_ids = torch.full(batch_shape, int(slot), dtype=torch.long, device=hidden_step.device)
            slot_feature = self.slot_embedding(slot_ids)
            if slot == 0:
                prefix = torch.zeros_like(hidden_step)
            else:
                previous_tokens = decoded_or_teacher[..., :slot].clamp(min=0, max=num_bins - 1)
                previous_slots = torch.arange(slot, device=hidden_step.device).view(*([1] * (previous_tokens.ndim - 1)), slot)
                previous = self.coord_embedding(previous_tokens) + self.prefix_slot_embedding(previous_slots)
                previous = previous.reshape(*batch_shape, slot * self.prefix_token_dim)
                prefix = self.prefix_projections[slot - 1](previous)
            return self.mlp(torch.cat([hidden_step, slot_feature, prefix], dim=-1))

        def forward(self, hidden, target_faces):  # type: ignore[no-untyped-def]
            clamped = target_faces.clamp(min=0, max=num_bins - 1)
            logits = [self._slot_logits(hidden, clamped, slot) for slot in range(9)]
            return torch.stack(logits, dim=2)

        def greedy_decode(self, hidden, limit_bins: int | None = None):  # type: ignore[no-untyped-def]
            if hidden.ndim != 3 or hidden.shape[1] != 1:
                raise ValueError(f"hidden must have shape (B, 1, H), got {tuple(hidden.shape)}")
            limit_bins = int(limit_bins or num_bins)
            hidden_step = hidden[:, 0, :]
            decoded = torch.full((hidden.shape[0], 9), -1, dtype=torch.long, device=hidden.device)
            for slot in range(9):
                logits = self._slot_logits(hidden_step, decoded, slot)
                decoded[:, slot] = torch.argmax(logits[:, :limit_bins], dim=-1)
            return decoded

    class ContinuousFaceEmbedding(nn.Module):
        """Legacy previous-face embedding over normalized coordinate scalars."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(9, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
            )

        def forward(self, input_faces):  # type: ignore[no-untyped-def]
            normalized = input_faces.clamp(min=0, max=num_bins - 1).to(dtype=self.net[0].weight.dtype)
            normalized = (normalized / max(1, num_bins - 1)) * 2.0 - 1.0
            return self.net(normalized)

    class DiscreteSumFaceEmbedding(nn.Module):
        """Embed each of the nine quantized face-coordinate tokens.

        FACE describes a face embedding layer over the nine quantized
        coordinate tokens. The paper does not publish the exact implementation;
        this variant follows the literal discrete-token interpretation while
        keeping one face-level vector per triangle for the autoregressive face
        decoder.
        """

        def __init__(self) -> None:
            super().__init__()
            self.coord_embeddings = nn.ModuleList([nn.Embedding(num_bins, hidden_size) for _ in range(9)])
            self.slot_embedding = nn.Embedding(9, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )

        def forward(self, input_faces):  # type: ignore[no-untyped-def]
            clamped = input_faces.clamp(min=0, max=num_bins - 1)
            embedded = 0
            for slot, embedding in enumerate(self.coord_embeddings):
                slot_ids = torch.full_like(clamped[:, :, slot], slot)
                embedded = embedded + embedding(clamped[:, :, slot]) + self.slot_embedding(slot_ids)
            embedded = embedded / math.sqrt(9.0)
            return self.proj(self.norm(embedded))

    class TokenConcatProjectFaceEmbedding(nn.Module):
        """Paper-faithful previous-face embedding over nine discrete tokens.

        FACE describes a face embedding layer that encodes the nine quantized
        coordinate tokens. We keep the token identities instead of sum-pooling:
        each ZYX slot gets a shared coordinate-token embedding plus explicit
        slot embedding, the nine slot vectors are concatenated in order, and the
        concatenation is projected to the decoder hidden size.
        """

        def __init__(self) -> None:
            super().__init__()
            self.token_dim = max(1, hidden_size // 9)
            self.coord_embedding = nn.Embedding(num_bins, self.token_dim)
            self.slot_embedding = nn.Embedding(9, self.token_dim)
            self.proj = nn.Sequential(
                nn.Linear(9 * self.token_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )

        def forward(self, input_faces):  # type: ignore[no-untyped-def]
            clamped = input_faces.clamp(min=0, max=num_bins - 1)
            slot_ids = torch.arange(9, device=input_faces.device).view(1, 1, 9)
            embedded = self.coord_embedding(clamped) + self.slot_embedding(slot_ids)
            return self.proj(embedded.reshape(input_faces.shape[0], input_faces.shape[1], 9 * self.token_dim))

    class PaperFaceARAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_bins = int(num_bins)
            self.max_faces = int(max_faces)
            self.hidden_size = int(hidden_size)
            self.encoder_hidden_size = int(encoder_hidden_size)
            self.latent_dim = int(latent_dim)
            self.encoder_backend = encoder_backend
            self.causal_mlp_variant = causal_mlp_variant
            self.face_embedding_variant = face_embedding_variant
            self.encoder = Shape2VecSetEncoder() if encoder_backend == "shape2vecset" else NativeVecSetEncoder()
            if face_embedding_variant == "token_concat_project":
                self.face_embedding = TokenConcatProjectFaceEmbedding()
            elif face_embedding_variant == "discrete_sum":
                self.face_embedding = DiscreteSumFaceEmbedding()
            else:
                self.face_embedding = ContinuousFaceEmbedding()
            self.bos_face = nn.Parameter(torch.randn(hidden_size) * 0.02)
            self.face_position_embedding = nn.Embedding(max_faces, hidden_size)
            self.decoder_blocks = nn.ModuleList([FaceDecoderBlock() for _ in range(decoder_layers)])
            self.decoder_norm = nn.LayerNorm(hidden_size)
            self.causal_mlp = PaperChainCausalCoordinateMLP() if causal_mlp_variant == "paper_chain" else LegacyConcatCausalCoordinateMLP()
            self.parallel_head = nn.Linear(hidden_size, 9 * num_bins)
            self.eos_head = nn.Linear(hidden_size, 1) if enable_eos_head else None

        def _face_inputs(self, input_faces):  # type: ignore[no-untyped-def]
            batch, face_count, coords = input_faces.shape
            if coords != 9:
                raise ValueError(f"input_faces must have 9 coordinate tokens, got {coords}")
            if face_count > self.max_faces:
                raise ValueError(f"face_count {face_count} exceeds max_faces={self.max_faces}")
            face_tokens = self.face_embedding(input_faces)
            face_tokens[:, 0, :] = self.bos_face.unsqueeze(0)
            positions = torch.arange(face_count, device=input_faces.device).unsqueeze(0).expand(batch, -1)
            return face_tokens + self.face_position_embedding(positions)

        def _single_face_input(self, input_face, position: int):  # type: ignore[no-untyped-def]
            batch, coords = input_face.shape
            if coords != 9:
                raise ValueError(f"input_face must have 9 coordinate tokens, got {coords}")
            if position >= self.max_faces:
                raise ValueError(f"position {position} exceeds max_faces={self.max_faces}")
            if position == 0:
                face_token = self.bos_face.unsqueeze(0).expand(batch, -1)
            else:
                face_token = self.face_embedding(input_face.unsqueeze(1))[:, 0, :]
            positions = torch.full((batch,), int(position), device=input_face.device, dtype=torch.long)
            return (face_token + self.face_position_embedding(positions)).unsqueeze(1)

        def _causal_mask(self, face_count: int, device):  # type: ignore[no-untyped-def]
            return torch.triu(torch.full((face_count, face_count), float("-inf"), device=device), diagonal=1)

        def hidden(self, point_features, input_faces, query_indices=None):  # type: ignore[no-untyped-def]
            vecset = self.encoder(point_features, query_indices=query_indices)
            x = self._face_inputs(input_faces)
            mask = self._causal_mask(x.shape[1], input_faces.device)
            for block in self.decoder_blocks:
                x = block(x, vecset, mask)
            return self.decoder_norm(x)

        def forward(self, point_features, input_faces, target_faces, query_indices=None):  # type: ignore[no-untyped-def]
            return self.forward_causal(point_features, input_faces, target_faces, query_indices=query_indices)

        def forward_causal(self, point_features, input_faces, target_faces, query_indices=None):  # type: ignore[no-untyped-def]
            hidden = self.hidden(point_features, input_faces, query_indices=query_indices)
            return self._causal_logits_from_hidden(hidden, target_faces)

        def forward_parallel(self, point_features, input_faces, query_indices=None):  # type: ignore[no-untyped-def]
            hidden = self.hidden(point_features, input_faces, query_indices=query_indices)
            return self.parallel_head(hidden).reshape(hidden.shape[0], hidden.shape[1], 9, self.num_bins)

        def eos_logits_from_hidden(self, hidden):  # type: ignore[no-untyped-def]
            if self.eos_head is None:
                return None
            return self.eos_head(hidden).squeeze(-1)

        def eos_logits(self, point_features, input_faces, query_indices=None):  # type: ignore[no-untyped-def]
            return self.eos_logits_from_hidden(self.hidden(point_features, input_faces, query_indices=query_indices))

        def next_face_logits(self, point_features, input_faces, prefix_face, query_indices=None):  # type: ignore[no-untyped-def]
            hidden = self.hidden(point_features, input_faces, query_indices=query_indices)[:, -1:, :]
            prefix = prefix_face.reshape(prefix_face.shape[0], 1, 9)
            return self._causal_logits_from_hidden(hidden, prefix)[:, 0, :, :]

        def init_incremental_cache(self, point_features, query_indices=None):  # type: ignore[no-untyped-def]
            return {
                "vecset": self.encoder(point_features, query_indices=query_indices),
                "block_inputs": [None for _ in self.decoder_blocks],
            }

        def incremental_hidden_step(self, input_face, position: int, cache):  # type: ignore[no-untyped-def]
            vecset = cache["vecset"]
            block_inputs = cache["block_inputs"]
            x = self._single_face_input(input_face, position)
            for idx, block in enumerate(self.decoder_blocks):
                block_input = x
                previous = block_inputs[idx]
                if previous is None:
                    key_value_input = block_input
                else:
                    key_value_input = torch.cat([previous, block_input], dim=1)
                query_norm = block.norm_self(block_input)
                key_value_norm = block.norm_self(key_value_input)
                self_out, _ = block.self_attn(
                    query_norm,
                    key_value_norm,
                    key_value_norm,
                    need_weights=False,
                )
                x = block_input + self_out
                cross_out, _ = block.cross_attn(block.norm_cross(x), vecset, vecset, need_weights=False)
                x = x + cross_out
                x = x + block.ff(block.norm_ff(x))
                block_inputs[idx] = block_input if previous is None else torch.cat([previous, block_input], dim=1)
            return self.decoder_norm(x)

        def _causal_logits_from_hidden(self, hidden, target_faces):  # type: ignore[no-untyped-def]
            return self.causal_mlp(hidden, target_faces)

        def greedy_face_from_hidden(self, hidden, limit_bins: int | None = None):  # type: ignore[no-untyped-def]
            return self.causal_mlp.greedy_decode(hidden, limit_bins=limit_bins)

    return PaperFaceARAE()


def _farthest_point_indices(points, count: int):  # type: ignore[no-untyped-def]
    import torch

    batch, point_count, _ = points.shape
    count = int(count)
    if point_count <= 0:
        raise ValueError("point cloud must contain at least one point")
    centroid = points.mean(dim=1, keepdim=True)
    distances = torch.sum((points - centroid) ** 2, dim=-1)
    current = torch.argmax(distances, dim=1)
    selected = []
    min_distances = torch.full((batch, point_count), float("inf"), device=points.device, dtype=points.dtype)
    for _ in range(count):
        selected.append(current)
        current_points = points[torch.arange(batch, device=points.device), current].unsqueeze(1)
        dist = torch.sum((points - current_points) ** 2, dim=-1)
        min_distances = torch.minimum(min_distances, dist)
        current = torch.argmax(min_distances, dim=1)
    return torch.stack(selected, dim=1)
