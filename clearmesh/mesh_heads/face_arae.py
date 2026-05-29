"""Tiny point-conditioned FACE decoder.

This is the next rung above ``face_tiny``: it conditions canonical FACE tokens
on sampled surface points and normals. It is not yet the full paper-scale ARAE,
but it preserves the key contract we need for scaling:

surface points/normals -> compact condition tokens -> causal face-token decoder.
"""

from __future__ import annotations

from .face_tiny import FaceTinyVocabulary


def build_tiny_point_conditioned_face_decoder(
    num_bins: int,
    max_tokens: int,
    point_feature_dim: int = 6,
    hidden_size: int = 192,
    layers: int = 4,
    heads: int = 6,
    condition_tokens: int = 8,
):
    """Build a small causal decoder conditioned on a point/normal set."""

    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional ML env
        raise RuntimeError("PyTorch is required for FACE ARAE training") from exc

    vocab = FaceTinyVocabulary(num_bins)

    class TinyPointConditionedFaceDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.condition_tokens = int(condition_tokens)
            self.max_tokens = int(max_tokens)
            self.point_encoder = nn.Sequential(
                nn.Linear(point_feature_dim, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            self.condition_queries = nn.Parameter(torch.randn(condition_tokens, hidden_size) * 0.02)
            self.condition_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.token_embedding = nn.Embedding(vocab.vocab_size, hidden_size)
            self.position_embedding = nn.Embedding(max_tokens, hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(hidden_size)
            self.output = nn.Linear(hidden_size, vocab.vocab_size)

        def _condition(self, point_features):  # type: ignore[no-untyped-def]
            encoded = self.point_encoder(point_features)
            pooled_mean = encoded.mean(dim=1)
            pooled_max = encoded.max(dim=1).values
            pooled = self.condition_projection(torch.cat([pooled_mean, pooled_max], dim=-1))
            return pooled.unsqueeze(1) + self.condition_queries.unsqueeze(0)

        def _attention_mask(self, token_count: int, device):  # type: ignore[no-untyped-def]
            total = self.condition_tokens + token_count
            mask = torch.zeros((total, total), device=device)
            # Condition tokens summarize the point set and do not inspect target tokens.
            mask[: self.condition_tokens, self.condition_tokens :] = float("-inf")
            token_mask = torch.triu(
                torch.full((token_count, token_count), float("-inf"), device=device),
                diagonal=1,
            )
            mask[self.condition_tokens :, self.condition_tokens :] = token_mask
            return mask

        def forward(self, point_features, input_ids):  # type: ignore[no-untyped-def]
            batch, seq_len = input_ids.shape
            if seq_len > self.max_tokens:
                raise ValueError(f"input sequence length {seq_len} exceeds max_tokens={self.max_tokens}")
            condition = self._condition(point_features)
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
            tokens = self.token_embedding(input_ids) + self.position_embedding(positions)
            x = torch.cat([condition, tokens], dim=1)
            x = self.blocks(x, mask=self._attention_mask(seq_len, input_ids.device))
            return self.output(self.norm(x[:, self.condition_tokens :, :]))

    return TinyPointConditionedFaceDecoder()


def build_tiny_point_conditioned_face_level_decoder(
    num_bins: int,
    max_faces: int,
    point_feature_dim: int = 6,
    hidden_size: int = 192,
    layers: int = 4,
    heads: int = 6,
    condition_tokens: int = 8,
):
    """Build a FACE-like decoder that autoregresses one triangle per step."""

    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional ML env
        raise RuntimeError("PyTorch is required for FACE ARAE training") from exc

    class TinyPointConditionedFaceLevelDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.condition_tokens = int(condition_tokens)
            self.max_faces = int(max_faces)
            self.num_bins = int(num_bins)
            self.hidden_size = int(hidden_size)
            self.point_encoder = nn.Sequential(
                nn.Linear(point_feature_dim, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            self.condition_queries = nn.Parameter(torch.randn(condition_tokens, hidden_size) * 0.02)
            self.condition_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.count_output = nn.Linear(hidden_size, max_faces + 1)
            self.coord_embedding = nn.Embedding(num_bins, hidden_size)
            self.face_projection = nn.Linear(9 * hidden_size, hidden_size)
            self.bos_face = nn.Parameter(torch.randn(hidden_size) * 0.02)
            self.position_embedding = nn.Embedding(max_faces, hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(hidden_size)
            self.output = nn.Linear(hidden_size, 9 * num_bins)
            self.reuse_vertex_output = nn.Linear(hidden_size, 3 * 2)
            self.edge_closure_output = nn.Linear(hidden_size, 4)

        def _pooled_condition(self, point_features):  # type: ignore[no-untyped-def]
            encoded = self.point_encoder(point_features)
            pooled_mean = encoded.mean(dim=1)
            pooled_max = encoded.max(dim=1).values
            return self.condition_projection(torch.cat([pooled_mean, pooled_max], dim=-1))

        def _condition(self, point_features):  # type: ignore[no-untyped-def]
            pooled = self._pooled_condition(point_features)
            return pooled.unsqueeze(1) + self.condition_queries.unsqueeze(0)

        def predict_face_count_logits(self, point_features):  # type: ignore[no-untyped-def]
            return self.count_output(self._pooled_condition(point_features))

        def _attention_mask(self, face_count: int, device):  # type: ignore[no-untyped-def]
            total = self.condition_tokens + face_count
            mask = torch.zeros((total, total), device=device)
            mask[: self.condition_tokens, self.condition_tokens :] = float("-inf")
            face_mask = torch.triu(
                torch.full((face_count, face_count), float("-inf"), device=device),
                diagonal=1,
            )
            mask[self.condition_tokens :, self.condition_tokens :] = face_mask
            return mask

        def _hidden(self, point_features, input_faces):  # type: ignore[no-untyped-def]
            batch, face_count, coords = input_faces.shape
            if coords != 9:
                raise ValueError(f"input_faces must have 9 coordinates per face, got {coords}")
            if face_count > self.max_faces:
                raise ValueError(f"face_count {face_count} exceeds max_faces={self.max_faces}")
            condition = self._condition(point_features)
            clamped = input_faces.clamp(min=0, max=self.num_bins - 1)
            face_tokens = self.coord_embedding(clamped).reshape(batch, face_count, 9 * self.hidden_size)
            face_tokens = self.face_projection(face_tokens)
            face_tokens[:, 0, :] = self.bos_face.unsqueeze(0)
            positions = torch.arange(face_count, device=input_faces.device).unsqueeze(0).expand(batch, -1)
            face_tokens = face_tokens + self.position_embedding(positions)
            x = torch.cat([condition, face_tokens], dim=1)
            x = self.blocks(x, mask=self._attention_mask(face_count, input_faces.device))
            return self.norm(x[:, self.condition_tokens :, :])

        def forward(self, point_features, input_faces):  # type: ignore[no-untyped-def]
            batch, face_count, _ = input_faces.shape
            hidden = self._hidden(point_features, input_faces)
            logits = self.output(hidden)
            return logits.reshape(batch, face_count, 9, self.num_bins)

        def forward_with_aux(self, point_features, input_faces):  # type: ignore[no-untyped-def]
            batch, face_count, _ = input_faces.shape
            hidden = self._hidden(point_features, input_faces)
            coord_logits = self.output(hidden).reshape(batch, face_count, 9, self.num_bins)
            reuse_logits = self.reuse_vertex_output(hidden).reshape(batch, face_count, 3, 2)
            closure_logits = self.edge_closure_output(hidden)
            return {
                "coord_logits": coord_logits,
                "reuse_vertex_logits": reuse_logits,
                "edge_closure_logits": closure_logits,
            }

    return TinyPointConditionedFaceLevelDecoder()


def build_tiny_point_conditioned_indexed_face_decoder(
    num_bins: int,
    max_vertices: int,
    max_faces: int,
    point_feature_dim: int = 6,
    hidden_size: int = 192,
    layers: int = 4,
    heads: int = 6,
    condition_tokens: int = 8,
    edge_head_mode: str = "geometry",
    condition_backend: str = "pooled",
    decoder_backend: str = "prefix",
    encoder_layers: int = 4,
    latent_dim: int = 64,
    face_output_mode: str = "linear",
    voxset_resolution: int = 16,
    spatial_gate_sigma: float = 0.35,
    spatial_gate_top_k: int = 0,
):
    """Build a FACE-lite v2 decoder over explicit vertex-table face indices.

    The vertex table is supplied as quantized coordinates. The autoregressive
    stream predicts triangle indices into that table, which makes vertex reuse a
    hard representation property instead of an emergent coordinate-weld event.
    """

    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional ML env
        raise RuntimeError("PyTorch is required for FACE indexed training") from exc

    condition_backend = condition_backend.strip().lower()
    decoder_backend = decoder_backend.strip().lower()
    face_output_mode = face_output_mode.strip().lower()
    if condition_backend not in {"pooled", "vecset", "voxset"}:
        raise ValueError(f"condition_backend must be 'pooled', 'vecset', or 'voxset', got {condition_backend!r}")
    spatial_decoder_backends = {
        "spatial_cross_attn",
        "spatial_modulated_cross_attn",
        "spatial_topology_modulated_cross_attn",
    }
    if decoder_backend not in {"prefix", "cross_attn", *spatial_decoder_backends}:
        raise ValueError(
            "decoder_backend must be 'prefix', 'cross_attn', 'spatial_cross_attn', "
            "'spatial_modulated_cross_attn', or 'spatial_topology_modulated_cross_attn', "
            f"got {decoder_backend!r}"
        )
    if decoder_backend in spatial_decoder_backends and condition_backend != "voxset":
        raise ValueError(f"decoder_backend={decoder_backend!r} requires condition_backend='voxset'")
    if face_output_mode not in {"linear", "geometry"}:
        raise ValueError(f"face_output_mode must be 'linear' or 'geometry', got {face_output_mode!r}")

    class VecSetConditionEncoder(nn.Module):
        """FACE-style FPS query VecSet encoder for indexed FACE-Q.

        The paper path uses many latent tokens and injects them into every decoder
        block through cross-attention. This lightweight implementation mirrors the
        existing paper FACE encoder topology without changing the indexed target
        representation.
        """

        def __init__(self) -> None:
            super().__init__()
            self.vecset_tokens = int(condition_tokens)
            self.query_projection = nn.Linear(point_feature_dim, hidden_size)
            self.point_projection = nn.Linear(point_feature_dim, hidden_size)
            self.cross_attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(encoder_layers))
            self.norm = nn.LayerNorm(hidden_size)
            self.bottleneck = nn.Linear(hidden_size, int(latent_dim))

        def forward(self, point_features):  # type: ignore[no-untyped-def]
            query_indices = _farthest_point_indices_local(point_features[..., :3], self.vecset_tokens)
            gather = query_indices.unsqueeze(-1).expand(-1, -1, point_features.shape[-1])
            query_points = torch.gather(point_features, 1, gather)
            queries = self.query_projection(query_points)
            points = self.point_projection(point_features)
            vecset, _ = self.cross_attention(queries, points, points, need_weights=False)
            return self.bottleneck(self.norm(self.encoder(vecset)))

    class VoxSetConditionEncoder(nn.Module):
        """Voxel-anchored Shape2VecSet variant for spatially grounded FACE.

        VecSet queries are sampled surface points, so the decoder sees a useful
        latent set but not a stable lattice of known positions. VoxSet queries
        are coarse active voxel centers derived from the conditioning surface,
        which gives each latent token an explicit 3D anchor for gated
        cross-attention.
        """

        def __init__(self) -> None:
            super().__init__()
            self.voxset_tokens = int(condition_tokens)
            self.resolution = max(1, int(voxset_resolution))
            self.query_projection = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            self.point_projection = nn.Linear(point_feature_dim, hidden_size)
            self.cross_attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(encoder_layers))
            self.norm = nn.LayerNorm(hidden_size)
            self.bottleneck = nn.Linear(hidden_size, int(latent_dim))

        def _voxel_centers(self, point_features):  # type: ignore[no-untyped-def]
            points = point_features[..., :3].detach()
            batch, _point_count, _ = points.shape
            centers = []
            resolution = int(self.resolution)
            for row in range(batch):
                normalized = torch.clamp((points[row] + 1.0) * 0.5, 0.0, 1.0 - 1.0e-6)
                coords = torch.floor(normalized * float(resolution)).to(torch.long)
                unique = torch.unique(coords, dim=0)
                if unique.numel() == 0:
                    center = points.new_zeros((1, 3))
                else:
                    center = ((unique.to(points.dtype) + 0.5) / float(resolution)) * 2.0 - 1.0
                if center.shape[0] >= self.voxset_tokens:
                    selected = _farthest_point_indices_local(center.unsqueeze(0), self.voxset_tokens)[0]
                    center = center[selected]
                else:
                    repeat = int((self.voxset_tokens + center.shape[0] - 1) // max(1, center.shape[0]))
                    center = center.repeat(repeat, 1)[: self.voxset_tokens]
                centers.append(center)
            return torch.stack(centers, dim=0)

        def forward(self, point_features):  # type: ignore[no-untyped-def]
            voxel_centers = self._voxel_centers(point_features)
            queries = self.query_projection(voxel_centers)
            points = self.point_projection(point_features)
            voxset, _ = self.cross_attention(queries, points, points, need_weights=False)
            latents = self.bottleneck(self.norm(self.encoder(voxset)))
            return latents, voxel_centers

    class IndexedDecoderBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = int(heads)
            self.self_attn = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(
                hidden_size,
                heads,
                batch_first=True,
                kdim=int(latent_dim),
                vdim=int(latent_dim),
            )
            self.norm_self = nn.LayerNorm(hidden_size)
            self.norm_cross = nn.LayerNorm(hidden_size)
            self.norm_ff = nn.LayerNorm(hidden_size)
            self.ff = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )

        def forward(self, x, context, causal_mask, cross_attn_mask=None):  # type: ignore[no-untyped-def]
            self_norm = self.norm_self(x)
            self_out, _ = self.self_attn(
                self_norm,
                self_norm,
                self_norm,
                attn_mask=causal_mask,
                need_weights=False,
            )
            x = x + self_out
            cross_out, _ = self.cross_attn(
                self.norm_cross(x),
                context,
                context,
                attn_mask=cross_attn_mask,
                need_weights=False,
            )
            x = x + cross_out
            return x + self.ff(self.norm_ff(x))

    def _farthest_point_indices_local(points, count: int):  # type: ignore[no-untyped-def]
        batch, point_count, _ = points.shape
        count = min(int(count), int(point_count))
        device = points.device
        selected = torch.zeros((batch, count), dtype=torch.long, device=device)
        farthest = torch.zeros((batch,), dtype=torch.long, device=device)
        distances = torch.full((batch, point_count), float("inf"), device=device)
        batch_indices = torch.arange(batch, device=device)
        for idx in range(count):
            selected[:, idx] = farthest
            centroid = points[batch_indices, farthest].unsqueeze(1)
            dist = torch.sum((points - centroid) ** 2, dim=-1)
            distances = torch.minimum(distances, dist)
            farthest = torch.max(distances, dim=1).indices
        return selected

    class TinyPointConditionedIndexedFaceDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.condition_tokens = int(condition_tokens)
            self.max_vertices = int(max_vertices)
            self.max_faces = int(max_faces)
            self.num_bins = int(num_bins)
            self.hidden_size = int(hidden_size)
            self.condition_backend = condition_backend
            self.decoder_backend = decoder_backend
            self.face_output_mode = face_output_mode
            self.latent_dim = int(latent_dim)
            self.spatial_gate_sigma = float(spatial_gate_sigma)
            self.spatial_gate_top_k = int(spatial_gate_top_k)
            if edge_head_mode not in {"index", "geometry"}:
                raise ValueError(f"edge_head_mode must be 'index' or 'geometry', got {edge_head_mode!r}")
            self.edge_head_mode = edge_head_mode
            if condition_backend == "vecset":
                self.condition_encoder = VecSetConditionEncoder()
                self.vecset_count_projection = nn.Linear(int(latent_dim), hidden_size)
            elif condition_backend == "voxset":
                self.condition_encoder = VoxSetConditionEncoder()
                self.vecset_count_projection = nn.Linear(int(latent_dim), hidden_size)
            else:
                self.point_encoder = nn.Sequential(
                    nn.Linear(point_feature_dim, hidden_size),
                    nn.GELU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.LayerNorm(hidden_size),
                )
                self.condition_queries = nn.Parameter(torch.randn(condition_tokens, hidden_size) * 0.02)
                self.condition_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.count_output = nn.Linear(hidden_size, max_faces + 1)
            self.seed_face_mlp = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, 3 * max_vertices),
            )
            self.coord_embedding = nn.Embedding(num_bins, hidden_size)
            self.vertex_projection = nn.Linear(3 * hidden_size, hidden_size)
            self.vertex_position_embedding = nn.Embedding(max_vertices, hidden_size)
            self.index_embedding = nn.Embedding(max_vertices + 1, hidden_size)
            self.face_projection = nn.Linear(3 * hidden_size, hidden_size)
            self.bos_face = nn.Parameter(torch.randn(hidden_size) * 0.02)
            self.face_position_embedding = nn.Embedding(max_faces, hidden_size)
            if decoder_backend in {"cross_attn", *spatial_decoder_backends}:
                self.decoder_blocks = nn.ModuleList([IndexedDecoderBlock() for _ in range(layers)])
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(hidden_size)
            if decoder_backend in {"spatial_modulated_cross_attn", "spatial_topology_modulated_cross_attn"}:
                self.spatial_local_projection = nn.Sequential(
                    nn.LayerNorm(int(latent_dim) + 4),
                    nn.Linear(int(latent_dim) + 4, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                self.spatial_local_scale = nn.Parameter(torch.tensor(0.1))
            if decoder_backend == "spatial_topology_modulated_cross_attn":
                self.spatial_topology_projection = nn.Sequential(
                    nn.LayerNorm(2 * hidden_size),
                    nn.Linear(2 * hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                self.spatial_topology_scale = nn.Parameter(torch.tensor(0.1))
            if face_output_mode == "geometry":
                self.face_query_mlp = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, 3 * hidden_size),
                )
                self.face_vertex_key = nn.Linear(hidden_size, hidden_size, bias=False)
            else:
                self.output = nn.Linear(hidden_size, 3 * max_vertices)
            self.topology_output = nn.Linear(hidden_size, 4)
            self.corner_bos = nn.Parameter(torch.randn(hidden_size) * 0.02)
            self.corner_position_embedding = nn.Embedding(3, hidden_size)
            if face_output_mode == "geometry":
                self.corner_query_mlp = nn.Sequential(
                    nn.Linear(3 * hidden_size, hidden_size),
                    nn.GELU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                )
                self.corner_vertex_key = nn.Linear(hidden_size, hidden_size, bias=False)
            else:
                self.corner_mlp = nn.Sequential(
                    nn.Linear(3 * hidden_size, hidden_size),
                    nn.GELU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, max_vertices),
                )
            self.edge_action_mlp = nn.Sequential(
                nn.Linear(3 * hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, max_vertices),
            )
            self.edge_choice_mlp = nn.Sequential(
                nn.Linear(3 * hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 1),
            )
            self.edge_action_context_mlp = nn.Sequential(
                nn.Linear(3 * hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            self.edge_action_query = nn.Linear(hidden_size, hidden_size, bias=False)
            self.edge_action_key = nn.Linear(hidden_size, hidden_size, bias=False)
            self.edge_choice_geometry_mlp = nn.Sequential(
                nn.Linear(3 * hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 1),
            )

        def _pooled_condition(self, point_features, vertex_table):  # type: ignore[no-untyped-def]
            if self.condition_backend in {"vecset", "voxset"}:
                encoded = self.condition_encoder(point_features)
                vecset = encoded[0] if isinstance(encoded, tuple) else encoded
                point_pooled = self.vecset_count_projection(vecset.mean(dim=1))
            else:
                encoded = self.point_encoder(point_features)
                pooled_mean = encoded.mean(dim=1)
                pooled_max = encoded.max(dim=1).values
                point_pooled = self.condition_projection(torch.cat([pooled_mean, pooled_max], dim=-1))
            vertex_hidden = self._vertex_hidden(vertex_table)
            vertex_mask = vertex_table[..., 0].ge(0).to(vertex_hidden.dtype).unsqueeze(-1)
            vertex_pooled = (vertex_hidden * vertex_mask).sum(dim=1) / vertex_mask.sum(dim=1).clamp(min=1.0)
            return point_pooled + vertex_pooled

        def _condition(self, point_features, vertex_table):  # type: ignore[no-untyped-def]
            if self.condition_backend == "vecset":
                return self.condition_encoder(point_features)
            if self.condition_backend == "voxset":
                return self.condition_encoder(point_features)[0]
            pooled = self._pooled_condition(point_features, vertex_table)
            return pooled.unsqueeze(1) + self.condition_queries.unsqueeze(0)

        def _condition_with_positions(self, point_features, vertex_table):  # type: ignore[no-untyped-def]
            if self.condition_backend == "voxset":
                return self.condition_encoder(point_features)
            return self._condition(point_features, vertex_table), None

        def _vertex_hidden(self, vertex_table):  # type: ignore[no-untyped-def]
            batch, vertex_count, coords = vertex_table.shape
            if coords != 3:
                raise ValueError(f"vertex_table must have 3 coordinates per vertex, got {coords}")
            if vertex_count > self.max_vertices:
                raise ValueError(f"vertex_count {vertex_count} exceeds max_vertices={self.max_vertices}")
            valid = vertex_table.ge(0)
            clamped = vertex_table.clamp(min=0, max=self.num_bins - 1)
            embedded = self.coord_embedding(clamped).reshape(batch, vertex_count, 3 * self.hidden_size)
            hidden = self.vertex_projection(embedded)
            positions = torch.arange(vertex_count, device=vertex_table.device).unsqueeze(0).expand(batch, -1)
            hidden = hidden + self.vertex_position_embedding(positions)
            return hidden * valid.all(dim=-1).to(hidden.dtype).unsqueeze(-1)

        def _vertex_positions(self, vertex_table):  # type: ignore[no-untyped-def]
            valid = vertex_table.ge(0).all(dim=-1)
            clamped = vertex_table.clamp(min=0, max=self.num_bins - 1).to(torch.float32)
            positions = ((clamped + 0.5) / float(max(1, self.num_bins))) * 2.0 - 1.0
            return positions, valid

        def _bos_face_position(self, vertex_table):  # type: ignore[no-untyped-def]
            positions, valid = self._vertex_positions(vertex_table)
            masked = positions.masked_fill(~valid.unsqueeze(-1), 1.0e6)
            min_pos = masked.amin(dim=1)
            fallback = torch.zeros_like(min_pos)
            has_valid = valid.any(dim=1).unsqueeze(-1)
            return torch.where(has_valid, min_pos.clamp(-1.0, 1.0), fallback)

        def _face_positions_from_prefix(self, input_faces, vertex_table):  # type: ignore[no-untyped-def]
            vertex_positions, valid_vertices = self._vertex_positions(vertex_table)
            batch, face_count, corners = input_faces.shape
            clamped = input_faces.clamp(min=0, max=self.max_vertices - 1)
            gathered = torch.gather(
                vertex_positions.unsqueeze(1).expand(-1, face_count, -1, -1),
                dim=2,
                index=clamped.unsqueeze(-1).expand(-1, -1, -1, 3),
            )
            vertex_valid = torch.gather(
                valid_vertices.unsqueeze(1).expand(-1, face_count, -1),
                dim=2,
                index=clamped,
            )
            face_valid = input_faces.ge(0).all(dim=-1) & vertex_valid.all(dim=-1)
            centers = gathered.mean(dim=2)
            bos = self._bos_face_position(vertex_table).unsqueeze(1).expand(-1, face_count, -1)
            return torch.where(face_valid.unsqueeze(-1), centers, bos)

        def _spatial_cross_attn_mask(self, face_positions, context_positions):  # type: ignore[no-untyped-def]
            if context_positions is None:
                return None
            sigma = max(float(self.spatial_gate_sigma), 1.0e-4)
            dist2 = torch.sum((face_positions.unsqueeze(2) - context_positions.unsqueeze(1)) ** 2, dim=-1)
            bias = -dist2 / (2.0 * sigma * sigma)
            top_k = int(self.spatial_gate_top_k)
            if top_k > 0 and top_k < bias.shape[-1]:
                keep = torch.topk(bias, k=top_k, dim=-1).indices
                hard = torch.full_like(bias, -1.0e4)
                bias = hard.scatter(-1, keep, torch.gather(bias, -1, keep))
            return bias.repeat_interleave(int(heads), dim=0)

        def _spatial_local_context(self, face_positions, context_positions, context):  # type: ignore[no-untyped-def]
            if context_positions is None:
                return None
            sigma = max(float(self.spatial_gate_sigma), 1.0e-4)
            rel = context_positions.unsqueeze(1) - face_positions.unsqueeze(2)
            dist2 = torch.sum(rel**2, dim=-1)
            logits = -dist2 / (2.0 * sigma * sigma)
            top_k = int(self.spatial_gate_top_k)
            if top_k > 0 and top_k < logits.shape[-1]:
                keep = torch.topk(logits, k=top_k, dim=-1).indices
                hard = torch.full_like(logits, -1.0e4)
                logits = hard.scatter(-1, keep, torch.gather(logits, -1, keep))
            weights = torch.softmax(logits, dim=-1)
            local_latent = torch.einsum("bft,btd->bfd", weights, context)
            local_rel = torch.einsum("bft,bftd->bfd", weights, rel)
            local_dist = torch.einsum("bft,bft->bf", weights, dist2).unsqueeze(-1)
            local = torch.cat([local_latent, local_rel, local_dist], dim=-1)
            return self.spatial_local_projection(local)

        def predict_face_count_logits(self, point_features, vertex_table):  # type: ignore[no-untyped-def]
            return self.count_output(self._pooled_condition(point_features, vertex_table))

        def seed_face_logits(self, point_features, vertex_table):  # type: ignore[no-untyped-def]
            pooled = self._pooled_condition(point_features, vertex_table)
            return self.seed_face_mlp(pooled).reshape(pooled.shape[0], 3, self.max_vertices)

        def _attention_mask(self, face_count: int, device):  # type: ignore[no-untyped-def]
            total = self.condition_tokens + face_count
            mask = torch.zeros((total, total), device=device)
            mask[: self.condition_tokens, self.condition_tokens :] = float("-inf")
            face_mask = torch.triu(
                torch.full((face_count, face_count), float("-inf"), device=device),
                diagonal=1,
            )
            mask[self.condition_tokens :, self.condition_tokens :] = face_mask
            return mask

        def _hidden(self, point_features, vertex_table, input_faces, *, return_spatial_local: bool = False):  # type: ignore[no-untyped-def]
            batch, face_count, corners = input_faces.shape
            if corners != 3:
                raise ValueError(f"input_faces must have 3 indices per face, got {corners}")
            if face_count > self.max_faces:
                raise ValueError(f"face_count {face_count} exceeds max_faces={self.max_faces}")
            # -1 is BOS/padding in batches; shift real indices by +1 for embedding.
            shifted = input_faces.clamp(min=-1, max=self.max_vertices - 1) + 1
            face_tokens = self.index_embedding(shifted).reshape(batch, face_count, 3 * self.hidden_size)
            face_tokens = self.face_projection(face_tokens)
            face_tokens[:, 0, :] = self.bos_face.unsqueeze(0)
            positions = torch.arange(face_count, device=input_faces.device).unsqueeze(0).expand(batch, -1)
            face_tokens = face_tokens + self.face_position_embedding(positions)
            if self.decoder_backend == "cross_attn":
                context = self._condition(point_features, vertex_table)
                mask = torch.triu(torch.full((face_count, face_count), float("-inf"), device=input_faces.device), diagonal=1)
                x = face_tokens
                for block in self.decoder_blocks:
                    x = block(x, context, mask)
                return self.norm(x)
            if self.decoder_backend == "spatial_cross_attn":
                context, context_positions = self._condition_with_positions(point_features, vertex_table)
                face_positions = self._face_positions_from_prefix(input_faces, vertex_table)
                cross_mask = self._spatial_cross_attn_mask(face_positions, context_positions)
                mask = torch.triu(torch.full((face_count, face_count), float("-inf"), device=input_faces.device), diagonal=1)
                x = face_tokens
                for block in self.decoder_blocks:
                    x = block(x, context, mask, cross_attn_mask=cross_mask)
                return self.norm(x)
            if self.decoder_backend in {"spatial_modulated_cross_attn", "spatial_topology_modulated_cross_attn"}:
                context, context_positions = self._condition_with_positions(point_features, vertex_table)
                face_positions = self._face_positions_from_prefix(input_faces, vertex_table)
                local_context = self._spatial_local_context(face_positions, context_positions, context)
                if local_context is not None:
                    face_tokens = face_tokens + self.spatial_local_scale * local_context
                cross_mask = self._spatial_cross_attn_mask(face_positions, context_positions)
                mask = torch.triu(torch.full((face_count, face_count), float("-inf"), device=input_faces.device), diagonal=1)
                x = face_tokens
                for block in self.decoder_blocks:
                    x = block(x, context, mask, cross_attn_mask=cross_mask)
                hidden = self.norm(x)
                if return_spatial_local:
                    return hidden, local_context
                return hidden
            condition = self._condition(point_features, vertex_table)
            x = torch.cat([condition, face_tokens], dim=1)
            x = self.blocks(x, mask=self._attention_mask(face_count, input_faces.device))
            hidden = self.norm(x[:, self.condition_tokens :, :])
            if return_spatial_local:
                return hidden, None
            return hidden

        def _topology_hidden_from_local(self, hidden, spatial_local_context=None):  # type: ignore[no-untyped-def]
            if self.decoder_backend != "spatial_topology_modulated_cross_attn" or spatial_local_context is None:
                return hidden
            local = self.spatial_topology_projection(torch.cat([hidden, spatial_local_context], dim=-1))
            return hidden + self.spatial_topology_scale * local

        def topology_hidden_from_hidden(
            self,
            hidden,
            point_features=None,
            vertex_table=None,
            input_faces=None,
            spatial_local_context=None,
        ):  # type: ignore[no-untyped-def]
            if (
                spatial_local_context is None
                and self.decoder_backend == "spatial_topology_modulated_cross_attn"
                and point_features is not None
                and vertex_table is not None
                and input_faces is not None
            ):
                context, context_positions = self._condition_with_positions(point_features, vertex_table)
                face_positions = self._face_positions_from_prefix(input_faces, vertex_table)
                spatial_local_context = self._spatial_local_context(face_positions, context_positions, context)
                if spatial_local_context.shape[1] != hidden.shape[1]:
                    spatial_local_context = spatial_local_context[:, -hidden.shape[1] :, :]
            return self._topology_hidden_from_local(hidden, spatial_local_context)

        def topology_logits_from_hidden(
            self,
            hidden,
            point_features=None,
            vertex_table=None,
            input_faces=None,
            spatial_local_context=None,
        ):  # type: ignore[no-untyped-def]
            topology_hidden = self.topology_hidden_from_hidden(
                hidden,
                point_features=point_features,
                vertex_table=vertex_table,
                input_faces=input_faces,
                spatial_local_context=spatial_local_context,
            )
            return self.topology_output(topology_hidden)

        def edge_action_logits_from_hidden(
            self,
            hidden,
            edge_indices,
            *,
            point_features=None,
            vertex_table=None,
            input_faces=None,
            spatial_local_context=None,
        ):  # type: ignore[no-untyped-def]
            topology_hidden = self.topology_hidden_from_hidden(
                hidden,
                point_features=point_features,
                vertex_table=vertex_table,
                input_faces=input_faces,
                spatial_local_context=spatial_local_context,
            )
            return self._edge_action_logits_from_hidden(topology_hidden, edge_indices, vertex_table=vertex_table)

        def edge_choice_logits_from_hidden(
            self,
            hidden,
            candidate_edges,
            *,
            point_features=None,
            vertex_table=None,
            input_faces=None,
            spatial_local_context=None,
        ):  # type: ignore[no-untyped-def]
            topology_hidden = self.topology_hidden_from_hidden(
                hidden,
                point_features=point_features,
                vertex_table=vertex_table,
                input_faces=input_faces,
                spatial_local_context=spatial_local_context,
            )
            return self._edge_choice_logits_from_hidden(topology_hidden, candidate_edges, vertex_table=vertex_table)

        def _valid_vertex_mask(self, vertex_table):  # type: ignore[no-untyped-def]
            return vertex_table[..., 0].ge(0)

        def _face_logits_from_hidden(self, hidden, vertex_table):  # type: ignore[no-untyped-def]
            if self.face_output_mode != "geometry":
                return self.output(hidden).reshape(hidden.shape[0], hidden.shape[1], 3, self.max_vertices)
            vertex_hidden = self._vertex_hidden(vertex_table)
            queries = self.face_query_mlp(hidden).reshape(hidden.shape[0], hidden.shape[1], 3, self.hidden_size)
            keys = self.face_vertex_key(vertex_hidden)
            logits = torch.einsum("bfch,bvh->bfcv", queries, keys) / float(self.hidden_size ** 0.5)
            return logits.masked_fill(~self._valid_vertex_mask(vertex_table).unsqueeze(1).unsqueeze(1), -1e9)

        def forward(
            self,
            point_features,
            vertex_table,
            input_faces,
            *,
            target_faces=None,
            return_topology: bool = False,
            edge_action_indices=None,
            edge_choice_candidates=None,
            return_count: bool = False,
            return_seed: bool = False,
            return_hidden: bool = False,
        ):  # type: ignore[no-untyped-def]
            want_spatial_local = bool(
                self.decoder_backend == "spatial_topology_modulated_cross_attn"
                and (return_topology or edge_action_indices is not None or edge_choice_candidates is not None)
            )
            if want_spatial_local:
                hidden, spatial_local_context = self._hidden(
                    point_features,
                    vertex_table,
                    input_faces,
                    return_spatial_local=True,
                )
            else:
                hidden = self._hidden(point_features, vertex_table, input_faces)
                spatial_local_context = None
            if target_faces is None:
                face_logits = self._face_logits_from_hidden(hidden, vertex_table)
            else:
                prefix = target_faces.masked_fill(target_faces.lt(0), -1)
                face_logits = self._corner_causal_logits_from_hidden(hidden, prefix, vertex_table=vertex_table)

            if not (
                return_topology
                or edge_action_indices is not None
                or edge_choice_candidates is not None
                or return_count
                or return_seed
                or return_hidden
            ):
                return face_logits

            outputs = {"face_logits": face_logits}
            if return_count:
                outputs["count_logits"] = self.predict_face_count_logits(point_features, vertex_table)
            if return_seed:
                outputs["seed_logits"] = self.seed_face_logits(point_features, vertex_table)
            topology_hidden = self._topology_hidden_from_local(hidden, spatial_local_context)
            if return_topology:
                outputs["closure_logits"] = self.topology_output(topology_hidden)
            if edge_action_indices is not None:
                outputs["edge_action_logits"] = self._edge_action_logits_from_hidden(
                    topology_hidden,
                    edge_action_indices,
                    vertex_table=vertex_table,
                )
            if edge_choice_candidates is not None:
                outputs["edge_choice_logits"] = self._edge_choice_logits_from_hidden(
                    topology_hidden,
                    edge_choice_candidates,
                    vertex_table=vertex_table,
                )
            if return_hidden:
                outputs["hidden"] = hidden
            return outputs

        def forward_with_topology(self, point_features, vertex_table, input_faces):  # type: ignore[no-untyped-def]
            return self.forward(point_features, vertex_table, input_faces, return_topology=True)

        def forward_corner_causal(self, point_features, vertex_table, input_faces, target_faces):  # type: ignore[no-untyped-def]
            return self.forward(point_features, vertex_table, input_faces, target_faces=target_faces)

        def corner_causal_next_logits(self, point_features, vertex_table, input_faces, prefix_faces):  # type: ignore[no-untyped-def]
            hidden = self._hidden(point_features, vertex_table, input_faces)[:, -1:, :]
            prefix = prefix_faces.reshape(prefix_faces.shape[0], 1, 3)
            return self._corner_causal_logits_from_hidden(hidden, prefix, vertex_table=vertex_table)[:, 0, :, :]

        def forward_edge_action(self, point_features, vertex_table, input_faces, edge_indices):  # type: ignore[no-untyped-def]
            return self.forward(
                point_features,
                vertex_table,
                input_faces,
                edge_action_indices=edge_indices,
            )["edge_action_logits"]

        def forward_edge_choice(self, point_features, vertex_table, input_faces, candidate_edges):  # type: ignore[no-untyped-def]
            return self.forward(
                point_features,
                vertex_table,
                input_faces,
                edge_choice_candidates=candidate_edges,
            )["edge_choice_logits"]

        def edge_action_next_logits(self, point_features, vertex_table, input_faces, edge_indices):  # type: ignore[no-untyped-def]
            hidden = self._hidden(point_features, vertex_table, input_faces)[:, -1:, :]
            edge_prefix = edge_indices.reshape(edge_indices.shape[0], 1, 2)
            return self.edge_action_logits_from_hidden(
                hidden,
                edge_prefix,
                point_features=point_features,
                vertex_table=vertex_table,
                input_faces=input_faces,
            )[:, 0, :]

        def edge_choice_next_logits(self, point_features, vertex_table, input_faces, candidate_edges):  # type: ignore[no-untyped-def]
            hidden = self._hidden(point_features, vertex_table, input_faces)[:, -1:, :]
            edge_prefix = candidate_edges.reshape(candidate_edges.shape[0], 1, candidate_edges.shape[-2], 2)
            return self.edge_choice_logits_from_hidden(
                hidden,
                edge_prefix,
                point_features=point_features,
                vertex_table=vertex_table,
                input_faces=input_faces,
            )[:, 0, :]

        def _corner_causal_logits_from_hidden(self, hidden, corner_prefix, vertex_table=None):  # type: ignore[no-untyped-def]
            batch, face_count, _ = hidden.shape
            shifted = corner_prefix.clamp(min=-1, max=self.max_vertices - 1) + 1
            embedded = self.index_embedding(shifted)
            bos = self.corner_bos.reshape(1, 1, self.hidden_size).expand(batch, face_count, -1)
            logits = []
            vertex_hidden = self._vertex_hidden(vertex_table) if self.face_output_mode == "geometry" else None
            vertex_keys = self.corner_vertex_key(vertex_hidden) if vertex_hidden is not None else None
            valid_vertices = self._valid_vertex_mask(vertex_table) if vertex_table is not None else None
            for corner in range(3):
                prev_a = bos if corner == 0 else embedded[:, :, 0, :]
                prev_b = bos if corner <= 1 else embedded[:, :, 1, :]
                pos = self.corner_position_embedding.weight[corner].reshape(1, 1, self.hidden_size)
                context = torch.cat([hidden + pos, prev_a, prev_b], dim=-1)
                if self.face_output_mode == "geometry":
                    query = self.corner_query_mlp(context)
                    slot_logits = torch.einsum("bfh,bvh->bfv", query, vertex_keys) / float(self.hidden_size ** 0.5)
                    slot_logits = slot_logits.masked_fill(~valid_vertices.unsqueeze(1), -1e9)
                    logits.append(slot_logits)
                else:
                    logits.append(self.corner_mlp(context))
            return torch.stack(logits, dim=2)

        def _edge_action_logits_from_hidden(self, hidden, edge_indices, vertex_table=None):  # type: ignore[no-untyped-def]
            if self.edge_head_mode == "geometry" and vertex_table is not None:
                if vertex_table.shape[0] == 1 and hidden.shape[0] > 1:
                    vertex_table = vertex_table.expand(hidden.shape[0], -1, -1)
                vertex_hidden = self._vertex_hidden(vertex_table)
                edge_a = self._gather_vertex_hidden(vertex_hidden, edge_indices[..., 0])
                edge_b = self._gather_vertex_hidden(vertex_hidden, edge_indices[..., 1])
                context = self.edge_action_context_mlp(torch.cat([hidden, edge_a, edge_b], dim=-1))
                query = self.edge_action_query(context)
                keys = self.edge_action_key(vertex_hidden)
                logits = torch.einsum("bfh,bvh->bfv", query, keys) / float(self.hidden_size ** 0.5)
                valid_vertices = vertex_table[..., 0].ge(0).unsqueeze(1)
                logits = logits.masked_fill(~valid_vertices, -1e9)
                for corner in range(2):
                    invalid = edge_indices[..., corner].lt(0)
                    endpoint = edge_indices[..., corner].clamp(min=0, max=self.max_vertices - 1).unsqueeze(-1)
                    endpoint_mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, endpoint, True)
                    logits = logits.masked_fill(endpoint_mask & ~invalid.unsqueeze(-1), -1e9)
                return logits
            shifted = edge_indices.clamp(min=-1, max=self.max_vertices - 1) + 1
            embedded = self.index_embedding(shifted)
            context = torch.cat([hidden, embedded[:, :, 0, :], embedded[:, :, 1, :]], dim=-1)
            return self.edge_action_mlp(context)

        def _edge_choice_logits_from_hidden(self, hidden, candidate_edges, vertex_table=None):  # type: ignore[no-untyped-def]
            if self.edge_head_mode == "geometry" and vertex_table is not None:
                if vertex_table.shape[0] == 1 and hidden.shape[0] > 1:
                    vertex_table = vertex_table.expand(hidden.shape[0], -1, -1)
                vertex_hidden = self._vertex_hidden(vertex_table)
                edge_a = self._gather_vertex_hidden(vertex_hidden, candidate_edges[..., 0])
                edge_b = self._gather_vertex_hidden(vertex_hidden, candidate_edges[..., 1])
                if candidate_edges.ndim == 3:
                    context = torch.cat([hidden, edge_a, edge_b], dim=-1)
                    return self.edge_choice_geometry_mlp(context).squeeze(-1)
                if candidate_edges.ndim == 4:
                    hidden_expanded = hidden.unsqueeze(2).expand(-1, -1, candidate_edges.shape[2], -1)
                    context = torch.cat([hidden_expanded, edge_a, edge_b], dim=-1)
                    return self.edge_choice_geometry_mlp(context).squeeze(-1)
                raise ValueError(f"candidate_edges must have shape (B,F,2) or (B,F,C,2), got {tuple(candidate_edges.shape)}")
            shifted = candidate_edges.clamp(min=-1, max=self.max_vertices - 1) + 1
            embedded = self.index_embedding(shifted)
            if embedded.ndim == 4:
                context = torch.cat([hidden, embedded[:, :, 0, :], embedded[:, :, 1, :]], dim=-1)
                return self.edge_choice_mlp(context).squeeze(-1)
            if embedded.ndim == 5:
                hidden_expanded = hidden.unsqueeze(2).expand(-1, -1, embedded.shape[2], -1)
                context = torch.cat([hidden_expanded, embedded[:, :, :, 0, :], embedded[:, :, :, 1, :]], dim=-1)
                return self.edge_choice_mlp(context).squeeze(-1)
            raise ValueError(f"candidate_edges must have shape (B,F,2) or (B,F,C,2), got {tuple(candidate_edges.shape)}")

        def _gather_vertex_hidden(self, vertex_hidden, indices):  # type: ignore[no-untyped-def]
            batch, vertex_count, channels = vertex_hidden.shape
            clamped = indices.clamp(min=0, max=vertex_count - 1)
            flat = clamped.reshape(batch, -1)
            gathered = torch.gather(
                vertex_hidden,
                dim=1,
                index=flat.unsqueeze(-1).expand(-1, -1, channels),
            )
            gathered = gathered.reshape(*indices.shape, channels)
            return gathered * indices.ge(0).unsqueeze(-1).to(gathered.dtype)

    return TinyPointConditionedIndexedFaceDecoder()
