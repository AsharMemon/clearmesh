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

    class TinyPointConditionedIndexedFaceDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.condition_tokens = int(condition_tokens)
            self.max_vertices = int(max_vertices)
            self.max_faces = int(max_faces)
            self.num_bins = int(num_bins)
            self.hidden_size = int(hidden_size)
            if edge_head_mode not in {"index", "geometry"}:
                raise ValueError(f"edge_head_mode must be 'index' or 'geometry', got {edge_head_mode!r}")
            self.edge_head_mode = edge_head_mode
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
            self.output = nn.Linear(hidden_size, 3 * max_vertices)
            self.topology_output = nn.Linear(hidden_size, 4)
            self.corner_bos = nn.Parameter(torch.randn(hidden_size) * 0.02)
            self.corner_position_embedding = nn.Embedding(3, hidden_size)
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
            encoded = self.point_encoder(point_features)
            pooled_mean = encoded.mean(dim=1)
            pooled_max = encoded.max(dim=1).values
            point_pooled = self.condition_projection(torch.cat([pooled_mean, pooled_max], dim=-1))
            vertex_hidden = self._vertex_hidden(vertex_table)
            vertex_mask = vertex_table[..., 0].ge(0).to(vertex_hidden.dtype).unsqueeze(-1)
            vertex_pooled = (vertex_hidden * vertex_mask).sum(dim=1) / vertex_mask.sum(dim=1).clamp(min=1.0)
            return point_pooled + vertex_pooled

        def _condition(self, point_features, vertex_table):  # type: ignore[no-untyped-def]
            pooled = self._pooled_condition(point_features, vertex_table)
            return pooled.unsqueeze(1) + self.condition_queries.unsqueeze(0)

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

        def _hidden(self, point_features, vertex_table, input_faces):  # type: ignore[no-untyped-def]
            batch, face_count, corners = input_faces.shape
            if corners != 3:
                raise ValueError(f"input_faces must have 3 indices per face, got {corners}")
            if face_count > self.max_faces:
                raise ValueError(f"face_count {face_count} exceeds max_faces={self.max_faces}")
            condition = self._condition(point_features, vertex_table)
            # -1 is BOS/padding in batches; shift real indices by +1 for embedding.
            shifted = input_faces.clamp(min=-1, max=self.max_vertices - 1) + 1
            face_tokens = self.index_embedding(shifted).reshape(batch, face_count, 3 * self.hidden_size)
            face_tokens = self.face_projection(face_tokens)
            face_tokens[:, 0, :] = self.bos_face.unsqueeze(0)
            positions = torch.arange(face_count, device=input_faces.device).unsqueeze(0).expand(batch, -1)
            face_tokens = face_tokens + self.face_position_embedding(positions)
            x = torch.cat([condition, face_tokens], dim=1)
            x = self.blocks(x, mask=self._attention_mask(face_count, input_faces.device))
            return self.norm(x[:, self.condition_tokens :, :])

        def forward(self, point_features, vertex_table, input_faces):  # type: ignore[no-untyped-def]
            batch, face_count, _ = input_faces.shape
            hidden = self._hidden(point_features, vertex_table, input_faces)
            logits = self.output(hidden)
            return logits.reshape(batch, face_count, 3, self.max_vertices)

        def forward_with_topology(self, point_features, vertex_table, input_faces):  # type: ignore[no-untyped-def]
            batch, face_count, _ = input_faces.shape
            hidden = self._hidden(point_features, vertex_table, input_faces)
            face_logits = self.output(hidden).reshape(batch, face_count, 3, self.max_vertices)
            closure_logits = self.topology_output(hidden)
            return {"face_logits": face_logits, "closure_logits": closure_logits}

        def forward_corner_causal(self, point_features, vertex_table, input_faces, target_faces):  # type: ignore[no-untyped-def]
            hidden = self._hidden(point_features, vertex_table, input_faces)
            prefix = target_faces.masked_fill(target_faces.lt(0), -1)
            return self._corner_causal_logits_from_hidden(hidden, prefix)

        def corner_causal_next_logits(self, point_features, vertex_table, input_faces, prefix_faces):  # type: ignore[no-untyped-def]
            hidden = self._hidden(point_features, vertex_table, input_faces)[:, -1:, :]
            prefix = prefix_faces.reshape(prefix_faces.shape[0], 1, 3)
            return self._corner_causal_logits_from_hidden(hidden, prefix)[:, 0, :, :]

        def forward_edge_action(self, point_features, vertex_table, input_faces, edge_indices):  # type: ignore[no-untyped-def]
            hidden = self._hidden(point_features, vertex_table, input_faces)
            return self._edge_action_logits_from_hidden(hidden, edge_indices, vertex_table=vertex_table)

        def forward_edge_choice(self, point_features, vertex_table, input_faces, candidate_edges):  # type: ignore[no-untyped-def]
            hidden = self._hidden(point_features, vertex_table, input_faces)
            return self._edge_choice_logits_from_hidden(hidden, candidate_edges, vertex_table=vertex_table)

        def edge_action_next_logits(self, point_features, vertex_table, input_faces, edge_indices):  # type: ignore[no-untyped-def]
            hidden = self._hidden(point_features, vertex_table, input_faces)[:, -1:, :]
            edge_prefix = edge_indices.reshape(edge_indices.shape[0], 1, 2)
            return self._edge_action_logits_from_hidden(hidden, edge_prefix, vertex_table=vertex_table)[:, 0, :]

        def edge_choice_next_logits(self, point_features, vertex_table, input_faces, candidate_edges):  # type: ignore[no-untyped-def]
            hidden = self._hidden(point_features, vertex_table, input_faces)[:, -1:, :]
            edge_prefix = candidate_edges.reshape(candidate_edges.shape[0], 1, candidate_edges.shape[-2], 2)
            return self._edge_choice_logits_from_hidden(hidden, edge_prefix, vertex_table=vertex_table)[:, 0, :]

        def _corner_causal_logits_from_hidden(self, hidden, corner_prefix):  # type: ignore[no-untyped-def]
            batch, face_count, _ = hidden.shape
            shifted = corner_prefix.clamp(min=-1, max=self.max_vertices - 1) + 1
            embedded = self.index_embedding(shifted)
            bos = self.corner_bos.reshape(1, 1, self.hidden_size).expand(batch, face_count, -1)
            logits = []
            for corner in range(3):
                prev_a = bos if corner == 0 else embedded[:, :, 0, :]
                prev_b = bos if corner <= 1 else embedded[:, :, 1, :]
                pos = self.corner_position_embedding.weight[corner].reshape(1, 1, self.hidden_size)
                context = torch.cat([hidden + pos, prev_a, prev_b], dim=-1)
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
