"""Tiny FACE-token autoregressor for smoke training.

This module is intentionally modest: it is a correctness scaffold, not the
paper-scale FACE ARAE. Its job is to prove that our FACE token stream can be
learned end-to-end, checkpointed, and moved to Thunder before we build the
larger autoencoder/image-conditioned model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .face_tokens import FaceTokenSequence


@dataclass(frozen=True)
class FaceTinyVocabulary:
    num_bins: int

    @property
    def bos(self) -> int:
        return int(self.num_bins)

    @property
    def eos(self) -> int:
        return int(self.num_bins + 1)

    @property
    def vocab_size(self) -> int:
        return int(self.num_bins + 2)


def sequence_to_autoregressive_tokens(sequence: FaceTokenSequence) -> tuple[np.ndarray, np.ndarray]:
    """Return teacher-forcing ``input_ids`` and ``target_ids`` for one mesh."""

    vocab = FaceTinyVocabulary(sequence.num_bins)
    flat = sequence.as_flat_tokens().astype(np.int64)
    input_ids = np.concatenate([[vocab.bos], flat])
    target_ids = np.concatenate([flat, [vocab.eos]])
    return input_ids.astype(np.int64), target_ids.astype(np.int64)


def build_tiny_face_autoregressor(
    num_bins: int,
    max_tokens: int,
    hidden_size: int = 128,
    layers: int = 2,
    heads: int = 4,
):
    """Build a small causal Transformer over FACE coordinate tokens."""

    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional ML env
        raise RuntimeError("PyTorch is required for FACE tiny training") from exc

    vocab = FaceTinyVocabulary(num_bins)

    class TinyFaceAutoregressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
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

        def forward(self, input_ids):  # type: ignore[no-untyped-def]
            batch, seq_len = input_ids.shape
            if seq_len > max_tokens:
                raise ValueError(f"input sequence length {seq_len} exceeds max_tokens={max_tokens}")
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                diagonal=1,
            )
            x = self.blocks(x, mask=causal_mask)
            return self.output(self.norm(x))

    return TinyFaceAutoregressor()
