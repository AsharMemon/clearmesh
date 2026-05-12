"""Tiny VDF regressor for LATTICE smoke training."""

from __future__ import annotations


def build_tiny_vdf_regressor(
    input_dim: int = 3,
    output_dim: int = 12,
    hidden_size: int = 128,
    layers: int = 4,
):
    """Build a small MLP that predicts vertex displacements and normals."""

    try:
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional ML env
        raise RuntimeError("PyTorch is required for LATTICE VDF training") from exc

    blocks: list[nn.Module] = []
    dim = input_dim
    for _ in range(max(1, layers)):
        blocks.extend([nn.Linear(dim, hidden_size), nn.SiLU(), nn.LayerNorm(hidden_size)])
        dim = hidden_size
    blocks.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*blocks)
