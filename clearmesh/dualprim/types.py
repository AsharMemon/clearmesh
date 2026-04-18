"""DualPrim data structures.

A DualPrimitive pairs a "positive" superquadric (PSQ) that adds shape
with a "negative" superquadric (NSQ) that carves it. Together with an
opacity α, render sharpness θ, and basic color, this is the full
per-primitive state described in Table 1 of the paper.

The scene-level state is a tensor of K dual-primitives plus a small
MLP that produces view-dependent color. We represent the primitive
state as a single torch.Tensor of shape (K, P) where P=27 so that the
full scene is optimized by a single Adam. Helpers below split the
flat tensor back into named fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------
# Flat parameter layout (one row per dual-primitive)
# ---------------------------------------------------------------------
# Index   Field
# [0:3]   PSQ scale (a_x, a_y, a_z)
# [3:6]   NSQ scale
# [6:8]   PSQ shape (ε1, ε2)
# [8:10]  NSQ shape
# [10]    alpha (opacity / pruning weight)
# [11]    theta (render sharpness θ_S)
# [12:15] PSQ translation
# [15:18] NSQ translation
# [18:21] PSQ rotation (XYZ Euler, radians internally; config range is deg)
# [21:24] NSQ rotation
# [24:27] basic color (RGB in [0, 1])
# ---------------------------------------------------------------------

IDX_PSQ_SCALE = slice(0, 3)
IDX_NSQ_SCALE = slice(3, 6)
IDX_PSQ_SHAPE = slice(6, 8)
IDX_NSQ_SHAPE = slice(8, 10)
IDX_ALPHA = 10
IDX_THETA = 11
IDX_PSQ_TRANSLATION = slice(12, 15)
IDX_NSQ_TRANSLATION = slice(15, 18)
IDX_PSQ_ROTATION = slice(18, 21)
IDX_NSQ_ROTATION = slice(21, 24)
IDX_COLOR = slice(24, 27)

DUAL_PRIM_DIM = 27


@dataclass
class DualPrimitive:
    """Single dual-primitive (Python-side view, used for export/IO).

    During optimization the full scene lives in a single
    (K, DUAL_PRIM_DIM) tensor; this dataclass is for inspecting and
    serializing individual primitives.
    """
    psq_scale: torch.Tensor             # (3,)
    nsq_scale: torch.Tensor             # (3,)
    psq_shape: torch.Tensor             # (2,) — (ε1, ε2)
    nsq_shape: torch.Tensor             # (2,)
    alpha: torch.Tensor                 # scalar
    theta: torch.Tensor                 # scalar — render sharpness
    psq_translation: torch.Tensor       # (3,)
    nsq_translation: torch.Tensor       # (3,)
    psq_rotation_rad: torch.Tensor      # (3,) — Euler XYZ in radians
    nsq_rotation_rad: torch.Tensor      # (3,)
    color: torch.Tensor                 # (3,) — basic RGB

    def to_vector(self) -> torch.Tensor:
        """Pack into a (DUAL_PRIM_DIM,) row of the scene tensor."""
        return torch.cat([
            self.psq_scale, self.nsq_scale,
            self.psq_shape, self.nsq_shape,
            self.alpha.reshape(1), self.theta.reshape(1),
            self.psq_translation, self.nsq_translation,
            self.psq_rotation_rad, self.nsq_rotation_rad,
            self.color,
        ]).flatten()

    @classmethod
    def from_vector(cls, v: torch.Tensor) -> "DualPrimitive":
        assert v.shape[-1] == DUAL_PRIM_DIM, f"expected {DUAL_PRIM_DIM}, got {v.shape}"
        return cls(
            psq_scale=v[IDX_PSQ_SCALE],
            nsq_scale=v[IDX_NSQ_SCALE],
            psq_shape=v[IDX_PSQ_SHAPE],
            nsq_shape=v[IDX_NSQ_SHAPE],
            alpha=v[IDX_ALPHA],
            theta=v[IDX_THETA],
            psq_translation=v[IDX_PSQ_TRANSLATION],
            nsq_translation=v[IDX_NSQ_TRANSLATION],
            psq_rotation_rad=v[IDX_PSQ_ROTATION],
            nsq_rotation_rad=v[IDX_NSQ_ROTATION],
            color=v[IDX_COLOR],
        )


# ---------------------------------------------------------------------
# Scene state
# ---------------------------------------------------------------------

class DualPrimScene:
    """The full scene: K dual-primitives + lighting MLP.

    The primitive parameters are a single nn.Parameter so a single Adam
    optimizes everything. Pruning zeros out rows — they remain in the
    tensor (to keep slot indices stable across the optimization) but
    their alpha is clamped to 0 and the optimizer ignores their
    gradients via a mask.
    """

    def __init__(self, params: torch.Tensor, lighting_mlp: Optional[torch.nn.Module] = None):
        # params: (K, DUAL_PRIM_DIM) torch.Tensor, requires_grad=True
        assert params.dim() == 2 and params.shape[-1] == DUAL_PRIM_DIM
        self.params = params
        self.lighting_mlp = lighting_mlp
        self.alive: torch.Tensor = torch.ones(
            params.shape[0], dtype=torch.bool, device=params.device,
        )

    @property
    def K(self) -> int:
        return self.params.shape[0]

    @property
    def num_alive(self) -> int:
        return int(self.alive.sum().item())

    # ---- Per-field views (no copy, just slices) ----------------------
    def psq_scale(self):          return self.params[:, IDX_PSQ_SCALE]
    def nsq_scale(self):          return self.params[:, IDX_NSQ_SCALE]
    def psq_shape(self):          return self.params[:, IDX_PSQ_SHAPE]
    def nsq_shape(self):          return self.params[:, IDX_NSQ_SHAPE]
    def alpha(self):              return self.params[:, IDX_ALPHA]
    def theta(self):              return self.params[:, IDX_THETA]
    def psq_translation(self):    return self.params[:, IDX_PSQ_TRANSLATION]
    def nsq_translation(self):    return self.params[:, IDX_NSQ_TRANSLATION]
    def psq_rotation(self):       return self.params[:, IDX_PSQ_ROTATION]
    def nsq_rotation(self):       return self.params[:, IDX_NSQ_ROTATION]
    def color(self):              return self.params[:, IDX_COLOR]

    def get_primitive(self, i: int) -> DualPrimitive:
        """Extract one dual-primitive as a dataclass (detached)."""
        return DualPrimitive.from_vector(self.params[i].detach())

    def live_primitives(self) -> list[DualPrimitive]:
        """All alive dual-primitives as dataclasses."""
        idx = self.alive.nonzero(as_tuple=True)[0].tolist()
        return [self.get_primitive(i) for i in idx]
