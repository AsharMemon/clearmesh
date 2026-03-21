"""Stage 2 v2 Flow Matching DiT — clean-room refinement in TRELLIS.2 SLAT space.

Uses the FULL pretrained TRELLIS.2 shape DiT backbone with minimal task-specific
modifications, matching UltraShape's approach of fine-tuning rather than redesigning.

Conditional generation approach: given coarse SLAT (from 512 model) + image
features, generate refined SLAT from noise at the same positions. The 1024
diffusion model output serves as the training target.

Architecture (inheriting from TRELLIS.2, verified by Gate 0C):
  - hidden_dim: 1536
  - num_heads: 12
  - num_layers: all blocks (30)
  - head_dim: 128
  - mlp_ratio: 5.3334
  - RoPE: 3D, freq_dim=21 per axis
  - cond_dim: 1024 (DINOv2 from TRELLIS.2 pipeline)

Task-specific modifications (the minimal changes):
  1. input_proj: Linear(64, 1536) — concatenated [noisy_slat(32) + coarse_slat(32)]
  2. out_head: LayerNorm(1536) + Linear(1536, 32) — zero-initialized
  3. Cross-attention: reused, same 1024-dim (DINOv2 from TRELLIS.2 pipeline)
  4. RoPE: reused, fed shared voxel coordinates (same for coarse and fine)
  5. Timestep embedding: reused, pretrained

Training formulation: rectified flow (velocity prediction) matching TRELLIS.2's
native training objective. CFG with 10% dropout, guidance_scale=5.0 at inference.

Weight loading: load ALL pretrained weights from TRELLIS.2's shape DiT checkpoint.
Only input_proj and out_head are trained from scratch.
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn

from clearmesh.stage2.model import (
    CrossAttention,
    DiTBlock,
    FeedForward,
    RMSNorm,
    SelfAttention,
    TimestepEmbedder,
    rope_3d,
)


class FlowMatchingDiT(nn.Module):
    """Stage 2 v2 Flow Matching DiT for SLAT refinement.

    Full TRELLIS.2 backbone with task-specific input/output projections.
    Predicts velocity v_t for flow matching in SLAT space.

    Forward signature:
        v_pred = model(noisy_slat, coarse_slat, positions, timestep, cond_features)

    The noisy_slat and coarse_slat are concatenated along the feature dim
    before projection into the hidden space.

    Args:
        voxel_dim: SLAT feature dimension (32 for TRELLIS.2)
        model_dim: Hidden dimension (1536 for TRELLIS.2)
        num_heads: Attention heads (12 for TRELLIS.2)
        num_layers: Transformer blocks (30 for TRELLIS.2, verify via Gate 0C)
        cond_dim: Conditioning dimension (1024 for DINOv2 from TRELLIS.2)
        mlp_ratio: FFN expansion ratio (5.3334 for TRELLIS.2)
        use_checkpoint: Gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        voxel_dim: int = 32,
        model_dim: int = 1536,
        num_heads: int = 12,
        num_layers: int = 30,
        cond_dim: int = 1024,
        mlp_ratio: float = 5.3334,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.voxel_dim = voxel_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.use_checkpoint = use_checkpoint

        # --- Task-specific: NEW input projection (trained from scratch) ---
        # Concatenated [noisy_slat(32) + coarse_slat(32)] = 64-dim → hidden
        self.input_proj = nn.Linear(voxel_dim * 2, model_dim)

        # --- Pretrained: Timestep embedding (matching TRELLIS.2 key layout) ---
        self.t_embedder = TimestepEmbedder(model_dim, sin_dim=256)

        # --- Pretrained: Shared AdaLN modulation ---
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 6 * model_dim),
        )

        # --- Pretrained: Transformer blocks (ALL from TRELLIS.2) ---
        self.blocks = nn.ModuleList(
            [DiTBlock(model_dim, num_heads, cond_dim, mlp_ratio)
             for _ in range(num_layers)]
        )

        # --- Task-specific: NEW output head (zero-initialized) ---
        # Predicts velocity v_t in SLAT space (32-dim)
        self.out_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, voxel_dim),
        )

        # --- Initialization ---
        # input_proj: small random init
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)

        # out_head: zero-init final Linear so initial v_pred ≈ 0
        nn.init.zeros_(self.out_head[1].weight)
        nn.init.zeros_(self.out_head[1].bias)

    def forward(
        self,
        noisy_slat: torch.Tensor,
        coarse_slat: torch.Tensor,
        positions: torch.Tensor,
        timestep: torch.Tensor,
        cond_features: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass — predicts velocity v_t for flow matching.

        Args:
            noisy_slat: (B, N, 32) noisy fine SLAT at time t
            coarse_slat: (B, N, 32) coarse SLAT conditioning
            positions: (B, N, 3) integer voxel coordinates
            timestep: (B,) flow matching timestep in [0, 1]
            cond_features: (B, M, 1024) DINOv2 image features
            cond_mask: (B, M) bool — True = foreground token

        Returns:
            (B, N, 32) predicted velocity v_t
        """
        # Concatenate noisy + coarse along feature dim → (B, N, 64)
        x = torch.cat([noisy_slat, coarse_slat], dim=-1)
        x = self.input_proj(x)  # (B, N, 1536)

        # Timestep → shared AdaLN modulation
        t_emb = self.t_embedder(timestep)
        shared_mod = self.adaLN_modulation(t_emb)  # (B, 6·D)

        # Transformer blocks
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, positions, shared_mod, cond_features, cond_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, positions, shared_mod, cond_features, cond_mask)

        return self.out_head(x)  # (B, N, 32) velocity prediction

    # ------------------------------------------------------------------
    # Parameter groups for differential learning rates
    # ------------------------------------------------------------------

    def param_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_new: float = 1e-4,
        lr_cross_attn_kv: float = 5e-5,
    ) -> list[dict]:
        """Create parameter groups with differential learning rates.

        Following UltraShape's fine-tuning approach:
          - Pretrained backbone blocks: small LR (1e-5)
          - New layers (input_proj, out_head): high LR (1e-4)
          - Cross-attention KV projections: moderate LR (5e-5)
        """
        new_params = []
        cross_attn_kv_params = []
        backbone_params = []

        new_modules = {"input_proj", "out_head"}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # New task-specific layers
            if any(name.startswith(m) for m in new_modules):
                new_params.append(param)
            # Cross-attention KV projections
            elif "cross_attn.to_kv" in name:
                cross_attn_kv_params.append(param)
            # Everything else (pretrained backbone)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": new_params, "lr": lr_new, "name": "new_layers"},
            {"params": cross_attn_kv_params, "lr": lr_cross_attn_kv, "name": "cross_attn_kv"},
        ]

    # ------------------------------------------------------------------
    # Pretrained weight loading from TRELLIS.2
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        num_layers: int = 30,
        **kwargs,
    ) -> "FlowMatchingDiT":
        """Create model and load TRELLIS.2 pretrained weights.

        Loads ALL transformer blocks, timestep embedder, and AdaLN modulation
        from TRELLIS.2's shape DiT checkpoint. Only input_proj and out_head
        are fresh-initialized (they don't exist in TRELLIS.2).

        Args:
            checkpoint_path: Path to TRELLIS.2 .safetensors or .pt checkpoint
            num_layers: Number of blocks in checkpoint (expected ~30)
        """
        model = cls(num_layers=num_layers, **kwargs)

        # Load checkpoint
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        model_state = model.state_dict()
        loaded, skipped_shape, skipped_missing = [], [], []

        for key, param in state_dict.items():
            # Skip blocks beyond our num_layers
            if key.startswith("blocks."):
                block_idx = int(key.split(".")[1])
                if block_idx >= num_layers:
                    skipped_missing.append(f"{key} (block {block_idx} >= {num_layers})")
                    continue

            # Map TRELLIS.2 key names to our model
            # TRELLIS.2 uses 'input_layer' for its input projection
            mapped_key = _map_trellis_key(key)

            if mapped_key in model_state:
                if param.shape == model_state[mapped_key].shape:
                    model_state[mapped_key] = param
                    loaded.append(mapped_key)
                else:
                    skipped_shape.append(
                        f"  {key} → {mapped_key}: ckpt {list(param.shape)} "
                        f"vs model {list(model_state[mapped_key].shape)}"
                    )
            else:
                skipped_missing.append(key)

        model.load_state_dict(model_state, strict=False)

        # Re-zero-init output head (in case partial load corrupted it)
        nn.init.zeros_(model.out_head[1].weight)
        nn.init.zeros_(model.out_head[1].bias)

        print(f"\n{'='*60}")
        print(f"FlowMatchingDiT pretrained weight loading")
        print(f"  Source:           {checkpoint_path}")
        print(f"  Loaded:           {len(loaded)} / {len(model_state)} keys")
        if skipped_shape:
            print(f"  Shape mismatch:   {len(skipped_shape)}")
            for s in skipped_shape:
                print(s)
        print(f"  Not in model:     {len(skipped_missing)}")
        print(f"  Fresh init:       input_proj, out_head")
        print(f"{'='*60}\n")

        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        num_layers: int = 30,
        **kwargs,
    ) -> "FlowMatchingDiT":
        """Load a Stage 2 v2 training checkpoint (full model state).

        Args:
            checkpoint_path: Path to Stage 2 v2 .pt checkpoint
            num_layers: Number of blocks in the model
        """
        model = cls(num_layers=num_layers, **kwargs)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict, strict=True)

        step = ckpt.get("global_step", "?")
        print(f"Loaded FlowMatchingDiT checkpoint (step {step}) from {checkpoint_path}")
        return model


class EMA:
    """Exponential Moving Average of model parameters.

    Usage:
        ema = EMA(model, decay=0.9999)
        # In training loop:
        ema.update()
        # For inference:
        ema.apply()      # swap EMA weights into model
        # ... run inference ...
        ema.restore()    # swap original weights back
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply(self):
        """Swap EMA weights into the model (for inference)."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights (after inference)."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {"shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state_dict: dict):
        shadow = state_dict["shadow"]
        for k, v in shadow.items():
            if k in self.shadow:
                self.shadow[k] = v.to(self.shadow[k].device)


# ---------------------------------------------------------------------------
# Key mapping: TRELLIS.2 → FlowMatchingDiT
# ---------------------------------------------------------------------------

def _map_trellis_key(key: str) -> str:
    """Map TRELLIS.2 state_dict key names to FlowMatchingDiT key names.

    TRELLIS.2 uses 'input_layer' for its native 32→1536 projection.
    We use 'input_proj' for our 64→1536 projection (different dim, so this
    will be a shape mismatch and skipped — which is correct).

    All other keys should match exactly since our DiTBlock, SelfAttention,
    CrossAttention, FeedForward, TimestepEmbedder, and adaLN_modulation
    use the same key layout as TRELLIS.2.
    """
    # TRELLIS.2's input_layer won't match our input_proj (dim mismatch)
    # but the key name also differs, so it just gets skipped.
    # No mapping needed for most keys.

    # Handle potential 'out_layer' → we use 'out_head' (skip, fresh init)
    # These are expected to not match.

    return key
