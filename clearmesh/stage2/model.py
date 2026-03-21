"""Stage 2 Refinement DiT — direct residual prediction with TRELLIS.2 backbone.

Predicts a SLAT delta (residual) that refines coarse SLAT features:
    refined_slat = coarse_slat + model(coarse_slat, cond_features)

Single forward pass, deterministic output, no noise schedule or DDIM sampling.
The predicted delta is decoded by TRELLIS.2's frozen FlexiDualGridVaeDecoder.

Key design decisions:
  - Matches TRELLIS.2's SLatFlowModel state_dict layout (hidden=1536, heads=12,
    fused QKV, shared AdaLN, QK RMS norm) so we can load pretrained weights
  - Loads first N of 30 TRELLIS.2 blocks (default 12 — ~528M params)
  - Frozen backbone: first N-K blocks frozen, last K blocks trainable (~50-80M)
  - Fixed dummy timestep t=0 for AdaLN compatibility with pretrained weights
  - out_head MLP predicts 32-dim SLAT residual (delta)
  - Dense attention (not sparse) — same weight shapes, works on batched tensors
  - Gradient checkpointing for memory efficiency during training
  - Image token masking for cleaner DINO conditioning
  - All operations in normalized SLAT space (zero-mean, unit-std per channel)
  - Output fed through TRELLIS.2's FlexiDualGridVaeDecoder at inference

Architecture (per block, matching TRELLIS.2):
  1. AdaLN-modulated self-attention with 3D RoPE + QK RMS Norm
  2. Cross-attention to DINO features (optional token masking) + QK RMS Norm
  3. AdaLN-modulated feed-forward (GELU-tanh, ratio=5.3334)

Weight key compatibility (all match TRELLIS.2 exactly):
  t_embedder.mlp.{0,2}.{weight,bias}
  adaLN_modulation.1.{weight,bias}
  input_layer.{weight,bias}
  blocks.{i}.modulation
  blocks.{i}.self_attn.to_qkv.{weight,bias}
  blocks.{i}.self_attn.{q,k}_rms_norm.gamma
  blocks.{i}.self_attn.to_out.{weight,bias}
  blocks.{i}.norm2.{weight,bias}
  blocks.{i}.cross_attn.to_q.{weight,bias}
  blocks.{i}.cross_attn.to_kv.{weight,bias}
  blocks.{i}.cross_attn.{q,k}_rms_norm.gamma
  blocks.{i}.cross_attn.to_out.{weight,bias}
  blocks.{i}.mlp.mlp.{0,2}.{weight,bias}
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# RoPE — 3D Rotary Position Embedding (no learnable params)
# ---------------------------------------------------------------------------

def rope_3d(
    x: torch.Tensor,
    positions: torch.Tensor,
    freq_dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply 3D Rotary Position Embedding matching TRELLIS.2.

    Uses complex multiplication. Each axis (x, y, z) gets ``freq_dim``
    frequency bands.  Remaining head dims are left untouched.

    Args:
        x: (B, H, N, D) per-head query or key
        positions: (B, N, 3) **integer** voxel coordinates (same scale as
            TRELLIS.2 training — typically 0 … resolution-1).
        freq_dim: Frequency bands per axis (= head_dim // 2 // 3 = 21)
        base: RoPE base frequency (10 000 matches TRELLIS.2)
    """
    B, H, N, D = x.shape
    device = x.device

    # Frequencies: base^(-2i / freq_dim), i ∈ [0, freq_dim)
    freqs = 1.0 / (
        base ** (torch.arange(0, freq_dim, device=device).float() / freq_dim)
    )  # (freq_dim,)

    # Per-axis angles → concatenate
    angles = []
    for axis in range(3):
        pos = positions[:, :, axis : axis + 1].float()  # (B, N, 1)
        angles.append(pos * freqs.view(1, 1, -1))       # (B, N, freq_dim)
    angles = torch.cat(angles, dim=-1)  # (B, N, freq_dim*3)

    # Split x into "rope part" (rotated) and "pass-through part"
    rope_dim = freq_dim * 3 * 2  # each freq rotates a pair of dims
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]

    # Complex-multiply rotation
    x_pairs = x_rope.float().reshape(B, H, N, -1, 2)
    x_complex = torch.view_as_complex(x_pairs)              # (B, H, N, rope_dim/2)
    phases = torch.polar(
        torch.ones_like(angles), angles
    ).unsqueeze(1)                                           # (B, 1, N, rope_dim/2)
    x_rotated = torch.view_as_real(x_complex * phases)
    x_rotated = x_rotated.reshape(B, H, N, rope_dim).to(x.dtype)

    return torch.cat([x_rotated, x_pass], dim=-1)


# ---------------------------------------------------------------------------
# QK RMS Norm (per-head, learnable gamma)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Per-head RMS normalisation with learnable scale.

    State-dict key: ``{q,k}_rms_norm.gamma``  shape ``[num_heads, head_dim]``
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, N, D)"""
        rms = x.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        return (x.float() * rms * self.gamma[None, :, None, :]).to(x.dtype)


# ---------------------------------------------------------------------------
# Self-Attention (fused QKV, QK RMS Norm, 3-D RoPE)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Matches TRELLIS.2 ``SparseMultiHeadAttention(type='self')``.

    Keys: to_qkv, q_rms_norm, k_rms_norm, to_out
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.freq_dim = self.head_dim // 2 // 3  # 128//2//3 = 21

        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.q_rms_norm = RMSNorm(num_heads, self.head_dim)
        self.k_rms_norm = RMSNorm(num_heads, self.head_dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.to_qkv(x).reshape(B, N, 3, H, D)
        q, k, v = qkv.unbind(2)                # each (B, N, H, D)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, N, D)

        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
        q = rope_3d(q, positions, self.freq_dim)
        k = rope_3d(k, positions, self.freq_dim)

        out = F.scaled_dot_product_attention(q, k, v)       # (B, H, N, D)
        return self.to_out(out.transpose(1, 2).reshape(B, N, self.dim))


# ---------------------------------------------------------------------------
# Cross-Attention (separate Q / KV, QK RMS Norm, token masking)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Matches TRELLIS.2 ``SparseMultiHeadAttention(type='cross')``.

    Keys: to_q, to_kv, q_rms_norm, k_rms_norm, to_out
    """

    def __init__(self, dim: int, cond_dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(cond_dim, 2 * dim)
        self.q_rms_norm = RMSNorm(num_heads, self.head_dim)
        self.k_rms_norm = RMSNorm(num_heads, self.head_dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            context: (B, M, C) DINO image features
            context_mask: (B, M) bool — True = foreground (keep), False = background
        """
        B, N, _ = x.shape
        M = context.shape[1]
        H, D = self.num_heads, self.head_dim

        if M == 0:
            return x.new_zeros(B, N, self.dim)

        q = self.to_q(x).reshape(B, N, H, D).transpose(1, 2)
        kv = self.to_kv(context).reshape(B, M, 2, H, D)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)

        # Image token masking: suppress background tokens
        attn_mask = None
        valid_context = None
        if context_mask is not None:
            # Guard against all-false rows: keep one dummy token alive to avoid NaNs,
            # then zero the whole output for samples with no valid conditioning.
            valid_context = context_mask.any(dim=1)
            safe_mask = context_mask
            if not valid_context.all():
                safe_mask = context_mask.clone()
                safe_mask[~valid_context, 0] = True
            attn_mask = safe_mask[:, None, None, :].expand(B, H, N, M)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        if valid_context is not None and not valid_context.all():
            out = out * valid_context[:, None, None, None].to(out.dtype)
        return self.to_out(out.transpose(1, 2).reshape(B, N, self.dim))


# ---------------------------------------------------------------------------
# Feed-Forward (GELU-tanh, matching TRELLIS.2 double-nested .mlp.mlp path)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Keys: mlp.mlp.{0,2}.{weight,bias}   (double-nested matches TRELLIS.2)"""

    def __init__(self, dim: int, mlp_ratio: float = 5.3334):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# DiT Block (shared AdaLN + per-block offset)
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Matches TRELLIS.2 ``ModulatedSparseTransformerBlock``.

    Flow (identical to TRELLIS.2):
      1. norm1 → AdaLN scale/shift → self_attn → gate → residual
      2. norm2 → cross_attn → residual  (NO AdaLN gate on cross-attn)
      3. norm3 → AdaLN scale/shift → mlp → gate → residual

    Keys per block:
      modulation, self_attn.*, norm2.*, cross_attn.*, mlp.*
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 5.3334,
    ):
        super().__init__()
        # Per-block learnable AdaLN offset (added to shared modulation)
        self.modulation = nn.Parameter(torch.randn(6 * dim) / dim ** 0.5)

        # Pre-norms (no elementwise_affine for self-attn & MLP, matches TRELLIS.2)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        # Cross-attn pre-norm WITH affine (matches TRELLIS.2)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)

        self.self_attn = SelfAttention(dim, num_heads)
        self.cross_attn = CrossAttention(dim, cond_dim, num_heads)
        self.mlp = FeedForward(dim, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        shared_mod: torch.Tensor,
        cond_features: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Combine shared (from timestep) + per-block modulation → 6 vectors
        mod = shared_mod + self.modulation.unsqueeze(0)     # (B, 6D)
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = [
            c.unsqueeze(1) for c in mod.chunk(6, dim=-1)    # each (B, 1, D)
        ]

        # --- Self-attention ---
        h = self.norm1(x) * (1 + sc_msa) + s_msa
        h = self.self_attn(h, positions)
        x = x + g_msa * h

        # --- Cross-attention (no AdaLN gate) ---
        if cond_features is not None:
            h = self.norm2(x)
            h = self.cross_attn(h, cond_features, cond_mask)
            x = x + h

        # --- MLP ---
        h = self.norm3(x) * (1 + sc_mlp) + s_mlp
        h = self.mlp(h)
        x = x + g_mlp * h

        return x


# ---------------------------------------------------------------------------
# Timestep Embedder (sinusoidal → MLP, matching TRELLIS.2 layout)
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Keys: t_embedder.mlp.{0,2}.{weight,bias}"""

    def __init__(self, dim: int, sin_dim: int = 256):
        super().__init__()
        self.sin_dim = sin_dim
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, dim),    # mlp.0
            nn.SiLU(),                  # mlp.1  (no params)
            nn.Linear(dim, dim),        # mlp.2
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.sin_dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / (half - 1))
        )
        emb = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# RefinementDiT — main model
# ---------------------------------------------------------------------------

class RefinementDiT(nn.Module):
    """Stage 2 Refinement DiT — direct residual prediction in SLAT space.

    Architecture matched to TRELLIS.2's ``SLatFlowModel`` (1.3 B shape model)
    so that pretrained weights can be loaded for the transformer body.

    Predicts a SLAT residual (delta) via single forward pass:
        refined_slat = coarse_slat + model(coarse_slat, cond_features)

    At inference, refined SLAT is decoded by TRELLIS.2's frozen decoder.

    Training strategy:
      - Freeze pretrained backbone blocks (first N-K of N total)
      - Train only last K blocks + out_head MLP (~50-80M trainable params)
      - Fixed dummy timestep t=0 for AdaLN compatibility with pretrained weights
      - Loss = L1(coarse + predicted_delta, fine_slat)

    Differences from TRELLIS.2:
      - ``noisy_slat_proj``: kept for state_dict compatibility, unused in residual mode
      - ``out_head``: MLP predicts SLAT delta (LN + 256 hidden + GELU + 32) (FRESH)
      - Dense attention instead of sparse  (same weight shapes)
      - Gradient checkpointing  (optional, for training)

    Args:
        voxel_dim:  Input feature dim (32 = TRELLIS.2 SLAT latent dim)
        model_dim:  Hidden dim (1536 = TRELLIS.2)
        num_heads:  Attention heads (12 = TRELLIS.2)
        num_layers: Blocks to use (12 default; TRELLIS.2 has 30)
        cond_dim:   Conditioning dim (1024 = DINOv2-ViT-L / DINOv3)
        mlp_ratio:  FFN ratio (5.3334 = TRELLIS.2, hidden 8192)
        use_checkpoint: Gradient checkpointing (saves ~60 % VRAM)
    """

    def __init__(
        self,
        voxel_dim: int = 32,
        model_dim: int = 1536,
        num_heads: int = 12,
        num_layers: int = 12,
        cond_dim: int = 1024,
        mlp_ratio: float = 5.3334,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.voxel_dim = voxel_dim
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

        # Input projection  (key: input_layer — matches TRELLIS.2 if voxel_dim=32)
        # Projects coarse SLAT (conditioning) into token space
        self.input_layer = nn.Linear(voxel_dim, model_dim)

        # Noisy SLAT projection — kept for state_dict compatibility with pretrained
        # weights and diffusion checkpoints. NOT used in residual prediction mode.
        self.noisy_slat_proj = nn.Linear(voxel_dim, model_dim)

        # Timestep embedding  (keys: t_embedder.mlp.*)
        # In residual mode: always receives t=0 so AdaLN modulation still functions
        self.t_embedder = TimestepEmbedder(model_dim, sin_dim=256)

        # Shared AdaLN modulation  (keys: adaLN_modulation.1.*)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 6 * model_dim),
        )

        # Transformer blocks  (keys: blocks.{i}.*)
        self.blocks = nn.ModuleList(
            [DiTBlock(model_dim, num_heads, cond_dim, mlp_ratio) for _ in range(num_layers)]
        )

        # Output head  (MLP with LayerNorm)
        # Predicts 32-dim SLAT residual (delta = fine - coarse)
        self.out_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 256),
            nn.GELU(),
            nn.Linear(256, voxel_dim),  # 32-dim SLAT delta
        )

        # Careful initialisation for fresh-init layers
        # out_head: near-zero init so initial prediction ≈ identity (delta ≈ 0)
        nn.init.normal_(self.out_head[1].weight, std=0.02)
        nn.init.zeros_(self.out_head[1].bias)
        nn.init.zeros_(self.out_head[3].weight)  # zero init → delta starts at 0
        nn.init.zeros_(self.out_head[3].bias)
        nn.init.normal_(self.noisy_slat_proj.weight, std=0.02)
        nn.init.zeros_(self.noisy_slat_proj.bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        coarse_voxels: torch.Tensor,
        positions: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        cond_features: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass — predicts SLAT delta (residual).

        In residual mode, timestep is fixed at t=0 internally. The noisy_slat_proj
        is NOT used. The model takes coarse SLAT as input and predicts a delta.

        Args:
            coarse_voxels: (B, N, voxel_dim) coarse SLAT features (normalized)
            positions: (B, N, 3) **integer** voxel coords (0…R-1)
            timestep: (B,) optional — if None, uses fixed t=0 (residual mode)
            cond_features: (B, M, cond_dim) DINO image features
            cond_mask: (B, M) bool — True = foreground token to keep

        Returns:
            (B, N, voxel_dim) predicted SLAT delta (residual)
        """
        B = coarse_voxels.shape[0]
        device = coarse_voxels.device

        x = self.input_layer(coarse_voxels)

        # Fixed t=0 for AdaLN compatibility with pretrained weights
        if timestep is None:
            timestep = torch.zeros(B, device=device)

        # Timestep → shared modulation for all blocks
        t_emb = self.t_embedder(timestep)
        shared_mod = self.adaLN_modulation(t_emb)  # (B, 6·D)

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(
                    block, x, positions, shared_mod, cond_features, cond_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, positions, shared_mod, cond_features, cond_mask)

        return self.out_head(x)

    # ------------------------------------------------------------------

    def freeze_backbone(self, trainable_blocks: int = 3):
        """Freeze pretrained backbone, keeping only last K blocks + out_head trainable.

        Freezes: input_layer, t_embedder, adaLN_modulation, noisy_slat_proj,
                 blocks[0 : num_layers - trainable_blocks]
        Trainable: blocks[num_layers - trainable_blocks :], out_head

        Args:
            trainable_blocks: Number of final blocks to keep trainable (default 3).
                With 12 blocks: freezes 0-8, trains 9-11 + out_head.
                Each block ≈ 28.9M params → 3 blocks + head ≈ 87M trainable.
        """
        freeze_until = self.num_layers - trainable_blocks

        # Freeze shared layers
        for param in self.input_layer.parameters():
            param.requires_grad = False
        for param in self.t_embedder.parameters():
            param.requires_grad = False
        for param in self.adaLN_modulation.parameters():
            param.requires_grad = False
        for param in self.noisy_slat_proj.parameters():
            param.requires_grad = False

        # Freeze early blocks
        for i, block in enumerate(self.blocks):
            if i < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True

        # out_head always trainable
        for param in self.out_head.parameters():
            param.requires_grad = True

        # Summary
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = frozen + trainable
        print(f"Backbone frozen: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable "
              f"(blocks {freeze_until}-{self.num_layers-1} + out_head)")
        return trainable

    # ------------------------------------------------------------------

    @torch.no_grad()
    def refine_residual(
        self,
        coarse_voxels: torch.Tensor,
        positions: torch.Tensor,
        cond_features: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        delta_scale: float = 1.0,
    ) -> torch.Tensor:
        """Direct residual refinement — single forward pass.

        Computes: refined_slat = coarse_slat + delta_scale * model(coarse_slat, cond)

        Args:
            coarse_voxels: (B, N, 32) **normalized** coarse SLAT features
            positions: (B, N, 3) integer voxel coords
            cond_features: (B, M, cond_dim) DINO image conditioning
            cond_mask: (B, M) bool foreground mask
            delta_scale: Scale factor for predicted delta (1.0 = full refinement,
                         <1.0 = conservative, >1.0 = aggressive). Default 1.0.

        Returns:
            (B, N, 32) refined SLAT features in normalized space
        """
        delta = self.forward(
            coarse_voxels, positions,
            cond_features=cond_features,
            cond_mask=cond_mask,
        )
        return coarse_voxels + delta_scale * delta

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        num_layers: int = 12,
        **kwargs,
    ) -> "RefinementDiT":
        """Create model and load TRELLIS.2 pretrained weights.

        Loads the first ``num_layers`` blocks from TRELLIS.2's shape DiT.
        Re-initialises ``out_layer`` (dim mismatch) and ``noisy_slat_proj`` (new).

        Args:
            checkpoint_path: Path to .safetensors or .pt checkpoint
            num_layers: Blocks to load (default 12 of 30)
        """
        model = cls(num_layers=num_layers, **kwargs)

        # Load checkpoint
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        model_state = model.state_dict()
        loaded, skipped_shape, skipped_extra = [], [], []

        for key, param in state_dict.items():
            # Skip blocks beyond our num_layers
            if key.startswith("blocks."):
                block_idx = int(key.split(".")[1])
                if block_idx >= num_layers:
                    continue

            if key in model_state:
                if param.shape == model_state[key].shape:
                    model_state[key] = param
                    loaded.append(key)
                else:
                    skipped_shape.append(
                        f"  {key}: ckpt {list(param.shape)} vs model {list(model_state[key].shape)}"
                    )
            else:
                skipped_extra.append(key)

        model.load_state_dict(model_state, strict=False)

        print(f"\n{'='*60}")
        print(f"Pretrained weight loading from: {checkpoint_path}")
        print(f"  Loaded:           {len(loaded)} / {len(model_state)} keys")
        print(f"  Shape mismatch:   {len(skipped_shape)}")
        for s in skipped_shape:
            print(s)
        print(f"  Not in model:     {len(skipped_extra)} (higher blocks, etc.)")
        print(f"  Fresh init:       out_head (MLP), noisy_slat_proj")
        print(f"{'='*60}\n")

        return model

    @classmethod
    def from_residual_checkpoint(
        cls,
        checkpoint_path: str,
        num_layers: int = 12,
        trainable_blocks: int = 3,
        **kwargs,
    ) -> "RefinementDiT":
        """Load a Stage 2 residual prediction checkpoint.

        Loads the full model state (including frozen backbone weights)
        from a Stage 2 training checkpoint, then re-freezes the backbone.

        Args:
            checkpoint_path: Path to Stage 2 .pt checkpoint
            num_layers: Blocks in the model
            trainable_blocks: Number of final blocks that were trainable
        """
        model = cls(num_layers=num_layers, **kwargs)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict, strict=True)

        step = ckpt.get("global_step", "?")
        print(f"Loaded residual checkpoint (step {step}) from {checkpoint_path}")

        model.freeze_backbone(trainable_blocks)
        return model
