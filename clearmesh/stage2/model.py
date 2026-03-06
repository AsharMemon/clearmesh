"""Stage 2 Refinement DiT — architecture matched to TRELLIS.2 for pretrained weight loading.

Key design decisions:
  - Matches TRELLIS.2's SLatFlowModel state_dict layout (hidden=1536, heads=12,
    fused QKV, shared AdaLN, QK RMS norm) so we can load pretrained weights
  - Loads first N of 30 TRELLIS.2 blocks (default 12 — ~528M params)
  - Adds refinement-specific: sdf_proj (diffusion input), out_layer predicts SDF (dim=1)
  - Dense attention (not sparse) — same weight shapes, works on batched tensors
  - Gradient checkpointing for memory efficiency during training
  - Image token masking for cleaner DINO conditioning

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

        q = self.to_q(x).reshape(B, N, H, D).transpose(1, 2)
        kv = self.to_kv(context).reshape(B, M, 2, H, D)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)

        # Image token masking: suppress background tokens
        attn_mask = None
        if context_mask is not None:
            # (B, M) → (B, 1, 1, M) broadcast over heads and queries
            attn_mask = context_mask[:, None, None, :].expand(B, H, N, M)
            attn_mask = torch.where(attn_mask, 0.0, float("-inf")).to(q.dtype)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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
    """Stage 2 Refinement Diffusion Transformer.

    Architecture matched to TRELLIS.2's ``SLatFlowModel`` (1.3 B shape model)
    so that pretrained weights can be loaded for the transformer body.

    Differences from TRELLIS.2:
      - ``sdf_proj``  : projects noisy SDF into token space  (NEW — fresh init)
      - ``out_layer``  : predicts SDF (dim 1) not SLAT (dim 32)  (RE-INIT)
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
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

        # Input projection  (key: input_layer — matches TRELLIS.2 if voxel_dim=32)
        self.input_layer = nn.Linear(voxel_dim, model_dim)

        # SDF projection  (NOT in TRELLIS.2 — always freshly initialised)
        self.sdf_proj = nn.Linear(1, model_dim)

        # Timestep embedding  (keys: t_embedder.mlp.*)
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

        # Output head  (predicts SDF — dim 1, NOT 32 like TRELLIS.2)
        self.out_layer = nn.Linear(model_dim, 1)

        # Careful initialisation for fresh-init layers
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)
        nn.init.normal_(self.sdf_proj.weight, std=0.02)
        nn.init.zeros_(self.sdf_proj.bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        coarse_voxels: torch.Tensor,
        positions: torch.Tensor,
        timestep: torch.Tensor,
        cond_features: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        noisy_sdf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            coarse_voxels: (B, N, voxel_dim) coarse voxel features
            positions: (B, N, 3) **integer** voxel coords (0…R-1)
            timestep: (B,) diffusion timestep in [0, 1]
            cond_features: (B, M, cond_dim) DINO image features
            cond_mask: (B, M) bool — True = foreground token to keep
            noisy_sdf: (B, N, 1) noisy SDF (diffusion training)

        Returns:
            (B, N, 1) predicted noise
        """
        x = self.input_layer(coarse_voxels)

        if noisy_sdf is not None:
            x = x + self.sdf_proj(noisy_sdf)

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

        return self.out_layer(x)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def refine(
        self,
        coarse_voxels: torch.Tensor,
        positions: torch.Tensor,
        cond_features: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """DDPM denoising inference — refine coarse voxels to SDF."""
        B, N, _ = coarse_voxels.shape
        device = coarse_voxels.device

        sdf = torch.randn(B, N, 1, device=device)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t = timesteps[i].expand(B)
            t_next = timesteps[i + 1].expand(B)

            noise_pred = self.forward(
                coarse_voxels, positions, t, cond_features, cond_mask, sdf
            )

            alpha_t = (1 - t).view(B, 1, 1)
            alpha_next = (1 - t_next).view(B, 1, 1)

            sdf = (sdf - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            if i < num_steps - 1:
                sdf = alpha_next.sqrt() * sdf + (1 - alpha_next).sqrt() * torch.randn_like(sdf)

        return sdf

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
        Re-initialises ``out_layer`` (dim mismatch) and ``sdf_proj`` (new).

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
        print(f"  Fresh init:       out_layer, sdf_proj")
        print(f"{'='*60}\n")

        return model
