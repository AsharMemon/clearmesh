#!/usr/bin/env python3
"""Gate 0C: TRELLIS.2 DiT Architecture Inspection.

Loads the TRELLIS.2 shape DiT checkpoint and documents:
  - Exact number of transformer blocks
  - Hidden dimension (expected: 1536)
  - Number of attention heads (expected: 12)
  - MLP ratio and hidden dim
  - Cross-attention conditioning dimension (expected: 1024)
  - Input projection: what does it take as input?
  - Output projection: what does it predict?
  - Timestep embedding structure
  - RoPE implementation details
  - Flow matching convention (t=0 data vs t=0 noise)

Also loads the config JSON to verify architectural parameters.

Usage:
    python scripts/data/inspect_trellis_dit.py \\
        --model_dir /workspace/models/trellis2-4b

    # Or specify checkpoint directly:
    python scripts/data/inspect_trellis_dit.py \\
        --checkpoint /path/to/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors \\
        --config_json /path/to/slat_flow_img2shape_dit_1_3B_512_bf16.json
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def inspect_config(config_path: str):
    """Inspect TRELLIS.2 DiT config JSON."""
    print(f"\n{'='*60}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")

    with open(config_path) as f:
        config = json.load(f)

    print(json.dumps(config, indent=2))

    # Extract key parameters
    print(f"\n--- Key Parameters ---")
    for key in ["d_model", "n_heads", "n_layers", "d_ff", "d_cond",
                 "d_in", "d_out", "mlp_ratio", "n_blocks",
                 "hidden_size", "num_heads", "num_layers"]:
        val = config.get(key)
        if val is not None:
            print(f"  {key}: {val}")

    # Search nested dicts
    def search_nested(d, prefix=""):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                search_nested(v, full_key)
            elif isinstance(v, (int, float, str, bool)):
                if any(term in k.lower() for term in
                       ["dim", "head", "layer", "block", "ratio", "cond",
                        "channel", "hidden", "embed", "input", "output"]):
                    print(f"  {full_key}: {v}")

    search_nested(config)
    return config


def inspect_checkpoint(ckpt_path: str):
    """Inspect TRELLIS.2 DiT checkpoint state dict."""
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    else:
        import torch
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

    # --- Print all keys with shapes ---
    print(f"\nTotal keys: {len(state_dict)}")
    print(f"\n--- Full State Dict ---")
    for key in sorted(state_dict.keys()):
        shape = list(state_dict[key].shape)
        dtype = state_dict[key].dtype
        print(f"  {key:60s} {str(shape):20s} {dtype}")

    # --- Analyze architecture ---
    print(f"\n{'='*60}")
    print("Architecture Analysis")
    print(f"{'='*60}")

    # Count blocks
    block_indices = set()
    for key in state_dict:
        m = re.match(r"blocks\.(\d+)\.", key)
        if m:
            block_indices.add(int(m.group(1)))

    if block_indices:
        n_blocks = max(block_indices) + 1
        print(f"\n  Number of blocks: {n_blocks} (indices 0-{max(block_indices)})")
    else:
        print("\n  No 'blocks.X.' pattern found in keys")
        n_blocks = 0

    # Hidden dimension (from self_attn.to_qkv weight)
    for key in state_dict:
        if "blocks.0.self_attn.to_qkv.weight" in key:
            shape = state_dict[key].shape
            hidden_dim = shape[1]  # input dim
            qkv_dim = shape[0]  # output dim = 3 * hidden
            n_heads_guess = hidden_dim // 128  # assuming head_dim=128
            print(f"\n  Hidden dim: {hidden_dim} (from to_qkv input)")
            print(f"  QKV dim: {qkv_dim} (= 3 × {qkv_dim // 3})")
            print(f"  Num heads: {n_heads_guess} (assuming head_dim=128)")
            print(f"  Head dim: {hidden_dim // n_heads_guess}")
            break

    # MLP dimension
    for key in state_dict:
        if "blocks.0.mlp.mlp.0.weight" in key:
            shape = state_dict[key].shape
            mlp_hidden = shape[0]
            mlp_input = shape[1]
            ratio = mlp_hidden / mlp_input
            print(f"\n  MLP hidden: {mlp_hidden}")
            print(f"  MLP input: {mlp_input}")
            print(f"  MLP ratio: {ratio:.4f}")
            break

    # Cross-attention conditioning dim
    for key in state_dict:
        if "blocks.0.cross_attn.to_kv.weight" in key:
            shape = state_dict[key].shape
            cond_dim = shape[1]
            kv_dim = shape[0]
            print(f"\n  Cross-attn cond_dim: {cond_dim} (KV input)")
            print(f"  Cross-attn KV out: {kv_dim} (= 2 × {kv_dim // 2})")
            break

    # Input projection
    for key in state_dict:
        if key == "input_layer.weight" or key == "input_proj.weight":
            shape = state_dict[key].shape
            print(f"\n  Input projection: {key}")
            print(f"    Shape: {list(shape)} ({shape[1]}→{shape[0]})")
            print(f"    Input dim (SLAT dim): {shape[1]}")
            break

    # Output projection
    for key in state_dict:
        if key.startswith("out_layer") or key.startswith("out_head") or key.startswith("output"):
            shape = state_dict[key].shape
            print(f"\n  Output projection: {key}")
            print(f"    Shape: {list(shape)}")
            break

    # Timestep embedder
    for key in state_dict:
        if "t_embedder" in key and "weight" in key:
            shape = state_dict[key].shape
            print(f"\n  Timestep embedder: {key}")
            print(f"    Shape: {list(shape)}")
            break

    # AdaLN modulation
    for key in state_dict:
        if "adaLN_modulation" in key and "weight" in key:
            shape = state_dict[key].shape
            print(f"\n  AdaLN modulation: {key}")
            print(f"    Shape: {list(shape)} ({shape[1]}→{shape[0]})")
            print(f"    Output = 6 × {shape[0] // 6} (6 modulation vectors)")
            break

    # Per-block modulation (offset)
    for key in state_dict:
        if "blocks.0.modulation" in key and "modulation" in key:
            shape = state_dict[key].shape
            print(f"\n  Per-block modulation: {key}")
            print(f"    Shape: {list(shape)}")
            break

    # RMSNorm gamma shapes
    for key in state_dict:
        if "blocks.0.self_attn.q_rms_norm.gamma" in key:
            shape = state_dict[key].shape
            print(f"\n  QK RMSNorm gamma: {list(shape)} (num_heads × head_dim)")
            break

    # Check for any unexpected keys (MoE, U-Net skips, etc.)
    print(f"\n--- Unique key prefixes (top-level) ---")
    prefixes = defaultdict(int)
    for key in state_dict:
        prefix = key.split(".")[0]
        prefixes[prefix] += 1
    for prefix, count in sorted(prefixes.items()):
        print(f"  {prefix}: {count} keys")

    # Check for MoE-related keys
    moe_keys = [k for k in state_dict if "expert" in k.lower() or "moe" in k.lower() or "router" in k.lower()]
    if moe_keys:
        print(f"\n  WARNING: MoE-related keys found: {moe_keys[:5]}...")
    else:
        print(f"\n  No MoE keys found (expected — TRELLIS.2 doesn't use MoE)")

    # Check for skip connection keys
    skip_keys = [k for k in state_dict if "skip" in k.lower()]
    if skip_keys:
        print(f"  Skip connection keys found: {skip_keys[:5]}...")
    else:
        print(f"  No skip connection keys (expected — TRELLIS.2 doesn't use U-Net skips)")

    # Total parameter count
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"\n  Total parameters: {total_params / 1e6:.1f}M ({total_params / 1e9:.2f}B)")

    return state_dict


def find_checkpoint_files(model_dir: str) -> tuple[str | None, str | None]:
    """Find shape DiT checkpoint and config in model directory."""
    model_path = Path(model_dir)
    ckpt_dir = model_path / "ckpts"

    ckpt_path = None
    config_path = None

    # Look for shape DiT checkpoint (not texture DiT)
    search_dirs = [ckpt_dir, model_path]
    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            name = f.name.lower()
            if "slat_flow" in name and "img2shape" in name:
                if f.suffix in (".safetensors", ".pt", ".bin"):
                    ckpt_path = str(f)
                elif f.suffix == ".json":
                    config_path = str(f)

    return ckpt_path, config_path


def main():
    parser = argparse.ArgumentParser(
        description="Gate 0C: Inspect TRELLIS.2 shape DiT architecture"
    )
    parser.add_argument("--model_dir", default="/workspace/models/trellis2-4b",
                        help="TRELLIS.2 model directory")
    parser.add_argument("--checkpoint", default=None,
                        help="Direct path to checkpoint (.safetensors or .pt)")
    parser.add_argument("--config_json", default=None,
                        help="Direct path to config JSON")
    args = parser.parse_args()

    # Find files
    ckpt_path = args.checkpoint
    config_path = args.config_json

    if ckpt_path is None or config_path is None:
        auto_ckpt, auto_config = find_checkpoint_files(args.model_dir)
        if ckpt_path is None:
            ckpt_path = auto_ckpt
        if config_path is None:
            config_path = auto_config

    print(f"\n{'='*60}")
    print(f"Gate 0C: TRELLIS.2 Shape DiT Architecture Inspection")
    print(f"{'='*60}")

    # List available files
    ckpt_dir = Path(args.model_dir) / "ckpts"
    if ckpt_dir.exists():
        print(f"\nAvailable checkpoints in {ckpt_dir}:")
        for f in sorted(ckpt_dir.iterdir()):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.0f} MB)")

    # Inspect config
    if config_path:
        inspect_config(config_path)
    else:
        print("\nNo config JSON found — will infer from checkpoint")

    # Inspect checkpoint
    if ckpt_path:
        inspect_checkpoint(ckpt_path)
    else:
        print("\nERROR: No checkpoint found")
        print(f"  Searched: {args.model_dir}")
        print(f"  Expected: slat_flow_img2shape_dit_*.safetensors")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("ACTION ITEMS after reviewing output:")
    print("  1. Confirm num_layers matches model_v2.py default (30)")
    print("  2. Confirm hidden_dim=1536, num_heads=12, head_dim=128")
    print("  3. Confirm MLP ratio (expected 5.3334)")
    print("  4. Confirm cond_dim=1024 (DINOv2 features)")
    print("  5. Check input_layer dim (expected 32 for SLAT)")
    print("  6. Note any unexpected keys (MoE, skip connections)")
    print("  7. Update configs/train_stage2_v2.yaml if any values differ")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
