"""End-to-end inference for Stage 2 v2 FlowMatchingDiT.

Pipeline:
  1. Input image → TRELLIS.2 (512) → coarse SLAT (N, 32) + positions + DINOv2 cond
  2. Initialize x_T ~ N(0, I) with same shape as coarse SLAT
  3. 50 Euler steps from t=1 (noise) → t=0 (data) with CFG
  4. Refined SLAT → denormalize → reconstruct SparseTensor → TRELLIS.2 decoder → mesh

Also supports SDEdit-style inference: start from partially noised coarse SLAT
instead of pure noise, then denoise from t=t_start → t=0.

Usage:
    python -m clearmesh.stage2.infer_v2 \\
        --config configs/train_stage2_v2.yaml \\
        --checkpoint checkpoints/stage2_v2/checkpoint_final.pt \\
        --image path/to/image.png \\
        --output_dir results/

    # SDEdit mode (start from coarse with partial noise)
    python -m clearmesh.stage2.infer_v2 \\
        --config configs/train_stage2_v2.yaml \\
        --checkpoint checkpoints/stage2_v2/checkpoint_final.pt \\
        --image path/to/image.png \\
        --output_dir results/ \\
        --sdedit --t_start 0.3

    # Compare with baselines
    python -m clearmesh.stage2.infer_v2 \\
        --config configs/train_stage2_v2.yaml \\
        --checkpoint checkpoints/stage2_v2/checkpoint_final.pt \\
        --image path/to/image.png \\
        --output_dir results/ \\
        --save_baselines
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from clearmesh.stage2.infer_slat import (
    SLAT_MEAN,
    SLAT_STD,
    denormalize_slat,
    extract_coarse_slat,
    load_image,
    load_trellis2_pipeline,
    normalize_slat,
    reconstruct_sparse_tensor,
    setup_trellis2,
)
from clearmesh.stage2.model_v2 import EMA, FlowMatchingDiT


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_stage2_v2_model(
    config: dict,
    checkpoint_path: str,
    device: str = "cuda",
    use_ema: bool = True,
) -> tuple[FlowMatchingDiT, int]:
    """Load Stage 2 v2 FlowMatchingDiT from training checkpoint."""
    model = FlowMatchingDiT(
        voxel_dim=config.get("voxel_dim", 32),
        model_dim=config.get("model_dim", 1536),
        num_heads=config.get("num_heads", 12),
        num_layers=config.get("num_layers", 30),
        cond_dim=config.get("cond_dim", 1024),
        mlp_ratio=config.get("mlp_ratio", 5.3334),
        use_checkpoint=False,  # no checkpointing at inference
    )

    print(f"Loading Stage 2 v2 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=True)
    step = ckpt.get("global_step", "?")

    # Apply EMA weights if available
    if use_ema and "ema" in ckpt:
        ema = EMA(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply()
        print(f"  Loaded with EMA weights (step {step})")
    else:
        print(f"  Loaded (step {step}, no EMA)")

    return model.to(device).eval(), step


# ---------------------------------------------------------------------------
# Flow matching inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def refine_flow_matching(
    model: FlowMatchingDiT,
    coarse_slat_raw: torch.Tensor,
    positions: np.ndarray,
    cond_features: np.ndarray | None = None,
    steps: int = 50,
    guidance_scale: float = 5.0,
    max_tokens: int = 32768,
    sdedit: bool = False,
    t_start: float = 1.0,
    seed: int = 42,
    device: str = "cuda",
) -> torch.Tensor:
    """Run flow matching refinement: 50 Euler steps from noise → refined SLAT.

    Args:
        coarse_slat_raw: (N, 32) raw SLAT features (NOT normalized)
        positions: (N, 3) int32 voxel coordinates
        cond_features: (M, 1024) DINOv2 features or None
        steps: Number of Euler ODE steps
        guidance_scale: CFG scale (5.0 default, matching UltraShape)
        max_tokens: Maximum token budget for inference
        sdedit: If True, start from partially noised coarse SLAT
        t_start: Starting timestep for SDEdit (0.3 = light noise, 1.0 = pure noise)
        seed: Random seed for noise

    Returns:
        (N, 32) refined SLAT in raw (denormalized) space, ready for decoder
    """
    N = coarse_slat_raw.shape[0]
    torch.manual_seed(seed)

    # Normalize coarse SLAT to training space
    coarse_norm = normalize_slat(coarse_slat_raw)  # (N, 32)
    pos = torch.from_numpy(positions.astype(np.float32))

    # Subsample if too many tokens
    if N > max_tokens:
        idx = torch.randperm(N)[:max_tokens].sort().values
        coarse_norm = coarse_norm[idx]
        pos = pos[idx]
        subsampled = True
    else:
        idx = None
        subsampled = False

    orig_n = coarse_norm.shape[0]

    # Add batch dim
    coarse_t = coarse_norm.unsqueeze(0).to(device)  # (1, N, 32)
    pos_t = pos.unsqueeze(0).to(device)  # (1, N, 3)

    # Conditioning
    cond_t = None
    cond_zero = None
    cond_mask = None
    cond_mask_zero = None
    if cond_features is not None:
        cond_np = cond_features.astype(np.float32)
        cond_t = torch.from_numpy(cond_np).unsqueeze(0).to(device)
        cond_zero = torch.zeros_like(cond_t)
        cond_mask = torch.ones(1, cond_np.shape[0], dtype=torch.bool, device=device)
        cond_mask_zero = torch.zeros_like(cond_mask)

    # Initialize: pure noise or SDEdit
    if sdedit and t_start < 1.0:
        # SDEdit: start from partially noised coarse SLAT
        noise = torch.randn_like(coarse_t)
        x_t = (1 - t_start) * coarse_t + t_start * noise
        print(f"    SDEdit mode: starting from t={t_start}")
    else:
        # Standard: start from pure noise at t=1
        x_t = torch.randn_like(coarse_t)
        t_start = 1.0

    # Euler ODE solve from t=t_start → t=0
    actual_steps = int(steps * t_start)  # fewer steps if SDEdit
    sigmas = torch.linspace(t_start, 0, actual_steps + 1, device=device)

    t0 = time.time()
    for i in range(actual_steps):
        t = sigmas[i].unsqueeze(0)  # (1,)
        dt = sigmas[i + 1] - sigmas[i]  # negative

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Conditional prediction
            v_cond = model(x_t, coarse_t, pos_t, t, cond_features=cond_t, cond_mask=cond_mask)

            if guidance_scale != 1.0 and cond_t is not None:
                # Unconditional prediction
                v_uncond = model(
                    x_t,
                    coarse_t,
                    pos_t,
                    t,
                    cond_features=cond_zero,
                    cond_mask=cond_mask_zero,
                )
                # CFG
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond

        # Euler step
        x_t = x_t + dt * v.float()

    elapsed = time.time() - t0

    # Extract refined SLAT
    refined_norm = x_t.float().squeeze(0)  # (N, 32) normalized

    # Denormalize back to raw SLAT space
    refined_raw = denormalize_slat(refined_norm)

    if subsampled:
        # Reconstruct full-size SLAT: use coarse for non-refined positions
        full_refined = coarse_slat_raw.clone().to(device)
        full_refined[idx] = refined_raw
        refined_raw = full_refined
        print(f"    Refined {orig_n}/{N} tokens in {elapsed:.1f}s "
              f"({actual_steps} steps, {elapsed/actual_steps:.2f}s/step)")
    else:
        print(f"    Refined {orig_n} tokens in {elapsed:.1f}s "
              f"({actual_steps} steps, {elapsed/actual_steps:.2f}s/step)")

    return refined_raw


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_single_image(
    trellis_pipeline,
    stage2_model: FlowMatchingDiT,
    image: Image.Image,
    name: str,
    output_dir: str,
    steps: int = 50,
    guidance_scale: float = 5.0,
    max_tokens: int = 32768,
    sdedit: bool = False,
    t_start: float = 1.0,
    seed: int = 42,
    save_baselines: bool = False,
    device: str = "cuda",
):
    """Full end-to-end: image → coarse SLAT → flow matching → decoded mesh."""
    out_path = Path(output_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)

    image.save(out_path / "input.png")

    # --- Step 1: TRELLIS.2 → coarse SLAT ---
    print(f"\n  [1/3] TRELLIS.2 (512) → coarse SLAT...")
    t0 = time.time()
    intermediates = extract_coarse_slat(trellis_pipeline, image, seed=seed)
    trellis_time = time.time() - t0
    n_points = intermediates["positions"].shape[0]
    print(f"        {n_points} sparse points, {trellis_time:.1f}s")

    # Save intermediates
    np.save(out_path / "positions.npy", intermediates["positions"])
    np.save(out_path / "coarse_slat.npy", intermediates["coarse_slat"].cpu().numpy())
    if intermediates["cond_features"] is not None:
        np.save(out_path / "cond_features.npy", intermediates["cond_features"])

    # --- Optional baselines ---
    if save_baselines:
        import trimesh as _trimesh

        def _export_mesh(mesh_obj, path):
            if hasattr(mesh_obj, "vertices") and isinstance(mesh_obj.vertices, torch.Tensor):
                v = mesh_obj.vertices.detach().cpu().float().numpy()
                f = mesh_obj.faces.detach().cpu().numpy()
                _trimesh.Trimesh(vertices=v, faces=f).export(str(path))
                return len(v), len(f)
            mesh_obj.export(str(path))
            return len(mesh_obj.vertices), len(mesh_obj.faces)

        print(f"  [baseline] Decoding TRELLIS.2 512 baseline...")
        try:
            with torch.no_grad():
                meshes_512, _ = trellis_pipeline.decode_shape_slat(
                    intermediates["shape_slat_obj"], 512
                )
            nv, nf = _export_mesh(meshes_512[0], out_path / "baseline_512.glb")
            print(f"        Baseline 512: {nv} verts, {nf} faces")
        except Exception as e:
            print(f"        Baseline 512 failed: {e}")

    # --- Step 2: Flow matching refinement (50 Euler steps) ---
    mode = f"SDEdit t={t_start}" if sdedit else f"full denoise"
    print(f"  [2/3] Flow matching refinement ({mode}, {steps} steps, cfg={guidance_scale})...")
    refined_slat_raw = refine_flow_matching(
        stage2_model,
        intermediates["coarse_slat"],
        intermediates["positions"],
        cond_features=intermediates["cond_features"],
        steps=steps,
        guidance_scale=guidance_scale,
        max_tokens=max_tokens,
        sdedit=sdedit,
        t_start=t_start,
        seed=seed,
        device=device,
    )

    np.save(out_path / "refined_slat.npy", refined_slat_raw.cpu().numpy())

    # Delta stats
    coarse_raw = intermediates["coarse_slat"].to(device)
    delta = (refined_slat_raw - coarse_raw).abs()
    print(f"        SLAT delta: mean={delta.mean():.3f}, max={delta.max():.3f}")

    # --- Step 3: Decode through TRELLIS.2 ---
    print(f"  [3/3] Decoding through TRELLIS.2 decoder...")
    t0 = time.time()

    modified_slat_obj = reconstruct_sparse_tensor(
        intermediates["shape_slat_obj"], refined_slat_raw
    )

    try:
        with torch.no_grad():
            meshes, _ = trellis_pipeline.decode_shape_slat(modified_slat_obj, 512)
        decode_time = time.time() - t0
        mesh = meshes[0]

        try:
            mesh.fill_holes(max_hole_perimeter=3e-2)
        except Exception:
            pass
        try:
            mesh.simplify(target=500_000)
        except Exception:
            pass

        mesh_path = out_path / "refined_mesh.glb"
        import trimesh

        verts = mesh.vertices.detach().cpu().float().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        trimesh.Trimesh(vertices=verts, faces=faces).export(str(mesh_path))
        n_verts, n_faces = len(verts), len(faces)
        print(f"        Mesh: {n_verts} verts, {n_faces} faces, {decode_time:.1f}s")
        print(f"        Saved: {mesh_path}")
        mesh_info = f"{n_verts} verts, {n_faces} faces"
    except Exception as e:
        decode_time = time.time() - t0
        print(f"        Decode failed: {e}")
        mesh_info = f"FAILED: {e}"

    # Summary
    summary = {
        "name": name,
        "n_points": int(n_points),
        "trellis_time": round(trellis_time, 2),
        "steps": steps,
        "guidance_scale": guidance_scale,
        "sdedit": sdedit,
        "t_start": t_start,
        "decode_time": round(decode_time, 2),
        "slat_delta_mean": float(delta.mean()),
        "slat_delta_max": float(delta.max()),
        "mesh": mesh_info,
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    del intermediates, refined_slat_raw, coarse_raw, delta
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ClearMesh Stage 2 v2 flow matching inference"
    )
    parser.add_argument("--config", required=True, help="Stage 2 v2 config YAML")
    parser.add_argument("--checkpoint", required=True, help="Stage 2 v2 checkpoint .pt")

    # Input
    parser.add_argument("--image", default=None, help="Single image path or URL")
    parser.add_argument("--image_dir", default=None, help="Directory of images")

    # TRELLIS.2
    parser.add_argument("--trellis2_dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--model_dir", default="/workspace/models/trellis2-4b")

    # Inference params
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--sdedit", action="store_true",
                        help="SDEdit mode: start from partially noised coarse SLAT")
    parser.add_argument("--t_start", type=float, default=0.3,
                        help="Starting timestep for SDEdit (0.3 = light noise)")
    parser.add_argument("--no_ema", action="store_true",
                        help="Don't use EMA weights even if available")

    # Output
    parser.add_argument("--output_dir", default="results_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_baselines", action="store_true")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"ClearMesh Stage 2 v2 — Flow Matching Inference")
    print(f"{'='*60}")

    # Load TRELLIS.2
    print(f"\nLoading TRELLIS.2 pipeline...")
    setup_trellis2(args.trellis2_dir)
    trellis_pipeline = load_trellis2_pipeline(args.model_dir, device=device)

    # Load Stage 2 v2
    print(f"\nLoading Stage 2 v2 model...")
    stage2_model, step = load_stage2_v2_model(
        config, args.checkpoint, device=device, use_ema=not args.no_ema,
    )

    # Gather images
    images = []
    if args.image:
        name = Path(args.image).stem if not args.image.startswith("http") else "url_image"
        images.append((name, args.image))
    elif args.image_dir:
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            for p in sorted(Path(args.image_dir).glob(ext)):
                images.append((p.stem, str(p)))
    else:
        print("Error: specify --image or --image_dir")
        sys.exit(1)

    print(f"\nProcessing {len(images)} image(s)...")
    print(f"  Mode:          Flow matching ({args.steps} Euler steps)")
    print(f"  CFG scale:     {args.guidance_scale}")
    print(f"  Max tokens:    {args.max_tokens}")
    if args.sdedit:
        print(f"  SDEdit:        t_start={args.t_start}")
    print(f"  Output:        {args.output_dir}")

    results = []
    for i, (name, path_or_url) in enumerate(images):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(images)}] {name}")
        print(f"{'='*60}")

        try:
            img = load_image(path_or_url)
            summary = process_single_image(
                trellis_pipeline, stage2_model, img, name,
                output_dir=args.output_dir,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                max_tokens=args.max_tokens,
                sdedit=args.sdedit,
                t_start=args.t_start if args.sdedit else 1.0,
                seed=args.seed,
                save_baselines=args.save_baselines,
                device=device,
            )
            results.append(summary)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": name, "error": str(e)})

    results_path = Path(args.output_dir) / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
