#!/usr/bin/env python3
"""End-to-end image → mesh inference using TRELLIS.2 + ClearMesh Stage 2 residual.

Pipeline:
  1. Input image → TRELLIS.2 (512) → coarse SLAT (N, 32) + positions + DINOv2 cond
  2. Coarse SLAT → normalize → ClearMesh DiT (single forward pass) → refined SLAT
  3. Refined SLAT → denormalize → reconstruct SparseTensor → TRELLIS.2 decoder → mesh

The model predicts a residual (delta) in normalized SLAT space:
    refined_slat = coarse_slat + model(coarse_slat, cond_features)

Single forward pass. Deterministic. ~50× faster than diffusion DDIM sampling.

Environment requirements:
  - Python 3.10 (for o_voxel and flex_gemm cp310 wheels)
  - pip install flex_gemm and o_voxel from TRELLIS.2 Space_Wheels release
  - TRELLIS.2 repo stubs (flex_gemm/, o_voxel/) must be renamed/removed
    so the real pip-installed packages are found on sys.path
  - The decoder uses mixed precision: blocks in fp16, norms in fp32.
  - Decode resolution must be ≥256 (decoder native config) for mesh faces.

Usage:
    python -m clearmesh.stage2.infer_slat \\
        --config configs/train_stage2_residual.yaml \\
        --checkpoint checkpoints/clearmesh_stage2_residual/checkpoint_final.pt \\
        --image path/to/image.png \\
        --output_dir results/

    # With delta scaling (0.5 = conservative, 1.0 = full, 1.5 = aggressive)
    python -m clearmesh.stage2.infer_slat \\
        --config configs/train_stage2_residual.yaml \\
        --checkpoint checkpoints/clearmesh_stage2_residual/checkpoint_final.pt \\
        --image path/to/image.png \\
        --output_dir results/ \\
        --delta_scale 0.8

    # Compare with TRELLIS.2 baseline (512 and 1024)
    python -m clearmesh.stage2.infer_slat \\
        --config configs/train_stage2_residual.yaml \\
        --checkpoint checkpoints/clearmesh_stage2_residual/checkpoint_final.pt \\
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


# ---------------------------------------------------------------------------
# SLAT normalization constants (TRELLIS.2 4B shape_slat_normalization)
# ---------------------------------------------------------------------------

SLAT_MEAN = torch.tensor([
    0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252,
    1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904,
    0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463,
    1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
    -0.059621, 2.22078, 1.621212, 0.87723, 0.567247, -3.175944,
    -3.186688, 1.578665,
], dtype=torch.float32)

SLAT_STD = torch.tensor([
    5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237,
    5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469,
    5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013,
    5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194,
    5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207,
    5.347008, 5.405691,
], dtype=torch.float32)


def normalize_slat(slat: torch.Tensor) -> torch.Tensor:
    """Normalize raw SLAT to zero-mean, unit-std per channel."""
    mean = SLAT_MEAN.to(slat.device)
    std = SLAT_STD.to(slat.device)
    return (slat - mean) / std


def denormalize_slat(slat: torch.Tensor) -> torch.Tensor:
    """Denormalize from training space back to raw SLAT for decoder."""
    mean = SLAT_MEAN.to(slat.device)
    std = SLAT_STD.to(slat.device)
    return slat * std + mean


# ---------------------------------------------------------------------------
# TRELLIS.2 helpers
# ---------------------------------------------------------------------------

def setup_trellis2(trellis2_dir: str):
    """Add TRELLIS.2 to sys.path and configure sparse backend.

    Note: Do NOT set SPARSE_CONV_BACKEND=spconv — flex_gemm is the correct
    backend for TRELLIS.2 decode on modern GPUs (H100, A100). The spconv
    backend causes 'can't find suitable algorithm' errors.
    """
    if trellis2_dir not in sys.path:
        sys.path.insert(0, trellis2_dir)
    # flex_gemm is the default and correct backend; don't override
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")


def load_trellis2_pipeline(model_dir: str, device: str = "cuda"):
    """Load the TRELLIS.2 image-to-3D pipeline."""
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if os.path.exists(model_dir):
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_dir)
    else:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")

    # Disable low_vram mode for faster inference
    pipeline.low_vram = False
    for m in pipeline.models.values():
        if hasattr(m, "low_vram"):
            m.low_vram = False
    if hasattr(pipeline, "image_cond_model") and pipeline.image_cond_model is not None:
        if hasattr(pipeline.image_cond_model, "low_vram"):
            pipeline.image_cond_model.low_vram = False

    pipeline.to(device)
    return pipeline


@torch.no_grad()
def extract_coarse_slat(
    pipeline,
    image: Image.Image,
    seed: int = 42,
):
    """Run TRELLIS.2 (512 model) to get coarse SLAT + sparse structure.

    Returns dict with:
      - positions:      (N, 3) int32 — voxel coordinates
      - coarse_slat:    (N, 32) float32 — raw SLAT features (NOT normalized)
      - cond_features:  (M, 1024) float32 — DINOv2 conditioning
      - coords:         original spconv coords object (for decoder)
      - shape_slat_obj: original spconv SparseTensor (for decoder)
    """
    processed_image = pipeline.preprocess_image(image)
    torch.manual_seed(seed)

    cond_512 = pipeline.get_cond([processed_image], 512)

    coords = pipeline.sample_sparse_structure(
        cond_512, 32, 1, {"steps": 12, "guidance_strength": 9.0},
    )

    shape_slat = pipeline.sample_shape_slat(
        cond_512,
        pipeline.models["shape_slat_flow_model_512"],
        coords,
        {"steps": 12, "guidance_strength": 4.5},
    )

    # Extract positions as numpy
    if isinstance(coords, torch.Tensor):
        if coords.dim() == 2 and coords.shape[1] == 4:
            positions_np = coords[:, 1:].cpu().numpy().astype(np.int32)
        else:
            positions_np = coords.cpu().numpy().astype(np.int32)
    elif hasattr(coords, 'coords'):
        c = coords.coords
        positions_np = (c[:, 1:] if c.shape[1] == 4 else c).cpu().numpy().astype(np.int32)
    else:
        raise RuntimeError(f"Cannot extract coords from {type(coords)}")

    # Extract SLAT features as float32 tensor (NOT numpy, to keep on GPU)
    if hasattr(shape_slat, 'feats'):
        slat_tensor = shape_slat.feats.float()
    elif hasattr(shape_slat, 'F'):
        slat_tensor = shape_slat.F.float()
    elif isinstance(shape_slat, torch.Tensor):
        slat_tensor = shape_slat.squeeze(0).float()
    else:
        raise RuntimeError(f"Cannot extract SLAT from {type(shape_slat)}")

    # DINOv2 conditioning
    if isinstance(cond_512, dict):
        cond_tensor = cond_512.get('cond', cond_512.get('image_cond'))
        if cond_tensor is None:
            cond_tensor = next(iter(cond_512.values()))
    elif isinstance(cond_512, (list, tuple)):
        cond_tensor = cond_512[0]
    else:
        cond_tensor = cond_512

    cond_feats = None
    if isinstance(cond_tensor, torch.Tensor):
        ct = cond_tensor.squeeze(0) if cond_tensor.dim() == 3 else cond_tensor
        cond_feats = ct.float().cpu().numpy()

    return {
        "positions": positions_np,
        "coarse_slat": slat_tensor,  # (N, 32) float32 tensor, raw (NOT normalized)
        "cond_features": cond_feats,  # (M, 1024) numpy
        "coords": coords,             # keep for decoder
        "shape_slat_obj": shape_slat,  # keep for decoder / baseline
        "cond_512": cond_512,          # keep for 1024 baseline
    }


def reconstruct_sparse_tensor(shape_slat_obj, refined_slat_raw: torch.Tensor):
    """Replace the SLAT features in a SparseTensor with refined values.

    This modifies the existing SparseTensor in-place (or returns a new one)
    so it can be passed to pipeline.decode_shape_slat().
    """
    if hasattr(shape_slat_obj, 'feats'):
        # spconv SparseTensor — replace features
        shape_slat_obj.feats = refined_slat_raw.to(shape_slat_obj.feats.device)
        return shape_slat_obj
    elif hasattr(shape_slat_obj, 'F'):
        # Alternative spconv API
        shape_slat_obj.F = refined_slat_raw.to(shape_slat_obj.F.device)
        return shape_slat_obj
    elif isinstance(shape_slat_obj, torch.Tensor):
        # Plain tensor — just return refined values with matching shape
        if shape_slat_obj.dim() == 3:
            return refined_slat_raw.unsqueeze(0).to(shape_slat_obj.device)
        return refined_slat_raw.to(shape_slat_obj.device)
    else:
        raise RuntimeError(f"Cannot reconstruct SparseTensor from {type(shape_slat_obj)}")


# ---------------------------------------------------------------------------
# ClearMesh Stage 2
# ---------------------------------------------------------------------------

def load_stage2_model(config: dict, checkpoint_path: str, device: str = "cuda"):
    """Load ClearMesh Stage 2 RefinementDiT (residual prediction mode)."""
    from clearmesh.stage2.model import RefinementDiT

    model = RefinementDiT(
        voxel_dim=config.get("voxel_dim", 32),
        model_dim=config.get("model_dim", 1536),
        num_heads=config.get("num_heads", 12),
        num_layers=config.get("num_layers", 12),
        cond_dim=config.get("cond_dim", 1024),
        mlp_ratio=config.get("mlp_ratio", 5.3334),
        use_checkpoint=False,
    )

    print(f"Loading Stage 2 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=True)
    step = ckpt.get("global_step", "?")
    print(f"  Stage 2 loaded (step {step}, residual mode)")

    return model.to(device).eval(), step


@torch.no_grad()
def refine_slat(
    model,
    coarse_slat_raw: torch.Tensor,
    positions: np.ndarray,
    cond_features: np.ndarray | None = None,
    max_tokens: int = 8192,
    delta_scale: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Run Stage 2 direct residual refinement in SLAT space.

    Single forward pass: refined = coarse + model(coarse, cond)
    Deterministic output. No noise, no sampling, no strength tuning.

    Args:
        coarse_slat_raw: (N, 32) raw SLAT features (NOT normalized)
        positions: (N, 3) int32 voxel coordinates
        cond_features: (M, 1024) DINOv2 features or None
        delta_scale: Scale factor for predicted delta (1.0 = full refinement,
                     <1.0 = conservative, >1.0 = aggressive)

    Returns:
        (N, 32) refined SLAT in raw (denormalized) space, ready for decoder
    """
    N = coarse_slat_raw.shape[0]

    # Normalize to training space
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

    # Pad to max_tokens for batched inference
    if orig_n < max_tokens:
        pad = max_tokens - orig_n
        coarse_norm = torch.nn.functional.pad(coarse_norm, (0, 0, 0, pad))
        pos = torch.nn.functional.pad(pos, (0, 0, 0, pad))

    # Batch dim
    coarse_t = coarse_norm.unsqueeze(0).to(device)
    pos_t = pos.unsqueeze(0).to(device)

    # Conditioning
    cond_t = None
    cond_mask = None
    if cond_features is not None:
        cond_np = cond_features.astype(np.float32)
        cond_t = torch.from_numpy(cond_np).unsqueeze(0).to(device)
        cond_mask = torch.ones(1, cond_np.shape[0], dtype=torch.bool, device=device)

    # Single forward pass — direct residual prediction
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        refined_norm = model.refine_residual(
            coarse_t, pos_t,
            cond_features=cond_t,
            cond_mask=cond_mask,
            delta_scale=delta_scale,
        )
    elapsed = time.time() - t0

    # Remove batch dim and padding
    refined_norm = refined_norm.float().squeeze(0)[:orig_n]  # (orig_n, 32)

    # Denormalize back to raw SLAT space for decoder
    refined_raw = denormalize_slat(refined_norm)  # (orig_n, 32)

    if subsampled:
        # Reconstruct full-size SLAT: use coarse for non-refined positions
        full_refined = coarse_slat_raw.clone().to(device)
        full_refined[idx] = refined_raw
        refined_raw = full_refined

    print(f"    Refined {orig_n} points in {elapsed:.2f}s (single pass, scale={delta_scale})")
    return refined_raw


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_single_image(
    trellis_pipeline,
    stage2_model,
    image: Image.Image,
    name: str,
    output_dir: str,
    max_tokens: int = 8192,
    delta_scale: float = 1.0,
    seed: int = 42,
    save_baselines: bool = False,
    device: str = "cuda",
):
    """Full end-to-end: image → coarse SLAT → refined SLAT → decoded mesh."""
    out_path = Path(output_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)

    # Save input image
    image.save(out_path / "input.png")

    # --- Step 1: TRELLIS.2 → coarse SLAT ---
    print(f"\n  [1/3] TRELLIS.2 (512) → coarse SLAT...")
    t0 = time.time()
    intermediates = extract_coarse_slat(trellis_pipeline, image, seed=seed)
    trellis_time = time.time() - t0
    n_points = intermediates["positions"].shape[0]
    print(f"        {n_points} sparse points, {trellis_time:.1f}s")

    # Save coarse SLAT for debugging
    np.save(out_path / "positions.npy", intermediates["positions"])
    np.save(out_path / "coarse_slat.npy", intermediates["coarse_slat"].cpu().numpy())
    if intermediates["cond_features"] is not None:
        np.save(out_path / "cond_features.npy", intermediates["cond_features"])

    # --- Optional: save TRELLIS.2 512 baseline mesh ---
    if save_baselines:
        import trimesh as _trimesh

        def _export_mesh(mesh_obj, path):
            """Export TRELLIS.2 mesh (handles .detach() for grad tensors)."""
            if hasattr(mesh_obj, 'vertices') and isinstance(mesh_obj.vertices, torch.Tensor):
                v = mesh_obj.vertices.detach().cpu().float().numpy()
                f = mesh_obj.faces.detach().cpu().numpy()
                _trimesh.Trimesh(vertices=v, faces=f).export(str(path))
                return len(v), len(f)
            mesh_obj.export(str(path))
            return len(mesh_obj.vertices), len(mesh_obj.faces)

        print(f"  [baseline] Decoding TRELLIS.2 512 baseline mesh...")
        t_b = time.time()
        try:
            with torch.no_grad():
                meshes_512, _ = trellis_pipeline.decode_shape_slat(
                    intermediates["shape_slat_obj"], 512
                )
            nv, nf = _export_mesh(meshes_512[0], out_path / "baseline_512.glb")
            print(f"        Baseline 512: {nv} verts, {nf} faces, {time.time()-t_b:.1f}s")
        except Exception as e:
            print(f"        Baseline 512 failed: {e}")

        # Also generate 1024 baseline for comparison
        print(f"  [baseline] Generating TRELLIS.2 1024 baseline...")
        t_b = time.time()
        try:
            cond_1024 = trellis_pipeline.get_cond(
                [trellis_pipeline.preprocess_image(image)], 1024
            )
            shape_slat_1024 = trellis_pipeline.sample_shape_slat(
                cond_1024,
                trellis_pipeline.models["shape_slat_flow_model_1024"],
                intermediates["coords"],
                {"steps": 12, "guidance_strength": 4.5},
            )
            with torch.no_grad():
                meshes_1024, _ = trellis_pipeline.decode_shape_slat(shape_slat_1024, 512)
            nv, nf = _export_mesh(meshes_1024[0], out_path / "baseline_1024.glb")
            print(f"        Baseline 1024: {nv} verts, {nf} faces, {time.time()-t_b:.1f}s")
            del shape_slat_1024, meshes_1024, cond_1024
        except Exception as e:
            print(f"        Baseline 1024 failed: {e}")

    # --- Step 2: ClearMesh DiT residual refinement (single forward pass) ---
    print(f"  [2/3] ClearMesh Stage 2 residual refinement (scale={delta_scale:.2f})...")
    refined_slat_raw = refine_slat(
        stage2_model,
        intermediates["coarse_slat"],
        intermediates["positions"],
        cond_features=intermediates["cond_features"],
        max_tokens=max_tokens,
        delta_scale=delta_scale,
        device=device,
    )

    # Save refined SLAT
    np.save(out_path / "refined_slat.npy", refined_slat_raw.cpu().numpy())

    # Compute refinement delta stats
    coarse_raw = intermediates["coarse_slat"].to(device)
    delta = (refined_slat_raw - coarse_raw).abs()
    print(f"        SLAT delta: mean={delta.mean():.3f}, max={delta.max():.3f}")

    # --- Step 3: Decode refined SLAT through TRELLIS.2's frozen decoder ---
    print(f"  [3/3] Decoding through TRELLIS.2 FlexiDualGridVaeDecoder...")
    t0 = time.time()

    # Replace features in the original SparseTensor
    modified_slat_obj = reconstruct_sparse_tensor(
        intermediates["shape_slat_obj"], refined_slat_raw
    )

    try:
        # Resolution MUST be 512 — the decoder's sparse UNet upsamples through 4
        # levels (32→64→128→256→512), so output coords go up to 511. Passing a
        # smaller grid_size (e.g., 256) corrupts the hashmap in mesh extraction,
        # causing stretched/degenerate geometry.
        with torch.no_grad():
            meshes, _ = trellis_pipeline.decode_shape_slat(modified_slat_obj, 512)
        decode_time = time.time() - t0
        mesh = meshes[0]

        # Post-process: fill holes to fix wireframe appearance, then simplify
        try:
            mesh.fill_holes(max_hole_perimeter=3e-2)
            print(f"        Holes filled")
        except Exception as e_fill:
            print(f"        fill_holes skipped: {e_fill}")

        try:
            mesh.simplify(target=500_000)
            print(f"        Simplified to ~500k faces")
        except Exception as e_simp:
            print(f"        simplify skipped: {e_simp}")

        mesh_path = out_path / "refined_mesh.glb"
        import trimesh
        verts = mesh.vertices.detach().cpu().float().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        tm = trimesh.Trimesh(vertices=verts, faces=faces)
        tm.export(str(mesh_path))
        n_verts = len(verts)
        n_faces = len(faces)
        print(f"        Mesh: {n_verts} verts, {n_faces} faces, {decode_time:.1f}s")
        print(f"        Saved: {mesh_path}")

        mesh_info = f"{n_verts} verts, {n_faces} faces"
    except Exception as e:
        decode_time = time.time() - t0
        print(f"        Decode failed: {e}")
        mesh_info = f"FAILED: {e}"

    # Summary
    total_time = trellis_time + decode_time
    summary = {
        "name": name,
        "n_points": int(n_points),
        "trellis_time": round(trellis_time, 2),
        "delta_scale": delta_scale,
        "decode_time": round(decode_time, 2),
        "total_time": round(total_time, 2),
        "slat_delta_mean": float(delta.mean()),
        "slat_delta_max": float(delta.max()),
        "mesh": mesh_info,
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Cleanup
    del intermediates, refined_slat_raw, coarse_raw, delta
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path_or_url: str) -> Image.Image:
    """Load image from local path or URL."""
    if path_or_url.startswith(("http://", "https://")):
        import io
        import urllib.request
        req = urllib.request.Request(path_or_url, headers={"User-Agent": "ClearMesh/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGBA")
    return Image.open(path_or_url).convert("RGBA")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ClearMesh SLAT-space inference: image → refined mesh"
    )
    parser.add_argument("--config", required=True, help="Stage 2 training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Stage 2 checkpoint .pt")

    # Input
    parser.add_argument("--image", default=None, help="Single image path or URL")
    parser.add_argument("--image_dir", default=None, help="Directory of images")

    # TRELLIS.2
    parser.add_argument("--trellis2_dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--model_dir", default="/workspace/models/trellis2-4b")

    # Stage 2
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--delta_scale", type=float, default=1.0,
                        help="Scale factor for predicted delta. 1.0 = full refinement, "
                             "<1.0 = conservative, >1.0 = aggressive")

    # Output
    parser.add_argument("--output_dir", default="slat_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_baselines", action="store_true",
                        help="Also save TRELLIS.2 512/1024 baseline meshes for A/B comparison")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"ClearMesh SLAT-Space Inference")
    print(f"{'='*60}")

    # Load TRELLIS.2
    print(f"\nLoading TRELLIS.2 pipeline...")
    setup_trellis2(args.trellis2_dir)
    trellis_pipeline = load_trellis2_pipeline(args.model_dir, device=device)
    print(f"  TRELLIS.2 loaded")

    # Load Stage 2
    print(f"\nLoading ClearMesh Stage 2...")
    stage2_model, step = load_stage2_model(config, args.checkpoint, device=device)

    # Gather images
    images = []
    if args.image:
        name = Path(args.image).stem if not args.image.startswith("http") else "url_image"
        images.append((name, args.image))
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            for p in sorted(img_dir.glob(ext)):
                images.append((p.stem, str(p)))
    else:
        print("Error: specify --image or --image_dir")
        sys.exit(1)

    print(f"\nProcessing {len(images)} image(s)...")
    print(f"  Mode:          Direct residual prediction (single pass)")
    print(f"  Delta scale:   {args.delta_scale}")
    print(f"  Max tokens:    {args.max_tokens}")
    print(f"  Save baselines: {args.save_baselines}")
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
                max_tokens=args.max_tokens,
                delta_scale=args.delta_scale,
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

    # Save all results
    results_path = Path(args.output_dir) / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
