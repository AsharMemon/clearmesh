#!/usr/bin/env python3
"""Generate coarse/fine SLAT training pairs for Stage 2 v2 refinement DiT.

Approach: conditional generation (UltraShape-like).
  - Coarse SLAT from 512 model = conditioning signal
  - Fine SLAT from 1024 model = training target
  - Both share the same sparse structure positions (from same coords)
  - Model learns to generate refined SLAT from noise, conditioned on coarse + image

For each input image:
  1. Sparse structure generation → voxel positions (N, 3)
  2. Shape SLAT flow model at 512 → coarse_slat (N, 32) — conditioning
  3. Shape SLAT flow model at 1024 → fine_slat (N, 32) — target
  4. Extract image conditioning features (DINOv2 from TRELLIS.2 pipeline)

Output per model:
  <uid>/
    coarse_slat.npy        (N, 32)   float16 — 512-model SLAT features (conditioning)
    fine_slat.npy          (N, 32)   float16 — 1024-model SLAT features (target)
    positions.npy          (N, 3)    float32 — sparse voxel coordinates (shared)
    cond_features.npy      (M, 1024) float16 — DINOv2 image features
    image.png              input image (for feature re-extraction if needed)
    metadata.json          generation params, timing, shapes

Usage:
    # Single GPU
    python generate_slat_pairs.py \\
        --image_dir /workspace/data/rendered_views \\
        --output_dir /workspace/data/slat_pairs_v2 \\
        --gpu 0

    # Sharded multi-GPU run (8 pods)
    python generate_slat_pairs.py \\
        --image_dir /workspace/data/rendered_views \\
        --output_dir /workspace/data/slat_pairs_v2 \\
        --shard_id 0 --num_shards 8 --gpu 0
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────

SPARSE_SAMPLER_PARAMS = {
    "steps": 12,
    "guidance_strength": 9.0,
}
SHAPE_SAMPLER_PARAMS = {
    "steps": 12,
    "guidance_strength": 4.5,
}
SS_RESOLUTION = 32  # Sparse structure resolution for both 512 and 1024


# ── Memory helpers ────────────────────────────────────────────────────────

def cleanup_memory():
    """Aggressive memory cleanup between models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Try malloc_trim on Linux
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def get_rss_gb() -> float:
    """Get current RSS in GiB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# ── SLAT extraction helpers ──────────────────────────────────────────────

def extract_coords(coords) -> np.ndarray | None:
    """Extract (N, 3) integer voxel coordinates from TRELLIS.2 coords object."""
    if isinstance(coords, torch.Tensor):
        if coords.dim() == 2 and coords.shape[1] == 4:
            return coords[:, 1:].cpu().numpy().astype(np.int32)
        elif coords.dim() == 2 and coords.shape[1] == 3:
            return coords.cpu().numpy().astype(np.int32)
        else:
            return coords.cpu().numpy().astype(np.int32)
    elif hasattr(coords, 'coords'):
        # SparseTensor from spconv
        c = coords.coords
        return (c[:, 1:] if c.shape[1] == 4 else c).cpu().numpy().astype(np.int32)
    return None


def extract_slat_feats(shape_slat) -> np.ndarray | None:
    """Extract (N, 32) SLAT features from TRELLIS.2 shape_slat object."""
    if hasattr(shape_slat, 'feats'):
        return shape_slat.feats.cpu().numpy().astype(np.float16)
    elif hasattr(shape_slat, 'F'):
        return shape_slat.F.cpu().numpy().astype(np.float16)
    elif isinstance(shape_slat, torch.Tensor):
        t = shape_slat.squeeze(0) if shape_slat.dim() == 3 else shape_slat
        return t.cpu().numpy().astype(np.float16)
    return None


def extract_cond_feats(cond) -> np.ndarray | None:
    """Extract (M, 1024) DINOv2 conditioning features (legacy)."""
    if isinstance(cond, dict):
        cond_tensor = cond.get('cond', cond.get('image_cond'))
        if cond_tensor is None:
            cond_tensor = next(iter(cond.values()))
    elif isinstance(cond, (list, tuple)):
        cond_tensor = cond[0]
    else:
        cond_tensor = cond

    if isinstance(cond_tensor, torch.Tensor):
        ct = cond_tensor.squeeze(0) if cond_tensor.dim() == 3 else cond_tensor
        feats = ct.cpu().numpy().astype(np.float16)
        if feats.ndim == 2 and feats.shape[1] == 1024:
            return feats
    return None


# (DINOv3 feature extraction and encoder-aligned targets removed in v2.1.
#  DINOv3 was replaced by TRELLIS.2's built-in DINOv2 pipeline conditioning.
#  Encoder-aligned targets replaced by 1024 diffusion targets — same positions,
#  no alignment needed. See Gate 0D-1/0D-2 for alignment viability analysis.)


# ── Core SLAT pair generation ────────────────────────────────────────────

def generate_slat_pair(
    pipeline,
    image: Image.Image,
    seed: int = 42,
    sparse_sampler_params: dict | None = None,
    shape_sampler_params: dict | None = None,
) -> dict | None:
    """Generate a coarse/fine SLAT pair from a single image.

    Conditional generation approach (UltraShape-like):
      - Coarse SLAT from 512 diffusion model → conditioning signal
      - Fine SLAT from 1024 diffusion model → training target
      - Both share the SAME positions (from the same sparse structure)
      - DINOv2 features from TRELLIS.2's built-in conditioning pipeline

    Returns dict with:
        positions:       (N, 3) float32 — shared voxel coordinates
        coarse_slat:     (N, 32) float16 — 512 model (conditioning)
        fine_slat:       (N, 32) float16 — 1024 model (target)
        cond_features:   (M, 1024) float16 — DINOv2 image features
        n_points:        int
    or None on failure.
    """
    ss_params = sparse_sampler_params or SPARSE_SAMPLER_PARAMS
    sh_params = shape_sampler_params or SHAPE_SAMPLER_PARAMS

    with torch.no_grad():
        processed_image = pipeline.preprocess_image(image)
        torch.manual_seed(seed)

        # Get conditioning at 512 resolution
        cond_512 = pipeline.get_cond([processed_image], 512)

        # Sample sparse structure — defines positions shared by both 512 and 1024
        coords = pipeline.sample_sparse_structure(
            cond_512, SS_RESOLUTION, 1, ss_params,
        )

        # Extract positions
        positions = extract_coords(coords)
        if positions is None or positions.shape[0] == 0:
            return None

        # Coarse SLAT: 512 flow model (conditioning signal)
        coarse_shape_slat = pipeline.sample_shape_slat(
            cond_512,
            pipeline.models["shape_slat_flow_model_512"],
            coords,
            sh_params,
        )
        coarse_slat = extract_slat_feats(coarse_shape_slat)
        if coarse_slat is None:
            return None

        # Fine SLAT: 1024 flow model (training target, same positions)
        cond_1024 = pipeline.get_cond([processed_image], 1024)
        fine_shape_slat = pipeline.sample_shape_slat(
            cond_1024,
            pipeline.models["shape_slat_flow_model_1024"],
            coords,  # same sparse structure → same positions
            sh_params,
        )
        fine_slat = extract_slat_feats(fine_shape_slat)
        del cond_1024, fine_shape_slat

        if fine_slat is None:
            return None

        # DINOv2 conditioning features from TRELLIS.2 pipeline
        cond_features = extract_cond_feats(cond_512)

        # Validate shapes — coarse and fine must match exactly (same positions)
        if coarse_slat.shape != fine_slat.shape:
            print(f"    [slat] shape mismatch: coarse {coarse_slat.shape} vs fine {fine_slat.shape}")
            return None
        if positions.shape[0] != coarse_slat.shape[0]:
            print(f"    [slat] count mismatch: positions {positions.shape[0]} vs slat {coarse_slat.shape[0]}")
            return None
        if coarse_slat.shape[1] != 32:
            print(f"    [slat] unexpected SLAT dim: {coarse_slat.shape[1]} (expected 32)")
            return None

        # Convert positions to float32
        positions = positions.astype(np.float32)

        # Cleanup GPU
        del processed_image, cond_512, coords, coarse_shape_slat

        return {
            "positions": positions,
            "coarse_slat": coarse_slat,
            "fine_slat": fine_slat,
            "cond_features": cond_features,
            "n_points": positions.shape[0],
        }


def save_slat_pair(
    output_dir: Path,
    uid: str,
    pair: dict,
    metadata: dict,
    image: Image.Image | None = None,
) -> bool:
    """Save SLAT pair to disk. Returns True on success."""
    pair_dir = output_dir / uid
    pair_dir.mkdir(parents=True, exist_ok=True)

    try:
        np.save(pair_dir / "positions.npy", pair["positions"])
        np.save(pair_dir / "coarse_slat.npy", pair["coarse_slat"])
        np.save(pair_dir / "fine_slat.npy", pair["fine_slat"])
        if pair["cond_features"] is not None:
            np.save(pair_dir / "cond_features.npy", pair["cond_features"])

        # Save input image for potential re-extraction
        if image is not None:
            image.save(str(pair_dir / "image.png"))

        with open(pair_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return True
    except Exception as e:
        print(f"    [save] failed for {uid}: {e}")
        return False


# ── Image loading ────────────────────────────────────────────────────────

def load_image_for_uid(uid: str, image_dir: Path | None, models_json: list | None) -> Image.Image | None:
    """Load the conditioning image for a given UID.

    Strategy:
      1. If image_dir is set, look for <image_dir>/<uid>/*.png (pre-rendered views)
      2. If models_json is set, load & render the 3D model (requires trimesh + blender)
    """
    if image_dir is not None:
        uid_dir = image_dir / uid
        if uid_dir.is_dir():
            # Use first PNG/JPG found
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                images = sorted(uid_dir.glob(ext))
                if images:
                    return Image.open(images[0]).convert("RGBA")
        # Try flat layout: <image_dir>/<uid>.png
        for ext in (".png", ".jpg", ".jpeg"):
            flat_path = image_dir / f"{uid}{ext}"
            if flat_path.exists():
                return Image.open(flat_path).convert("RGBA")

    return None


# ── Progress tracking ────────────────────────────────────────────────────

class ProgressTracker:
    """Track completed UIDs to support resume."""

    def __init__(self, output_dir: Path):
        self.progress_file = output_dir / "slat_progress.json"
        self.completed: set[str] = set()
        self.failed: set[str] = set()
        self.stats = {
            "total_attempted": 0,
            "total_success": 0,
            "total_failed": 0,
            "total_skipped": 0,
        }
        self._load()

    def _load(self):
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                data = json.load(f)
            self.completed = set(data.get("completed", []))
            self.failed = set(data.get("failed", []))
            self.stats = data.get("stats", self.stats)

    def save(self):
        data = {
            "completed": sorted(self.completed),
            "failed": sorted(self.failed),
            "stats": self.stats,
        }
        with open(self.progress_file, "w") as f:
            json.dump(data, f, indent=2)

    def mark_complete(self, uid: str):
        self.completed.add(uid)
        self.stats["total_success"] += 1
        self.stats["total_attempted"] += 1

    def mark_failed(self, uid: str):
        self.failed.add(uid)
        self.stats["total_failed"] += 1
        self.stats["total_attempted"] += 1

    def is_done(self, uid: str) -> bool:
        return uid in self.completed


# ── Main generation loop ─────────────────────────────────────────────────

def run_generation(
    input_json: str | None,
    image_dir: str | None,
    output_dir: str,
    shard_id: int = 0,
    num_shards: int = 1,
    gpu: int = 0,
    seed: int = 42,
    trellis2_dir: str = "/workspace/TRELLIS.2",
    model_dir: str = "/workspace/models/trellis2-4b",
    save_every: int = 25,
    max_models: int = 0,
):
    """Main generation loop — 512 coarse + 1024 fine SLAT pairs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model list
    if input_json:
        with open(input_json) as f:
            models = json.load(f)
        if isinstance(models, dict):
            # Handle {uid: info} format
            models = [{"uid": k, **v} if isinstance(v, dict) else {"uid": k, "path": v}
                      for k, v in models.items()]
    else:
        models = []

    # If using image_dir without input_json, discover UIDs from directory
    if not models and image_dir:
        img_path = Path(image_dir)
        for d in sorted(img_path.iterdir()):
            if d.is_dir():
                models.append({"uid": d.name})
            elif d.suffix.lower() in (".png", ".jpg", ".jpeg"):
                models.append({"uid": d.stem})

    if not models:
        print("ERROR: No models found. Provide --input_json or --image_dir.")
        return

    # Shard selection
    models = [m for i, m in enumerate(models) if i % num_shards == shard_id]
    if max_models > 0:
        models = models[:max_models]

    print(f"\n{'='*60}")
    print(f"SLAT Pair Generation — conditional generation (512 coarse + 1024 fine)")
    print(f"  Shard:        {shard_id}/{num_shards}")
    print(f"  Models:       {len(models)}")
    print(f"  Output:       {output_dir}")
    print(f"  GPU:          {gpu}")
    print(f"  Seed:         {seed}")
    print(f"{'='*60}\n")

    # Resume tracking
    tracker = ProgressTracker(output_path)
    already_done = sum(1 for m in models if tracker.is_done(m.get("uid", "")))
    if already_done > 0:
        print(f"  Resuming: {already_done} already completed, {len(models) - already_done} remaining\n")

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Load TRELLIS.2 pipeline
    print("Loading TRELLIS.2 pipeline...")
    sys.path.insert(0, trellis2_dir)
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if Path(model_dir).exists():
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_dir)
    else:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")

    # Require both 512 and 1024 flow models
    required_models = {
        "sparse_structure_flow_model",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
    }

    loaded_models = set(pipeline.models.keys())
    missing = required_models - loaded_models
    if missing:
        print(f"ERROR: Missing required models: {missing}")
        print(f"  Loaded: {loaded_models}")
        return

    # Prune unnecessary models to save VRAM
    keep_models = set(required_models)
    for key in list(pipeline.models.keys()):
        if key not in keep_models:
            print(f"  Pruning unused model: {key}")
            del pipeline.models[key]

    # NOTE: Do NOT set pipeline.rembg_model = None — preprocess_image() calls it
    # and will crash with TypeError if it's None. Keep rembg loaded.

    # Move to GPU
    pipeline.low_vram = False
    for m in pipeline.models.values():
        if hasattr(m, "low_vram"):
            m.low_vram = False
    if hasattr(pipeline, "image_cond_model") and pipeline.image_cond_model is not None:
        if hasattr(pipeline.image_cond_model, "low_vram"):
            pipeline.image_cond_model.low_vram = False
    pipeline.cuda()
    print(f"  Pipeline loaded. RSS: {get_rss_gb():.1f} GiB\n")

    # Parse directories
    img_dir = Path(image_dir) if image_dir else None

    # Main loop
    pbar = tqdm(models, desc="Generating SLAT pairs", disable=False)
    batch_start = time.time()
    batch_success = 0

    for i, model_info in enumerate(pbar):
        uid = model_info.get("uid", f"model_{i}")

        if tracker.is_done(uid):
            tracker.stats["total_skipped"] += 1
            continue

        t0 = time.time()

        # Load image
        image = load_image_for_uid(uid, img_dir, models)
        if image is None:
            print(f"  [{uid}] no image found, skipping")
            tracker.mark_failed(uid)
            continue

        # Generate SLAT pair
        try:
            pair = generate_slat_pair(
                pipeline, image, seed=seed,
                sparse_sampler_params=SPARSE_SAMPLER_PARAMS,
                shape_sampler_params=SHAPE_SAMPLER_PARAMS,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  [{uid}] OOM")
            cleanup_memory()
            tracker.mark_failed(uid)
            continue
        except Exception as e:
            print(f"  [{uid}] error: {type(e).__name__}: {e}")
            tracker.mark_failed(uid)
            continue

        if pair is None:
            print(f"  [{uid}] empty result (no sparse structure)")
            tracker.mark_failed(uid)
            continue

        # Save
        elapsed = time.time() - t0
        metadata = {
            "uid": uid,
            "seed": seed,
            "n_points": pair["n_points"],
            "coarse_slat_shape": list(pair["coarse_slat"].shape),
            "fine_slat_shape": list(pair["fine_slat"].shape),
            "has_cond_features": pair["cond_features"] is not None,
            "cond_features_source": "dinov2",
            "generation_time_sec": round(elapsed, 1),
            "sparse_sampler_params": SPARSE_SAMPLER_PARAMS,
            "shape_sampler_params": SHAPE_SAMPLER_PARAMS,
        }

        if save_slat_pair(output_path, uid, pair, metadata, image=image):
            tracker.mark_complete(uid)
            batch_success += 1
            pbar.set_postfix_str(
                f"ok={tracker.stats['total_success']} "
                f"fail={tracker.stats['total_failed']} "
                f"pts={pair['n_points']} "
                f"{elapsed:.1f}s"
            )
        else:
            tracker.mark_failed(uid)

        # Periodic progress save
        if (i + 1) % save_every == 0:
            tracker.save()
            batch_elapsed = time.time() - batch_start
            rate = batch_success / max(batch_elapsed, 1) * 3600
            print(f"  [progress] {tracker.stats['total_success']} ok, "
                  f"{tracker.stats['total_failed']} fail, "
                  f"~{rate:.0f} pairs/hr, RSS={get_rss_gb():.1f} GiB")

        # Cleanup
        del image, pair
        cleanup_memory()

    # Final save
    tracker.save()
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Success: {tracker.stats['total_success']}")
    print(f"  Failed:  {tracker.stats['total_failed']}")
    print(f"  Skipped: {tracker.stats['total_skipped']}")
    print(f"{'='*60}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate coarse/fine SLAT pairs for Stage 2 v2 training"
    )

    # Input
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input_json", type=str,
        help="JSON file listing models to process (uid → path)")
    input_group.add_argument("--image_dir", type=str,
        help="Directory of pre-rendered images (one subdir per UID)")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
        help="Output directory for SLAT pairs")

    # Sharding
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    # GPU
    parser.add_argument("--gpu", type=int, default=0)

    # Generation params
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_models", type=int, default=0,
        help="Max models to process (0=all)")
    parser.add_argument("--save_every", type=int, default=25,
        help="Save progress every N models")

    # TRELLIS.2 paths
    parser.add_argument("--trellis2_dir", type=str,
        default="/workspace/TRELLIS.2")
    parser.add_argument("--model_dir", type=str,
        default="/workspace/models/trellis2-4b")

    args = parser.parse_args()

    if not args.input_json and not args.image_dir:
        parser.error("Must provide either --input_json or --image_dir")

    run_generation(
        input_json=args.input_json,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        gpu=args.gpu,
        seed=args.seed,
        trellis2_dir=args.trellis2_dir,
        model_dir=args.model_dir,
        save_every=args.save_every,
        max_models=args.max_models,
    )


if __name__ == "__main__":
    main()
