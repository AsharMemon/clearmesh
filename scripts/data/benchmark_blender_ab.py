#!/usr/bin/env python3
"""A/B benchmark trimesh vs Blender rendering on the same sampled models."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image

from scripts.data import generate_pairs as gp


def load_uid_list(path: Path) -> set[str]:
    try:
        with open(path) as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()


def collect_remaining_models(input_json: Path, shared_output_dir: Path, shard_id: int, num_shards: int) -> list[dict]:
    with open(input_json) as f:
        models = json.load(f)
    models = [m for i, m in enumerate(models) if i % num_shards == shard_id]

    all_completed = set()
    all_failed = set()
    for progress_file in shared_output_dir.rglob("progress.json"):
        all_completed.update(load_uid_list(progress_file))
    for failed_file in shared_output_dir.rglob("failed.json"):
        all_failed.update(load_uid_list(failed_file))

    skip_uids = all_completed | all_failed
    return [m for m in models if m["uid"] not in skip_uids]


def export_blender_asset(mesh, temp_root: Path) -> str:
    cached = mesh.metadata.get("_blender_asset") if hasattr(mesh, "metadata") else None
    if cached and os.path.exists(cached):
        return cached
    temp_root.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".glb", dir=str(temp_root))
    os.close(fd)
    mesh.export(path)
    mesh.metadata["_blender_asset"] = path
    return path


def make_blender_renderers(blender_bin: str, blender_script: str, temp_root: Path, samples: int):
    def render_multiview_blender(mesh, num_views: int = 12, image_size: int = 1024, views=None, fill_fraction: float = 0.75):
        views = views if views is not None else gp.CAMERA_VIEWS[:num_views]
        asset_path = export_blender_asset(mesh, temp_root)
        render_dir = Path(tempfile.mkdtemp(dir=str(temp_root), prefix="blender_views_"))
        cmd = [
            blender_bin,
            "--background",
            "--factory-startup",
            "--python",
            blender_script,
            "--",
            "--mesh",
            asset_path,
            "--output_dir",
            str(render_dir),
            "--views_json",
            json.dumps(views),
            "--resolution",
            str(image_size),
            "--fill_fraction",
            str(fill_fraction),
            "--samples",
            str(samples),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            tail = "\n".join((result.stderr or result.stdout).splitlines()[-10:])
            print(f"[blender] render failed: {tail}")
            shutil.rmtree(render_dir, ignore_errors=True)
            return [], gp.FAIL_RENDER_GL

        images = []
        for idx in range(len(views)):
            path = render_dir / f"view_{idx:02d}.png"
            if not path.exists():
                continue
            img = Image.open(path).convert("RGBA")
            arr = gp.np.array(img)
            transparent_mask = arr[:, :, 3] == 0
            arr[transparent_mask, :3] = 0
            images.append(Image.fromarray(arr))

        shutil.rmtree(render_dir, ignore_errors=True)
        if not images:
            return [], gp.FAIL_RENDER_ZERO_FG
        return images, None

    def render_single_view_blender(mesh, view, image_size: int = 1024, fill_fraction: float = 0.9):
        images, fail_reason = render_multiview_blender(
            mesh,
            num_views=1,
            image_size=image_size,
            views=[view],
            fill_fraction=fill_fraction,
        )
        if images:
            return images[0], None
        return None, fail_reason

    return render_multiview_blender, render_single_view_blender


def summarize_run(output_dir: Path, elapsed_sec: float) -> dict:
    progress = load_uid_list(output_dir / "progress.json")
    failed = load_uid_list(output_dir / "failed.json")
    return {
        "completed": len(progress),
        "failed": len(failed),
        "processed": len(progress) + len(failed),
        "success_rate": (len(progress) / max(len(progress) + len(failed), 1)),
        "wall_sec": elapsed_sec,
        "sec_per_model": elapsed_sec / max(len(progress) + len(failed), 1),
        "sec_per_success": elapsed_sec / max(len(progress), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", default="/workspace/data/filtered/valid_models.json")
    parser.add_argument("--shared_output_dir", default="/workspace/data/training_pairs")
    parser.add_argument("--benchmark_root", default="/workspace/data/benchmarks/blender_ab")
    parser.add_argument("--trellis2_dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--model_dir", default="/workspace/models/trellis2-4b")
    parser.add_argument("--renderer_size", type=int, default=1024)
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--pipeline_type", default="512")
    parser.add_argument("--sample_size", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--blender_bin", default="/workspace/tools/blender-3.0.1-linux-x64/blender")
    parser.add_argument("--blender_samples", type=int, default=16)
    args = parser.parse_args()

    remaining = collect_remaining_models(
        Path(args.input_json),
        Path(args.shared_output_dir),
        args.shard_id,
        args.num_shards,
    )
    sample = remaining[: args.sample_size]
    if not sample:
        raise SystemExit("No remaining models available for benchmark")

    bench_root = Path(args.benchmark_root) / time.strftime("%Y%m%d_%H%M%S")
    bench_root.mkdir(parents=True, exist_ok=True)
    sample_json = bench_root / "sample.json"
    with open(sample_json, "w") as f:
        json.dump(sample, f)

    original_render_multiview = gp.render_multiview
    original_render_single = gp.render_single_view

    def run_variant(name: str, use_blender: bool) -> dict:
        output_dir = bench_root / name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_root = bench_root / f"{name}_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)

        if use_blender:
            render_mv, render_sv = make_blender_renderers(
                args.blender_bin,
                str(Path(__file__).with_name("blender_render_views.py")),
                temp_root,
                args.blender_samples,
            )
            gp.render_multiview = render_mv
            gp.render_single_view = render_sv
        else:
            gp.render_multiview = original_render_multiview
            gp.render_single_view = original_render_single

        t0 = time.time()
        gp.generate_pairs(
            input_json=str(sample_json),
            output_dir=str(output_dir),
            pipeline_type=args.pipeline_type,
            render_size=args.renderer_size,
            num_views=args.num_views,
            limit=None,
            trellis2_dir=args.trellis2_dir,
            model_dir=args.model_dir,
            min_fg_frac=gp.DEFAULT_MIN_FG_FRAC,
            max_fg_frac=gp.DEFAULT_MAX_FG_FRAC,
            edge_margin=gp.DEFAULT_EDGE_MARGIN,
            shard_id=0,
            num_shards=1,
            gpu=args.gpu,
            retry_failed=False,
            geometry_only=True,
            disable_rembg_model=True,
            log_rss_every=5,
            rss_hard_limit_gb=0.0,
            rss_emergency_limit_gb=0.0,
            rss_watch_interval_sec=0.25,
            max_models_per_run=0,
            max_emergency_recycles_per_uid=3,
            recycle_exit_code=75,
            low_vram=False,
            use_malloc_trim=True,
            verbose_trellis_progress=False,
        )
        return summarize_run(output_dir, time.time() - t0)

    try:
        opengl_summary = run_variant("trimesh", use_blender=False)
        blender_summary = run_variant("blender", use_blender=True)
    finally:
        gp.render_multiview = original_render_multiview
        gp.render_single_view = original_render_single

    report = {
        "sample_size": len(sample),
        "sample_uids": [m["uid"] for m in sample],
        "num_views": args.num_views,
        "renderer_size": args.renderer_size,
        "blender_samples": args.blender_samples,
        "trimesh": opengl_summary,
        "blender": blender_summary,
    }
    report_path = bench_root / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
