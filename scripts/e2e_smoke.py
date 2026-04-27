#!/usr/bin/env python3
"""ClearMesh end-to-end GPU smoke driver.

Wraps ``ClearMeshPipeline.generate`` with structured pass/fail tracking
and emits a machine-readable JSON report at ``--report`` so a teammate
can attach the file rather than copy-paste logs.

What ends up in the report:
  - env: torch / CUDA / GPU / TRELLIS.2 commit + version
  - repo: git commit
  - stages: per-stage status + duration + failure reason
  - outputs: exported file path + size + sha256
  - mesh_stats: vertex/face counts + watertightness
  - overall_pass: True only if every enabled stage ran AND the output
    exists AND the mesh is non-degenerate

Designed to be launched by ``scripts/run_e2e_gpu.sh`` but usable
standalone::

    python3 scripts/e2e_smoke.py \\
        --input photo.png \\
        --output /tmp/clearmesh_e2e_out.glb \\
        --report /tmp/clearmesh_e2e_report.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def _run_git(args: list[str], cwd: str) -> str:
    """Return ``git <args>`` output, or 'unknown' on failure."""
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Return hex sha256 of a file; 'missing' if not found."""
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_env(repo_root: str, trellis2_dir: str) -> dict:
    """Capture torch / CUDA / GPU / TRELLIS.2 / repo fingerprints."""
    env: dict = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "repo_commit": _run_git(["rev-parse", "HEAD"], repo_root),
        "repo_branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
        "trellis2_commit": _run_git(["rev-parse", "HEAD"], trellis2_dir),
    }

    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except Exception as e:
        env["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        import trellis2

        env["trellis2_version"] = getattr(trellis2, "__version__", "unknown")
        env["trellis2_path"] = getattr(trellis2, "__file__", "unknown")
    except Exception as e:
        env["trellis2_error"] = f"{type(e).__name__}: {e}"

    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="ClearMesh e2e smoke with JSON report.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output mesh path")
    parser.add_argument(
        "--report",
        default="/tmp/clearmesh_e2e_report.json",
        help="Where to write the machine-readable report",
    )
    parser.add_argument(
        "--ultrashape-dir", default="/workspace/UltraShape-1.0"
    )
    parser.add_argument(
        "--ultrashape-checkpoint", default="/workspace/checkpoints/ultrashape_v1.pt"
    )
    parser.add_argument("--trellis2-dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--format", default="glb", choices=["stl", "glb", "obj", "fbx"])
    parser.add_argument("--resolution", type=int, default=512, choices=[512, 1024, 1536])
    parser.add_argument("--octree-res", type=int, default=512, choices=[512, 1024])
    args = parser.parse_args()

    repo_root = str(Path(__file__).resolve().parent.parent)

    report: dict = {
        "overall_pass": False,
        "env": collect_env(repo_root, args.trellis2_dir),
        "args": vars(args),
        "stages": {},
        "outputs": {},
        "mesh_stats": {},
        "errors": [],
    }

    def _write_report():
        """Write report atomically so partial failures still produce a file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
            tmp = args.report + ".tmp"
            with open(tmp, "w") as f:
                json.dump(report, f, indent=2, default=str)
            os.replace(tmp, args.report)
        except Exception as e:
            print(f"[e2e_smoke] could not write report: {e}", file=sys.stderr)

    # Preflight checkpoint existence — clearest signal for "install first".
    ckpt_exists = os.path.exists(args.ultrashape_checkpoint)
    report["stages"]["preflight_ultrashape_ckpt"] = {
        "pass": ckpt_exists,
        "path": args.ultrashape_checkpoint,
        "reason": None if ckpt_exists else "ultrashape checkpoint not found",
    }

    try:
        # Late import so `--help` works without torch installed.
        from clearmesh.pipeline import ClearMeshPipeline, GenerationOptions
    except Exception as e:
        report["errors"].append(f"import ClearMeshPipeline failed: {e}")
        report["stages"]["import"] = {"pass": False, "reason": str(e)}
        _write_report()
        print(f"[e2e_smoke] FAIL: pipeline import failed — {e}", file=sys.stderr)
        return 2

    report["stages"]["import"] = {"pass": True}

    pipeline = ClearMeshPipeline(
        ultrashape_dir=args.ultrashape_dir,
        ultrashape_checkpoint=args.ultrashape_checkpoint,
    )
    options = GenerationOptions(
        resolution=args.resolution,
        refinement_octree_res=args.octree_res,
        export_format=args.format,
    )

    # Run the actual pipeline. Any exception is caught, reported, and the
    # partial report is still written so the teammate has something to
    # attach even in the failure case.
    try:
        result = pipeline.generate(args.input, args.output, options)
    except Exception as e:
        report["errors"].append(f"pipeline.generate failed: {e}")
        report["traceback"] = traceback.format_exc()
        report["stages"]["generate"] = {"pass": False, "reason": str(e)}
        _write_report()
        print(f"[e2e_smoke] FAIL: {e}", file=sys.stderr)
        return 1

    # Fill in per-stage timings (stage names come from ClearMeshPipeline).
    for stage, duration in (result.timings or {}).items():
        report["stages"][stage] = {"pass": True, "duration_s": round(duration, 3)}

    # Output artifact checks.
    out_path = result.output_path or args.output
    out_exists = os.path.exists(out_path)
    out_size = os.path.getsize(out_path) if out_exists else 0
    report["outputs"] = {
        "path": out_path,
        "exists": out_exists,
        "size_bytes": out_size,
        "size_kb": round(out_size / 1024, 1),
        "sha256": _sha256_file(out_path),
    }

    # Mesh sanity.
    try:
        mesh = result.mesh
        report["mesh_stats"] = {
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
            "watertight": bool(mesh.is_watertight),
            "volume": (
                float(mesh.volume) if getattr(mesh, "is_watertight", False) else None
            ),
        }
    except Exception as e:
        report["errors"].append(f"mesh stats failed: {e}")

    # Overall pass: every stage green AND output exists AND mesh is non-degenerate.
    stages_ok = all(s.get("pass", False) for s in report["stages"].values())
    mesh_ok = report["mesh_stats"].get("vertices", 0) > 0
    report["overall_pass"] = stages_ok and out_exists and mesh_ok

    _write_report()

    # One-line human summary, full detail in JSON.
    verdict = "PASS" if report["overall_pass"] else "FAIL"
    print(
        f"\n[e2e_smoke] {verdict}  "
        f"output={out_path} ({report['outputs']['size_kb']} KB)  "
        f"verts={report['mesh_stats'].get('vertices', 'n/a')}  "
        f"faces={report['mesh_stats'].get('faces', 'n/a')}  "
        f"report={args.report}"
    )
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
