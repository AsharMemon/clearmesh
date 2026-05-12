#!/usr/bin/env python3
"""Compare reference-surface builders for one proxy mesh."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.eval.mesh_quality import evaluate_mesh, evaluate_mesh_pair  # noqa: E402
from clearmesh.mesh.normalization import SurfaceNormalizationOptions, normalize_surface_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--target-faces", type=int, default=50000)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--sample-points", type=int, default=120000)
    parser.add_argument("--manifoldplus-binary")
    parser.add_argument("--manifoldplus-depth", type=int, default=8)
    parser.add_argument("--ultrashape-dir")
    parser.add_argument("--ultrashape-checkpoint")
    parser.add_argument("--ultrashape-python", default=sys.executable)
    parser.add_argument("--ultrashape-remove-bg", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    results.append(
        _run_local_normalization(
            "poisson",
            args.input,
            args.output_dir / "poisson" / "reference.obj",
            SurfaceNormalizationOptions(
                engine="poisson",
                target_faces=args.target_faces,
                poisson_depth=args.poisson_depth,
                sample_points=args.sample_points,
            ),
        )
    )
    if args.manifoldplus_binary:
        results.append(
            _run_command_reference(
                "manifoldplus",
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/product/run_manifoldplus.py"),
                    "--input",
                    str(args.input),
                    "--output-dir",
                    str(args.output_dir / "manifoldplus"),
                    "--binary",
                    args.manifoldplus_binary,
                    "--depth",
                    str(args.manifoldplus_depth),
                ],
                args.output_dir / "manifoldplus" / "manifold_reference.obj",
                args.input,
            )
        )
    else:
        results.append({"engine": "manifoldplus", "status": "skipped", "reason": "manifoldplus_binary_not_supplied"})
    if args.ultrashape_dir and args.ultrashape_checkpoint and args.image:
        results.append(
            _run_command_reference(
                "ultrashape",
                [
                    args.ultrashape_python,
                    str(REPO_ROOT / "scripts/product/run_ultrashape_refinement.py"),
                    "--mesh",
                    str(args.input),
                    "--image",
                    str(args.image),
                    "--output-dir",
                    str(args.output_dir / "ultrashape"),
                    "--output-name",
                    "ultrashape_reference.glb",
                    "--ultrashape-dir",
                    args.ultrashape_dir,
                    "--checkpoint",
                    args.ultrashape_checkpoint,
                ]
                + (["--remove-bg"] if args.ultrashape_remove_bg else []),
                args.output_dir / "ultrashape" / "ultrashape_reference.glb",
                args.input,
            )
        )
    else:
        results.append({"engine": "ultrashape", "status": "skipped", "reason": "ultrashape_dir_checkpoint_or_image_not_supplied"})

    payload = {"input": str(args.input), "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(args.report)


def _run_local_normalization(engine: str, input_path: Path, output_path: Path, options: SurfaceNormalizationOptions) -> dict:
    started = time.time()
    try:
        report = normalize_surface_file(input_path, output_path, options)
        return {
            "engine": engine,
            "status": "succeeded",
            "runtime_seconds": time.time() - started,
            "output_path": str(output_path),
            "normalization_report": asdict(report),
            "pair_metrics": _safe_pair(output_path, input_path),
        }
    except Exception as exc:  # noqa: BLE001
        return {"engine": engine, "status": "failed", "runtime_seconds": time.time() - started, "error": f"{type(exc).__name__}: {exc}"}


def _run_command_reference(engine: str, command: list[str], output_path: Path, input_path: Path) -> dict:
    started = time.time()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            return {
                "engine": engine,
                "status": "failed",
                "runtime_seconds": time.time() - started,
                "command": command,
                "stdout_tail": (result.stdout or "")[-2000:],
                "stderr_tail": (result.stderr or "")[-2000:],
            }
        return {
            "engine": engine,
            "status": "succeeded",
            "runtime_seconds": time.time() - started,
            "command": command,
            "output_path": str(output_path),
            "mesh_metrics": evaluate_mesh(output_path),
            "pair_metrics": _safe_pair(output_path, input_path),
            "stdout_tail": (result.stdout or "")[-2000:],
            "stderr_tail": (result.stderr or "")[-2000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"engine": engine, "status": "failed", "runtime_seconds": time.time() - started, "command": command, "error": f"{type(exc).__name__}: {exc}"}


def _safe_pair(generated: Path, reference: Path) -> dict:
    try:
        return evaluate_mesh_pair(generated, reference, samples=10000)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    main()
