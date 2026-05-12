#!/usr/bin/env python3
"""Check whether the TRELLIS.2 GPU runtime is ready."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default="/home/ubuntu/TRELLIS.2", type=Path)
    parser.add_argument("--python", default="/home/ubuntu/trellis2-venv/bin/python", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, object] = {
        "repo_dir": str(args.repo_dir),
        "python": str(args.python),
        "checks": {},
        "errors": [],
    }
    checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
    errors: list[str] = report["errors"]  # type: ignore[assignment]

    checks["repo_exists"] = args.repo_dir.exists()
    checks["example_exists"] = (args.repo_dir / "example.py").exists()
    checks["setup_exists"] = (args.repo_dir / "setup.sh").exists()
    checks["python_exists"] = args.python.exists()

    if not args.repo_dir.exists():
        errors.append(f"TRELLIS.2 repo not found: {args.repo_dir}")
    if not args.python.exists():
        errors.append(f"TRELLIS.2 Python not found: {args.python}")

    if args.python.exists():
        code = """
import importlib, json
mods = ["torch", "PIL", "cv2", "trellis2", "o_voxel", "trimesh"]
out = {"modules": {}}
for mod in mods:
    try:
        m = importlib.import_module(mod)
        out["modules"][mod] = {"ok": True, "version": getattr(m, "__version__", None)}
    except Exception as exc:
        out["modules"][mod] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
try:
    import torch
    out["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
except Exception as exc:
    out["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(out, sort_keys=True))
"""
        proc = subprocess.run(
            [str(args.python), "-c", code],
            cwd=str(args.repo_dir) if args.repo_dir.exists() else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            probe = json.loads(proc.stdout)
            checks["python_probe"] = probe
            for name, module in probe.get("modules", {}).items():
                if not module.get("ok"):
                    errors.append(f"missing module {name}: {module.get('error')}")
            if not probe.get("torch", {}).get("cuda_available"):
                errors.append("torch CUDA is not available")
        else:
            checks["python_probe_error"] = proc.stderr.strip()
            errors.append(f"python probe failed with exit code {proc.returncode}")

    report["ok"] = not errors
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
