#!/usr/bin/env python3
"""Preflight checks for running MeshRipple through ClearMesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REQUIRED_FILES = [
    "main.py",
    "config_loader/config_10k_full_dense_mesh.yaml",
    "config_loader/config_20k_nsa.yaml",
]
REQUIRED_MODULES = [
    "torch",
    "accelerate",
    "trimesh",
    "open3d",
    "timm",
    "transformers",
    "einops",
    "omegaconf",
    "beartype",
    "yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path("/home/ubuntu/mesh-heads/MeshRipple"))
    parser.add_argument("--python", default="python")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, object] = {"ok": True, "errors": [], "warnings": []}
    repo_dir = args.repo_dir
    if not repo_dir.exists():
        report["ok"] = False
        report["errors"].append(f"repo_dir missing: {repo_dir}")
    else:
        missing_files = [name for name in REQUIRED_FILES if not (repo_dir / name).exists()]
        if missing_files:
            report["ok"] = False
            report["errors"].append(f"missing repo files: {missing_files}")

    checkpoint_dir = args.checkpoint_dir or repo_dir / "ckpt"
    ckpt_10k = checkpoint_dir / "meshRipple_10k.pth"
    ckpt_nsa = checkpoint_dir / "meshRipple_nsa.pth"
    report["checkpoints"] = {
        "checkpoint_dir": str(checkpoint_dir),
        "meshRipple_10k.pth": ckpt_10k.exists(),
        "meshRipple_nsa.pth": ckpt_nsa.exists(),
    }
    if not ckpt_10k.exists() and not ckpt_nsa.exists():
        report["ok"] = False
        report["errors"].append(f"no MeshRipple checkpoints found in {checkpoint_dir}")

    code = """
import importlib, json
mods = __MODS__
result = {'modules': {}, 'torch': {}}
for mod in mods:
    try:
        imported = importlib.import_module(mod)
        result['modules'][mod] = {'ok': True, 'version': getattr(imported, '__version__', None)}
    except Exception as exc:
        result['modules'][mod] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
try:
    import torch
    result['torch'] = {'version': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}
except Exception as exc:
    result['torch'] = {'error': f'{type(exc).__name__}: {exc}'}
print(json.dumps(result, sort_keys=True))
""".replace("__MODS__", repr(REQUIRED_MODULES))
    proc = subprocess.run([args.python, "-c", code], cwd=repo_dir if repo_dir.exists() else None, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        report["ok"] = False
        report["errors"].append(f"python import probe failed: {proc.stderr.strip()[-1000:]}")
    else:
        probe = json.loads(proc.stdout)
        report["python_probe"] = probe
        missing = [name for name, value in probe["modules"].items() if not value.get("ok")]
        if missing:
            report["ok"] = False
            report["errors"].append(f"missing Python modules: {missing}")
        if not probe.get("torch", {}).get("cuda_available"):
            report["warnings"].append("torch.cuda.is_available() is false")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
