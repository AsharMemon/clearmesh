#!/usr/bin/env python3
"""Run ManifoldPlus as a watertight reference-surface command hook."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import trimesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", default="manifold_reference.obj")
    parser.add_argument("--binary", default="ManifoldPlus")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    binary = shutil.which(args.binary) or (str(Path(args.binary).expanduser()) if Path(args.binary).expanduser().exists() else None)
    if binary is None:
        raise SystemExit(f"ManifoldPlus binary not found: {args.binary}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    input_path = _prepare_obj_input(args.input, args.output_dir)
    command = [binary, "--input", str(input_path), "--output", str(output_path), "--depth", str(args.depth)]
    subprocess.run(command, check=True, timeout=max(1, args.timeout_seconds))
    print(output_path)
    return 0


def _prepare_obj_input(input_path: Path, output_dir: Path) -> Path:
    """ManifoldPlus is OBJ-oriented; normalize containers before native code."""
    if input_path.suffix.lower() == ".obj":
        return input_path
    mesh = trimesh.load(str(input_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    converted = output_dir / "manifoldplus_input.obj"
    mesh.export(converted)
    return converted


if __name__ == "__main__":
    raise SystemExit(main())
