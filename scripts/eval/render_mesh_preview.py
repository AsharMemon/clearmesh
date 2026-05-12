#!/usr/bin/env python3
"""Invoke Blender to render one mesh preview."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()

    blender = shutil.which(args.blender) or args.blender
    script = Path(__file__).with_name("render_mesh_preview_blender.py")
    proc = subprocess.run(
        [blender, "--background", "--python", str(script), "--", str(args.mesh), str(args.output)],
        text=True,
        check=False,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
