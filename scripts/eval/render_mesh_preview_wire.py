#!/usr/bin/env python3
"""Render a lightweight mesh wire preview without Blender."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh.preview import render_wire_preview_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    render_wire_preview_file(args.mesh, args.output, size=args.size)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
