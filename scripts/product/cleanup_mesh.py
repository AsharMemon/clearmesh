#!/usr/bin/env python3
"""Clean a generated mesh and write a cleanup report."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh.cleanup import CleanupOptions, cleanup_mesh_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--min-component-faces", type=int, default=8)
    parser.add_argument("--min-component-face-ratio", type=float, default=0.0)
    parser.add_argument("--keep-largest-components", type=int)
    parser.add_argument("--split-nonmanifold-vertices", action="store_true")
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--no-fix-normals", action="store_true")
    parser.add_argument("--no-merge-vertices", action="store_true")
    args = parser.parse_args()

    options = CleanupOptions(
        min_component_faces=args.min_component_faces,
        min_component_face_ratio=args.min_component_face_ratio,
        keep_largest_components=args.keep_largest_components,
        split_nonmanifold_vertices=args.split_nonmanifold_vertices,
        fill_holes=args.fill_holes,
        fix_normals=not args.no_fix_normals,
        merge_vertices=not args.no_merge_vertices,
    )
    report = cleanup_mesh_file(args.input, args.output, options)
    payload = asdict(report)
    report_path = args.report_json or Path(args.output).with_suffix(Path(args.output).suffix + ".cleanup.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
