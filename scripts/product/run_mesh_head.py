#!/usr/bin/env python3
"""Run one external mesh-head adapter against a point cloud.

This is the narrow bridge between ClearMesh manifests and public GitHub repos. It
keeps repo-specific execution isolated so the product worker can call it on GPU
machines without importing those projects into ClearMesh core.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clearmesh.mesh_heads import MeshHeadInput, build_mesh_head  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="meshripple", help="Mesh head adapter name")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--point-cloud", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--proxy-mesh", type=Path)
    parser.add_argument("--part-id")
    parser.add_argument("--config-json", type=Path, help="Optional adapter config JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = {}
    if args.config_json:
        config = json.loads(args.config_json.read_text(encoding="utf-8"))
    adapter = build_mesh_head(args.head, config)
    result = adapter.run(
        MeshHeadInput(
            case_id=args.case_id,
            point_cloud_path=args.point_cloud,
            proxy_mesh_path=args.proxy_mesh,
            output_dir=args.output_dir,
            part_id=args.part_id,
        )
    )
    print(
        json.dumps(
            {
                "adapter": result.adapter_name,
                "mesh_path": str(result.mesh_path),
                "command": result.command,
                "stdout_path": str(result.stdout_path),
                "stderr_path": str(result.stderr_path),
                "metadata": result.metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
