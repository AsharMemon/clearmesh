#!/usr/bin/env python3
"""Sample mesh surfaces at fixed point budgets for mesh-head bake-offs.

Input CSV columns:
  case_id,method,mesh_path,reference_path

Only mesh_path is required. Outputs one row per case and point budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.pointcloud import sample_to_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample point clouds from mesh bake-off manifests")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=[16_384, 40_960, 100_000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--formats", nargs="+", choices=["npz", "ply"], default=["npz", "ply"])
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    output_rows = []
    for row_index, row in enumerate(rows):
        mesh_path = row["mesh_path"]
        case_id = row.get("case_id") or f"case_{row_index:04d}"
        method = row.get("method") or "unknown"
        for budget in args.budgets:
            stem = args.output_dir / case_id / method / f"points_{budget}"
            paths = sample_to_files(
                mesh_path,
                stem,
                count=budget,
                seed=args.seed + row_index,
                formats=tuple(args.formats),
            )
            output_rows.append(
                {
                    "case_id": case_id,
                    "source_method": method,
                    "source_mesh_path": mesh_path,
                    "point_budget": str(budget),
                    "npz_path": paths.get("npz", ""),
                    "ply_path": paths.get("ply", ""),
                    "reference_path": row.get("reference_path", ""),
                }
            )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()) if output_rows else [])
        writer.writeheader()
        writer.writerows(output_rows)

    metadata_path = args.output_manifest.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "source_manifest": str(args.manifest),
                "output_manifest": str(args.output_manifest),
                "budgets": args.budgets,
                "rows": len(output_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output_manifest} ({len(output_rows)} point-cloud rows)")


if __name__ == "__main__":
    main()
