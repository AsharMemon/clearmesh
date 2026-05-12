#!/usr/bin/env python3
"""Batch mesh evaluation for ClearMesh mesh-head bake-offs.

Manifest format (CSV):
  case_id,method,mesh_path,reference_path

reference_path is optional. If present, Chamfer/Hausdorff/normal consistency are
computed against the proxy or ground-truth mesh.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.eval import evaluate_mesh, evaluate_mesh_pair


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated meshes for bake-off metrics")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with case_id,method,mesh_path[,reference_path]")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON report path")
    parser.add_argument("--samples", type=int, default=20_000, help="Surface samples for pair metrics")
    parser.add_argument("--blender", default=None, help="Optional Blender executable for import/export roundtrip")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    report: dict[str, Any] = {"manifest": str(args.manifest), "results": []}

    for row in rows:
        mesh_path = row["mesh_path"]
        result: dict[str, Any] = {
            "case_id": row.get("case_id", ""),
            "method": row.get("method", ""),
            "mesh_path": mesh_path,
            "mesh_metrics": evaluate_mesh(mesh_path, blender=args.blender),
        }
        reference_path = row.get("reference_path") or ""
        if reference_path:
            result["reference_path"] = reference_path
            try:
                result["pair_metrics"] = evaluate_mesh_pair(
                    mesh_path,
                    reference_path,
                    samples=args.samples,
                )
            except Exception as exc:  # noqa: BLE001 - one failed row should not kill a bake-off.
                result["pair_metrics"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        report["results"].append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} meshes)")


if __name__ == "__main__":
    main()
