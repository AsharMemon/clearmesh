#!/usr/bin/env python3
"""Summarize ClearMesh job records and mesh eval reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def metric(report: dict[str, Any], path: str) -> Any:
    current: Any = report
    for part in path.split('.'):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", default=".clearmesh_state")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    jobs_dir = Path(args.state_root) / "jobs"
    rows = []
    for path in sorted(jobs_dir.glob("job_*.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        eval_asset = next((asset for asset in job.get("assets", []) if asset.get("kind") == "mesh_eval_report"), None)
        report = {}
        if eval_asset and Path(eval_asset["uri"]).exists():
            report = json.loads(Path(eval_asset["uri"]).read_text(encoding="utf-8"))
        rows.append(
            {
                "job_id": job["id"],
                "case_id": job["request"].get("metadata", {}).get("case_id"),
                "status": job["status"],
                "mesh_head": next((asset.get("metadata", {}).get("mesh_head") for asset in job.get("assets", []) if asset.get("kind") == "artist_mesh"), None),
                "export": next((asset.get("uri") for asset in job.get("assets", []) if asset.get("kind") == "export_mesh"), None),
                "connected_components": metric(report, "mesh_metrics.connected_components"),
                "watertight": metric(report, "mesh_metrics.watertight"),
                "boundary_loop_count": metric(report, "mesh_metrics.boundary_loop_count"),
                "nonmanifold_edge_count": metric(report, "mesh_metrics.nonmanifold_edge_count"),
                "face_count": metric(report, "mesh_metrics.face_count"),
                "chamfer_l2": metric(report, "pair_metrics.chamfer_l2"),
                "normal_consistency": metric(report, "pair_metrics.normal_consistency"),
            }
        )
    payload = {"jobs": rows}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
