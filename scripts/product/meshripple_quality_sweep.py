#!/usr/bin/env python3
"""Run a MeshRipple quality/runtime sweep against one proxy mesh."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.eval.mesh_quality import evaluate_mesh, evaluate_mesh_pair
from clearmesh.mesh.cleanup import CleanupOptions, cleanup_mesh_file
from clearmesh.mesh_heads import MeshHeadInput, build_mesh_head
from clearmesh.pointcloud import sample_to_files


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-mesh", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-id", default="quality_sweep")
    parser.add_argument("--config", action="append", required=True, help="label=/path/to/config.json")
    parser.add_argument("--point-budget", type=int, default=40960)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--cleanup-min-component-faces", type=int, default=8)
    parser.add_argument("--cleanup-keep-largest-components", type=int)
    parser.add_argument("--skip-cleanup", action="store_true")
    args = parser.parse_args()

    proxy_mesh = args.proxy_mesh.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    point_prefix = output_dir / "pointcloud" / f"{args.case_id}_{args.point_budget}"

    point_started = time.time()
    point_paths = sample_to_files(proxy_mesh, point_prefix, count=args.point_budget)
    point_seconds = time.time() - point_started

    results: list[dict[str, Any]] = []
    for config_spec in args.config:
        if "=" not in config_spec:
            raise SystemExit(f"--config must be label=path, got: {config_spec}")
        label, raw_config_path = config_spec.split("=", 1)
        config_path = Path(raw_config_path).expanduser().resolve()
        run_dir = output_dir / label
        mesh_dir = run_dir / "mesh_head"
        config = load_json(config_path)
        adapter = build_mesh_head("meshripple", config)

        started = time.time()
        result = adapter.run(
            MeshHeadInput(
                case_id=f"{args.case_id}_{label}",
                point_cloud_path=point_paths["ply"],
                proxy_mesh_path=proxy_mesh,
                output_dir=mesh_dir,
                metadata={"sweep_label": label},
            )
        )
        mesh_seconds = time.time() - started

        raw_metrics = evaluate_mesh(result.mesh_path)
        raw_pair = evaluate_mesh_pair(result.mesh_path, proxy_mesh, samples=args.samples)
        item: dict[str, Any] = {
            "label": label,
            "config_path": str(config_path),
            "mesh_path": str(result.mesh_path),
            "stdout_path": str(result.stdout_path),
            "stderr_path": str(result.stderr_path),
            "mesh_seconds": mesh_seconds,
            "mesh_metrics": raw_metrics,
            "pair_metrics": raw_pair,
        }

        if not args.skip_cleanup:
            clean_path = run_dir / "cleanup" / f"{Path(result.mesh_path).stem}_cleaned.obj"
            cleanup_started = time.time()
            cleanup_report = cleanup_mesh_file(
                result.mesh_path,
                clean_path,
                CleanupOptions(
                    min_component_faces=args.cleanup_min_component_faces,
                    keep_largest_components=args.cleanup_keep_largest_components,
                ),
            )
            cleanup_seconds = time.time() - cleanup_started
            item["cleaned_mesh_path"] = str(clean_path)
            item["cleanup_seconds"] = cleanup_seconds
            item["cleanup_report"] = asdict(cleanup_report)
            item["cleaned_mesh_metrics"] = evaluate_mesh(clean_path)
            item["cleaned_pair_metrics"] = evaluate_mesh_pair(clean_path, proxy_mesh, samples=args.samples)
        results.append(item)

        report = {
            "case_id": args.case_id,
            "proxy_mesh": str(proxy_mesh),
            "point_cloud": {key: str(value) for key, value in point_paths.items()},
            "point_seconds": point_seconds,
            "results": results,
        }
        (output_dir / "quality_sweep.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"completed": label, "mesh_seconds": mesh_seconds, "mesh_path": str(result.mesh_path)}, indent=2))

    print(output_dir / "quality_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
