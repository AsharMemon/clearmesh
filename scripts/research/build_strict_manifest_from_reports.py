#!/usr/bin/env python3
"""Build a strict target manifest from copied strict target reports.

Shard workers write ``strict_target_manifest.json`` only after strict target
preparation fully completes. For partial scale gates, B2 may already contain
many accepted strict meshes plus their per-mesh reports. This helper rebuilds
the manifest from those reports without interrupting the active workers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MESH_SUFFIXES = (".glb", ".gltf", ".obj", ".ply", ".stl")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mesh_for_report(report: dict[str, Any], report_path: Path, strict_root: Path) -> Path | None:
    raw_output = str(report.get("output_path") or "")
    if raw_output:
        candidate = Path(raw_output)
        if candidate.exists():
            return candidate
        basename = candidate.name
        if basename:
            local = strict_root / "meshes" / basename
            if local.exists():
                return local
            matches = list(strict_root.rglob(basename))
            if matches:
                return matches[0]

    stem = report_path.stem
    for suffix in MESH_SUFFIXES:
        candidate = strict_root / "meshes" / f"{stem}_strict{suffix}"
        if candidate.exists():
            return candidate
    matches = [path for path in strict_root.rglob(f"{stem}*") if path.suffix.lower() in MESH_SUFFIXES]
    return matches[0] if matches else None


def _build_records(strict_root: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    report_dir = strict_root / "reports"
    if not report_dir.exists():
        raise SystemExit(f"missing report dir: {report_dir}")
    records: list[dict[str, Any]] = []
    accepted = rejected = missing_mesh = errors = 0
    for report_path in sorted(report_dir.glob("*.json")):
        try:
            report = _read_json(report_path)
        except Exception:
            errors += 1
            continue
        if not bool(report.get("accepted")):
            rejected += 1
            continue
        mesh_path = _mesh_for_report(report, report_path, strict_root)
        if mesh_path is None:
            missing_mesh += 1
            continue
        records.append(
            {
                "source_path": str(report.get("input_path") or ""),
                "target_path": str(mesh_path),
                "report_path": str(report_path),
                "status": "accepted",
                "accepted": True,
                "engine": report.get("engine"),
                "output_metrics": report.get("output_metrics") or {},
            }
        )
        accepted += 1
        if limit > 0 and len(records) >= limit:
            break
    return records, {
        "accepted": accepted,
        "rejected": rejected,
        "missing_mesh": missing_mesh,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    all_records: list[dict[str, Any]] = []
    sources = []
    for strict_root in args.strict_root:
        records, summary = _build_records(strict_root.resolve(), max(0, args.limit - len(all_records)) if args.limit else 0)
        all_records.extend(records)
        sources.append({"strict_root": str(strict_root), "records": len(records), **summary})
        if args.limit > 0 and len(all_records) >= args.limit:
            break

    mesh_dir = str(args.output.parent / "meshes")
    summary = {
        "input_dir": None,
        "manifest": None,
        "mesh_dir": mesh_dir,
        "candidate_count": sum(int(source["records"]) for source in sources),
        "accepted": len(all_records),
        "sources": sources,
        "records": all_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2, sort_keys=True))
    return 0 if all_records else 2


if __name__ == "__main__":
    raise SystemExit(main())
