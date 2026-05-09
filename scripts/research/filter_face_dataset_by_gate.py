#!/usr/bin/env python3
"""Promote only FACE-token shards that passed a dataset gate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_path(row: dict, manifest_path: Path) -> Path:
    path = Path(str(row["path"]))
    if path.is_absolute() or path.exists():
        return path
    return manifest_path.parent / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    args = parser.parse_args()

    manifest_path = args.dataset_dir / "manifest.jsonl"
    rows = _read_jsonl(manifest_path)
    rows_by_resolved = {str(_source_path(row, manifest_path).resolve()): row for row in rows}
    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    passing_results = [result for result in gate.get("results", []) if result.get("passes")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    promoted_rows = []
    missing = []
    for result in passing_results:
        source = Path(str(result["path"]))
        row = rows_by_resolved.get(str(source.resolve()))
        if row is None:
            missing.append(str(source))
            continue
        destination = args.output_dir / source.name
        if source.resolve() != destination.resolve():
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            if args.copy_mode == "symlink":
                destination.symlink_to(source)
            else:
                shutil.copy2(source, destination)
        updated = dict(row)
        updated["path"] = str(destination)
        promoted_rows.append(updated)

    with (args.output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in promoted_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "gate_report": str(args.gate_report),
        "output_dir": str(args.output_dir),
        "input_count": len(rows),
        "gate_sample_count": int(gate.get("sample_count", len(gate.get("results", [])))),
        "passing_count": len(passing_results),
        "promoted_count": len(promoted_rows),
        "missing_count": len(missing),
        "missing": missing,
    }
    (args.output_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if promoted_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
