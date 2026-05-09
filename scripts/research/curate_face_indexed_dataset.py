#!/usr/bin/env python3
"""Promote topology-safe FACE-indexed shards into a curated dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_path(row: dict[str, Any], base: Path) -> Path:
    path = Path(str(row["path"]))
    if path.is_absolute() or path.exists():
        return path
    return base / path


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    return default if value is None else int(value)


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return default if value is None else float(value)


def _violations(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    violations: list[str] = []
    if not bool(row.get("indexed_decoded_watertight")):
        violations.append("indexed decoded mesh is not watertight")
    if not bool(row.get("indexed_token_watertight_edge_graph")):
        violations.append("indexed token edge graph is not watertight")
    if _int(row, "indexed_token_boundary_edge_count") > args.max_boundary_edges:
        violations.append("indexed token boundary edges above limit")
    if _int(row, "indexed_token_nonmanifold_edge_count") > args.max_nonmanifold_edges:
        violations.append("indexed token nonmanifold edges above limit")
    if _float(row, "indexed_token_edge_pairing_ratio") < args.min_edge_pairing_ratio:
        violations.append("indexed edge pairing ratio below limit")
    if args.require_boundary_growth and _int(row, "indexed_zero_closure_after_first") != 0:
        violations.append("indexed sequence has extra zero-closure jumps")
    if args.max_faces and _int(row, "indexed_faces") > args.max_faces:
        violations.append("indexed face count above limit")
    if args.max_vertices and _int(row, "indexed_vertices") > args.max_vertices:
        violations.append("indexed vertex count above limit")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--max-faces", type=int, default=512)
    parser.add_argument("--max-vertices", type=int, default=0)
    parser.add_argument("--max-boundary-edges", type=int, default=0)
    parser.add_argument("--max-nonmanifold-edges", type=int, default=0)
    parser.add_argument("--min-edge-pairing-ratio", type=float, default=1.0)
    parser.add_argument("--require-boundary-growth", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    manifest_path = args.input_dir / "manifest.jsonl"
    rows = _read_jsonl(manifest_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    promoted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        source = _resolve_path(row, args.input_dir)
        violations = _violations(row, args)
        if violations:
            rejected.append({"path": str(source), "violations": violations})
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
        promoted.append(updated)

    with (args.output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in promoted:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    faces = [_int(row, "indexed_faces") for row in promoted]
    vertices = [_int(row, "indexed_vertices") for row in promoted]
    zero_closure = [_int(row, "indexed_zero_closure_after_first") for row in promoted]
    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "input_count": int(len(rows)),
        "written": int(len(promoted)),
        "failed": int(len(rejected)),
        "fail_paths": [item["path"] for item in rejected],
        "rejections": rejected,
        "max_faces": int(max(faces) if faces else 0),
        "max_vertices": int(max(vertices) if vertices else 0),
        "mean_faces": float(sum(faces) / len(faces)) if faces else 0.0,
        "zero_closure_after_first_sum": int(sum(zero_closure)),
    }
    (args.output_dir / "curation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
