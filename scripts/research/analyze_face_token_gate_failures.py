#!/usr/bin/env python3
"""Analyze FACE-token gate failures after corpus construction.

The strict target mesh can be watertight while the quantized FACE token graph is
not. This script separates those cases and reports the token-level failure modes
that matter for scaling decisions: boundary edges, nonmanifold edges, duplicate
faces, degenerate faces, and quantized vertex collapse.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_topology import face_token_topology_report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_path(path: str | Path, manifest_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return manifest_path.parent / candidate


def _token_key(token_family: str) -> str:
    token_family = token_family.strip().lower()
    if token_family in {"paper", "paper_tokens"}:
        return "paper_tokens"
    if token_family in {"coordinate", "face", "tokens"}:
        return "tokens"
    if token_family in {"indexed", "indexed_tokens"}:
        return "indexed_tokens"
    raise ValueError(f"unsupported token family: {token_family}")


def _load_tokens(path: Path, token_family: str) -> np.ndarray:
    data = np.load(path)
    key = _token_key(token_family)
    if key not in data.files:
        raise ValueError(f"{path} is missing {key}")
    tokens = np.asarray(data[key], dtype=np.int64)
    if tokens.ndim == 1:
        tokens = tokens.reshape(-1, 9)
    return tokens


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _compact_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    numeric_keys = [
        "faces",
        "unique_vertex_count",
        "unique_vertex_ratio",
        "degenerate_face_count",
        "duplicate_face_count",
        "boundary_edge_count",
        "nonmanifold_edge_count",
        "edge_pairing_ratio",
        "max_edge_use",
    ]
    out: dict[str, Any] = {"count": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        out[f"mean_{key}"] = _mean(values)
        out[f"median_{key}"] = _median(values)
        out[f"max_{key}"] = max(values) if values else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-family", choices=["paper", "coordinate", "indexed"], default="paper")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    manifest_path = args.dataset_dir / "manifest.jsonl"
    manifest_rows = _read_jsonl(manifest_path)
    rows_by_resolved = {
        str(_resolve_path(row["path"], manifest_path).resolve()): row for row in manifest_rows
    }
    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))

    analyzed: list[dict[str, Any]] = []
    violation_counts: Counter[str] = Counter()
    missing: list[str] = []
    for result in gate.get("results", []):
        source = Path(str(result.get("path", "")))
        row = rows_by_resolved.get(str(source.resolve()))
        if row is None:
            missing.append(str(source))
            continue
        shard = _resolve_path(row["path"], manifest_path)
        tokens = _load_tokens(shard, args.token_family)
        report = face_token_topology_report(tokens).to_dict()
        violations = list(result.get("violations") or [])
        violation_counts.update(violations)
        face_count = int(report["face_count"])
        unique_vertices = int(report["unique_vertex_count"])
        item = {
            "path": str(shard),
            "source_name": row.get("source_name"),
            "passes": bool(result.get("passes")),
            "violations": violations,
            "faces": face_count,
            "unique_vertex_count": unique_vertices,
            "unique_vertex_ratio": float(unique_vertices / max(1, face_count * 3)),
            "degenerate_face_count": int(report["degenerate_face_count"]),
            "duplicate_face_count": int(report["duplicate_face_count"]),
            "boundary_edge_count": int(report["boundary_edge_count"]),
            "nonmanifold_edge_count": int(report["nonmanifold_edge_count"]),
            "edge_pairing_ratio": float(report["edge_pairing_ratio"]),
            "max_edge_use": int(report["max_edge_use"]),
            "decoded_watertight": result.get("decoded_watertight"),
            "token_watertight_edge_graph": result.get("token_watertight_edge_graph"),
        }
        analyzed.append(item)

    passing = [row for row in analyzed if row["passes"]]
    failing = [row for row in analyzed if not row["passes"]]
    worst = sorted(
        failing,
        key=lambda row: (
            int(row["boundary_edge_count"]) + 4 * int(row["nonmanifold_edge_count"]),
            int(row["degenerate_face_count"]) + int(row["duplicate_face_count"]),
        ),
        reverse=True,
    )[: max(0, args.top_k)]

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "gate_report": str(args.gate_report),
        "token_family": args.token_family,
        "sample_count": len(analyzed),
        "passing": len(passing),
        "failing": len(failing),
        "pass_rate": len(passing) / len(analyzed) if analyzed else 0.0,
        "missing_count": len(missing),
        "missing": missing,
        "violation_counts": dict(violation_counts),
        "passing_stats": _compact_stats(passing),
        "failing_stats": _compact_stats(failing),
        "worst_failures": worst,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["sample_count", "passing", "failing", "pass_rate", "violation_counts"]}, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0 if analyzed else 2


if __name__ == "__main__":
    raise SystemExit(main())
