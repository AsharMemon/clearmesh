#!/usr/bin/env python3
"""Source a large high-quality Objaverse++ candidate pool for FACE training.

This script does not download 3D assets. It streams Objaverse++ annotations,
keeps high/superior standalone candidates, writes a deterministic selected
annotation pool, and shards it for parallel Thunder corpus workers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


QUALITY_MAP = {
    "low": 0,
    "low quality": 0,
    "medium": 1,
    "medium quality": 1,
    "high": 2,
    "high quality": 2,
    "superior": 3,
    "superior quality": 3,
    "excellent": 3,
}

UID_KEYS = ("uid", "UID", "objaverse_uid", "Objaverse UID", "object_uid", "object_id")
QUALITY_KEYS = ("quality_score", "score", "quality", "Quality", "quality_label", "Quality Score")
REJECT_KEYS = (
    "transparent",
    "transparency",
    "Transparency",
    "is_transparent",
    "scene",
    "Scene",
    "is_scene",
    "not_single_object",
    "not_a_single_object",
    "Not a Single Object",
    "multi_object",
    "is_multi_object",
    "multiple_objects",
    "multi-object",
)
OPTIONAL_KEEP_KEYS = (
    "aesthetic_score",
    "category",
    "categories",
    "caption",
    "name",
    "tags",
    "license",
    "source",
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _uid(row: dict[str, Any]) -> str:
    for key in UID_KEYS:
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def _quality(row: dict[str, Any]) -> int:
    for key in QUALITY_KEYS:
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        text = str(value).strip().lower()
        if text in QUALITY_MAP:
            return QUALITY_MAP[text]
    return -1


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _reject_reasons(row: dict[str, Any], *, min_quality: int, reject_regex: re.Pattern[str] | None) -> list[str]:
    reasons: list[str] = []
    quality = _quality(row)
    if quality < min_quality:
        reasons.append(f"quality {quality} < {min_quality}")
    for key in REJECT_KEYS:
        if _truthy(row.get(key)):
            reasons.append(f"reject flag: {key}")
    if reject_regex is not None:
        haystack = " ".join(str(value) for value in row.values() if value is not None)
        if reject_regex.search(haystack):
            reasons.append("reject regex")
    return reasons


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    uid = _uid(row)
    out: dict[str, Any] = {"uid": uid, "quality_score": _quality(row)}
    for key in (*QUALITY_KEYS, *REJECT_KEYS, *OPTIONAL_KEEP_KEYS):
        if key in row and key not in out:
            out[key] = _jsonable(row[key])
    for key, value in row.items():
        lower = key.lower()
        if lower.startswith(("quality", "aesthetic", "category", "tag", "license", "source")) and key not in out:
            out[key] = _jsonable(value)
    return out


def _row_score(row: dict[str, Any]) -> float:
    quality = _quality(row)
    aesthetic = _float_or_none(row.get("aesthetic_score"))
    aesthetic_term = 0.0 if aesthetic is None else max(0.0, min(1.0, (aesthetic - 4.0) / 5.0))
    return float(quality * 100.0 + aesthetic_term)


def _iter_annotation_rows(source: str, split: str) -> Iterable[dict[str, Any]]:
    path = Path(source)
    if path.exists():
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield json.loads(line)
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for uid, value in data.items():
                row = dict(value) if isinstance(value, dict) else {"score": value}
                row.setdefault("uid", uid)
                yield row
            return
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    yield row
            return
        raise ValueError(f"unsupported annotation JSON structure: {source}")

    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise SystemExit("Install datasets first: pip install datasets") from exc

    dataset = load_dataset(source, split=split, streaming=True)
    for row in dataset:
        yield dict(row)


def _select_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    reject_regex = re.compile(args.reject_regex, re.IGNORECASE) if args.reject_regex else None
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned = 0
    eligible = 0
    duplicate_uids = 0
    quality_distribution: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()
    schema_keys: Counter[str] = Counter()

    for row in _iter_annotation_rows(args.annotations, args.split):
        if args.scan_limit and scanned >= args.scan_limit:
            break
        scanned += 1
        schema_keys.update(row.keys())
        uid = _uid(row)
        if not uid:
            reject_counts["missing uid"] += 1
            continue
        if uid in seen:
            duplicate_uids += 1
            continue
        seen.add(uid)
        reasons = _reject_reasons(row, min_quality=args.min_quality, reject_regex=reject_regex)
        if reasons:
            reject_counts.update(reasons)
            continue
        eligible += 1
        quality_distribution[str(_quality(row))] += 1
        record = dict(row) if args.store_full_annotation else _compact_row(row)
        record.setdefault("uid", uid)
        record.setdefault("quality_score", _quality(row))
        record["source_rank_score"] = _row_score(row)

        if args.selection_mode == "first" or args.target <= 0:
            selected.append(record)
        elif len(selected) < args.target:
            selected.append(record)
        else:
            replacement_index = rng.randint(0, eligible - 1)
            if replacement_index < args.target:
                selected[replacement_index] = record

        if args.progress_every > 0 and scanned % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "scanned": scanned,
                        "eligible": eligible,
                        "selected": len(selected),
                        "top_reject_reasons": reject_counts.most_common(5),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.selection_mode == "first" and args.target > 0 and len(selected) >= args.target:
            break

    if args.selection_mode == "quality_ranked" and args.target > 0:
        selected.sort(key=lambda item: (float(item.get("source_rank_score", 0.0)), str(item.get("uid", ""))), reverse=True)
        selected = selected[: args.target]
    elif args.shuffle:
        rng.shuffle(selected)

    summary = {
        "annotations": args.annotations,
        "split": args.split,
        "scan_limit": int(args.scan_limit),
        "scanned": scanned,
        "eligible": eligible,
        "selected": len(selected),
        "target": int(args.target),
        "min_quality": int(args.min_quality),
        "selection_mode": args.selection_mode,
        "seed": int(args.seed),
        "duplicate_uids": duplicate_uids,
        "quality_distribution": dict(sorted(quality_distribution.items())),
        "top_reject_reasons": reject_counts.most_common(30),
        "top_schema_keys": schema_keys.most_common(80),
    }
    return selected, summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_shards(rows: list[dict[str, Any]], *, output_dir: Path, num_shards: int) -> list[dict[str, Any]]:
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_rows: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    for index, row in enumerate(rows):
        shard_rows[index % num_shards].append(row)
    summaries = []
    for shard_id, rows_for_shard in enumerate(shard_rows):
        shard_path = shard_dir / f"shard_{shard_id:04d}.jsonl"
        _write_jsonl(shard_path, rows_for_shard)
        summaries.append({"shard_id": shard_id, "path": str(shard_path), "count": len(rows_for_shard)})
    return summaries


def _write_worker_plan(output_dir: Path, shards: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    plan_path = output_dir / "worker_plan.jsonl"
    per_worker_target = max(1, int(args.worker_select_target))
    with plan_path.open("w", encoding="utf-8") as handle:
        for shard in shards:
            row = {
                "shard_id": shard["shard_id"],
                "annotations": shard["path"],
                "select_target": min(per_worker_target, int(shard["count"])),
                "scan_limit": int(shard["count"]),
                "min_quality": int(args.min_quality),
                "oversample_factor": 1,
                "paper_within_face_order": args.paper_within_face_order,
                "num_bins": int(args.num_bins),
                "target_faces": int(args.target_faces),
                "token_max_faces": int(args.token_max_faces),
                "model_max_faces": int(args.model_max_faces),
                "point_samples": int(args.point_samples),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return plan_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="cindyxl/ObjaversePlusPlus")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=600_000)
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--min-quality", type=int, default=2)
    parser.add_argument("--selection-mode", choices=["reservoir", "first", "quality_ranked"], default="reservoir")
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle selected rows before sharding.")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--worker-select-target", type=int, default=10_000)
    parser.add_argument("--reject-regex", default="", help="Optional regex over raw annotation values, e.g. 'scan|photogrammetry'.")
    parser.add_argument("--store-full-annotation", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--paper-within-face-order", default="rotate_min_zyx")
    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--target-faces", type=int, default=512)
    parser.add_argument("--token-max-faces", type=int, default=512)
    parser.add_argument("--model-max-faces", type=int, default=512)
    parser.add_argument("--point-samples", type=int, default=8192)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected, summary = _select_rows(args)
    selected_path = args.output_dir / "selected_annotations.jsonl"
    _write_jsonl(selected_path, selected)
    shards = _write_shards(selected, output_dir=args.output_dir, num_shards=args.num_shards)
    worker_plan = _write_worker_plan(args.output_dir, shards, args)

    summary.update(
        {
            "selected_annotations": str(selected_path),
            "num_shards": int(args.num_shards),
            "worker_plan": str(worker_plan),
            "shards": shards,
            "estimated_final_usable_at_55pct": int(round(len(selected) * 0.55)),
            "estimated_final_usable_at_65pct": int(round(len(selected) * 0.65)),
        }
    )
    summary_path = args.output_dir / "source_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
