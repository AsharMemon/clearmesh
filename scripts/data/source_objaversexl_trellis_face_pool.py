#!/usr/bin/env python3
"""Build a sharded Objaverse-XL/TRELLIS source pool for FACE corpus workers.

This does not download meshes. It reads the TRELLIS-500K ObjaverseXL metadata
CSVs, applies a lightweight metadata/aesthetic prefilter, excludes source IDs
that are already represented in durable B2 corpora, and writes JSONL shards for
the FACE queue worker with SOURCE_KIND=objaversexl.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


TRELLIS_500K_REPO = "JeffreyXiang/TRELLIS-500K"
CSV_FILES = {
    "sketchfab": "ObjaverseXL_sketchfab.csv",
    "github": "ObjaverseXL_github.csv",
}
OBJAVERSE_UID_RE = re.compile(r"([0-9a-f]{32})(?:\?|$|#)", re.IGNORECASE)


def _normalize_id(raw: str) -> str:
    text = raw.strip().lower()
    if ":" in text:
        text = text.split(":", 1)[1]
    return text


def _load_id_set(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {_normalize_id(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _extract_objaverse_uid(file_identifier: Any) -> str:
    if not isinstance(file_identifier, str):
        return ""
    match = OBJAVERSE_UID_RE.search(file_identifier)
    if match:
        return match.group(1).lower()
    last = file_identifier.rstrip("/").split("/")[-1]
    if len(last) >= 32:
        candidate = last[-32:].lower()
        if re.fullmatch(r"[0-9a-f]{32}", candidate):
            return candidate
    return ""


def _read_trellis_csv(source: str, cache_dir: Path) -> pd.DataFrame:
    csv_name = CSV_FILES[source]
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / csv_name
    if local_path.exists():
        df = pd.read_csv(local_path)
    else:
        df = pd.read_csv(f"hf://datasets/{TRELLIS_500K_REPO}/{csv_name}")
        df.to_csv(local_path, index=False)
    df["trellis_source"] = source
    return df


def _quality_from_aesthetic(score: float | None, superior_threshold: float) -> int:
    if score is None:
        return 1
    return 3 if score >= superior_threshold else 2


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exclude_ids = _load_id_set(args.exclude_id_list)
    frames = [_read_trellis_csv(source, args.cache_dir) for source in args.sources]
    df = pd.concat(frames, ignore_index=True)
    scanned = len(df)

    df = df[df["sha256"].notna() & (df["sha256"] != "")].copy()
    df = df[df["file_identifier"].notna() & (df["file_identifier"] != "")].copy()
    df = df.drop_duplicates(subset="sha256", keep="first")
    after_basic = len(df)

    rows: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for record in df.to_dict("records"):
        sha = str(record.get("sha256") or "").strip().lower()
        if not sha:
            reject_counts["missing sha256"] += 1
            continue
        objaverse_uid = _extract_objaverse_uid(record.get("file_identifier"))
        if sha in exclude_ids or (objaverse_uid and objaverse_uid in exclude_ids):
            reject_counts["excluded existing source id"] += 1
            continue
        try:
            aesthetic = float(record.get("aesthetic_score")) if pd.notna(record.get("aesthetic_score")) else None
        except (TypeError, ValueError):
            aesthetic = None
        if aesthetic is None:
            if args.require_aesthetic:
                reject_counts["missing aesthetic"] += 1
                continue
        elif aesthetic < args.min_aesthetic:
            reject_counts["aesthetic below threshold"] += 1
            continue
        trellis_source = str(record.get("trellis_source") or "unknown")
        row = {
            "uid": sha,
            "sha256": sha,
            "source_dataset": "ObjaverseXL",
            "source_kind": "objaversexl",
            "trellis_source": trellis_source,
            "file_identifier": str(record.get("file_identifier") or ""),
            "aesthetic_score": aesthetic,
            "quality_score": _quality_from_aesthetic(aesthetic, args.superior_aesthetic),
        }
        if objaverse_uid:
            row["objaverse_uid"] = objaverse_uid
        captions = record.get("captions")
        if isinstance(captions, str) and captions:
            row["captions"] = captions
        for key in ("license", "category", "tags"):
            if key in record and pd.notna(record[key]):
                row[key] = _jsonable(record[key])
        rows.append(row)
        source_counts[trellis_source] += 1

    if args.selection_mode == "aesthetic_ranked":
        rows.sort(
            key=lambda item: (
                float(item["aesthetic_score"]) if item.get("aesthetic_score") is not None else -1.0,
                str(item.get("uid", "")),
            ),
            reverse=True,
        )
    elif args.selection_mode == "sha":
        rows.sort(key=lambda item: str(item.get("uid", "")))

    if args.target > 0:
        rows = rows[: args.target]

    summary = {
        "repo": TRELLIS_500K_REPO,
        "sources": args.sources,
        "scanned": scanned,
        "after_basic_filter": after_basic,
        "exclude_id_count": len(exclude_ids),
        "selected_rows": len(rows),
        "target": int(args.target),
        "min_aesthetic": float(args.min_aesthetic),
        "selection_mode": args.selection_mode,
        "source_counts": dict(sorted(source_counts.items())),
        "top_reject_reasons": reject_counts.most_common(30),
    }
    return rows, summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_shards(rows: list[dict[str, Any]], output_dir: Path, shard_size: int) -> list[dict[str, Any]]:
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for shard_id, start in enumerate(range(0, len(rows), max(1, shard_size))):
        shard_rows = rows[start : start + max(1, shard_size)]
        shard_path = shard_dir / f"shard_{shard_id:04d}.jsonl"
        _write_jsonl(shard_path, shard_rows)
        summaries.append({"shard_id": shard_id, "path": str(shard_path), "count": len(shard_rows)})
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", choices=sorted(CSV_FILES), default=["sketchfab", "github"])
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--exclude-id-list", type=Path, default=None)
    parser.add_argument("--target", type=int, default=0, help="Max selected rows; 0 keeps all rows.")
    parser.add_argument("--shard-size", type=int, default=5990)
    parser.add_argument("--min-aesthetic", type=float, default=5.5)
    parser.add_argument("--superior-aesthetic", type=float, default=7.0)
    parser.add_argument("--require-aesthetic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selection-mode", choices=["aesthetic_ranked", "sha"], default="aesthetic_ranked")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = args.cache_dir or args.output_dir / "metadata_cache"
    rows, summary = _build_rows(args)
    selected_path = args.output_dir / "source_candidates.jsonl"
    _write_jsonl(selected_path, rows)
    shards = _write_shards(rows, args.output_dir, args.shard_size)
    summary.update(
        {
            "output_dir": str(args.output_dir),
            "manifest": str(selected_path),
            "shard_size": int(args.shard_size),
            "shards": len(shards),
            "shard_counts": [int(row["count"]) for row in shards],
        }
    )
    (args.output_dir / "pool_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
