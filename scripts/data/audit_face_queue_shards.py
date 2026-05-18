#!/usr/bin/env python3
"""Audit FACE shard queue launch metadata for duplicate work assignments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit should keep scanning.
        return {"_path": str(path), "_error": f"{type(exc).__name__}: {exc}"}


def _iter_queue_info(root: Path):
    for path in sorted(root.glob("*/queue_info.json")):
        payload = _load_json(path)
        payload["_path"] = str(path)
        yield payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".codex_outputs/face_corpus_queue_setup"),
        help="Directory containing per-launch queue_info.json files.",
    )
    parser.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Optional active Thunder instance ids to audit. If omitted, all launch metadata is scanned.",
    )
    parser.add_argument(
        "--source-kind",
        default="",
        help="Optional source kind filter, e.g. texverse or objaversexl.",
    )
    parser.add_argument(
        "--latest-per-instance",
        action="store_true",
        help="Only audit the most recently modified queue_info.json per instance after filtering.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    active_instances = {str(item) for item in args.instances or []}
    wanted_source = args.source_kind.lower().strip()
    candidates: list[dict[str, Any]] = []
    for info in _iter_queue_info(args.root):
        if "_error" in info:
            candidates.append(info)
            continue
        instance = str(info.get("instance_id", ""))
        source_kind = str(info.get("source_kind", "")).lower()
        if active_instances and instance not in active_instances:
            continue
        if wanted_source and source_kind != wanted_source:
            continue
        candidates.append(info)

    if args.latest_per_instance:
        latest: dict[str, dict[str, Any]] = {}
        for info in candidates:
            if "_error" in info:
                continue
            instance = str(info.get("instance_id", ""))
            path = Path(str(info.get("_path", "")))
            mtime = path.stat().st_mtime if path.exists() else 0.0
            prev = latest.get(instance)
            if prev is None:
                info["_mtime"] = mtime
                latest[instance] = info
                continue
            prev_path = Path(str(prev.get("_path", "")))
            prev_mtime = prev_path.stat().st_mtime if prev_path.exists() else 0.0
            if mtime > prev_mtime:
                info["_mtime"] = mtime
                latest[instance] = info
        candidates = list(latest.values())

    assignments: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    scanned = 0
    skipped = 0

    for info in candidates:
        if "_error" in info:
            skipped += 1
            continue
        instance = str(info.get("instance_id", ""))
        source_kind = str(info.get("source_kind", "")).lower()
        if active_instances and instance not in active_instances:
            continue
        if wanted_source and source_kind != wanted_source:
            continue
        scanned += 1
        prefix_root = str(info.get("b2_prefix_root", ""))
        for shard_id in str(info.get("queue_shard_ids", "")).split():
            key = (source_kind, prefix_root, shard_id)
            assignments[key].append(
                {
                    "instance_id": instance,
                    "queue_info": str(info.get("_path", "")),
                    "run_stamp_prefix": str(info.get("run_stamp_prefix", "")),
                }
            )

    duplicates = {
        "|".join(key): values
        for key, values in sorted(assignments.items())
        if len({value["instance_id"] for value in values}) > 1
    }
    result = {
        "root": str(args.root),
        "instances": sorted(active_instances),
        "source_kind": wanted_source,
        "latest_per_instance": bool(args.latest_per_instance),
        "scanned_queue_infos": scanned,
        "skipped_bad_json": skipped,
        "assignment_count": len(assignments),
        "duplicate_assignment_count": len(duplicates),
        "duplicates": duplicates,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"scanned_queue_infos={scanned} assignment_count={len(assignments)} duplicate_assignment_count={len(duplicates)}")
        if duplicates:
            for key, values in duplicates.items():
                print(f"DUPLICATE {key}")
                for value in values:
                    print(f"  instance={value['instance_id']} run={value['run_stamp_prefix']} info={value['queue_info']}")
    return 1 if duplicates else 0


if __name__ == "__main__":
    raise SystemExit(main())
