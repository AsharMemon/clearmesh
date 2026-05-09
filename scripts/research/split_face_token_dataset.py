#!/usr/bin/env python3
"""Split FACE-token NPZ shards into train/test dataset directories."""

from __future__ import annotations

import argparse
import json
import re
import random
import shutil
from collections import defaultdict
from pathlib import Path


def _read_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _source_path(row: dict[str, object], manifest_path: Path) -> Path:
    raw = Path(str(row["path"]))
    if raw.is_absolute() or raw.exists():
        return raw
    return manifest_path.parent / raw


def _write_split(rows: list[dict[str, object]], *, manifest_path: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = output_dir / "manifest.jsonl"
    written: list[str] = []
    with manifest_out.open("w", encoding="utf-8") as handle:
        for row in rows:
            src = _source_path(row, manifest_path)
            if not src.exists():
                raise FileNotFoundError(f"manifest shard does not exist: {src}")
            dst = output_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            updated = dict(row)
            updated["path"] = str(dst)
            handle.write(json.dumps(updated, sort_keys=True) + "\n")
            written.append(str(dst))
    return written


def _group_key(row: dict[str, object], *, field: str, regex: str | None, manifest_path: Path) -> str:
    if field == "path_stem":
        raw = _source_path(row, manifest_path).stem
    else:
        raw = str(row.get(field) or _source_path(row, manifest_path).stem)
    if regex:
        match = re.match(regex, raw)
        if match:
            if "group" in match.groupdict():
                return str(match.group("group"))
            if match.groups():
                return str(match.group(1))
    return raw


def _split_rows(
    rows: list[dict[str, object]],
    *,
    manifest_path: Path,
    test_count: int,
    test_ratio: float,
    seed: int,
    shuffle: bool,
    group_field: str | None,
    group_regex: str | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if not group_field:
        order = list(range(len(rows)))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(order)
        if test_count > 0:
            selected_test_count = test_count
        else:
            selected_test_count = max(1, round(len(rows) * test_ratio))
        selected_test_count = min(max(1, int(selected_test_count)), len(rows) - 1)
        test_indices = set(order[-selected_test_count:])
        train_rows = [row for idx, row in enumerate(rows) if idx not in test_indices]
        test_rows = [row for idx, row in enumerate(rows) if idx in test_indices]
        return train_rows, test_rows, {
            "split_mode": "row",
            "train_groups": [],
            "test_groups": [],
            "group_count": 0,
            "test_group_count": 0,
        }

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row, field=group_field, regex=group_regex, manifest_path=manifest_path)].append(row)
    groups = sorted(grouped)
    if len(groups) < 2:
        raise SystemExit("need at least two groups for a grouped split")
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(groups)
    if test_count > 0:
        selected_test_groups = test_count
    else:
        selected_test_groups = max(1, round(len(groups) * test_ratio))
    selected_test_groups = min(max(1, int(selected_test_groups)), len(groups) - 1)
    test_groups = set(groups[-selected_test_groups:])
    train_group_names = [group for group in sorted(grouped) if group not in test_groups]
    test_group_names = [group for group in sorted(grouped) if group in test_groups]
    train_rows = [row for group in train_group_names for row in grouped[group]]
    test_rows = [row for group in test_group_names for row in grouped[group]]
    return train_rows, test_rows, {
        "split_mode": "group",
        "group_field": group_field,
        "group_regex": group_regex,
        "group_count": len(grouped),
        "test_group_count": len(test_group_names),
        "train_groups": train_group_names,
        "test_groups": test_group_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--group-field",
        default="",
        help="Optional manifest field used for grouped splitting. Use path_stem to group by NPZ stem.",
    )
    parser.add_argument(
        "--group-regex",
        default="",
        help="Optional regex applied to the group field. Use a named group 'group' or the first capture group.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy"],
        default="copy",
        help="Reserved for future symlink/hardlink modes; copy is safest across machines.",
    )
    args = parser.parse_args()

    manifest_path = args.dataset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    rows = _read_manifest(manifest_path)
    if len(rows) < 2:
        raise SystemExit("need at least two shards for a train/test split")
    train_rows, test_rows, split_info = _split_rows(
        rows,
        manifest_path=manifest_path,
        test_count=args.test_count,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle=args.shuffle,
        group_field=args.group_field or None,
        group_regex=args.group_regex or None,
    )

    train_written = _write_split(train_rows, manifest_path=manifest_path, output_dir=args.output_dir / "train")
    test_written = _write_split(test_rows, manifest_path=manifest_path, output_dir=args.output_dir / "test")
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "seed": int(args.seed),
        "shuffle": bool(args.shuffle),
        "total": len(rows),
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "train_manifest": str(args.output_dir / "train" / "manifest.jsonl"),
        "test_manifest": str(args.output_dir / "test" / "manifest.jsonl"),
        "train_paths": train_written,
        "test_paths": test_written,
        **split_info,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
