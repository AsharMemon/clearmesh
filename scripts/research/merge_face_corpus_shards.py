#!/usr/bin/env python3
"""Merge parallel FACE corpus shards into one strict token dataset.

Production-scale corpus prep should be embarrassingly parallel: many workers
build independent Objaverse++ FACE corpora, then we merge their strict passing
token shards and create one global train/test split.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return stem.strip("._") or "shard"


def _source_path(row: dict[str, object], manifest_path: Path) -> Path:
    raw = Path(str(row["path"]))
    if raw.is_absolute() and raw.exists():
        return raw
    if raw.exists():
        return raw
    manifest_relative = manifest_path.parent / raw
    if manifest_relative.exists():
        return manifest_relative
    # Lean corpus archives are often extracted on a different machine than the
    # original shard worker, so manifests may preserve stale absolute paths.
    # The NPZ files are still co-located with the extracted manifest.
    basename_relative = manifest_path.parent / raw.name
    if basename_relative.exists():
        return basename_relative
    return manifest_relative


def _find_manifest(input_path: Path, source: str) -> Path:
    if input_path.is_file():
        return input_path
    candidates: list[Path] = []
    if source in {"auto", "tokens_pass"}:
        candidates.append(input_path / "tokens_pass" / "manifest.jsonl")
    if source in {"auto", "tokens"}:
        candidates.append(input_path / "tokens" / "manifest.jsonl")
    if source in {"auto", "split_pass"}:
        candidates.extend(
            [
                input_path / "split_pass" / "train" / "manifest.jsonl",
                input_path / "split_pass" / "test" / "manifest.jsonl",
            ]
        )
    if source in {"auto", "manifest"}:
        candidates.append(input_path / "manifest.jsonl")
    existing = [path for path in candidates if path.exists()]
    if existing:
        return existing[0]
    raise FileNotFoundError(f"no FACE manifest found for {input_path} with source={source}")


def _copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def _dedup_key(row: dict[str, object], src: Path) -> str:
    for key in ("uid", "objaverse_uid", "source_uid", "sha256", "sha256sum"):
        value = row.get(key)
        if value:
            return f"{key}:{value}"
    return f"path:{src.resolve()}"


def _merge_inputs(
    inputs: list[Path],
    *,
    output_tokens_dir: Path,
    source: str,
    copy_mode: str,
    dedupe: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    seen: set[str] = set()
    source_summaries = []
    missing: list[str] = []
    duplicates = 0
    name_counts: dict[str, int] = {}

    for input_index, input_path in enumerate(inputs):
        manifest = _find_manifest(input_path, source)
        rows = _read_jsonl(manifest)
        prefix = _safe_stem(input_path.name if input_path.is_dir() else manifest.parent.name)
        accepted = 0
        for row_index, row in enumerate(rows):
            src = _source_path(row, manifest)
            if not src.exists():
                missing.append(str(src))
                continue
            key = _dedup_key(row, src)
            if dedupe and key in seen:
                duplicates += 1
                continue
            seen.add(key)

            base_name = f"{prefix}__{src.name}"
            if base_name in name_counts:
                name_counts[base_name] += 1
                base_name = f"{prefix}__{name_counts[base_name]:06d}__{src.name}"
            else:
                name_counts[base_name] = 0
            dst = output_tokens_dir / base_name
            _copy_file(src, dst, copy_mode)
            updated = dict(row)
            updated.update(
                {
                    "path": str(dst),
                    "merge_input_index": input_index,
                    "merge_row_index": row_index,
                    "merge_source_manifest": str(manifest),
                    "merge_source_path": str(src),
                }
            )
            rows_out.append(updated)
            accepted += 1
        source_summaries.append(
            {
                "input": str(input_path),
                "manifest": str(manifest),
                "input_count": len(rows),
                "accepted_count": accepted,
            }
        )

    summary = {
        "source_count": len(inputs),
        "sources": source_summaries,
        "merged_count": len(rows_out),
        "missing_count": len(missing),
        "missing": missing[:100],
        "duplicate_count": duplicates,
        "dedupe": bool(dedupe),
    }
    return rows_out, summary


def _split_rows(
    rows: list[dict[str, object]],
    *,
    test_count: int,
    test_ratio: float,
    seed: int,
    shuffle: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(rows) < 2:
        raise SystemExit("need at least two merged rows for train/test split")
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    selected_test_count = int(test_count) if test_count > 0 else max(1, round(len(rows) * test_ratio))
    selected_test_count = min(max(1, selected_test_count), len(rows) - 1)
    test_indices = set(order[-selected_test_count:])
    train = [row for idx, row in enumerate(rows) if idx not in test_indices]
    test = [row for idx, row in enumerate(rows) if idx in test_indices]
    return train, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="inputs", type=Path, action="append", default=[], help="Corpus root or manifest. Repeatable.")
    parser.add_argument("--input-list", type=Path, default=None, help="Text file of corpus roots/manifests, one per line.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", choices=["auto", "tokens_pass", "tokens", "split_pass", "manifest"], default="auto")
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="copy")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument("--test-ratio", type=float, default=0.02, help="Default 2%% holdout for large production corpora.")
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    inputs = list(args.inputs)
    if args.input_list:
        inputs.extend(Path(line.strip()) for line in args.input_list.read_text(encoding="utf-8").splitlines() if line.strip())
    if not inputs:
        raise SystemExit("provide at least one --input or --input-list")

    output = args.output_dir
    tokens_dir = output / "tokens_pass"
    split_dir = output / "split_pass"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    rows, merge_summary = _merge_inputs(
        inputs,
        output_tokens_dir=tokens_dir,
        source=args.source,
        copy_mode=args.copy_mode,
        dedupe=not args.no_dedupe,
    )
    _write_jsonl(tokens_dir / "manifest.jsonl", rows)

    train_rows, test_rows = _split_rows(
        rows,
        test_count=args.test_count,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    _write_jsonl(split_dir / "train" / "manifest.jsonl", train_rows)
    _write_jsonl(split_dir / "test" / "manifest.jsonl", test_rows)

    summary = {
        "output_dir": str(output),
        "tokens_manifest": str(tokens_dir / "manifest.jsonl"),
        "split_dir": str(split_dir),
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "test_ratio": float(args.test_ratio),
        "test_count_requested": int(args.test_count),
        "seed": int(args.seed),
        "copy_mode": args.copy_mode,
        **merge_summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
