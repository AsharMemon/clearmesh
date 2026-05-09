#!/usr/bin/env python3
"""Create a FACE token split with globally unique paper-token hashes.

The FACE training gate intentionally fails when identical tokenized meshes leak
across train/test. Objaverse-style sources can contain duplicates even after
quality filtering, so this helper deduplicates by the same paper-token hash used
by ``check_face_token_leakage.py`` and writes a fresh train/test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
    return rows


def _resolve_manifest_path(row: dict[str, Any], manifest_path: Path) -> Path:
    raw = Path(str(row.get("path") or ""))
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw
    manifest_relative = manifest_path.parent / raw
    if manifest_relative.exists():
        return manifest_relative
    basename_relative = manifest_path.parent / raw.name
    if basename_relative.exists():
        return basename_relative
    return manifest_relative


def _load_rows(input_dir: Path | None, manifest: Path | None) -> list[dict[str, Any]]:
    if input_dir is None and manifest is None:
        raise SystemExit("pass --input-dir or --manifest")
    if input_dir is not None and manifest is not None:
        raise SystemExit("pass only one of --input-dir or --manifest")

    if manifest is None:
        assert input_dir is not None
        manifest = input_dir / "manifest.jsonl"
        if not manifest.exists():
            rows = [{"path": str(path)} for path in sorted(input_dir.glob("*.npz"))]
            for row in rows:
                row["_resolved_path"] = row["path"]
            return rows

    rows = _iter_jsonl(manifest)
    for row in rows:
        row["_resolved_path"] = str(_resolve_manifest_path(row, manifest))
    return rows


def _load_token_hash(path: Path) -> tuple[str, int, tuple[int, ...]]:
    data = np.load(path)
    if "paper_tokens" not in data.files:
        raise ValueError(f"{path} is missing paper_tokens")
    tokens = np.asarray(data["paper_tokens"], dtype="<i8", order="C")
    if tokens.ndim != 2 or tokens.shape[1] != 9:
        raise ValueError(f"{path} has invalid paper_tokens shape {tokens.shape}")
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0]) if "num_bins" in data.files else 0

    digest = hashlib.sha256()
    digest.update(str(num_bins).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(tokens.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tokens.tobytes())
    return digest.hexdigest(), num_bins, tuple(int(x) for x in tokens.shape)


def _copy_sample(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:  # pragma: no cover - argparse choices prevent this.
        raise ValueError(f"unknown copy mode {mode}")


def _manifest_path(dst: Path, output_dir: Path, path_mode: str) -> str:
    if path_mode == "absolute":
        return str(dst)
    if path_mode == "relative":
        return os.path.relpath(dst, output_dir)
    raise ValueError(f"unknown path mode {path_mode}")  # pragma: no cover


def _write_split(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    split_name: str,
    copy_mode: str,
    path_mode: str,
) -> dict[str, Any]:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.jsonl"
    written = []

    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            src = Path(str(row["_resolved_path"]))
            # Prefix the original basename with the split-local row index. This
            # avoids accidental overwrites if two source manifests contain the
            # same basename but different content.
            dst = split_dir / f"{idx:07d}_{src.name}"
            _copy_sample(src, dst, copy_mode)
            out_row = {key: value for key, value in row.items() if not key.startswith("_")}
            out_row["path"] = _manifest_path(dst, output_dir, path_mode)
            out_row["source_path"] = str(src)
            handle.write(json.dumps(out_row, sort_keys=True) + "\n")
            written.append(str(dst))

    return {
        "count": len(rows),
        "dir": str(split_dir),
        "manifest": str(manifest_path),
        "paths_preview": written[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--test-count", type=int, default=0, help="Explicit test count; overrides --test-ratio when >0.")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--path-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.test_ratio < 1.0:
        raise SystemExit("--test-ratio must be between 0 and 1")

    rows = _load_rows(args.input_dir, args.manifest)
    unique_rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicate_examples: list[dict[str, str]] = []
    skipped = []

    for row in rows:
        path = Path(str(row["_resolved_path"]))
        try:
            token_hash, num_bins, token_shape = _load_token_hash(path)
        except Exception as exc:  # noqa: BLE001 - report bad corpus rows.
            skipped.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if token_hash in seen:
            if len(duplicate_examples) < 20:
                duplicate_examples.append({"token_hash": token_hash, "kept": seen[token_hash], "dropped": str(path)})
            continue
        out_row = dict(row)
        out_row["token_hash"] = token_hash
        out_row["num_bins"] = int(num_bins)
        out_row["token_shape"] = list(token_shape)
        unique_rows.append(out_row)
        seen[token_hash] = str(path)

    if len(unique_rows) < 2:
        raise SystemExit(f"need at least 2 unique token rows, got {len(unique_rows)}")

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(unique_rows)

    test_count = int(args.test_count) if args.test_count > 0 else int(round(len(unique_rows) * args.test_ratio))
    test_count = max(1, min(test_count, len(unique_rows) - 1))
    test_rows = unique_rows[:test_count]
    train_rows = unique_rows[test_count:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_summary = _write_split(
        train_rows,
        output_dir=args.output_dir,
        split_name="train",
        copy_mode=args.copy_mode,
        path_mode=args.path_mode,
    )
    test_summary = _write_split(
        test_rows,
        output_dir=args.output_dir,
        split_name="test",
        copy_mode=args.copy_mode,
        path_mode=args.path_mode,
    )
    summary = {
        "input_count": len(rows),
        "unique_token_hashes": len(unique_rows),
        "removed_duplicates": len(rows) - len(unique_rows) - len(skipped),
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:20],
        "duplicate_examples": duplicate_examples,
        "seed": args.seed,
        "shuffle": not args.no_shuffle,
        "test_ratio": args.test_ratio,
        "test_count": test_count,
        "copy_mode": args.copy_mode,
        "path_mode": args.path_mode,
        "train": train_summary,
        "test": test_summary,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    (args.output_dir / "split_summary.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
