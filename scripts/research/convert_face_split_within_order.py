#!/usr/bin/env python3
"""Convert an existing FACE paper-token split to another within-face order.

The FACE paper specifies global face ordering by each triangle's minimum ZYX
vertex, but does not spell out the cyclic order of the three vertices inside a
triangle. For the rotate-min hypothesis, we keep face order and winding, then
cyclically rotate each face so slot 0 is the minimum ZYX vertex.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_path(row: dict[str, Any], manifest_path: Path) -> Path:
    raw = Path(str(row.get("path") or ""))
    if raw.is_absolute() and raw.exists():
        return raw
    candidates = [
        raw,
        manifest_path.parent / raw,
        manifest_path.parent / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"manifest shard does not exist: {raw}")


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.asarray(array, dtype="<i8", order="C")
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _min_offsets_zyx(tokens: np.ndarray) -> np.ndarray:
    faces = np.asarray(tokens, dtype=np.int64).reshape(-1, 3, 3)
    return np.asarray(
        [np.lexsort((face[:, 2], face[:, 1], face[:, 0]))[0] for face in faces],
        dtype=np.int64,
    )


def _starts_min_ratio(tokens: np.ndarray) -> float:
    if len(tokens) == 0:
        return 1.0
    return float(np.mean(_min_offsets_zyx(tokens) == 0))


def _rotate_min_zyx(tokens: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(tokens)
    faces = np.asarray(source, dtype=np.int64).reshape(-1, 3, 3).copy()
    offsets = _min_offsets_zyx(source)
    for index, offset in enumerate(offsets):
        if int(offset):
            faces[index] = np.roll(faces[index], -int(offset), axis=0)
    converted = faces.reshape(-1, 9).astype(source.dtype, copy=False)
    return converted, {
        "faces": int(len(faces)),
        "starts_min_before": _starts_min_ratio(source),
        "starts_min_after": _starts_min_ratio(converted),
        "rotation_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(offsets, return_counts=True))},
        "changed_faces": int(np.sum(offsets != 0)),
    }


def _convert_npz(src: Path, dst: Path, *, target_order: str) -> dict[str, Any]:
    data = np.load(src, allow_pickle=False)
    arrays = {key: data[key] for key in data.files}
    if "paper_tokens" not in arrays:
        raise ValueError(f"{src} is missing paper_tokens")
    original = np.asarray(arrays["paper_tokens"])
    if original.ndim != 2 or original.shape[1] != 9:
        raise ValueError(f"{src} has invalid paper_tokens shape {original.shape}")
    if target_order != "rotate_min_zyx":
        raise ValueError(f"unsupported target order: {target_order}")

    converted, stats = _rotate_min_zyx(original)
    source_order = str(np.asarray(arrays.get("paper_within_face_order", ["unknown"])).reshape(-1)[0])
    arrays["paper_tokens"] = converted
    arrays["paper_within_face_order"] = np.asarray([target_order])
    arrays["paper_within_face_order_source"] = np.asarray([source_order])
    arrays["paper_tokens_source_sha256"] = np.asarray([_sha256_array(original)])
    arrays["paper_tokens_converted_sha256"] = np.asarray([_sha256_array(converted)])
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **arrays)
    return {
        "source": str(src),
        "path": dst.name,
        "target": str(dst),
        "source_order": source_order,
        "target_order": target_order,
        **stats,
    }


def _convert_split_dir(input_dir: Path, output_dir: Path, *, target_order: str) -> dict[str, Any]:
    manifest = input_dir / "manifest.jsonl"
    rows = _read_jsonl(manifest)
    if rows:
        source_paths = [_resolve_path(row, manifest) for row in rows]
    else:
        source_paths = sorted(input_dir.glob("*.npz"))
    if not source_paths:
        raise SystemExit(f"no npz shards found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    converted_rows = []
    summaries = []
    for row, src in zip(rows or [{} for _ in source_paths], source_paths):
        dst = output_dir / src.name
        summary = _convert_npz(src, dst, target_order=target_order)
        updated = dict(row)
        updated["path"] = dst.name
        updated["paper_within_face_order"] = target_order
        updated["paper_within_face_order_source"] = summary["source_order"]
        converted_rows.append(updated)
        summaries.append(summary)

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in converted_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "count": len(summaries),
        "starts_min_before_mean": float(np.mean([row["starts_min_before"] for row in summaries])),
        "starts_min_after_mean": float(np.mean([row["starts_min_after"] for row in summaries])),
        "changed_faces": int(sum(row["changed_faces"] for row in summaries)),
        "examples": summaries[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-split-dir", type=Path, required=True)
    parser.add_argument("--output-split-dir", type=Path, required=True)
    parser.add_argument("--target-order", choices=["rotate_min_zyx"], default="rotate_min_zyx")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not (args.input_split_dir / "train").is_dir() or not (args.input_split_dir / "test").is_dir():
        raise SystemExit(f"expected train/test dirs under {args.input_split_dir}")
    if args.output_split_dir.exists() and any(args.output_split_dir.iterdir()):
        if not args.force:
            raise SystemExit(f"{args.output_split_dir} is not empty; pass --force to replace it")
        shutil.rmtree(args.output_split_dir)

    summary = {
        "input_split_dir": str(args.input_split_dir),
        "output_split_dir": str(args.output_split_dir),
        "target_order": args.target_order,
        "train": _convert_split_dir(args.input_split_dir / "train", args.output_split_dir / "train", target_order=args.target_order),
        "test": _convert_split_dir(args.input_split_dir / "test", args.output_split_dir / "test", target_order=args.target_order),
    }
    if (args.input_split_dir / "split_summary.json").exists():
        shutil.copy2(args.input_split_dir / "split_summary.json", args.output_split_dir / "split_summary.source.json")
    (args.output_split_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
