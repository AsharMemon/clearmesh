#!/usr/bin/env python3
"""Download a curated Objaverse++ candidate subset for FACE training.

This script uses Objaverse++ annotations to choose promising standalone,
high-quality assets before spending bandwidth/storage on raw Objaverse files.
It is deliberately schema-tolerant because annotation column names can vary
between JSON exports and HuggingFace dataset revisions.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import signal
import sys
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

UID_KEYS = ("uid", "UID", "objaverse_uid", "Objaverse UID", "object_uid")
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


def _reasons(row: dict[str, Any], min_quality: int) -> list[str]:
    reasons = []
    quality = _quality(row)
    if quality < min_quality:
        reasons.append(f"quality {quality} < {min_quality}")
    for key in REJECT_KEYS:
        if _truthy(row.get(key)):
            reasons.append(f"reject flag: {key}")
    return reasons


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


def _select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = []
    scanned = 0
    for row in _iter_annotation_rows(args.annotations, args.split):
        scanned += 1
        uid = _uid(row)
        if not uid:
            continue
        if _reasons(row, args.min_quality):
            continue
        selected.append({"uid": uid, "quality_score": _quality(row), "annotation": row})
        if args.scan_limit and scanned >= args.scan_limit:
            break
        if args.target and len(selected) >= args.target * max(1, args.oversample_factor):
            break
    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(selected)
    if args.target:
        selected = selected[: args.target]
    return selected


def _download_batch_worker(batch: list[str], processes: int, queue: Any) -> None:
    """Run one Objaverse batch in an isolated process so it can be killed on hangs."""
    try:
        if hasattr(os, "setsid"):
            os.setsid()
    except Exception:
        pass
    try:
        import objaverse

        paths = objaverse.load_objects(batch, download_processes=processes)
        queue.put({"ok": True, "paths": {str(uid): str(path) for uid, path in paths.items()}})
    except BaseException as exc:  # noqa: BLE001 - isolate failures from parent.
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _load_batch(batch: list[str], processes: int, batch_timeout_seconds: int) -> dict[str, str]:
    if batch_timeout_seconds <= 0:
        import objaverse

        return {str(uid): str(path) for uid, path in objaverse.load_objects(batch, download_processes=processes).items()}

    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    queue: Any = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_download_batch_worker, args=(batch, processes, queue))
    proc.start()
    proc.join(batch_timeout_seconds)
    if proc.is_alive():
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.terminate()
        else:
            proc.terminate()
        proc.join(10)
        raise TimeoutError(f"download batch timed out after {batch_timeout_seconds}s")
    if proc.exitcode not in (0, None) and queue.empty():
        raise RuntimeError(f"download batch subprocess exited with code {proc.exitcode}")
    if queue.empty():
        return {}
    result = queue.get()
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error", "unknown batch error")))
    return dict(result.get("paths", {}))


def _download(
    selected: list[dict[str, Any]],
    output_dir: Path,
    processes: int,
    batch_size: int,
    batch_timeout_seconds: int,
) -> dict[str, str]:
    try:
        import objaverse
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise SystemExit("Install objaverse first: pip install objaverse") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".objaverse_cache"
    hf_cache = output_dir / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    home_cache = Path.home() / ".objaverse"
    if home_cache.is_symlink():
        # A previous Thunder run may have left ~/.objaverse pointing at a
        # deleted per-run cache. pathlib.exists() is false for broken
        # symlinks, but symlink_to() would still raise FileExistsError.
        if not home_cache.resolve(strict=False).exists():
            home_cache.unlink()
            home_cache.symlink_to(cache_dir)
    elif not home_cache.exists():
        home_cache.symlink_to(cache_dir)
    print(f"Objaverse cache: {home_cache} -> {home_cache.resolve(strict=False)}")

    manifest: dict[str, str] = {}
    uids = [row["uid"] for row in selected]
    for start in range(0, len(uids), max(1, batch_size)):
        batch = uids[start : start + max(1, batch_size)]
        batch_index = start // max(1, batch_size)
        print(
            f"starting download batch {batch_index} ({len(batch)} objects, timeout={batch_timeout_seconds}s)",
            flush=True,
        )
        try:
            paths = _load_batch(batch, processes, batch_timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - keep partial progress.
            print(f"download batch {batch_index} failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue
        downloaded = 0
        for uid, path in paths.items():
            if path and Path(path).exists():
                manifest[str(uid)] = str(path)
                downloaded += 1
        print(
            f"completed download batch {batch_index}: downloaded={downloaded} manifest_total={len(manifest)}",
            flush=True,
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="cindyxl/ObjaversePlusPlus", help="HF dataset id or local JSON/JSONL annotations.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--min-quality", type=int, default=2)
    parser.add_argument("--oversample-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--batch-timeout-seconds",
        type=int,
        default=0,
        help="Kill and skip a download batch if objaverse.load_objects hangs. 0 disables timeout.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_rows(args)
    selected_path = args.output_dir / "selected_annotations.jsonl"
    with selected_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {}
    if args.download and selected:
        manifest = _download(
            selected,
            args.output_dir / "downloads",
            args.processes,
            args.batch_size,
            args.batch_timeout_seconds,
        )
        (args.output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        candidates_path = args.output_dir / "downloaded_candidates.json"
        candidates = []
        selected_by_uid = {row["uid"]: row for row in selected}
        for uid, path in manifest.items():
            row = selected_by_uid.get(uid, {"quality_score": -1, "annotation": {}})
            candidates.append({"uid": uid, "path": path, "quality_score": row.get("quality_score", -1), **row.get("annotation", {})})
        candidates_path.write_text(json.dumps(candidates, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "selected": len(selected),
        "downloaded": len(manifest),
        "selected_annotations": str(selected_path),
        "download_manifest": str(args.output_dir / "download_manifest.json") if args.download else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
