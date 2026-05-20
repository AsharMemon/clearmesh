#!/usr/bin/env python3
"""Download TexVerse GLB candidates from a sharded FACE source manifest."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from source_skip_registry import filter_excluded_rows, load_source_skip_ids


def _iter_jsonl(path: Path):
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc


def _select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = list(_iter_jsonl(args.source_manifest))
    if args.scan_limit:
        rows = rows[: args.scan_limit]
    rows = [row for row in rows if int(row.get("quality_score", -1)) >= args.min_quality]
    if args.max_size_mb > 0:
        max_size_bytes = int(args.max_size_mb * 1024 * 1024)
        rows = [
            row
            for row in rows
            if int(row.get("hf_size_bytes") or 0) <= 0 or int(row.get("hf_size_bytes") or 0) <= max_size_bytes
        ]
    skip_ids = load_source_skip_ids(args.exclude_source_ids)
    rows, excluded = filter_excluded_rows(rows, skip_ids)
    args.excluded_source_ids_count = excluded
    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(rows)
    if args.target:
        rows = rows[: args.target]
    return rows


def _materialize_download(source: Path, target: Path, *, hardlink: bool) -> None:
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if hardlink:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _download_one(
    row: dict[str, Any],
    cache_dir: Path,
    copy_dir: Path,
    retries: int,
    retry_sleep_seconds: float,
    *,
    hardlink_cache: bool,
    cleanup_cache_each: bool,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    repo_id = str(row.get("repo_id") or row.get("hf_repo") or "YiboZhang2001/TexVerse")
    hf_path = str(row.get("hf_path") or row.get("path") or "")
    if not hf_path:
        raise ValueError("missing hf_path/path")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resolved = hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                repo_type="dataset",
                cache_dir=str(cache_dir),
                token=token,
                local_files_only=False,
            )
            # hf_hub_download often returns a snapshot symlink into the blob
            # cache. Resolve it before hardlinking so cache cleanup cannot
            # leave a dangling symlink in downloads.
            source = Path(resolved).resolve()
            suffix = source.suffix or ".glb"
            uid = str(row.get("uid") or source.stem.split("_")[0])
            target = copy_dir / f"{uid}_{row.get('download_resolution', '')}{suffix}"
            _materialize_download(source, target, hardlink=hardlink_cache)
            out = dict(row)
            out["uid"] = uid
            out["path"] = str(target)
            out["local_path"] = str(target)
            out["hf_path"] = hf_path
            out["repo_id"] = repo_id
            out["downloaded_size_bytes"] = target.stat().st_size
            if cleanup_cache_each:
                # Avoid doubling TexVerse disk use with a persistent HF cache.
                # After a hardlink/copy lands in downloads, the cache copy is
                # no longer needed for this single-worker shard.
                shutil.rmtree(cache_dir, ignore_errors=True)
                cache_dir.mkdir(parents=True, exist_ok=True)
            return out
        except Exception as exc:  # noqa: BLE001 - keep shard downloads moving.
            last_exc = exc
            if attempt < retries:
                text = str(exc).lower()
                multiplier = 4 if "429" in text or "rate" in text else 1
                wait = retry_sleep_seconds * multiplier * (2**attempt)
                print(f"download failed attempt={attempt + 1} path={hf_path}: {type(exc).__name__}: {exc}; sleep={wait:.1f}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--min-quality", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep-seconds", type=float, default=20.0)
    parser.add_argument("--max-size-mb", type=float, default=0.0, help="Skip rows with hf_size_bytes above this threshold; 0 disables.")
    parser.add_argument("--exclude-source-ids", nargs="*", type=Path, default=[], help="Text/JSON/JSONL source-ID registries to skip before download.")
    parser.add_argument("--hardlink-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-cache-each", action="store_true", help="Clear HF cache after each successful file; requires --workers 1.")
    args = parser.parse_args()
    if args.cleanup_cache_each and args.workers != 1:
        raise SystemExit("--cleanup-cache-each is only safe with --workers 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / ".hf_cache"
    copy_dir = args.output_dir / "downloads"
    selected = _select_rows(args)
    selected_path = args.output_dir / "selected_annotations.jsonl"
    with selected_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    downloaded: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_row = {
            executor.submit(
                _download_one,
                row,
                cache_dir,
                copy_dir,
                args.retries,
                args.retry_sleep_seconds,
                hardlink_cache=args.hardlink_cache,
                cleanup_cache_each=args.cleanup_cache_each,
            ): row
            for row in selected
        }
        for index, future in enumerate(futures.as_completed(future_to_row), start=1):
            row = future_to_row[future]
            try:
                downloaded.append(future.result())
            except Exception as exc:  # noqa: BLE001 - record rejects for this shard.
                errors.append({"uid": row.get("uid"), "hf_path": row.get("hf_path") or row.get("path"), "error": f"{type(exc).__name__}: {exc}"})
            if index % 25 == 0 or index == len(future_to_row):
                print(json.dumps({"processed": index, "downloaded": len(downloaded), "errors": len(errors)}), flush=True)

    downloaded.sort(key=lambda row: str(row.get("uid", "")))
    candidates_path = args.output_dir / "downloaded_candidates.json"
    candidates_path.write_text(json.dumps(downloaded, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "download_errors.json").write_text(json.dumps(errors, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {str(row["uid"]): row["path"] for row in downloaded if row.get("uid") and row.get("path")}
    (args.output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "source_manifest": str(args.source_manifest),
        "selected": len(selected),
        "downloaded": len(downloaded),
        "errors": len(errors),
        "excluded_source_ids": int(getattr(args, "excluded_source_ids_count", 0)),
        "max_size_mb": args.max_size_mb,
        "hardlink_cache": args.hardlink_cache,
        "cleanup_cache_each": args.cleanup_cache_each,
        "selected_annotations": str(selected_path),
        "download_manifest": str(args.output_dir / "download_manifest.json"),
        "downloaded_candidates": str(candidates_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if downloaded else 1


if __name__ == "__main__":
    raise SystemExit(main())
