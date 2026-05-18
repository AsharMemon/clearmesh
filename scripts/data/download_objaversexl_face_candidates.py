#!/usr/bin/env python3
"""Download Objaverse-XL candidates from a FACE source manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


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
    selected: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("quality_score", -1)) < args.min_quality:
            continue
        aesthetic = row.get("aesthetic_score")
        if aesthetic is not None and aesthetic != "":
            try:
                if float(aesthetic) < args.min_aesthetic:
                    continue
            except (TypeError, ValueError):
                pass
        selected.append(row)
    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(selected)
    if args.target:
        selected = selected[: args.target]
    return selected


def _load_annotations(download_dir: Path) -> pd.DataFrame:
    import objaverse.xl as oxl

    annotations = oxl.get_annotations(download_dir=str(download_dir))
    if isinstance(annotations, pd.DataFrame):
        return annotations
    if isinstance(annotations, dict):
        frames = [frame for frame in annotations.values() if isinstance(frame, pd.DataFrame)]
        if frames:
            return pd.concat(frames, ignore_index=True)
    raise TypeError(f"unexpected Objaverse-XL annotations type: {type(annotations)!r}")


def _download_batch(batch_df: pd.DataFrame, download_dir: Path, processes: int) -> dict[str, str]:
    import objaverse.xl as oxl

    paths = oxl.download_objects(
        batch_df,
        download_dir=str(download_dir),
        processes=processes,
        save_repo_format="zip",
    )
    out: dict[str, str] = {}
    for _, row in batch_df.iterrows():
        file_identifier = str(row.get("file_identifier") or row.get("fileIdentifier") or "")
        sha = str(row.get("sha256") or "")
        path = paths.get(file_identifier)
        if sha and path and Path(path).exists():
            out[sha] = str(path)
    return out


def _download(
    selected: list[dict[str, Any]],
    output_dir: Path,
    processes: int,
    batch_size: int,
    retries: int,
    retry_sleep_seconds: float,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".objaversexl_cache"
    models_dir = output_dir / "downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(output_dir / ".hf_cache"))

    target_sha = {str(row.get("sha256") or row.get("uid") or "").lower() for row in selected}
    target_sha.discard("")
    annotations = _load_annotations(cache_dir)
    if "sha256" not in annotations.columns:
        raise KeyError("Objaverse-XL annotations missing sha256 column")
    matched = annotations[annotations["sha256"].astype(str).str.lower().isin(target_sha)].copy()
    print(json.dumps({"selected": len(selected), "matched_objaversexl_annotations": int(len(matched))}), flush=True)

    manifest: dict[str, str] = {}
    records = matched.to_dict("records")
    for start in range(0, len(records), max(1, batch_size)):
        batch = pd.DataFrame(records[start : start + max(1, batch_size)])
        batch_index = start // max(1, batch_size)
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                paths = _download_batch(batch, models_dir, processes)
                manifest.update(paths)
                break
            except Exception as exc:  # noqa: BLE001 - keep shard moving.
                last_exc = exc
                if attempt >= retries:
                    print(
                        f"ObjaverseXL batch {batch_index} failed after {attempt + 1} attempts: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                wait = retry_sleep_seconds * (2**attempt)
                text = str(exc).lower()
                if "429" in text or "rate" in text or "too many requests" in text:
                    wait *= 4
                print(
                    f"ObjaverseXL batch {batch_index} attempt {attempt + 1} failed: {type(exc).__name__}: {exc}; sleep={wait:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
        print(
            json.dumps(
                {
                    "batch": batch_index,
                    "processed": min(start + len(batch), len(records)),
                    "manifest_total": len(manifest),
                    "last_error": f"{type(last_exc).__name__}: {last_exc}" if last_exc else "",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--min-quality", type=int, default=2)
    parser.add_argument("--min-aesthetic", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=20.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_rows(args)
    selected_path = args.output_dir / "selected_annotations.jsonl"
    with selected_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = _download(
        selected,
        args.output_dir,
        args.processes,
        args.batch_size,
        args.retries,
        args.retry_sleep_seconds,
    )
    (args.output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    selected_by_sha = {str(row.get("sha256") or row.get("uid") or "").lower(): row for row in selected}
    candidates: list[dict[str, Any]] = []
    for sha, path in sorted(manifest.items()):
        row = dict(selected_by_sha.get(sha.lower(), {}))
        row.update(
            {
                "uid": sha,
                "sha256": sha,
                "path": path,
                "local_path": path,
                "quality_score": int(row.get("quality_score", 2)),
                "source_dataset": "ObjaverseXL",
            }
        )
        candidates.append(row)
    candidates_path = args.output_dir / "downloaded_candidates.json"
    candidates_path.write_text(json.dumps(candidates, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "source_manifest": str(args.source_manifest),
        "selected": len(selected),
        "downloaded": len(candidates),
        "selected_annotations": str(selected_path),
        "download_manifest": str(args.output_dir / "download_manifest.json"),
        "downloaded_candidates": str(candidates_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
