#!/usr/bin/env python3
"""Build a sharded TexVerse source pool for strict FACE corpus preparation.

TexVerse contains multiple texture-resolution variants for many of the same
objects. FACE geometry/token preparation does not use texture pixels, so this
builder ranks objects by quality signals (PBR membership, high-resolution
variant availability) while selecting the smallest available GLB variant by
default to keep bandwidth and GPU-disk costs sane.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MAIN_REPO = "YiboZhang2001/TexVerse"
ONEK_REPO = "YiboZhang2001/TexVerse-1K"
USER_AGENT = "clearmesh-texverse-source-pool/1.0"
RESOLUTION_SUFFIX = {
    8192: "8192",
    4096: "4096",
    2048: "2048",
    1024: "1024",
}
RESOLUTION_QUALITY_ORDER = (8192, 4096, 2048, 1024)
RESOLUTION_DOWNLOAD_ORDER = (1024, 2048, 4096, 8192)
PATH_RE = re.compile(r"(?P<uid>[0-9a-fA-F]{32})_(?P<resolution>1024|2048|4096|8192)\.glb$")


@dataclass(frozen=True)
class Variant:
    repo_id: str
    path: str
    resolution: int
    size: int | None = None


def _auth_headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str, retries: int, sleep_seconds: float) -> tuple[list[dict[str, Any]], str | None, dict[str, str]]:
    headers = _auth_headers()
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
                link = response.headers.get("Link")
                response_headers = {key: value for key, value in response.headers.items()}
            if not isinstance(payload, list):
                raise RuntimeError(f"expected list payload from {url}, got {type(payload).__name__}")
            return payload, link, response_headers
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                wait = sleep_seconds * (2**attempt)
                print(f"HF API {exc.code}; sleeping {wait:.1f}s before retry", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise
        except Exception as exc:  # noqa: BLE001 - public API can transiently fail.
            last_exc = exc
            if attempt < retries:
                wait = sleep_seconds * (2**attempt)
                print(f"HF API error {type(exc).__name__}: {exc}; sleeping {wait:.1f}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' not in section and "rel=next" not in section:
            continue
        match = re.search(r"<([^>]+)>", section)
        if match:
            return match.group(1)
    return None


def _tree_url(repo_id: str, rel_path: str) -> str:
    encoded = "/".join(urllib.parse.quote(piece) for piece in rel_path.strip("/").split("/") if piece)
    suffix = f"/{encoded}" if encoded else ""
    return f"https://huggingface.co/api/datasets/{repo_id}/tree/main{suffix}?recursive=false&expand=false&limit=1000"


def _cache_name(repo_id: str, rel_path: str, page_index: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{repo_id}_{rel_path}_{page_index}")
    return f"{safe}.json"


def _iter_tree_pages(
    repo_id: str,
    rel_path: str,
    cache_dir: Path,
    retries: int,
    sleep_seconds: float,
    api_pause_seconds: float,
) -> Iterable[list[dict[str, Any]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    url: str | None = _tree_url(repo_id, rel_path)
    page_index = 0
    while url:
        cache_path = cache_dir / _cache_name(repo_id, rel_path, page_index)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            payload = cached["payload"]
            url = cached.get("next_url")
        else:
            payload, link, headers = _request_json(url, retries=retries, sleep_seconds=sleep_seconds)
            next_url = _next_link(link)
            cache_path.write_text(
                json.dumps({"url": url, "next_url": next_url, "headers": headers, "payload": payload}, sort_keys=True),
                encoding="utf-8",
            )
            url = next_url
            if api_pause_seconds > 0:
                time.sleep(api_pause_seconds)
        yield payload
        page_index += 1


def _list_child_folders(repo_id: str, rel_path: str, cache_dir: Path, args: argparse.Namespace) -> list[str]:
    folders: list[str] = []
    for page in _iter_tree_pages(repo_id, rel_path, cache_dir, args.retries, args.retry_sleep_seconds, args.api_pause_seconds):
        for item in page:
            if item.get("type") == "directory":
                folders.append(str(item.get("path", "")))
    return sorted(folder for folder in folders if folder)


def _parse_variant(repo_id: str, item: dict[str, Any]) -> tuple[str, Variant] | None:
    if item.get("type") not in {"file", None}:
        return None
    path = str(item.get("path") or "")
    match = PATH_RE.search(Path(path).name)
    if not match:
        return None
    uid = match.group("uid").lower()
    resolution = int(match.group("resolution"))
    size = item.get("size")
    return uid, Variant(repo_id=repo_id, path=path, resolution=resolution, size=int(size) if isinstance(size, int) else None)


def _normalize_id(raw: str) -> str:
    text = raw.strip().lower()
    if ":" in text:
        text = text.split(":", 1)[1]
    return text


def _load_id_set(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {_normalize_id(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _discover_variants(args: argparse.Namespace, cache_dir: Path) -> dict[str, dict[int, Variant]]:
    variants: dict[str, dict[int, Variant]] = {}
    roots: list[tuple[str, str]] = []
    if args.include_main:
        roots.extend((args.main_repo, root) for root in args.main_roots)
    if args.include_1k:
        roots.extend((args.onek_repo, root) for root in args.onek_roots)

    for repo_id, root in roots:
        folders = _list_child_folders(repo_id, root, cache_dir, args)
        print(json.dumps({"event": "folder_listed", "repo_id": repo_id, "root": root, "folders": len(folders)}), flush=True)
        for folder_index, folder in enumerate(folders, start=1):
            page_count = 0
            file_count = 0
            for page in _iter_tree_pages(repo_id, folder, cache_dir, args.retries, args.retry_sleep_seconds, args.api_pause_seconds):
                page_count += 1
                for item in page:
                    parsed = _parse_variant(repo_id, item)
                    if parsed is None:
                        continue
                    uid, variant = parsed
                    variants.setdefault(uid, {})[variant.resolution] = variant
                    file_count += 1
            if folder_index % max(1, args.progress_every_folders) == 0 or folder_index == len(folders):
                print(
                    json.dumps(
                        {
                            "event": "folder_scanned",
                            "repo_id": repo_id,
                            "folder": folder,
                            "folder_index": folder_index,
                            "folder_count": len(folders),
                            "pages": page_count,
                            "files": file_count,
                            "unique_uids_so_far": len(variants),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if args.max_discovery_folders and folder_index >= args.max_discovery_folders:
                break
    return variants


def _choose_download_variant(variant_by_resolution: dict[int, Variant], policy: str) -> Variant:
    if policy == "highest":
        order = RESOLUTION_QUALITY_ORDER
    elif policy == "lowest":
        order = RESOLUTION_DOWNLOAD_ORDER
    else:
        raise ValueError(f"unsupported download policy: {policy}")
    for resolution in order:
        if resolution in variant_by_resolution:
            return variant_by_resolution[resolution]
    return next(iter(variant_by_resolution.values()))


def _rank_rows(rows: list[dict[str, Any]], seed: int, shuffle_ties: bool) -> list[dict[str, Any]]:
    if shuffle_ties:
        rng = random.Random(seed)
        rng.shuffle(rows)
    rows.sort(
        key=lambda row: (
            int(row["is_pbr"]),
            int(row["max_resolution"]),
            int(row["variant_count"]),
            float(row["source_rank_score"]),
            str(row["uid"]),
        ),
        reverse=True,
    )
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-source", type=int, default=675_000)
    parser.add_argument("--shard-size", type=int, default=5_990)
    parser.add_argument("--main-repo", default=MAIN_REPO)
    parser.add_argument("--onek-repo", default=ONEK_REPO)
    parser.add_argument("--include-main", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-1k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--main-roots", nargs="+", default=["glbs/glbs_8k", "glbs/glbs_4k", "glbs/glbs_2k"])
    parser.add_argument("--onek-roots", nargs="+", default=["glbs/glbs_1k"])
    parser.add_argument("--pbr-id-list", type=Path, default=Path(".codex_outputs/texverse_hf_probe_20260516/TexVerse_pbr_id_list.txt"))
    parser.add_argument(
        "--exclude-id-list",
        type=Path,
        default=None,
        help="Optional newline-delimited UID list to exclude. Lines may be bare UIDs or dataset:UID identities.",
    )
    parser.add_argument("--download-resolution-policy", choices=["lowest", "highest"], default="lowest")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=20.0)
    parser.add_argument("--api-pause-seconds", type=float, default=0.75)
    parser.add_argument("--progress-every-folders", type=int, default=2)
    parser.add_argument("--max-discovery-folders", type=int, default=0, help="Debug limit per root; 0 means no limit.")
    parser.add_argument("--seed", type=int, default=675)
    parser.add_argument("--shuffle-ties", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.output_dir / "hf_tree_cache"
    pbr_ids = _load_id_set(args.pbr_id_list)
    exclude_ids = _load_id_set(args.exclude_id_list)
    variants = _discover_variants(args, cache_dir)

    rows: list[dict[str, Any]] = []
    resolution_hist: Counter[str] = Counter()
    max_resolution_hist: Counter[str] = Counter()
    for uid, variant_by_resolution in variants.items():
        if uid.lower() in exclude_ids:
            continue
        if not variant_by_resolution:
            continue
        chosen = _choose_download_variant(variant_by_resolution, args.download_resolution_policy)
        max_resolution = max(variant_by_resolution)
        is_pbr = uid in pbr_ids
        # This score is intentionally source-side only. Real geometric quality is
        # enforced later by strict target preparation and token-gate checks.
        source_rank_score = (
            (1000.0 if is_pbr else 0.0)
            + max_resolution / 10.0
            + 5.0 * len(variant_by_resolution)
            - chosen.resolution / 100000.0
        )
        row = {
            "uid": uid,
            "repo_id": chosen.repo_id,
            "hf_path": chosen.path,
            "path": chosen.path,
            "download_resolution": chosen.resolution,
            "max_resolution": max_resolution,
            "available_resolutions": sorted(variant_by_resolution),
            "variant_count": len(variant_by_resolution),
            "is_pbr": bool(is_pbr),
            "quality_score": 3 if is_pbr or max_resolution >= 4096 else 2,
            "source_rank_score": round(source_rank_score, 6),
            "source_dataset": "TexVerse",
        }
        if chosen.size is not None:
            row["hf_size_bytes"] = chosen.size
        rows.append(row)
        resolution_hist[str(chosen.resolution)] += 1
        max_resolution_hist[str(max_resolution)] += 1

    _rank_rows(rows, seed=args.seed, shuffle_ties=args.shuffle_ties)
    selected = rows[: args.max_source] if args.max_source > 0 else rows

    manifest_path = args.output_dir / "source_candidates.jsonl"
    _write_jsonl(manifest_path, selected)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_counts = []
    for shard_index, start in enumerate(range(0, len(selected), args.shard_size)):
        shard_rows = selected[start : start + args.shard_size]
        shard_path = shard_dir / f"shard_{shard_index:04d}.jsonl"
        shard_counts.append(_write_jsonl(shard_path, shard_rows))

    summary = {
        "dataset": "TexVerse",
        "main_repo": args.main_repo,
        "onek_repo": args.onek_repo,
        "output_dir": str(args.output_dir),
        "manifest": str(manifest_path),
        "all_unique_uids_discovered": len(rows),
        "selected_rows": len(selected),
        "max_source": args.max_source,
        "shard_size": args.shard_size,
        "shards": len(shard_counts),
        "shard_counts": shard_counts,
        "pbr_id_count": len(pbr_ids),
        "exclude_id_count": len(exclude_ids),
        "selected_pbr_count": sum(1 for row in selected if row["is_pbr"]),
        "download_resolution_policy": args.download_resolution_policy,
        "download_resolution_histogram_all": dict(sorted(resolution_hist.items())),
        "max_resolution_histogram_all": dict(sorted(max_resolution_hist.items())),
        "selected_download_resolution_histogram": dict(Counter(str(row["download_resolution"]) for row in selected).most_common()),
        "selected_max_resolution_histogram": dict(Counter(str(row["max_resolution"]) for row in selected).most_common()),
    }
    (args.output_dir / "pool_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
