#!/usr/bin/env python3
"""Plan FACE corpus shards without replaying B2-completed source shards.

The corpus workers are intentionally cheap and dumb once launched. This planner
is the guardrail that runs before launch: it compares local source-shard pools
against B2 shard archives, builds a completed source-UID registry, and emits
only shards that still contain unseen source assets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


UID_KEYS = ("uid", "UID", "objaverse_uid", "object_uid", "id", "sha256", "path", "hf_path")
SHARD_RE = re.compile(r"shard[_-]?(?P<id>\d+)")


@dataclass(frozen=True)
class Lane:
    name: str
    shard_dir: Path
    b2_prefix: str
    dataset: str


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, text=True, capture_output=True, env=env)


def _rclone_env(remote: str) -> dict[str, str]:
    env = os.environ.copy()
    key_id = env.get("B2_KEYID") or env.get("B2_KEY_ID") or env.get("B2_ACCOUNT_ID")
    app_key = env.get("B2_APPKEY") or env.get("B2_APP_KEY") or env.get("B2_APPLICATION_KEY")
    remote_name = remote[:-1] if remote.endswith(":") else remote
    if key_id and app_key and remote_name:
        config_prefix = f"RCLONE_CONFIG_{re.sub(r'[^A-Za-z0-9]', '_', remote_name).upper()}"
        env.setdefault(f"{config_prefix}_TYPE", "b2")
        env.setdefault(f"{config_prefix}_ACCOUNT", key_id)
        env.setdefault(f"{config_prefix}_KEY", app_key)
    return env


def _identity(row: dict, default_dataset: str) -> str | None:
    dataset = str(row.get("source_dataset") or default_dataset or "unknown").strip().lower()
    for key in UID_KEYS:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return f"{dataset}:{text.lower()}"
    return None


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(payload, dict):
                yield payload


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _shard_id(path: Path) -> str:
    match = SHARD_RE.search(path.stem)
    if not match:
        raise ValueError(f"cannot parse shard id from {path}")
    return match.group("id").zfill(4)


def _list_b2_completed(remote: str, bucket: str, prefix: str, env: dict[str, str]) -> set[str]:
    base = f"{remote}{bucket}/{prefix.strip('/')}"
    proc = _run(["rclone", "lsf", base, "--dirs-only"], env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"rclone lsf failed for {base}: {proc.stderr.strip()}")
    completed: set[str] = set()
    for raw in proc.stdout.splitlines():
        name = raw.strip().rstrip("/")
        if not name:
            continue
        sid_match = SHARD_RE.search(name)
        if not sid_match:
            continue
        archive = f"{base}/{name}/lean_face_corpus.tar.gz"
        check = _run(["rclone", "lsf", archive], env=env)
        if check.returncode == 0:
            completed.add(sid_match.group("id").zfill(4))
    return completed


def _parse_lane(spec: str) -> Lane:
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("lane must be name=shard_dir,b2_prefix,dataset")
    name, rest = parts
    values = [item.strip() for item in rest.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("lane must be name=shard_dir,b2_prefix,dataset")
    return Lane(name=name.strip(), shard_dir=Path(values[0]), b2_prefix=values[1].strip("/"), dataset=values[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", action="append", type=_parse_lane, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="clearmesh-pairs")
    parser.add_argument("--rclone-remote", default="b2env:", help="Configured rclone remote, e.g. b2env: or CMB2:")
    parser.add_argument("--max-planned-shards-per-lane", type=int, default=0)
    parser.add_argument("--min-rows-per-planned-shard", type=int, default=1)
    args = parser.parse_args()

    env = _rclone_env(args.rclone_remote)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed_identities: set[str] = set()
    completed_rows: list[dict] = []
    lane_summaries: list[dict] = []

    # First pass: build source identity registry from local source shards whose
    # matching B2 shard archive is already durable.
    lane_completed: dict[str, set[str]] = {}
    for lane in args.lane:
        if not lane.shard_dir.exists():
            raise FileNotFoundError(f"missing shard dir for lane {lane.name}: {lane.shard_dir}")
        completed = _list_b2_completed(args.rclone_remote, args.bucket, lane.b2_prefix, env)
        lane_completed[lane.name] = completed
        local_shards = sorted(lane.shard_dir.glob("shard_*.jsonl"))
        for shard_path in local_shards:
            sid = _shard_id(shard_path)
            if sid not in completed:
                continue
            source_count = 0
            added_count = 0
            for row in _read_jsonl(shard_path):
                source_count += 1
                ident = _identity(row, lane.dataset)
                if ident and ident not in completed_identities:
                    completed_identities.add(ident)
                    added_count += 1
            completed_rows.append(
                {
                    "lane": lane.name,
                    "dataset": lane.dataset,
                    "shard_id": sid,
                    "source_count": source_count,
                    "new_completed_source_identities": added_count,
                    "b2_prefix": f"{lane.b2_prefix}/shard{sid}",
                    "local_shard": str(shard_path),
                }
            )

    planned_rows: list[dict] = []
    planned_manifest_rows: list[dict] = []
    # Second pass: emit filtered shard files for local source shards that are
    # not complete in B2 and still contain unseen source identities.
    for lane in args.lane:
        completed = lane_completed[lane.name]
        local_shards = sorted(lane.shard_dir.glob("shard_*.jsonl"))
        planned_for_lane = 0
        complete_local = 0
        exhausted_by_source_registry = 0
        missing_b2 = 0
        for shard_path in local_shards:
            sid = _shard_id(shard_path)
            if sid in completed:
                complete_local += 1
                continue
            missing_b2 += 1
            filtered: list[dict] = []
            input_count = 0
            duplicate_source_count = 0
            for row in _read_jsonl(shard_path):
                input_count += 1
                ident = _identity(row, lane.dataset)
                if ident and ident in completed_identities:
                    duplicate_source_count += 1
                    continue
                filtered.append(row)
            if len(filtered) < args.min_rows_per_planned_shard:
                exhausted_by_source_registry += 1
                continue
            if args.max_planned_shards_per_lane and planned_for_lane >= args.max_planned_shards_per_lane:
                continue
            out_shard = args.output_dir / "planned_shards" / lane.name / f"shard_{sid}.jsonl"
            output_count = _write_jsonl(out_shard, filtered)
            planned_for_lane += 1
            row = {
                "lane": lane.name,
                "dataset": lane.dataset,
                "shard_id": sid,
                "input_count": input_count,
                "output_count": output_count,
                "duplicate_source_count": duplicate_source_count,
                "source_shard": str(shard_path),
                "planned_shard": str(out_shard),
                "b2_prefix": f"{lane.b2_prefix}/shard{sid}",
            }
            planned_rows.append(row)
            planned_manifest_rows.append(row)
        lane_summaries.append(
            {
                "lane": lane.name,
                "dataset": lane.dataset,
                "shard_dir": str(lane.shard_dir),
                "b2_prefix": lane.b2_prefix,
                "local_shards": len(local_shards),
                "b2_completed_shards": len(completed),
                "local_shards_complete_in_b2": complete_local,
                "local_shards_missing_in_b2": missing_b2,
                "planned_shards": planned_for_lane,
                "missing_shards_exhausted_by_source_registry": exhausted_by_source_registry,
            }
        )

    completed_path = args.output_dir / "completed_shards.jsonl"
    planned_path = args.output_dir / "planned_shards.jsonl"
    identity_path = args.output_dir / "completed_source_identities.txt"
    summary_path = args.output_dir / "summary.json"
    _write_jsonl(completed_path, completed_rows)
    _write_jsonl(planned_path, planned_manifest_rows)
    identity_path.write_text("\n".join(sorted(completed_identities)) + ("\n" if completed_identities else ""), encoding="utf-8")
    summary = {
        "bucket": args.bucket,
        "rclone_remote": args.rclone_remote,
        "completed_shards": len(completed_rows),
        "completed_source_identities": len(completed_identities),
        "planned_shards": len(planned_rows),
        "planned_source_rows": sum(int(row["output_count"]) for row in planned_rows),
        "lanes": lane_summaries,
        "completed_shards_manifest": str(completed_path),
        "planned_shards_manifest": str(planned_path),
        "completed_source_identities_path": str(identity_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not planned_rows:
        print("No non-duplicate local shards remain to launch.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
