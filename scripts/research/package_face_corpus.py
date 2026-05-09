#!/usr/bin/env python3
"""Package a prepared FACE corpus for train-only A100 runs.

The training job only needs the strict split plus a few gate summaries. This
script creates a lean, portable archive from a corpus directory produced by
`face_objaversepp_corpus_pilot.sh`, avoiding raw downloads and intermediate
meshes unless explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


DEFAULT_INCLUDE_PATHS = (
    "split_pass",
    "pilot_summary.json",
    "strict_gate.json",
    "train_strict_gate.json",
    "test_strict_gate.json",
    "tokens_pass/filter_summary.json",
    "tokens_pass/manifest.jsonl",
)


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_existing(tar: tarfile.TarFile, root: Path, relative: str, archive_root: str) -> list[str]:
    source = root / relative
    if not source.exists():
        return []
    added: list[str] = []
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file():
                arcname = str(Path(archive_root) / path.relative_to(root))
                tar.add(path, arcname=arcname, recursive=False)
                added.append(str(path.relative_to(root)))
    else:
        arcname = str(Path(archive_root) / source.relative_to(root))
        tar.add(source, arcname=arcname, recursive=False)
        added.append(str(source.relative_to(root)))
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run", type=Path, required=True, help="Prepared FACE corpus root.")
    parser.add_argument("--output", type=Path, required=True, help="Output .tar.gz path.")
    parser.add_argument("--archive-root", default=None, help="Root directory name inside the archive.")
    parser.add_argument("--include-raw", action="store_true", help="Also include raw/, strict_targets/, and tokens/.")
    parser.add_argument("--manifest-output", type=Path, default=None)
    args = parser.parse_args()

    data_run = args.data_run.resolve()
    if not data_run.exists():
        raise SystemExit(f"data run not found: {data_run}")
    split = data_run / "split_pass"
    train_manifest = split / "train" / "manifest.jsonl"
    test_manifest = split / "test" / "manifest.jsonl"
    if not train_manifest.exists() or not test_manifest.exists():
        raise SystemExit(f"expected split manifests under {split}")

    archive_root = args.archive_root or data_run.name
    include_paths = list(DEFAULT_INCLUDE_PATHS)
    if args.include_raw:
        include_paths.extend(["raw", "curated_candidates.jsonl", "curated_rejects.json", "strict_targets", "tokens"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    added: list[str] = []
    with tarfile.open(args.output, "w:gz") as tar:
        for relative in include_paths:
            added.extend(_add_existing(tar, data_run, relative, archive_root))

    gate = _read_json(data_run / "strict_gate.json") or {}
    summary = {
        "archive": str(args.output),
        "archive_root": archive_root,
        "data_run": str(data_run),
        "sha256": _sha256(args.output),
        "size_bytes": args.output.stat().st_size,
        "included_files": len(added),
        "include_raw": bool(args.include_raw),
        "train_count": _count_jsonl(train_manifest),
        "test_count": _count_jsonl(test_manifest),
        "strict_gate": {
            "sample_count": gate.get("sample_count"),
            "passing": gate.get("passing"),
            "failing": gate.get("failing"),
            "pass_rate": gate.get("pass_rate"),
        },
    }
    manifest_output = args.manifest_output or args.output.with_suffix(args.output.suffix + ".json")
    manifest_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
