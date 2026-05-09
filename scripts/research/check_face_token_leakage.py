#!/usr/bin/env python3
"""Check FACE paper-token splits for leakage and identity retokenization drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_tokens import (  # noqa: E402
    canonicalize_mesh_faces_paper_zyx,
    dequantize_normalized_points,
)


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
    # Split manifests copied from another machine may preserve the original
    # relative dataset path. The split step also copies each NPZ beside the
    # manifest, so fall back to the basename for portable Thunder uploads.
    basename_relative = manifest_path.parent / raw.name
    if basename_relative.exists():
        return basename_relative
    return manifest_relative


def _paths_from_manifest(manifest_path: Path) -> list[Path]:
    return [_resolve_manifest_path(row, manifest_path) for row in _iter_jsonl(manifest_path)]


def _paths_from_dir(dataset_dir: Path) -> list[Path]:
    manifest_path = dataset_dir / "manifest.jsonl"
    if manifest_path.exists():
        return _paths_from_manifest(manifest_path)
    return sorted(dataset_dir.glob("*.npz"))


def _sample_paths(label: str, *, dataset_dir: Path | None, manifest_path: Path | None) -> list[Path]:
    if dataset_dir is None and manifest_path is None:
        raise SystemExit(f"missing {label} dataset: pass --{label}-dir or --{label}-manifest")
    if dataset_dir is not None and manifest_path is not None:
        raise SystemExit(f"pass only one of --{label}-dir or --{label}-manifest")
    paths = _paths_from_manifest(manifest_path) if manifest_path is not None else _paths_from_dir(dataset_dir)  # type: ignore[arg-type]
    return [path for path in paths if path.suffix.lower() == ".npz"]


def _load_tokens(path: Path) -> tuple[np.ndarray, int, str]:
    data = np.load(path)
    if "paper_tokens" not in data.files:
        raise ValueError(f"{path} is missing paper_tokens")
    tokens = np.asarray(data["paper_tokens"], dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] != 9:
        raise ValueError(f"{path} has invalid paper_tokens shape {tokens.shape}")
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0]) if "num_bins" in data.files else 0
    within_face_order = (
        str(np.asarray(data["paper_within_face_order"]).reshape(-1)[0])
        if "paper_within_face_order" in data.files
        else "preserve"
    )
    return tokens, num_bins, within_face_order


def _token_hash(tokens: np.ndarray, num_bins: int) -> str:
    tokens = np.asarray(tokens, dtype="<i8", order="C")
    digest = hashlib.sha256()
    digest.update(str(int(num_bins)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(tokens.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tokens.tobytes())
    return digest.hexdigest()


def _identity_retokenization_report(tokens: np.ndarray, num_bins: int, within_face_order: str) -> dict[str, Any]:
    if len(tokens) == 0:
        return {
            "checked": False,
            "reason": "empty token sequence",
            "exact": False,
            "face_count_delta": 0,
            "token_accuracy": None,
            "max_abs_delta": None,
        }
    q_faces_zyx = np.asarray(tokens, dtype=np.int64).reshape(-1, 3, 3)
    face_vertices = dequantize_normalized_points(
        q_faces_zyx.reshape(-1, 3)[:, [2, 1, 0]],
        num_bins=num_bins,
    ).reshape(-1, 3, 3)
    flat_vertices = face_vertices.reshape(-1, 3)
    # The decoded points are already normalized bin centers. Re-fitting a new
    # mesh transform here can expand the bin-center bounds and shift tokens by
    # one bin, producing false "identity drift" reports on valid token files.
    normalized_vertices = flat_vertices
    faces = np.arange(len(flat_vertices), dtype=np.int64).reshape(-1, 3)
    roundtrip, _ = canonicalize_mesh_faces_paper_zyx(
        normalized_vertices,
        faces,
        num_bins=num_bins,
        within_face_order=within_face_order,
    )
    compared = int(min(len(tokens), len(roundtrip)))
    if compared <= 0:
        token_accuracy = None
        max_abs_delta = None
    else:
        token_accuracy = float(np.mean(tokens[:compared] == roundtrip[:compared]))
        max_abs_delta = int(np.max(np.abs(tokens[:compared] - roundtrip[:compared])))
    return {
        "checked": True,
        "exact": bool(len(tokens) == len(roundtrip) and np.array_equal(tokens, roundtrip)),
        "face_count_delta": int(len(roundtrip) - len(tokens)),
        "token_accuracy": token_accuracy,
        "max_abs_delta": max_abs_delta,
    }


def _scan_split(label: str, paths: list[Path], identity_limit: int) -> dict[str, Any]:
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    identity_reports = []
    errors = []
    face_counts = []
    for idx, path in enumerate(paths):
        try:
            tokens, num_bins, within_face_order = _load_tokens(path)
            token_hash = _token_hash(tokens, num_bins)
            hash_to_paths[token_hash].append(str(path))
            face_counts.append(int(len(tokens)))
            if identity_limit <= 0 or idx < identity_limit:
                identity_reports.append(
                    {
                        "path": str(path),
                        "within_face_order": within_face_order,
                        **_identity_retokenization_report(tokens, num_bins, within_face_order),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - this is a gate; report all bad rows.
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    duplicate_hashes = {key: value for key, value in hash_to_paths.items() if len(value) > 1}
    exact_identity = [row for row in identity_reports if row.get("exact")]
    drifted_identity = [row for row in identity_reports if row.get("checked") and not row.get("exact")]
    return {
        "label": label,
        "sample_count": len(paths),
        "valid_count": sum(len(value) for value in hash_to_paths.values()),
        "error_count": len(errors),
        "errors": errors[:20],
        "unique_token_hashes": len(hash_to_paths),
        "duplicate_hash_count": len(duplicate_hashes),
        "duplicate_hash_examples": dict(list(duplicate_hashes.items())[:10]),
        "face_count_min": min(face_counts) if face_counts else None,
        "face_count_max": max(face_counts) if face_counts else None,
        "face_count_mean": float(np.mean(face_counts)) if face_counts else None,
        "identity_roundtrip_checked": len(identity_reports),
        "identity_roundtrip_exact": len(exact_identity),
        "identity_roundtrip_drifted": len(drifted_identity),
        "identity_roundtrip_drift_examples": drifted_identity[:10],
        "_hash_to_paths": hash_to_paths,
    }


def _strip_private(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, action="append", default=[], help="Compatibility form: pass train then test manifest.")
    parser.add_argument("--identity-limit", type=int, default=64, help="Number of samples per split to retokenize; 0 means all.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--warn-only", action="store_true")
    parser.add_argument("--fail-on-identity-drift", action="store_true")
    args = parser.parse_args()

    if args.manifest:
        if len(args.manifest) != 2:
            raise SystemExit("--manifest compatibility mode expects exactly two paths: train then test")
        if args.train_manifest is not None or args.test_manifest is not None or args.train_dir is not None or args.test_dir is not None:
            raise SystemExit("--manifest cannot be mixed with explicit --train/--test inputs")
        args.train_manifest, args.test_manifest = args.manifest

    train_paths = _sample_paths("train", dataset_dir=args.train_dir, manifest_path=args.train_manifest)
    test_paths = _sample_paths("test", dataset_dir=args.test_dir, manifest_path=args.test_manifest)
    train_report = _scan_split("train", train_paths, int(args.identity_limit))
    test_report = _scan_split("test", test_paths, int(args.identity_limit))

    train_hashes = set(train_report["_hash_to_paths"])
    test_hashes = set(test_report["_hash_to_paths"])
    leaked_hashes = sorted(train_hashes.intersection(test_hashes))
    leak_examples = {
        key: {
            "train": train_report["_hash_to_paths"][key][:5],
            "test": test_report["_hash_to_paths"][key][:5],
        }
        for key in leaked_hashes[:20]
    }
    identity_drift_count = int(train_report["identity_roundtrip_drifted"]) + int(test_report["identity_roundtrip_drifted"])
    error_count = int(train_report["error_count"]) + int(test_report["error_count"])
    payload = {
        "ok": not leaked_hashes and error_count == 0 and (not args.fail_on_identity_drift or identity_drift_count == 0),
        "leaked_token_hash_count": len(leaked_hashes),
        "leak_examples": leak_examples,
        "identity_drift_count": identity_drift_count,
        "error_count": error_count,
        "train": _strip_private(train_report),
        "test": _strip_private(test_report),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if args.warn_only:
        return 0
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
