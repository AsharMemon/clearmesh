from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_token_npz(path: Path, tokens: np.ndarray) -> None:
    np.savez(
        path,
        paper_tokens=np.asarray(tokens, dtype=np.int64),
        num_bins=np.asarray([128], dtype=np.int64),
        paper_within_face_order=np.asarray(["rotate_min_zyx"]),
    )


def test_dedupe_face_token_split_removes_train_test_hash_leaks(tmp_path: Path) -> None:
    token_a = np.asarray(
        [
            [10, 10, 10, 10, 10, 40, 10, 40, 10],
            [30, 30, 30, 30, 30, 60, 30, 60, 30],
        ],
        dtype=np.int64,
    )
    token_b = np.asarray(
        [
            [20, 20, 20, 20, 20, 52, 20, 52, 20],
            [50, 50, 50, 50, 50, 82, 50, 82, 50],
        ],
        dtype=np.int64,
    )
    token_c = np.asarray(
        [
            [15, 15, 15, 15, 15, 48, 15, 48, 15],
            [70, 70, 70, 70, 70, 104, 70, 104, 70],
        ],
        dtype=np.int64,
    )

    src = tmp_path / "src"
    src.mkdir()
    _write_token_npz(src / "a_train.npz", token_a)
    _write_token_npz(src / "b_train.npz", token_b)
    _write_token_npz(src / "a_test_duplicate.npz", token_a)
    _write_token_npz(src / "c_test.npz", token_c)

    manifest = tmp_path / "combined_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for path in sorted(src.glob("*.npz")):
            handle.write(json.dumps({"path": str(path)}) + "\n")

    output_dir = tmp_path / "deduped"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/research/dedupe_face_token_split.py"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--seed",
            "7",
            "--test-count",
            "1",
            "--copy-mode",
            "copy",
            "--path-mode",
            "absolute",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    summary = json.loads((output_dir / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["input_count"] == 4
    assert summary["unique_token_hashes"] == 3
    assert summary["removed_duplicates"] == 1
    assert summary["train"]["count"] == 2
    assert summary["test"]["count"] == 1

    leakage_report = output_dir / "leakage_check.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/research/check_face_token_leakage.py"),
            "--train-dir",
            str(output_dir / "train"),
            "--test-dir",
            str(output_dir / "test"),
            "--output",
            str(leakage_report),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    leakage = json.loads(leakage_report.read_text(encoding="utf-8"))
    assert leakage["ok"] is True
    assert leakage["leaked_token_hash_count"] == 0
    assert leakage["train"]["duplicate_hash_count"] == 0
    assert leakage["test"]["duplicate_hash_count"] == 0
