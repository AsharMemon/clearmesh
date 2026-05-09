import json
import subprocess
import sys
from pathlib import Path


def _write_corpus(root: Path, name: str, count: int) -> None:
    tokens = root / name / "tokens_pass"
    tokens.mkdir(parents=True)
    with (tokens / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(count):
            shard = tokens / f"{name}_{idx}.npz"
            shard.write_bytes(f"{name}:{idx}".encode("utf-8"))
            handle.write(json.dumps({"path": str(shard), "uid": f"{name}-{idx}"}, sort_keys=True) + "\n")


def test_merge_face_corpus_shards_creates_global_split(tmp_path):
    _write_corpus(tmp_path, "shard_a", 3)
    _write_corpus(tmp_path, "shard_b", 2)
    output = tmp_path / "merged"
    script = Path(__file__).resolve().parents[1] / "scripts" / "research" / "merge_face_corpus_shards.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(tmp_path / "shard_a"),
            "--input",
            str(tmp_path / "shard_b"),
            "--output-dir",
            str(output),
            "--test-count",
            "1",
            "--seed",
            "7",
        ],
        check=True,
    )

    merged_manifest = output / "tokens_pass" / "manifest.jsonl"
    train_manifest = output / "split_pass" / "train" / "manifest.jsonl"
    test_manifest = output / "split_pass" / "test" / "manifest.jsonl"
    summary = json.loads((output / "merge_summary.json").read_text(encoding="utf-8"))

    merged_rows = [json.loads(line) for line in merged_manifest.read_text(encoding="utf-8").splitlines()]
    train_rows = [json.loads(line) for line in train_manifest.read_text(encoding="utf-8").splitlines()]
    test_rows = [json.loads(line) for line in test_manifest.read_text(encoding="utf-8").splitlines()]

    assert summary["merged_count"] == 5
    assert summary["train_count"] == 4
    assert summary["test_count"] == 1
    assert len(merged_rows) == 5
    assert len(train_rows) == 4
    assert len(test_rows) == 1
    assert all(Path(row["path"]).exists() for row in merged_rows)
    assert all("merge_source_manifest" in row for row in merged_rows)
