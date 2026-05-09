import json
import subprocess
import sys
import tarfile
from pathlib import Path


def test_package_face_corpus_includes_strict_split_without_raw(tmp_path):
    data = tmp_path / "corpus"
    train = data / "split_pass" / "train"
    test = data / "split_pass" / "test"
    raw = data / "raw"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    raw.mkdir(parents=True)
    (train / "sample_train.npz").write_bytes(b"train")
    (test / "sample_test.npz").write_bytes(b"test")
    (train / "manifest.jsonl").write_text(json.dumps({"path": str(train / "sample_train.npz")}) + "\n", encoding="utf-8")
    (test / "manifest.jsonl").write_text(json.dumps({"path": str(test / "sample_test.npz")}) + "\n", encoding="utf-8")
    (raw / "download.bin").write_bytes(b"raw")
    (data / "strict_gate.json").write_text(
        json.dumps({"sample_count": 2, "passing": 2, "failing": 0, "pass_rate": 1.0}),
        encoding="utf-8",
    )

    archive = tmp_path / "corpus.tar.gz"
    script = Path(__file__).resolve().parents[1] / "scripts" / "research" / "package_face_corpus.py"
    subprocess.run(
        [sys.executable, str(script), "--data-run", str(data), "--output", str(archive), "--archive-root", "packed"],
        check=True,
    )

    manifest = json.loads((tmp_path / "corpus.tar.gz.json").read_text(encoding="utf-8"))
    assert manifest["train_count"] == 1
    assert manifest["test_count"] == 1
    assert manifest["strict_gate"]["pass_rate"] == 1.0
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert "packed/split_pass/train/manifest.jsonl" in names
    assert "packed/split_pass/test/manifest.jsonl" in names
    assert "packed/raw/download.bin" not in names
