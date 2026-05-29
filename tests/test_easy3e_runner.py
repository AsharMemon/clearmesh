from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/product/run_easy3e_edit.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_easy3e_edit_test", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_easy3e_preflight_rejects_generation_only_trellis_snapshot(tmp_path) -> None:
    runner = load_runner_module()
    (tmp_path / "pipeline.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ckpts").mkdir()
    (tmp_path / "ckpts" / "shape_dec_next_dc_f16c32_fp16.json").write_text("{}", encoding="utf-8")

    with pytest.raises(runner.Easy3EPreflightError) as exc_info:
        runner.validate_easy3e_model_dir(tmp_path)

    message = str(exc_info.value)
    assert "shape_enc_next_dc_f16c32_fp16.json" in message
    assert "shape_enc_next_dc_f16c32_fp16.safetensors" in message
    assert "CLEARMESH_EASY3E_ENABLED=0" in message


def test_easy3e_preflight_accepts_snapshot_with_shape_encoder(tmp_path) -> None:
    runner = load_runner_module()
    ckpts = tmp_path / "ckpts"
    ckpts.mkdir()
    (tmp_path / "pipeline.json").write_text("{}", encoding="utf-8")
    (ckpts / "shape_enc_next_dc_f16c32_fp16.json").write_text("{}", encoding="utf-8")
    (ckpts / "shape_enc_next_dc_f16c32_fp16.safetensors").write_bytes(b"placeholder")

    runner.validate_easy3e_model_dir(tmp_path)


def test_easy3e_resolves_cached_hf_snapshot_without_env_download(monkeypatch, tmp_path) -> None:
    runner = load_runner_module()
    snapshot = tmp_path / "models--microsoft--TRELLIS.2-4B" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "pipeline.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    assert runner.resolve_model_dir("microsoft/TRELLIS.2-4B") == snapshot.resolve()
