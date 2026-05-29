from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/inference/clearmesh_pipeline_server.py"


def load_bridge_module():
    spec = importlib.util.spec_from_file_location("clearmesh_pipeline_server_test", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_prompt_discourages_bad_trellis_inputs() -> None:
    bridge = load_bridge_module()
    prompt = bridge.build_text_to_image_prompt("windmill")
    assert "exactly one object" in prompt
    assert "object floating in empty studio space" in prompt
    assert "no tabletop" in prompt
    assert "matte neutral material" in prompt
    assert "no black glossy silhouette" in prompt
    assert "no duplicate objects" in prompt


def test_reference_prompt_normalizes_duplicate_subject_prefixes() -> None:
    bridge = load_bridge_module()
    prompt = bridge.build_text_to_image_prompt("single isolated windmill")

    assert prompt.startswith("single isolated windmill, exactly one object")
    assert "single isolated single" not in prompt
    assert "isolated isolated" not in prompt


def test_trellis_qc_rejects_fragmented_mesh(monkeypatch) -> None:
    bridge = load_bridge_module()

    def fake_stats(_path):
        return {
            "ok": True,
            "vertices": 1000,
            "faces": 900,
            "components": 40,
            "tiny_component_count": 30,
            "largest_component_face_ratio": 0.2,
            "extent_ratio": 120.0,
            "bbox_diag": 1.0,
        }

    monkeypatch.setattr(bridge, "mesh_basic_stats", fake_stats)
    report = bridge.inspect_trellis_candidate(Path("bad.glb"))
    assert report["accepted"] is False
    assert any("too few faces" in reason for reason in report["reasons"])
    assert any("too fragmented" in reason for reason in report["reasons"])
    assert any("needle-like" in reason for reason in report["reasons"])


def test_trellis_qc_accepts_dominant_mesh(monkeypatch) -> None:
    bridge = load_bridge_module()

    def fake_stats(_path):
        return {
            "ok": True,
            "vertices": 8000,
            "faces": 12000,
            "components": 3,
            "tiny_component_count": 1,
            "largest_component_face_ratio": 0.92,
            "extent_ratio": 4.0,
            "bbox_diag": 1.5,
        }

    monkeypatch.setattr(bridge, "mesh_basic_stats", fake_stats)
    report = bridge.inspect_trellis_candidate(Path("good.glb"))
    assert report["accepted"] is True
    assert report["reasons"] == []
    assert report["score"] > 1.0


def test_trellis_qc_accepts_clean_two_component_mesh(monkeypatch) -> None:
    bridge = load_bridge_module()

    def fake_stats(_path):
        return {
            "ok": True,
            "vertices": 42000,
            "faces": 76000,
            "raw_components": 93,
            "components": 2,
            "tiny_component_count": 0,
            "largest_component_face_ratio": 0.502,
            "extent_ratio": 1.5,
            "bbox_diag": 1.4,
        }

    monkeypatch.setattr(bridge, "mesh_basic_stats", fake_stats)
    report = bridge.inspect_trellis_candidate(Path("clean_two_part.glb"))
    assert report["accepted"] is True
    assert report["reasons"] == []


def test_faceq_preview_skip_requires_server_opt_in(monkeypatch, tmp_path) -> None:
    bridge = load_bridge_module()
    server = bridge.PipelineServer(
        work_root=tmp_path / "work",
        state_root=tmp_path / "state",
        checkpoint=tmp_path / "checkpoint.pt",
        model_bundle=tmp_path / "bundle",
    )

    monkeypatch.delenv("CLEARMESH_ALLOW_TRELLIS_ONLY", raising=False)
    assert server.should_run_faceq({"quality_tier": "draft", "faceq": False}) is True

    monkeypatch.setenv("CLEARMESH_ALLOW_TRELLIS_ONLY", "1")
    assert server.should_run_faceq({"quality_tier": "draft"}) is False
    assert server.should_run_faceq({"quality_tier": "standard"}) is True
    assert server.should_run_faceq({"quality_tier": "standard", "faceq": False}) is False
    assert server.should_run_faceq({"quality_tier": "draft", "pipeline": {"faceq": True}}) is True


def test_trellis_command_disables_expensive_remesh_by_default(monkeypatch, tmp_path) -> None:
    bridge = load_bridge_module()
    server = bridge.PipelineServer(
        work_root=tmp_path / "work",
        state_root=tmp_path / "state",
        checkpoint=tmp_path / "checkpoint.pt",
        model_bundle=tmp_path / "bundle",
    )

    monkeypatch.delenv("CLEARMESH_TRELLIS_REMESH", raising=False)
    command = server.build_trellis_command(
        image_path=tmp_path / "input.png",
        output_dir=tmp_path / "trellis",
        preview_name="preview.glb",
        face_proxy_name="proxy.glb",
    )
    assert "--no-remesh" in command
    assert "--remesh" not in command

    monkeypatch.setenv("CLEARMESH_TRELLIS_REMESH", "1")
    command = server.build_trellis_command(
        image_path=tmp_path / "input.png",
        output_dir=tmp_path / "trellis",
        preview_name="preview.glb",
        face_proxy_name="proxy.glb",
    )
    assert "--remesh" in command
    assert "--no-remesh" not in command


def test_mesh_qc_counts_uv_seams_as_connected(monkeypatch, tmp_path) -> None:
    trimesh = pytest.importorskip("trimesh")

    bridge = load_bridge_module()
    monkeypatch.setenv("CLEARMESH_MESH_QC_SUBPROCESS", "0")

    # Two triangles share the same geometric edge, but the edge vertices are
    # intentionally duplicated as UV/material seams often are in textured GLBs.
    mesh = trimesh.Trimesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        faces=[[0, 1, 2], [3, 4, 5]],
        process=False,
    )
    path = tmp_path / "seam_split.obj"
    mesh.export(path)

    report = bridge.mesh_basic_stats(path)

    assert report["ok"] is True
    assert report["raw_components"] == 2
    assert report["components"] == 1
    assert report["largest_component_face_ratio"] == 1.0


def test_mesh_qc_subprocess_fails_safely_on_missing_mesh(monkeypatch, tmp_path) -> None:
    bridge = load_bridge_module()
    monkeypatch.setenv("CLEARMESH_MESH_QC_SUBPROCESS", "1")
    monkeypatch.setenv("CLEARMESH_MESH_QC_TIMEOUT_SECONDS", "5")

    report = bridge.mesh_basic_stats(tmp_path / "missing.glb")

    assert report["ok"] is False
    assert "load_error" in report


def test_chamfer_qc_subprocess_fails_safely_on_missing_meshes(monkeypatch, tmp_path) -> None:
    bridge = load_bridge_module()
    monkeypatch.setenv("CLEARMESH_MESH_QC_SUBPROCESS", "1")
    monkeypatch.setenv("CLEARMESH_MESH_QC_TIMEOUT_SECONDS", "5")

    value = bridge.normalized_surface_chamfer(tmp_path / "source.glb", tmp_path / "candidate.glb", 128)

    assert value is None


def test_texture_uv_copy_reference_reports_fallback(tmp_path) -> None:
    source = tmp_path / "source.glb"
    reference = tmp_path / "trellis_textured.glb"
    source.write_bytes(b"source")
    reference.write_bytes(b"reference-textured")
    output_dir = tmp_path / "texture_uv"

    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/product/run_texture_uv_postprocess.py"),
            "--input-mesh",
            str(source),
            "--reference-mesh",
            str(reference),
            "--output-dir",
            str(output_dir),
            "--mode",
            "copy-reference",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert proc.returncode == 0, proc.stderr
    assert (output_dir / "textured_mesh.glb").read_bytes() == b"reference-textured"
    report = json.loads((output_dir / "texture_uv_report.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["fallback"] is True
    assert report["uv_ready"] is True
    assert report["ai_textured"] is True
    assert report["texture_source"] == "trellis_pbr"
