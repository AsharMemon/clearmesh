from pathlib import Path

from clearmesh.mesh_heads import MeshHeadInput, build_mesh_head


def test_face_level_adapter_builds_sampler_command(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    proxy = tmp_path / "proxy.glb"
    checkpoint.write_bytes(b"placeholder")
    proxy.write_bytes(b"placeholder")
    adapter = build_mesh_head(
        "face-level",
        {
            "repo_dir": str(tmp_path),
            "python": "/usr/bin/python",
            "checkpoint": str(checkpoint),
            "point_samples": 256,
            "face_count": 128,
        },
    )

    command = adapter.build_command(
        MeshHeadInput(
            case_id="case_a",
            point_cloud_path=tmp_path / "unused.ply",
            proxy_mesh_path=proxy,
            output_dir=tmp_path / "out",
        ),
        tmp_path / "out" / "mesh.glb",
    )

    assert command[:2] == ["/usr/bin/python", "scripts/research/sample_face_level_from_mesh.py"]
    assert "--checkpoint" in command
    assert str(checkpoint) in command
    assert "--face-count" in command
    assert "128" in command
