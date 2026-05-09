import json

from scripts.research.select_face_oracle_passes import select_rows


def test_select_face_oracle_passes_filters_and_uses_repair(tmp_path):
    rows = [
        {
            "path": str(tmp_path / "ok.glb"),
            "source_name": "ok",
            "family": "indexed",
            "num_bins": 512,
            "within_face_order": "indexed",
            "source_watertight": True,
            "source_faces": 12,
            "token_faces": 12,
            "quantization_face_loss": 0,
            "watertight_edge_graph": False,
            "boundary_edge_count": 3,
            "nonmanifold_edge_count": 0,
            "edge_pairing_ratio": 0.9,
            "repair": {
                "manifold": {
                    "watertight_edge_graph": True,
                    "boundary_edge_count": 0,
                    "nonmanifold_edge_count": 0,
                    "edge_pairing_ratio": 1.0,
                }
            },
        },
        {
            "path": str(tmp_path / "bad.glb"),
            "source_name": "bad",
            "family": "indexed",
            "num_bins": 512,
            "within_face_order": "indexed",
            "source_watertight": True,
            "source_faces": 12,
            "token_faces": 10,
            "quantization_face_loss": 2,
            "watertight_edge_graph": False,
            "boundary_edge_count": 4,
            "nonmanifold_edge_count": 0,
            "edge_pairing_ratio": 0.8,
            "repair": {
                "manifold": {
                    "watertight_edge_graph": False,
                    "boundary_edge_count": 4,
                    "nonmanifold_edge_count": 0,
                    "edge_pairing_ratio": 0.8,
                }
            },
        },
        {
            "path": str(tmp_path / "paper.glb"),
            "source_name": "paper",
            "family": "paper",
            "num_bins": 512,
            "within_face_order": "preserve",
            "watertight_edge_graph": True,
            "repair": {},
        },
    ]

    result = select_rows(
        rows,
        families={"indexed"},
        bins={512},
        orders={"indexed"},
        repair_mode="manifold",
        max_quantization_face_loss=0,
        max_boundary_edges=0,
        max_nonmanifold_edges=0,
        min_edge_pairing_ratio=1.0,
        require_source_watertight=True,
        output_dir=tmp_path / "selection",
        copy_mode="none",
    )

    assert result["summary"]["selected_count"] == 2
    assert result["summary"]["passing_count"] == 1
    assert result["summary"]["pass_rate"] == 0.5
    assert result["passing"][0]["source_name"] == "ok"
    assert result["failing"][0]["source_name"] == "bad"
    assert "quantization_face_loss" in result["failing"][0]["reasons"]


def test_select_face_oracle_passes_cli_writes_manifests(tmp_path, capsys):
    from scripts.research import select_face_oracle_passes

    oracle_rows = tmp_path / "oracle_rows.jsonl"
    oracle_rows.write_text(
        json.dumps(
            {
                "path": str(tmp_path / "ok.glb"),
                "source_name": "ok",
                "family": "indexed",
                "num_bins": 512,
                "within_face_order": "indexed",
                "quantization_face_loss": 0,
                "watertight_edge_graph": True,
                "boundary_edge_count": 0,
                "nonmanifold_edge_count": 0,
                "edge_pairing_ratio": 1.0,
                "repair": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    import sys

    original_argv = sys.argv
    try:
        sys.argv = [
            "select_face_oracle_passes.py",
            "--oracle-rows",
            str(oracle_rows),
            "--output-dir",
            str(output_dir),
            "--families",
            "indexed",
            "--bins",
            "512",
            "--orders",
            "indexed",
            "--min-pass-rate",
            "1.0",
        ]
        assert select_face_oracle_passes.main() == 0
    finally:
        sys.argv = original_argv

    assert (output_dir / "selection_summary.json").exists()
    assert (output_dir / "pass_manifest.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert "passing_count" in capsys.readouterr().out
