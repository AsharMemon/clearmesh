import json

import numpy as np
import trimesh

from clearmesh.mesh_heads.face_indexed import encode_mesh_to_indexed_face_tokens
from clearmesh.mesh_heads.face_tokens import encode_mesh_to_paper_face_tokens
from scripts.research.face_token_oracle import (
    _analyze_npz_record,
    _records_from_dataset,
    _summarize,
)


def _write_shard(path, sequence, indexed, *, source_faces):
    np.savez_compressed(
        path,
        paper_tokens=np.asarray(sequence.tokens, dtype=np.int16),
        indexed_vertices=np.asarray(indexed.vertices, dtype=np.int16),
        indexed_faces=np.asarray(indexed.faces, dtype=np.int32),
        num_bins=np.asarray([sequence.num_bins], dtype=np.int32),
        paper_within_face_order=np.asarray(["preserve"]),
    )
    return {
        "path": str(path),
        "source_name": path.stem,
        "source_faces": int(source_faces),
    }


def test_face_token_oracle_reports_and_repairs_token_holes(tmp_path):
    mesh = trimesh.creation.box()
    sequence = encode_mesh_to_paper_face_tokens(mesh, num_bins=128)
    indexed = encode_mesh_to_indexed_face_tokens(mesh, num_bins=128)

    ok_path = tmp_path / "cube_ok.npz"
    damaged_path = tmp_path / "cube_missing_one_face.npz"
    ok_record = _write_shard(ok_path, sequence, indexed, source_faces=len(mesh.faces))

    damaged_sequence = encode_mesh_to_paper_face_tokens(mesh, num_bins=128)
    damaged_sequence = type(damaged_sequence)(
        tokens=np.concatenate([damaged_sequence.tokens[:-1], damaged_sequence.tokens[:1]], axis=0),
        num_bins=damaged_sequence.num_bins,
        transform=damaged_sequence.transform,
    )
    damaged_record = _write_shard(damaged_path, damaged_sequence, indexed, source_faces=len(mesh.faces))

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(ok_record) + "\n" + json.dumps(damaged_record) + "\n", encoding="utf-8")
    records = _records_from_dataset(tmp_path, manifest)
    rows = []
    for record in records:
        rows.extend(_analyze_npz_record((record, ["paper"], ["none", "dedupe", "manifold"])))

    summary = _summarize(rows, ["none", "dedupe", "manifold"])
    paper = summary["groups"][0]

    assert paper["family"] == "paper"
    assert paper["samples"] == 2
    assert paper["token_watertight"] == 1
    assert paper["repair_dedupe_watertight"] == 2
    assert paper["failure_causes"]["boundary_edges"] == 1
