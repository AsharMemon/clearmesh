import json

from clearmesh.mesh_heads.face_dataset_gate import (
    evaluate_face_dataset_records,
    thresholds_for_profile,
)


def test_strict_face_dataset_gate_accepts_closed_tokens():
    report = evaluate_face_dataset_records(
        [
            {
                "path": "cube.npz",
                "decoded_watertight": True,
                "token_watertight_edge_graph": True,
                "token_boundary_edge_count": 0,
                "token_nonmanifold_edge_count": 0,
                "token_edge_pairing_ratio": 1.0,
            }
        ],
        thresholds_for_profile("strict"),
    )

    assert report["passes"] is True
    assert report["passing"] == 1


def test_strict_face_dataset_gate_rejects_fragmented_proxy():
    report = evaluate_face_dataset_records(
        [
            {
                "path": "fragmented.npz",
                "decoded_watertight": False,
                "token_watertight_edge_graph": False,
                "token_boundary_edge_count": 41,
                "token_nonmanifold_edge_count": 3,
                "token_edge_pairing_ratio": 0.78,
            }
        ],
        thresholds_for_profile("strict"),
    )

    assert report["passes"] is False
    assert report["failing"] == 1
    assert any("not watertight" in item for item in report["results"][0]["violations"])


def test_proxy_face_dataset_gate_reports_without_rejecting():
    report = evaluate_face_dataset_records(
        [
            {
                "path": "fragmented.npz",
                "decoded_watertight": False,
                "token_watertight_edge_graph": False,
                "token_boundary_edge_count": 41,
                "token_nonmanifold_edge_count": 3,
                "token_edge_pairing_ratio": 0.78,
            }
        ],
        thresholds_for_profile("proxy"),
    )

    assert json.dumps(report)
    assert report["passes"] is True


def test_strict_face_dataset_gate_can_check_paper_token_fields():
    report = evaluate_face_dataset_records(
        [
            {
                "path": "paper_cube.npz",
                "decoded_watertight": False,
                "token_watertight_edge_graph": False,
                "token_boundary_edge_count": 99,
                "token_nonmanifold_edge_count": 9,
                "token_edge_pairing_ratio": 0.5,
                "paper_decoded_watertight": True,
                "paper_token_watertight_edge_graph": True,
                "paper_token_boundary_edge_count": 0,
                "paper_token_nonmanifold_edge_count": 0,
                "paper_token_edge_pairing_ratio": 1.0,
            }
        ],
        thresholds_for_profile("strict"),
        token_family="paper",
    )

    assert report["token_family"] == "paper"
    assert report["passes"] is True
    assert report["passing"] == 1


def test_face_dataset_gate_rejects_empty_manifests():
    report = evaluate_face_dataset_records([], thresholds_for_profile("strict"))

    assert report["sample_count"] == 0
    assert report["pass_rate"] == 0.0
    assert report["passes"] is False
