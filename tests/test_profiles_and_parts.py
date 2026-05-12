from clearmesh.parts.manifest import PartRecord, load_parts_manifest, write_parts_manifest
from clearmesh.product.models import GenerationRequest, JobRecord
from clearmesh.product.profiles import runtime_quote


def test_runtime_quote_includes_part_parallelism():
    quote = runtime_quote("high", includes_trellis=False, enable_parts=True, part_count=3)
    assert quote["profile"] == "high"
    assert quote["preview_seconds"]["low"] < quote["end_to_end_seconds"]["high"]
    assert quote["part_aware"]["estimated_part_count"] == 3
    assert quote["part_aware"]["serial_mesh_head_seconds"]["low"] == quote["mesh_head_seconds"]["low"] * 3
    assert quote["part_aware"]["ideal_parallel_mesh_head_seconds"]["low"] == quote["mesh_head_seconds"]["low"]


def test_job_steps_include_unified_surface_contract():
    job = JobRecord.create("team", "user", GenerationRequest(input_uri="local://input.png"))
    names = [step.name for step in job.steps]
    assert names[names.index("trellis_proxy") + 1 : names.index("point_cloud_bridge")] == [
        "coarse_adapter",
        "reference_refinement",
        "mesh_passport",
        "surface_normalization",
        "shrinkwrap_projection",
        "retopology_planning",
        "chart_remesh",
        "chart_stitch",
        "quad_remesh",
        "feature_projection",
        "preview_publish",
    ]
    assert "production_gate" in names


def test_parts_manifest_roundtrip(tmp_path):
    path = tmp_path / "parts_manifest.json"
    write_parts_manifest(
        path,
        [PartRecord(id="body", label="Body", point_cloud_path="body.ply", bbox=[0, 0, 0, 1, 1, 1])],
        source="unit-test",
    )
    parts = load_parts_manifest(path)
    assert len(parts) == 1
    assert parts[0].id == "body"
    assert parts[0].label == "Body"
    assert parts[0].point_cloud_path == "body.ply"
    assert parts[0].bbox == [0, 0, 0, 1, 1, 1]
