#!/usr/bin/env bash
# Full production-route smoke:
# TRELLIS.2 -> reference refinement -> retopo plan -> chart remesh/stitch
# -> quad remesh -> feature projection -> optional mesh head -> gates -> export.
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
STATE_ROOT="${STATE_ROOT:-/tmp/clearmesh_production_path_state}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/tmp/clearmesh_production_path_artifacts}"
INPUT_IMAGE="${INPUT_IMAGE:-/home/ubuntu/TRELLIS.2/assets/example_image/0e4984a9b3765ce80e9853443f9319ecedf90885c74b56cccfebc09402740f8a.webp}"
CONFIG_JSON="${CONFIG_JSON:-/home/ubuntu/clearmesh/configs/meshripple.thunder.smoke.json}"
REMOTE_ENV_PATH="${REMOTE_ENV_PATH:-/home/ubuntu/.clearmesh_hf.env}"
REFERENCE_MODE="${REFERENCE_MODE:-ultrashape}"
CHART_ENGINE="${CHART_ENGINE:-auto}"
QUAD_ENGINE="${QUAD_ENGINE:-auto}"
MESH_HEAD_POLICY="${MESH_HEAD_POLICY:-passport}"
POINT_BUDGET="${POINT_BUDGET:-40960}"
ULTRASHAPE_STEPS="${ULTRASHAPE_STEPS:-50}"
ULTRASHAPE_OCTREE_RES="${ULTRASHAPE_OCTREE_RES:-1024}"
ULTRASHAPE_NUM_LATENTS="${ULTRASHAPE_NUM_LATENTS:-32768}"
ULTRASHAPE_CHUNK_SIZE="${ULTRASHAPE_CHUNK_SIZE:-8000}"
ULTRASHAPE_SCALE="${ULTRASHAPE_SCALE:-0.99}"
ULTRASHAPE_SEED="${ULTRASHAPE_SEED:-42}"
COARSE_ADAPTER_ENGINE="${COARSE_ADAPTER_ENGINE:-auto}"
COARSE_ADAPTER_TARGET_FACES="${COARSE_ADAPTER_TARGET_FACES:-150000}"
COARSE_ADAPTER_SAMPLE_POINTS="${COARSE_ADAPTER_SAMPLE_POINTS:-180000}"
COARSE_ADAPTER_VOXEL_RESOLUTION="${COARSE_ADAPTER_VOXEL_RESOLUTION:-192}"
COARSE_ADAPTER_VOXEL_DILATE="${COARSE_ADAPTER_VOXEL_DILATE:-2}"
COARSE_ADAPTER_VOXEL_CLOSE="${COARSE_ADAPTER_VOXEL_CLOSE:-1}"
COARSE_ADAPTER_MESH_VOXEL_MAX_FACES="${COARSE_ADAPTER_MESH_VOXEL_MAX_FACES:-75000}"
COARSE_ADAPTER_HULL_MAX_POINTS="${COARSE_ADAPTER_HULL_MAX_POINTS:-20000}"
COARSE_ADAPTER_REQUIRE_WATERTIGHT="${COARSE_ADAPTER_REQUIRE_WATERTIGHT:-false}"
COARSE_ADAPTER_MAX_BOUNDARY_LOOPS="${COARSE_ADAPTER_MAX_BOUNDARY_LOOPS:-25}"
COARSE_ADAPTER_MAX_NONMANIFOLD_EDGES="${COARSE_ADAPTER_MAX_NONMANIFOLD_EDGES:-1000}"
REFERENCE_COMPONENT_FILTER_ENABLED="${REFERENCE_COMPONENT_FILTER_ENABLED:-true}"
REFERENCE_DOMINANT_COMPONENT_FACE_RATIO="${REFERENCE_DOMINANT_COMPONENT_FACE_RATIO:-0.9}"

emit_export() {
  printf 'export %s=%q\n' "$1" "$2"
}

{
  emit_export REMOTE_DIR "$REMOTE_DIR"
  emit_export STATE_ROOT "$STATE_ROOT"
  emit_export ARTIFACT_ROOT "$ARTIFACT_ROOT"
  emit_export INPUT_IMAGE "$INPUT_IMAGE"
  emit_export CONFIG_JSON "$CONFIG_JSON"
  emit_export REMOTE_ENV_PATH "$REMOTE_ENV_PATH"
  emit_export REFERENCE_MODE "$REFERENCE_MODE"
  emit_export CHART_ENGINE "$CHART_ENGINE"
  emit_export QUAD_ENGINE "$QUAD_ENGINE"
  emit_export MESH_HEAD_POLICY "$MESH_HEAD_POLICY"
  emit_export POINT_BUDGET "$POINT_BUDGET"
  emit_export ULTRASHAPE_STEPS "$ULTRASHAPE_STEPS"
  emit_export ULTRASHAPE_OCTREE_RES "$ULTRASHAPE_OCTREE_RES"
  emit_export ULTRASHAPE_NUM_LATENTS "$ULTRASHAPE_NUM_LATENTS"
  emit_export ULTRASHAPE_CHUNK_SIZE "$ULTRASHAPE_CHUNK_SIZE"
  emit_export ULTRASHAPE_SCALE "$ULTRASHAPE_SCALE"
  emit_export ULTRASHAPE_SEED "$ULTRASHAPE_SEED"
  emit_export COARSE_ADAPTER_ENGINE "$COARSE_ADAPTER_ENGINE"
  emit_export COARSE_ADAPTER_TARGET_FACES "$COARSE_ADAPTER_TARGET_FACES"
  emit_export COARSE_ADAPTER_SAMPLE_POINTS "$COARSE_ADAPTER_SAMPLE_POINTS"
  emit_export COARSE_ADAPTER_VOXEL_RESOLUTION "$COARSE_ADAPTER_VOXEL_RESOLUTION"
  emit_export COARSE_ADAPTER_VOXEL_DILATE "$COARSE_ADAPTER_VOXEL_DILATE"
  emit_export COARSE_ADAPTER_VOXEL_CLOSE "$COARSE_ADAPTER_VOXEL_CLOSE"
  emit_export COARSE_ADAPTER_MESH_VOXEL_MAX_FACES "$COARSE_ADAPTER_MESH_VOXEL_MAX_FACES"
  emit_export COARSE_ADAPTER_HULL_MAX_POINTS "$COARSE_ADAPTER_HULL_MAX_POINTS"
  emit_export COARSE_ADAPTER_REQUIRE_WATERTIGHT "$COARSE_ADAPTER_REQUIRE_WATERTIGHT"
  emit_export COARSE_ADAPTER_MAX_BOUNDARY_LOOPS "$COARSE_ADAPTER_MAX_BOUNDARY_LOOPS"
  emit_export COARSE_ADAPTER_MAX_NONMANIFOLD_EDGES "$COARSE_ADAPTER_MAX_NONMANIFOLD_EDGES"
  emit_export REFERENCE_COMPONENT_FILTER_ENABLED "$REFERENCE_COMPONENT_FILTER_ENABLED"
  emit_export REFERENCE_DOMINANT_COMPONENT_FACE_RATIO "$REFERENCE_DOMINANT_COMPONENT_FACE_RATIO"
  cat <<'REMOTE_SCRIPT'
set -euo pipefail
if [ -f "${REMOTE_ENV_PATH}" ]; then
  source "${REMOTE_ENV_PATH}"
fi
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/ubuntu/TRELLIS.2:${PYTHONPATH:-}
export ATTN_BACKEND=flash_attn

cd "${REMOTE_DIR}"
rm -rf "${STATE_ROOT}" "${ARTIFACT_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/uploads/production"
cp "${INPUT_IMAGE}" "${ARTIFACT_ROOT}/uploads/production/input.webp"

METADATA_JSON=$(mktemp /tmp/clearmesh_production_metadata.XXXXXX.json)
python3 - <<'PY' > "$METADATA_JSON"
import json
import os

reference_mode = os.environ.get("REFERENCE_MODE", "poisson")
metadata = {
    "project_id": "thunder_production_path",
    "trellis_command": "/home/ubuntu/trellis2-venv/bin/python -u /home/ubuntu/clearmesh/scripts/product/run_trellis2_proxy.py --input {input_path} --output-dir {output_dir} --output-name trellis_proxy.glb --decimation-target 250000 --texture-size 1024 --seed 0",
    "trellis_cwd": "/home/ubuntu/TRELLIS.2",
    "trellis_timeout_seconds": 7200,
    "coarse_adapter_enabled": True,
    "coarse_adapter_engine": os.environ.get("COARSE_ADAPTER_ENGINE", "auto"),
    "coarse_adapter_target_faces": int(os.environ.get("COARSE_ADAPTER_TARGET_FACES", "150000")),
    "coarse_adapter_sample_points": int(os.environ.get("COARSE_ADAPTER_SAMPLE_POINTS", "180000")),
    "coarse_adapter_min_component_faces": 64,
    "coarse_adapter_keep_largest_components": 128,
    "coarse_adapter_max_output_components": 1,
    "coarse_adapter_max_boundary_loops": int(os.environ.get("COARSE_ADAPTER_MAX_BOUNDARY_LOOPS", "25")),
    "coarse_adapter_max_nonmanifold_edges": int(os.environ.get("COARSE_ADAPTER_MAX_NONMANIFOLD_EDGES", "1000")),
    "coarse_adapter_require_watertight": os.environ.get("COARSE_ADAPTER_REQUIRE_WATERTIGHT", "false").lower() in {"1", "true", "yes", "y", "on"},
    "coarse_adapter_voxel_resolution": int(os.environ.get("COARSE_ADAPTER_VOXEL_RESOLUTION", "192")),
    "coarse_adapter_voxel_dilate": int(os.environ.get("COARSE_ADAPTER_VOXEL_DILATE", "2")),
    "coarse_adapter_voxel_close": int(os.environ.get("COARSE_ADAPTER_VOXEL_CLOSE", "1")),
    "coarse_adapter_mesh_voxel_max_faces": int(os.environ.get("COARSE_ADAPTER_MESH_VOXEL_MAX_FACES", "75000")),
    "coarse_adapter_hull_max_points": int(os.environ.get("COARSE_ADAPTER_HULL_MAX_POINTS", "20000")),
    "coarse_adapter_fallback": "convex_hull",
    "reference_refinement_enabled": True,
    "reference_component_filter_enabled": os.environ.get("REFERENCE_COMPONENT_FILTER_ENABLED", "true").lower() in {"1", "true", "yes", "y", "on"},
    "reference_dominant_component_face_ratio": float(os.environ.get("REFERENCE_DOMINANT_COMPONENT_FACE_RATIO", "0.9")),
    "surface_normalization_enabled": True,
    "surface_engine": "auto",
    "surface_target_faces": 50000,
    "retopology_planning_enabled": True,
    "retopo_merge_small_charts": True,
    "retopo_max_report_charts": 512,
    "chart_remesh_enabled": True,
    "chart_remesh_engine": os.environ.get("CHART_ENGINE", "auto"),
    "chart_remesh_max_charts": 8,
    "chart_remesh_min_chart_faces": 16,
    "chart_stitch_enabled": True,
    "chart_stitch_prefer_for_projection": True,
    "quad_remesh_enabled": True,
    "quad_remesh_engine": os.environ.get("QUAD_ENGINE", "auto"),
    "quad_target_faces": 5000,
    "feature_projection_enabled": True,
    "feature_projection_as_final": False,
    "part_structure_fallback": "meshmosaic_components",
    "component_part_max_parts": 8,
    "part_mesh_generation": False,
    "mesh_head_policy": os.environ.get("MESH_HEAD_POLICY", "passport"),
    "cleanup_enabled": True,
    "blender_gates_enabled": True,
    "production_require_blender": False,
}
if reference_mode == "ultrashape":
    steps = os.environ.get("ULTRASHAPE_STEPS", "50")
    octree_res = os.environ.get("ULTRASHAPE_OCTREE_RES", "1024")
    num_latents = os.environ.get("ULTRASHAPE_NUM_LATENTS", "32768")
    chunk_size = os.environ.get("ULTRASHAPE_CHUNK_SIZE", "8000")
    scale = os.environ.get("ULTRASHAPE_SCALE", "0.99")
    seed = os.environ.get("ULTRASHAPE_SEED", "42")
    metadata["reference_refinement_command"] = f"/home/ubuntu/ultrashape-venv/bin/python -u /home/ubuntu/clearmesh/scripts/product/run_ultrashape_refinement.py --mesh {{input_mesh}} --image {{reference_image}} --output-dir {{output_dir}} --ultrashape-dir /workspace/UltraShape-1.0 --checkpoint /workspace/checkpoints/ultrashape_v1.pt --num-steps {steps} --octree-resolution {octree_res} --num-latents {num_latents} --chunk-size {chunk_size} --scale {scale} --seed {seed} --remove-bg"
    metadata["reference_refinement_timeout_seconds"] = 7200
elif reference_mode == "manifoldplus":
    metadata["manifoldization_command"] = "/home/ubuntu/clearmesh-venv/bin/python /home/ubuntu/clearmesh/scripts/product/run_manifoldplus.py --input {input_mesh} --output-dir {output_dir} --binary /home/ubuntu/manifold-tools/ManifoldPlus/build/manifold --depth 8"
    metadata["reference_refinement_timeout_seconds"] = 3600
else:
    metadata["reference_refinement_engine"] = "poisson"
print(json.dumps(metadata, indent=2, sort_keys=True))
PY

JOB_ID=$(/home/ubuntu/clearmesh-venv/bin/python scripts/product/create_local_job.py \
  --state-root "${STATE_ROOT}" \
  --team-id team_dev \
  --user-id user_dev \
  --input-uri local://uploads/production/input.webp \
  --grant-credits 50 \
  --metadata-json "$METADATA_JSON" \
  --quality-tier draft)

/home/ubuntu/clearmesh-venv/bin/python scripts/product/run_pipeline_worker.py \
  --state-root "${STATE_ROOT}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --once \
  --execute-heavy \
  --mesh-head meshripple \
  --mesh-head-config-json "${CONFIG_JSON}" \
  --preferred-point-budget "${POINT_BUDGET}"

/home/ubuntu/clearmesh-venv/bin/python - <<PY
import json
import os
from pathlib import Path

job_id = "$JOB_ID"
state_root = Path(os.environ["STATE_ROOT"])
job = json.loads((state_root / "jobs" / f"{job_id}.json").read_text())
assets = job["assets"]
summary = {
    "job_id": job_id,
    "status": job["status"],
    "steps": {step["name"]: step["status"] for step in job["steps"]},
    "asset_kinds": [asset["kind"] for asset in assets],
    "coarse_proxy_mesh": next((asset["uri"] for asset in assets if asset["kind"] == "coarse_proxy_mesh"), None),
    "reference_mesh": next((asset["uri"] for asset in assets if asset["kind"] in {"reference_mesh", "refined_reference_mesh"}), None),
    "retopology_plan": next((asset["uri"] for asset in assets if asset["kind"] == "retopology_plan"), None),
    "chart_remesh_manifest": next((asset["uri"] for asset in assets if asset["kind"] == "chart_remesh_manifest"), None),
    "chart_stitched_mesh": next((asset["uri"] for asset in assets if asset["kind"] == "chart_stitched_mesh"), None),
    "projected_quad_mesh": next((asset["uri"] for asset in assets if asset["kind"] == "projected_quad_mesh"), None),
    "production_gate_report": next((asset["uri"] for asset in assets if asset["kind"] == "production_gate_report"), None),
    "mesh_eval_report": next((asset["uri"] for asset in assets if asset["kind"] == "mesh_eval_report"), None),
    "export_mesh": next((asset["uri"] for asset in assets if asset["kind"] == "export_mesh"), None),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
REMOTE_SCRIPT
} | "$(dirname "$0")/run_remote.sh" "$INSTANCE_ID"
