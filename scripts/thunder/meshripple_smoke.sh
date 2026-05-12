#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
PROXY_MESH="${PROXY_MESH:-/home/ubuntu/clearmesh/experiments/ultrashape/inputs/coarse_meshes/test_mug.glb}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/clearmesh_meshripple_smoke_micro}"
CONFIG_JSON="${CONFIG_JSON:-/home/ubuntu/clearmesh/configs/meshripple.thunder.smoke.json}"

"$(dirname "$0")/run_remote.sh" "$INSTANCE_ID" <<EOF
set -euo pipefail
cd "$REMOTE_DIR"
PROXY="$PROXY_MESH"
OUT="$OUTPUT_DIR"
rm -rf "\$OUT"
mkdir -p "\$OUT"

printf 'case_id,method,mesh_path,reference_path\nmeshripple_mug,trellis_proxy,%s,\n' "\$PROXY" > "\$OUT/manifest.csv"
/home/ubuntu/clearmesh-venv/bin/python scripts/data/sample_point_clouds.py \\
  --manifest "\$OUT/manifest.csv" \\
  --output-dir "\$OUT/pointclouds" \\
  --output-manifest "\$OUT/pointcloud_manifest.csv" \\
  --budgets 40960 \\
  --formats ply

PLY="\$OUT/pointclouds/meshripple_mug/trellis_proxy/points_40960.ply"
/home/ubuntu/clearmesh-venv/bin/python scripts/product/run_mesh_head.py \\
  --head meshripple \\
  --case-id meshripple_mug \\
  --point-cloud "\$PLY" \\
  --proxy-mesh "\$PROXY" \\
  --output-dir "\$OUT/mesh_head" \\
  --config-json "$CONFIG_JSON"

GEN="\$OUT/mesh_head/meshripple/_val_generate_k20_p0.9_t0.9/test_mug_y.obj"
printf 'case_id,method,mesh_path,reference_path\nmeshripple_mug,meshripple_micro,%s,%s\n' "\$GEN" "\$PROXY" > "\$OUT/eval_manifest.csv"
/home/ubuntu/clearmesh-venv/bin/python scripts/eval/evaluate_meshes.py \\
  --manifest "\$OUT/eval_manifest.csv" \\
  --output "\$OUT/eval_report.json" \\
  --samples 1000

/home/ubuntu/clearmesh-venv/bin/python - <<'PY'
import json
from pathlib import Path

out = Path("$OUTPUT_DIR")
report = json.loads((out / "eval_report.json").read_text())
result = report["results"][0]
print(json.dumps({
    "mesh_path": result["mesh_path"],
    "mesh_ok": result["mesh_metrics"]["ok"],
    "vertex_count": result["mesh_metrics"]["vertex_count"],
    "face_count": result["mesh_metrics"]["face_count"],
    "connected_components": result["mesh_metrics"]["connected_components"],
    "watertight": result["mesh_metrics"]["watertight"],
    "chamfer_l2": result["pair_metrics"]["chamfer_l2"],
    "normal_consistency": result["pair_metrics"]["normal_consistency"],
    "eval_report": str(out / "eval_report.json"),
}, indent=2, sort_keys=True))
PY
EOF
