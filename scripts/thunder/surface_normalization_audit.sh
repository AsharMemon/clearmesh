#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/meshripple-venv/bin/python}"
PROXY_MESH="${PROXY_MESH:-/tmp/clearmesh_batch2_artifacts/projects/trellis2_meshripple_batch/jobs/job_334383d7ebee43fdbc59d18a716aecf2/trellis_proxy/trellis_proxy.glb}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/clearmesh_surface_normalization_audit}"
TARGET_FACES="${TARGET_FACES:-50000}"
SAMPLE_POINTS="${SAMPLE_POINTS:-120000}"
POISSON_DEPTH="${POISSON_DEPTH:-8}"
DENSITY_QUANTILE="${DENSITY_QUANTILE:-0.02}"
MESHRIPPLE_REPO="${MESHRIPPLE_REPO:-/home/ubuntu/mesh-heads/MeshRipple}"
MESHRIPPLE_CONFIG="${MESHRIPPLE_CONFIG:-/home/ubuntu/clearmesh/configs/meshripple.thunder.official_dense.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/run_remote.sh" "$INSTANCE_ID" <<EOF
set -euo pipefail
cd "$REMOTE_DIR"
if [ ! -f "$PROXY_MESH" ]; then
  echo "proxy mesh not found: $PROXY_MESH" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
CONTROL_MESH="$OUTPUT_DIR/control_poisson_${TARGET_FACES}.obj"
REPORT="$OUTPUT_DIR/control_poisson_${TARGET_FACES}.json"
RAW_AUDIT="$OUTPUT_DIR/meshripple_preprocess_raw.json"
CONTROL_AUDIT="$OUTPUT_DIR/meshripple_preprocess_control.json"

"$PYTHON_BIN" scripts/product/create_mesh_passport.py \
  --mesh "$PROXY_MESH" \
  --output "$OUTPUT_DIR/raw_mesh_passport.json"

"$PYTHON_BIN" scripts/product/normalize_surface.py \
  --input "$PROXY_MESH" \
  --output "\$CONTROL_MESH" \
  --report "\$REPORT" \
  --engine poisson \
  --target-faces "$TARGET_FACES" \
  --sample-points "$SAMPLE_POINTS" \
  --poisson-depth "$POISSON_DEPTH" \
  --density-quantile "$DENSITY_QUANTILE"

"$PYTHON_BIN" scripts/product/create_mesh_passport.py \
  --mesh "\$CONTROL_MESH" \
  --output "$OUTPUT_DIR/control_mesh_passport.json"

"$PYTHON_BIN" scripts/product/meshripple_preprocess_audit.py \
  --meshripple-repo "$MESHRIPPLE_REPO" \
  --config "$MESHRIPPLE_CONFIG" \
  --mesh "$PROXY_MESH" \
  --dec-to-facenum 5000 > "\$RAW_AUDIT"

"$PYTHON_BIN" scripts/product/meshripple_preprocess_audit.py \
  --meshripple-repo "$MESHRIPPLE_REPO" \
  --config "$MESHRIPPLE_CONFIG" \
  --mesh "\$CONTROL_MESH" \
  --dec-to-facenum 5000 > "\$CONTROL_AUDIT"

printf '\\n=== raw passport ===\\n'
cat "$OUTPUT_DIR/raw_mesh_passport.json"
printf '\\n=== control normalization report ===\\n'
cat "\$REPORT"
printf '\\n=== control passport ===\\n'
cat "$OUTPUT_DIR/control_mesh_passport.json"
printf '\\n=== MeshRipple raw preprocess ===\\n'
cat "\$RAW_AUDIT"
printf '\\n=== MeshRipple control preprocess ===\\n'
cat "\$CONTROL_AUDIT"
printf '\\n=== output dir ===\\n%s\\n' "$OUTPUT_DIR"
EOF

