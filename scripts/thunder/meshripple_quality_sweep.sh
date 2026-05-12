#!/usr/bin/env bash
set -euo pipefail
INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
PROXY_MESH="${PROXY_MESH:-/tmp/clearmesh_batch2_artifacts/projects/trellis2_meshripple_batch/jobs/job_334383d7ebee43fdbc59d18a716aecf2/trellis_proxy/trellis_proxy.glb}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/clearmesh_meshripple_quality_sweep}"
POINT_BUDGET="${POINT_BUDGET:-40960}"
SAMPLES="${SAMPLES:-10000}"
CONFIG_SPECS="${CONFIG_SPECS:-512=/home/ubuntu/clearmesh/configs/meshripple.thunder.quality.json,1k=/home/ubuntu/clearmesh/configs/meshripple.thunder.1k.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/run_remote.sh" "$INSTANCE_ID" <<EOF
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/ubuntu/TRELLIS.2:${PYTHONPATH:-}
export ATTN_BACKEND=flash_attn
cd "$REMOTE_DIR"
if [ ! -f "$PROXY_MESH" ]; then
  echo "proxy mesh not found: $PROXY_MESH" >&2
  exit 1
fi
ARGS=()
IFS=',' read -ra SPECS <<< "$CONFIG_SPECS"
for spec in "\${SPECS[@]}"; do
  ARGS+=(--config "\$spec")
done
/home/ubuntu/clearmesh-venv/bin/python scripts/product/meshripple_quality_sweep.py \
  --proxy-mesh "$PROXY_MESH" \
  --output-dir "$OUTPUT_DIR" \
  --case-id thunder_proxy_case \
  --point-budget "$POINT_BUDGET" \
  --samples "$SAMPLES" \
  "\${ARGS[@]}"
cat "$OUTPUT_DIR/quality_sweep.json"
EOF
