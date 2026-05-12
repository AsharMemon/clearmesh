#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_ENV_PATH="${REMOTE_ENV_PATH:-/home/ubuntu/.clearmesh_hf.env}"
INPUT_IMAGE="${INPUT_IMAGE:-/home/ubuntu/TRELLIS.2/assets/example_image/0e4984a9b3765ce80e9853443f9319ecedf90885c74b56cccfebc09402740f8a.webp}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/clearmesh_trellis2_smoke}"
DECIMATION_TARGET="${DECIMATION_TARGET:-250000}"
TEXTURE_SIZE="${TEXTURE_SIZE:-1024}"

"$(dirname "$0")/run_remote.sh" "$INSTANCE_ID" <<EOF
set -euo pipefail
if [ -f "$REMOTE_ENV_PATH" ]; then
  source "$REMOTE_ENV_PATH"
fi
cd /home/ubuntu/TRELLIS.2
source /home/ubuntu/trellis2-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/ubuntu/TRELLIS.2:${PYTHONPATH:-}
export ATTN_BACKEND=flash_attn
rm -rf "$OUTPUT_DIR"
python /home/ubuntu/clearmesh/scripts/product/run_trellis2_proxy.py \\
  --input "$INPUT_IMAGE" \\
  --output-dir "$OUTPUT_DIR" \\
  --output-name trellis_proxy.glb \\
  --decimation-target "$DECIMATION_TARGET" \\
  --texture-size "$TEXTURE_SIZE" \\
  --seed 0
ls -lh "$OUTPUT_DIR"
EOF
