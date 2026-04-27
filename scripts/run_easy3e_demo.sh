#!/usr/bin/env bash
# ClearMesh Easy3E demo driver.
#
# Runs the full Easy3E demo end-to-end on a GPU host:
#   1. Image → base mesh  (TRELLIS.2 + UltraShape)
#   2. Source view render + InstructPix2Pix text→image edit
#   3. Easy3E text-guided 3D edit (SLAT encode → repaint → decode)
#
# Prereqs (assumed already done on the host):
#   - scripts/setup/setup_all.sh has completed
#   - /workspace/TRELLIS.2/, /workspace/UltraShape-1.0/,
#     /workspace/checkpoints/ultrashape_v1.pt, /workspace/models/trellis2-4b/
#   - conda activate clearmesh
#
# Usage:
#   # Default: test_mug.png, "paint the mug bright red with a glossy finish"
#   bash scripts/run_easy3e_demo.sh
#
#   # Override instruction + input:
#   INPUT=path/to/img.png INSTRUCTION="make it look like copper" \\
#     bash scripts/run_easy3e_demo.sh
#
#   # Re-run just the edit step on an already-generated base mesh:
#   SKIP_BASE=1 INSTRUCTION="add a floral pattern" \\
#     bash scripts/run_easy3e_demo.sh
#
# Expected runtime on A100 80GB: ~4–7 min total.
# Artifacts land in ${OUTPUT_DIR} (default: /tmp/clearmesh_easy3e_demo/).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT="${INPUT:-${REPO_ROOT}/experiments/ultrashape/inputs/images/test_mug.png}"
INSTRUCTION="${INSTRUCTION:-paint the mug bright red with a glossy finish}"
VIEW="${VIEW:-front}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/clearmesh_easy3e_demo}"

ULTRASHAPE_DIR="${ULTRASHAPE_DIR:-/workspace/UltraShape-1.0}"
ULTRASHAPE_CKPT="${ULTRASHAPE_CKPT:-/workspace/checkpoints/ultrashape_v1.pt}"
TRELLIS2_DIR="${TRELLIS2_DIR:-/workspace/TRELLIS.2}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/trellis2-4b}"

SKIP_BASE_FLAG=""
if [[ "${SKIP_BASE:-0}" == "1" ]]; then
    SKIP_BASE_FLAG="--skip-base"
fi

echo "=== ClearMesh Easy3E Demo ==="
echo "  Input:       $INPUT"
echo "  Instruction: $INSTRUCTION"
echo "  View:        $VIEW"
echo "  Output dir:  $OUTPUT_DIR"
echo "  TRELLIS.2:   $TRELLIS2_DIR"
echo "  UltraShape:  $ULTRASHAPE_DIR"
echo "  Model dir:   $MODEL_DIR"
[[ -n "$SKIP_BASE_FLAG" ]] && echo "  SKIP_BASE:   yes (reusing $OUTPUT_DIR/base_mesh.glb)"
echo ""

if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: input image not found: $INPUT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --- Environment fingerprint (matches run_e2e_gpu.sh so both demos share
# the same header format in the console log) ---
echo "--- Environment fingerprint ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null \
    || echo "nvidia-smi: not available"
python3 -c "
import torch
print(f'torch={torch.__version__}  cuda={torch.version.cuda}  avail={torch.cuda.is_available()}')
try:
    import trellis2
    print(f'trellis2={getattr(trellis2, \"__version__\", \"unknown\")} @ {trellis2.__file__}')
except Exception as e:
    print(f'trellis2: IMPORT FAILED — {e}')
try:
    import diffusers
    print(f'diffusers={diffusers.__version__}')
except Exception as e:
    print(f'diffusers: IMPORT FAILED — {e}')
"
if [[ -d "$TRELLIS2_DIR/.git" ]]; then
  echo "trellis2_commit=$(git -C "$TRELLIS2_DIR" rev-parse --short HEAD)"
fi
if [[ -d "$REPO_ROOT/.git" ]]; then
  echo "clearmesh_commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
fi
echo ""

# --- Run demo via the Python driver. Its exit code is authoritative:
# 0 = overall_pass, 1 = stage failure, 2 = preflight failure. ---
cd "$REPO_ROOT"
python3 scripts/easy3e_demo.py \
    --input "$INPUT" \
    --instruction "$INSTRUCTION" \
    --view "$VIEW" \
    --output-dir "$OUTPUT_DIR" \
    --ultrashape-dir "$ULTRASHAPE_DIR" \
    --ultrashape-checkpoint "$ULTRASHAPE_CKPT" \
    --trellis2-dir "$TRELLIS2_DIR" \
    --model-dir "$MODEL_DIR" \
    $SKIP_BASE_FLAG
rc=$?

echo ""
echo "=== Easy3E demo complete (exit=$rc) ==="
echo "  Artifacts in: $OUTPUT_DIR"
if [[ -d "$OUTPUT_DIR" ]]; then
    ls -lh "$OUTPUT_DIR"/ 2>/dev/null || true
fi

REPORT="$OUTPUT_DIR/report.json"
if [[ -f "$REPORT" ]]; then
    echo ""
    echo "--- Report summary ---"
    if command -v jq >/dev/null 2>&1; then
        jq '{overall_pass,
             instruction,
             view,
             stages: (.stages | to_entries | map({stage: .key, pass: .value.pass, duration_s: .value.duration_s})),
             edited_mesh_stats,
             errors}' "$REPORT"
    else
        cat "$REPORT"
    fi
fi

exit $rc
