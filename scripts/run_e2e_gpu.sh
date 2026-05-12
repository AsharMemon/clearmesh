#!/usr/bin/env bash
# ClearMesh end-to-end GPU smoke test.
#
# Purpose: Runs the full ClearMeshPipeline on a single reference image
# end-to-end, validating that every stage works with real TRELLIS.2 +
# UltraShape checkpoints. Meant to be run on a GPU host (RunPod / Vast)
# after setup_all.sh has completed.
#
# Emits two artifacts to attach to the runbook report:
#   - $OUTPUT               — the generated .glb
#   - $REPORT (JSON)        — machine-readable per-stage status
#                             (produced by scripts/e2e_smoke.py)
#
# What it exercises (in order):
#   1. Background removal (rembg)
#   2. TRELLIS.2 4B coarse generation
#   3. UltraShape Stage-2 refinement
#   4. Mesh extraction + repair
#   5. Print-preparation (orientation, watertightness check)
#   6. GLB export
#
# Intentionally *off* for the first smoke (enable after this passes):
#   - Part decomposition (--decompose)
#   - Super-resolution  (--super-res)
#   - Retopology        (--retopo)
#   - Textures          (--textures)
#   - Rigging           (--rig)
#
# Usage:
#   # From the repo root, on a GPU host:
#   bash scripts/run_e2e_gpu.sh
#
#   # With a custom input:
#   bash scripts/run_e2e_gpu.sh /path/to/my_image.png
#
# Expected runtime: ~90-180s on A100 (dominated by Stage 2 refinement).
# Expected output: $OUTPUT + $REPORT + stdout summary.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT="${1:-${REPO_ROOT}/experiments/ultrashape/inputs/images/test_mug.png}"
OUTPUT="${2:-/tmp/clearmesh_e2e_out.glb}"
REPORT="${REPORT:-/tmp/clearmesh_e2e_report.json}"

ULTRASHAPE_DIR="${ULTRASHAPE_DIR:-/workspace/UltraShape-1.0}"
ULTRASHAPE_CKPT="${ULTRASHAPE_CKPT:-/workspace/checkpoints/ultrashape_v1.pt}"
TRELLIS2_DIR="${TRELLIS2_DIR:-/workspace/TRELLIS.2}"

echo "=== ClearMesh E2E Smoke Test ==="
echo "  Input:       $INPUT"
echo "  Output:      $OUTPUT"
echo "  Report:      $REPORT"
echo "  UltraShape:  $ULTRASHAPE_DIR"
echo "  Checkpoint:  $ULTRASHAPE_CKPT"
echo "  TRELLIS.2:   $TRELLIS2_DIR"
echo ""

if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: input image not found: $INPUT" >&2
  exit 1
fi

# --- Environment fingerprint (runbook step 1 output) ---
# Printed inline so it lands in the teammate's console log alongside
# the structured report. The JSON report also captures this via
# e2e_smoke.py:collect_env — this block is just for a fast human glance.
echo "--- Environment fingerprint ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi: not available"
python3 -c "
import torch
print(f'torch={torch.__version__}  cuda={torch.version.cuda}  avail={torch.cuda.is_available()}')
try:
    import trellis2
    print(f'trellis2={getattr(trellis2, \"__version__\", \"unknown\")} @ {trellis2.__file__}')
except Exception as e:
    print(f'trellis2: IMPORT FAILED — {e}')
"
if [[ -d "$TRELLIS2_DIR/.git" ]]; then
  echo "trellis2_commit=$(git -C "$TRELLIS2_DIR" rev-parse --short HEAD)"
fi
if [[ -d "$REPO_ROOT/.git" ]]; then
  echo "clearmesh_commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
fi
echo ""

# --- Preflight: fail fast with a clear message if the heavy deps aren't
# installed yet, before we burn 60s loading TRELLIS.2.
python3 -c "
import importlib
missing = []
for mod in ('torch', 'trimesh', 'PIL'):
    try: importlib.import_module(mod)
    except ImportError: missing.append(mod)
try:
    import trellis2  # noqa: F401
except ImportError:
    missing.append('trellis2 (run setup_all.sh)')
if missing:
    raise SystemExit('Missing required modules: ' + ', '.join(missing))
print('Preflight OK — all required modules importable.')
"

# --- Run the e2e via the Python driver so we get the JSON report ---
# Exit code is propagated; the driver returns 0 on overall_pass==true.
cd "$REPO_ROOT"
python3 scripts/e2e_smoke.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --report "$REPORT" \
    --ultrashape-dir "$ULTRASHAPE_DIR" \
    --ultrashape-checkpoint "$ULTRASHAPE_CKPT" \
    --trellis2-dir "$TRELLIS2_DIR" \
    --format glb \
    --resolution 512 \
    --octree-res 1024
rc=$?

echo ""
echo "=== E2E smoke complete (exit=$rc) ==="
echo "  Output: $OUTPUT"
echo "  Report: $REPORT"
if [[ -f "$OUTPUT" ]]; then
  ls -lh "$OUTPUT"
fi
if [[ -f "$REPORT" ]]; then
  echo ""
  echo "--- Report summary (jq-friendly) ---"
  if command -v jq >/dev/null 2>&1; then
    jq '{overall_pass, stages: (.stages | to_entries | map({stage: .key, pass: .value.pass, duration_s: .value.duration_s})), outputs, mesh_stats, errors}' "$REPORT"
  else
    cat "$REPORT"
  fi
fi
exit $rc
