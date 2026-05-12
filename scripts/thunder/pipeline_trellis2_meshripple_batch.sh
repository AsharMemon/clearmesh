#!/usr/bin/env bash
# Run a CSV benchmark batch through TRELLIS.2 -> point cloud -> MeshRipple -> eval on Thunder.
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
STATE_ROOT="${STATE_ROOT:-/tmp/clearmesh_trellis2_batch_state}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/tmp/clearmesh_trellis2_batch_artifacts}"
CONFIG_JSON="${CONFIG_JSON:-/home/ubuntu/clearmesh/configs/meshripple.thunder.example.json}"
MANIFEST="${MANIFEST:-/tmp/clearmesh_trellis2_batch_manifest.csv}"
POINT_BUDGET="${POINT_BUDGET:-40960}"
LIMIT="${LIMIT:-6}"
GRANT_CREDITS="${GRANT_CREDITS:-200}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/run_remote.sh" "$INSTANCE_ID" <<CLEARMESH_REMOTE_SCRIPT
set -euo pipefail
if [ -f "/home/ubuntu/.clearmesh_hf.env" ]; then
  source "/home/ubuntu/.clearmesh_hf.env"
fi
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/ubuntu/TRELLIS.2:${PYTHONPATH:-}
export ATTN_BACKEND=flash_attn

cd /home/ubuntu/clearmesh
rm -rf "$STATE_ROOT" "$ARTIFACT_ROOT"
mkdir -p "$ARTIFACT_ROOT/uploads/benchmark"

python3 - <<'PY'
from pathlib import Path
import csv

source_dir = Path('/home/ubuntu/TRELLIS.2/assets/example_image')
dest_dir = Path('$ARTIFACT_ROOT/uploads/benchmark')
manifest = Path('$MANIFEST')
limit = int('$LIMIT')
images = sorted(source_dir.glob('*'))[:limit]
if not images:
    raise SystemExit(f'no example images found in {source_dir}')
with manifest.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=['case_id','input_uri','prompt','quality_tier','point_budgets','enable_parts','enable_rigging','project_id','trellis_command','trellis_cwd','trellis_timeout_seconds'])
    writer.writeheader()
    for index, image in enumerate(images):
        case_id = f'trellis2_case_{index:02d}'
        copied = dest_dir / f'{case_id}{image.suffix}'
        copied.write_bytes(image.read_bytes())
        writer.writerow({
            'case_id': case_id,
            'input_uri': f'local://uploads/benchmark/{copied.name}',
            'prompt': 'Benchmark image-to-artist-mesh case',
            'quality_tier': 'draft',
            'point_budgets': '$POINT_BUDGET',
            'enable_parts': 'false',
            'enable_rigging': 'false',
            'project_id': 'trellis2_meshripple_batch',
            'trellis_command': '/home/ubuntu/trellis2-venv/bin/python -u /home/ubuntu/clearmesh/scripts/product/run_trellis2_proxy.py --input {input_path} --output-dir {output_dir} --output-name trellis_proxy.glb --decimation-target 250000 --texture-size 1024 --seed 0',
            'trellis_cwd': '/home/ubuntu/TRELLIS.2',
            'trellis_timeout_seconds': '7200',
        })
print(manifest)
PY

/home/ubuntu/clearmesh-venv/bin/python scripts/product/create_batch_jobs.py \
  --manifest "$MANIFEST" \
  --state-root "$STATE_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --grant-credits "$GRANT_CREDITS" \
  --team-id team_dev \
  --user-id user_dev

/home/ubuntu/clearmesh-venv/bin/python scripts/product/run_pipeline_worker.py \
  --state-root "$STATE_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --execute-heavy \
  --mesh-head meshripple \
  --mesh-head-config-json "$CONFIG_JSON" \
  --preferred-point-budget "$POINT_BUDGET"

/home/ubuntu/clearmesh-venv/bin/python - <<'PY'
import json
from pathlib import Path
state = Path('$STATE_ROOT')
summary = []
for path in sorted((state / 'jobs').glob('job_*.json')):
    job = json.loads(path.read_text())
    eval_asset = next((asset['uri'] for asset in job['assets'] if asset['kind'] == 'mesh_eval_report'), None)
    item = {
        'job_id': job['id'],
        'case_id': job['request']['metadata'].get('case_id'),
        'status': job['status'],
        'steps': {step['name']: step['status'] for step in job['steps']},
        'eval': eval_asset,
        'export': next((asset['uri'] for asset in job['assets'] if asset['kind'] == 'export_mesh'), None),
    }
    if eval_asset and Path(eval_asset).exists():
        report = json.loads(Path(eval_asset).read_text())
        item['metrics'] = report.get('mesh_metrics', {})
        item['pair_metrics'] = report.get('pair_metrics', {})
    summary.append(item)
print(json.dumps({'state_root': '$STATE_ROOT', 'artifact_root': '$ARTIFACT_ROOT', 'jobs': summary}, indent=2, sort_keys=True))
PY
CLEARMESH_REMOTE_SCRIPT
