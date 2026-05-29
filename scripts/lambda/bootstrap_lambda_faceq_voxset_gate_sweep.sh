#!/usr/bin/env bash
# Launch a low-cost Lambda FACE-Q ablation that keeps FACE autoregression intact
# and sweeps LATTICE-style VoxSet spatial cross-attention gates.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-.codex_outputs/lambda_faceq_voxset_gate_sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LAMBDA_ENV_FILE="${LAMBDA_ENV_FILE:-.codex_secrets/lambda.env}"
B2_ENV_FILE="${B2_ENV_FILE:-.codex_secrets/b2.env}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519}"
SSH_KEY_NAME="${SSH_KEY_NAME:-clearmesh-codex-mac-id-ed25519}"
INSTANCE_TYPE="${INSTANCE_TYPE:-gpu_1x_gh200}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/clearmesh-voxset-gate-sweep-repo}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-voxset-gate-sweep-venv}"
REMOTE_ROOT="${REMOTE_ROOT:-/tmp/clearmesh_faceq_voxset_gate_sweep_$RUN_STAMP}"
REMOTE_LOG="${REMOTE_LOG:-/tmp/clearmesh_faceq_voxset_gate_sweep.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/tmp/clearmesh_faceq_voxset_gate_sweep.pid}"
REMOTE_B2_ENV="${REMOTE_B2_ENV:-$REMOTE_ROOT/.b2.env}"
API_BASE="${LAMBDA_API_BASE:-https://cloud.lambdalabs.com/api/v1}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_CORPUS_PREFIX="${B2_CORPUS_PREFIX:-face-corpora/paper-large-65k1024-repack8k128-v2/objpp-minquality1/shard0010}"
B2_CORPUS_ARCHIVE="${B2_CORPUS_ARCHIVE:-lean_face_corpus.tar.gz}"
ARCHIVE_DATASET_DIR="${ARCHIVE_DATASET_DIR:-corpus/split_pass/train}"
ARCHIVE_TEST_DIR="${ARCHIVE_TEST_DIR:-corpus/split_pass/test}"
B2_RUN_PREFIX="${B2_RUN_PREFIX:-face-runs/faceq-voxset-spatial-gate-sweep/lambda-$RUN_STAMP}"
STEPS="${STEPS:-2000}"
TRAIN_LIMIT="${TRAIN_LIMIT:-384}"
POINT_SAMPLES="${POINT_SAMPLES:-65536}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
LAYERS="${LAYERS:-4}"
HEADS="${HEADS:-8}"
CONDITION_TOKENS="${CONDITION_TOKENS:-256}"
ENCODER_LAYERS="${ENCODER_LAYERS:-3}"
LATENT_DIM="${LATENT_DIM:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_EVERY="${LOG_EVERY:-50}"
VOXSET_RESOLUTION="${VOXSET_RESOLUTION:-16}"
EVAL_LIMIT="${EVAL_LIMIT:-12}"
TOPOLOGY_LOSS_WEIGHT="${TOPOLOGY_LOSS_WEIGHT:-0.2}"
EDGE_ACTION_LOSS_WEIGHT="${EDGE_ACTION_LOSS_WEIGHT:-0.0}"
EDGE_CHOICE_LOSS_WEIGHT="${EDGE_CHOICE_LOSS_WEIGHT:-0.0}"
CORNER_CLOSURE_PRESENCE_LOSS_WEIGHT="${CORNER_CLOSURE_PRESENCE_LOSS_WEIGHT:-0.0}"
EDGE_CHOICE_CANDIDATES="${EDGE_CHOICE_CANDIDATES:-64}"
RUN_FREE_RUN_EVAL="${RUN_FREE_RUN_EVAL:-0}"
FREE_RUN_EVAL_LIMIT="${FREE_RUN_EVAL_LIMIT:-6}"
FREE_RUN_DECODE_MODE="${FREE_RUN_DECODE_MODE:-boundary_edge}"
FREE_RUN_CONSTRAINT_TOP_K="${FREE_RUN_CONSTRAINT_TOP_K:-24}"
FREE_RUN_EDGE_ACTION_BONUS="${FREE_RUN_EDGE_ACTION_BONUS:-0.75}"
FREE_RUN_EDGE_ACTION_CANDIDATE_TOP_K="${FREE_RUN_EDGE_ACTION_CANDIDATE_TOP_K:-16}"
FREE_RUN_EDGE_CHOICE_BONUS="${FREE_RUN_EDGE_CHOICE_BONUS:-0.5}"
FREE_RUN_EDGE_CHOICE_CANDIDATE_TOP_K="${FREE_RUN_EDGE_CHOICE_CANDIDATE_TOP_K:-16}"
FREE_RUN_CLOSURE_TARGET_BONUS="${FREE_RUN_CLOSURE_TARGET_BONUS:-1.0}"
VARIANTS="${VARIANTS:-baseline_vecset,vecset,cross_attn,0.35,0;voxset_modulated_soft,voxset,spatial_modulated_cross_attn,0.35,0;voxset_modulated_tight_k24,voxset,spatial_modulated_cross_attn,0.25,24;voxset_modulated_broad_k96,voxset,spatial_modulated_cross_attn,0.50,96}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/launch.log"
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

if [[ ! -f "$LAMBDA_ENV_FILE" ]]; then echo "missing $LAMBDA_ENV_FILE" >&2; exit 2; fi
if [[ ! -f "$B2_ENV_FILE" ]]; then echo "missing $B2_ENV_FILE" >&2; exit 2; fi
if [[ ! -f "$SSH_KEY_FILE" ]]; then echo "missing $SSH_KEY_FILE" >&2; exit 2; fi
source "$LAMBDA_ENV_FILE"
: "${LAMBDA_KEY:?missing LAMBDA_KEY}"

LAMBDA_KEY="$LAMBDA_KEY" python3 scripts/lambda/poll_lambda_capacity.py \
  --out-dir "$OUT_DIR" \
  --target "$INSTANCE_TYPE" \
  --ssh-key-name "$SSH_KEY_NAME" \
  --launch \
  --name-prefix clearmesh-faceq-voxset-gate-sweep > "$OUT_DIR/lambda_launch_summary.json"

if [[ ! -f "$OUT_DIR/lambda_launched.flag" ]]; then
  log "no Lambda capacity for instance_type=$INSTANCE_TYPE; see $OUT_DIR/lambda_launch_summary.json"
  echo "$OUT_DIR"
  exit 5
fi

INSTANCE_ID=$(python3 - "$OUT_DIR/lambda_launched.flag" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding='utf-8'))
data=p.get('response',{}).get('data',p.get('response',{}))
ids=data.get('instance_ids') or data.get('ids') or []
print(ids[0] if ids else (data.get('id') or data.get('instance_id') or ''))
PY
)
if [[ -z "$INSTANCE_ID" ]]; then echo "could not parse Lambda instance id" >&2; exit 3; fi
log "launched lambda instance=$INSTANCE_ID type=$INSTANCE_TYPE"

get_instances(){ curl -fsS -L --max-time 45 -u "$LAMBDA_KEY:" -H 'Accept: application/json' "$API_BASE/instances"; }
INSTANCE_JSON="$OUT_DIR/lambda_instance.latest.json"
deadline=$(( $(date +%s) + 1800 ))
REMOTE_HOST=""
while [[ $(date +%s) -lt $deadline ]]; do
  get_instances > "$INSTANCE_JSON.tmp" && mv "$INSTANCE_JSON.tmp" "$INSTANCE_JSON"
  parsed=$(python3 - "$INSTANCE_ID" "$INSTANCE_JSON" <<'PY'
import json, sys
iid,path=sys.argv[1:3]
p=json.load(open(path, encoding='utf-8'))
for item in p.get('data',[]):
    if str(item.get('id')) == str(iid):
        print((item.get('status') or '').lower(), item.get('ip') or item.get('ip_address') or '-')
        break
else:
    print('missing -')
PY
)
  st=$(awk '{print $1}' <<<"$parsed")
  host=$(awk '{print $2}' <<<"$parsed")
  log "lambda status=$st host=$host"
  if [[ "$host" != "-" && -n "$host" && ( "$st" = "active" || "$st" = "running" || "$st" = "booted" ) ]]; then REMOTE_HOST="$host"; break; fi
  sleep 15
done
if [[ -z "$REMOTE_HOST" ]]; then echo "lambda instance not ready" >&2; exit 4; fi

SSH_OPTS=(-i "$SSH_KEY_FILE" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/known_hosts" -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
SSH_TARGET="$REMOTE_USER@$REMOTE_HOST"
log "preflight $SSH_TARGET"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "hostname; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader; mkdir -p '$REMOTE_REPO' '$REMOTE_ROOT'" | tee -a "$LOG"

log "syncing repo"
COPYFILE_DISABLE=1 tar --no-xattrs \
  --exclude='.git' --exclude='.codex_outputs' --exclude='.codex_secrets' \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.venv' \
  --exclude='node_modules' --exclude='.DS_Store' \
  -czf - . | ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "rm -rf '$REMOTE_REPO' && mkdir -p '$REMOTE_REPO' && tar -xzf - -C '$REMOTE_REPO'"

tmp_env="$OUT_DIR/b2_remote_env.sh"
{
  source "$B2_ENV_FILE"
  printf 'export B2_KEY_ID=%q\n' "${B2_KEY_ID:-${B2_KEYID:-${B2_APPLICATION_KEY_ID:-}}}"
  printf 'export B2_APP_KEY=%q\n' "${B2_APP_KEY:-${B2_APPKEY:-${B2_APPLICATION_KEY:-}}}"
} > "$tmp_env"
chmod 600 "$tmp_env"
scp "${SSH_OPTS[@]}" "$tmp_env" "$SSH_TARGET:$REMOTE_B2_ENV" >/dev/null

remote_script="$OUT_DIR/remote_gate_sweep.sh"
cat > "$remote_script" <<REMOTE
#!/usr/bin/env bash
set -euo pipefail
ROOT='$REMOTE_ROOT'
REPO='$REMOTE_REPO'
VENV='$REMOTE_VENV'
B2_ENV='$REMOTE_B2_ENV'
B2_BUCKET='$B2_BUCKET'
B2_CORPUS_PREFIX='$B2_CORPUS_PREFIX'
B2_CORPUS_ARCHIVE='$B2_CORPUS_ARCHIVE'
ARCHIVE_DATASET_DIR='$ARCHIVE_DATASET_DIR'
ARCHIVE_TEST_DIR='$ARCHIVE_TEST_DIR'
B2_RUN_PREFIX='$B2_RUN_PREFIX'
VARIANTS='$VARIANTS'
mkdir -p "\$ROOT/logs" "\$ROOT/data" "\$ROOT/runs" "\$ROOT/eval_reports"
status(){ python3 - "\$ROOT/status.jsonl" "\$1" "\${2:-}" <<'PY'
import json,sys,time
p,e,d=sys.argv[1:4]
open(p,'a',encoding='utf-8').write(json.dumps({'time':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),'event':e,'detail':d})+'\n')
PY
}
cd "\$REPO"
export PYTHONPATH="\$REPO:\${PYTHONPATH:-}"
if ! command -v rclone >/dev/null 2>&1; then sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y rclone; fi
if [[ ! -x "\$VENV/bin/python" ]]; then python3 -m venv --system-site-packages "\$VENV" || (sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv && python3 -m venv --system-site-packages "\$VENV"); fi
source "\$VENV/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -q numpy trimesh scipy networkx tqdm
python - <<'PY' || python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch>=2.4.0'
import torch
print('torch_ready', torch.__version__, 'cuda', torch.cuda.is_available())
PY
source "\$B2_ENV"
export RCLONE_CONFIG_B2ENV_TYPE=b2
export RCLONE_CONFIG_B2ENV_ACCOUNT="\$B2_KEY_ID"
export RCLONE_CONFIG_B2ENV_KEY="\$B2_APP_KEY"
status b2_download_started "\$B2_CORPUS_PREFIX/\$B2_CORPUS_ARCHIVE"
rclone copyto "b2env:\$B2_BUCKET/\$B2_CORPUS_PREFIX/\$B2_CORPUS_ARCHIVE" "\$ROOT/data/\$B2_CORPUS_ARCHIVE" --stats 20s --transfers 4 --checkers 8 --multi-thread-streams 8 --multi-thread-cutoff 64M
status extract_started "\$B2_CORPUS_ARCHIVE"
mkdir -p "\$ROOT/data/extracted"
tar -xzf "\$ROOT/data/\$B2_CORPUS_ARCHIVE" -C "\$ROOT/data/extracted"
DATASET_DIR="\$ROOT/data/extracted/\$ARCHIVE_DATASET_DIR"
TEST_DIR="\$ROOT/data/extracted/\$ARCHIVE_TEST_DIR"
test -d "\$DATASET_DIR"
python - "\$ROOT" "\$DATASET_DIR" "\$TEST_DIR" '$EVAL_LIMIT' <<'PY'
from pathlib import Path
import os, sys, numpy as np
root=Path(sys.argv[1]); limit=int(sys.argv[4])
for name,src in [('train',Path(sys.argv[2])),('test',Path(sys.argv[3]))]:
    out=root/'eval_small'/name; out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for p in sorted(src.rglob('*.npz')):
        try:
            with np.load(p) as d: rows.append((len(d['indexed_faces']), p))
        except Exception: pass
    for i,(faces,p) in enumerate(sorted(rows)[:limit]):
        target=out/f'{i:04d}_faces{faces:04d}_{p.name}'
        if not target.exists(): os.symlink(p, target)
    print(name, len(list(out.glob('*.npz'))))
PY
run_train(){
  local name="\$1" condition="\$2" decoder="\$3" sigma="\$4" topk="\$5"
  local out="\$ROOT/runs/\$name"
  mkdir -p "\$out"
  status train_started "\$name condition=\$condition decoder=\$decoder sigma=\$sigma topk=\$topk"
  python scripts/research/train_face_indexed_conditioned_tiny.py \
    --dataset-dir "\$DATASET_DIR" \
    --output "\$out/checkpoint.pt" \
    --steps '$STEPS' \
    --batch-size '$BATCH_SIZE' \
    --limit '$TRAIN_LIMIT' \
    --point-samples '$POINT_SAMPLES' \
    --hidden-size '$HIDDEN_SIZE' \
    --layers '$LAYERS' \
    --heads '$HEADS' \
    --condition-tokens '$CONDITION_TOKENS' \
    --condition-backend "\$condition" \
    --decoder-backend "\$decoder" \
    --encoder-layers '$ENCODER_LAYERS' \
    --latent-dim '$LATENT_DIM' \
    --face-output-mode geometry \
    --voxset-resolution '$VOXSET_RESOLUTION' \
    --spatial-gate-sigma "\$sigma" \
    --spatial-gate-top-k "\$topk" \
    --corner-head causal \
    --count-loss-weight 0.05 \
    --topology-loss-weight '$TOPOLOGY_LOSS_WEIGHT' \
    --edge-action-loss-weight '$EDGE_ACTION_LOSS_WEIGHT' \
    --edge-choice-loss-weight '$EDGE_CHOICE_LOSS_WEIGHT' \
    --corner-closure-presence-loss-weight '$CORNER_CLOSURE_PRESENCE_LOSS_WEIGHT' \
    --edge-choice-candidates '$EDGE_CHOICE_CANDIDATES' \
    --seed-face-loss-weight 1.0 \
    --seed-face-loss-stop-step 200 \
    --early-face-count 16 \
    --early-face-loss-weight 4.0 \
    --optimizer muon \
    --precision bf16 \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --log-every '$LOG_EVERY' \
    --checkpoint-every 1000 \
    --save-current-checkpoint \
    --device cuda | tee "\$out/train.log"
  status train_completed "\$name \$(tail -n 1 "\$out/train.log")"
}
run_eval(){
  local name="\$1"
  for split in train test; do
    status eval_started "\$name \$split"
    python scripts/research/eval_face_indexed_conditioned_tiny.py \
      --checkpoint "\$ROOT/runs/\$name/checkpoint.pt" \
      --dataset-dir "\$ROOT/eval_small/\$split" \
      --output "\$ROOT/eval_reports/\${name}_\${split}_teacher_forced.json" \
      --decode-strategy teacher_forced \
      --face-count-mode gt \
      --point-samples '$POINT_SAMPLES' \
      --pair-samples 200 \
      --device cuda | tee "\$ROOT/eval_reports/\${name}_\${split}_teacher_forced.log"
  done
  if [[ '$RUN_FREE_RUN_EVAL' = '1' ]]; then
    for split in train test; do
      status eval_started "\$name \$split free_run"
      python scripts/research/eval_face_indexed_conditioned_tiny.py \
        --checkpoint "\$ROOT/runs/\$name/checkpoint.pt" \
        --dataset-dir "\$ROOT/eval_small/\$split" \
        --output "\$ROOT/eval_reports/\${name}_\${split}_free_run.json" \
        --decode-strategy free_run \
        --decode-mode '$FREE_RUN_DECODE_MODE' \
        --corner-decode causal \
        --face-count-mode gt \
        --limit '$FREE_RUN_EVAL_LIMIT' \
        --point-samples '$POINT_SAMPLES' \
        --pair-samples 200 \
        --constraint-top-k '$FREE_RUN_CONSTRAINT_TOP_K' \
        --edge-action-bonus '$FREE_RUN_EDGE_ACTION_BONUS' \
        --edge-action-candidate-top-k '$FREE_RUN_EDGE_ACTION_CANDIDATE_TOP_K' \
        --edge-choice-bonus '$FREE_RUN_EDGE_CHOICE_BONUS' \
        --edge-choice-candidate-top-k '$FREE_RUN_EDGE_CHOICE_CANDIDATE_TOP_K' \
        --closure-target-bonus '$FREE_RUN_CLOSURE_TARGET_BONUS' \
        --boundary-budget-constraint \
        --vertex-link-constraint \
        --device cuda | tee "\$ROOT/eval_reports/\${name}_\${split}_free_run.log"
    done
  fi
}
IFS=';' read -ra rows <<< "\$VARIANTS"
for row in "\${rows[@]}"; do
  IFS=',' read -r name condition decoder sigma topk <<< "\$row"
  run_train "\$name" "\$condition" "\$decoder" "\$sigma" "\$topk"
  run_eval "\$name"
done
python - "\$ROOT" "\$VARIANTS" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
variants=[]
for row in sys.argv[2].split(';'):
    name,condition,decoder,sigma,topk=row.split(',')
    variants.append({'name':name,'condition_backend':condition,'decoder_backend':decoder,'spatial_gate_sigma':float(sigma),'spatial_gate_top_k':int(topk)})
def parse_train(path):
    rows=[]
    for line in Path(path).read_text(errors='ignore').splitlines():
        if line.startswith('{'):
            try: row=json.loads(line)
            except Exception: continue
            if 'step' in row and 'loss' in row: rows.append(row)
    if not rows: return {}
    return {'first':rows[0], 'last':rows[-1], 'min_loss':min(r['loss'] for r in rows), 'steps_logged':len(rows)}
def parse_eval(path):
    p=Path(path)
    if not p.exists(): return None
    return json.loads(p.read_text()).get('summary')
out={'variants': variants, 'runs': {}}
for v in variants:
    name=v['name']
    out['runs'][name]={'config':v,'train_log':parse_train(root/'runs'/name/'train.log')}
    for split in ['train','test']:
        out['runs'][name][f'eval_{split}']=parse_eval(root/'eval_reports'/f'{name}_{split}_teacher_forced.json')
        out['runs'][name][f'eval_{split}_free_run']=parse_eval(root/'eval_reports'/f'{name}_{split}_free_run.json')
Path(root/'ablation_summary.json').write_text(json.dumps(out, indent=2, sort_keys=True)+'\n')
print(json.dumps(out, sort_keys=True))
PY
status summary_written "\$ROOT/ablation_summary.json"
rclone copy "\$ROOT/status.jsonl" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/" --stats 0 || true
rclone copy "\$ROOT/ablation_summary.json" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/" --stats 0 || true
rclone copy "\$ROOT/runs" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/runs" --stats 0 || true
rclone copy "\$ROOT/eval_reports" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/eval_reports" --stats 0 || true
status b2_upload_completed "\$B2_RUN_PREFIX"
rclone copy "\$ROOT/status.jsonl" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/" --stats 0 || true
REMOTE
chmod +x "$remote_script"
scp "${SSH_OPTS[@]}" "$remote_script" "$SSH_TARGET:$REMOTE_ROOT/remote_gate_sweep.sh" >/dev/null
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cd '$REMOTE_ROOT'; nohup setsid bash remote_gate_sweep.sh > '$REMOTE_LOG' 2>&1 < /dev/null & echo \$! > '$REMOTE_PID'; exit 0"
cat > "$OUT_DIR/run_info.json" <<JSON
{"provider":"lambda","instance_id":"$INSTANCE_ID","instance_type":"$INSTANCE_TYPE","host":"$REMOTE_HOST","remote_user":"$REMOTE_USER","remote_lab_root":"$REMOTE_ROOT","remote_log":"$REMOTE_LOG","remote_pid":"$REMOTE_PID","b2_run_prefix":"$B2_RUN_PREFIX","steps":$STEPS,"train_limit":$TRAIN_LIMIT,"point_samples":$POINT_SAMPLES,"hidden_size":$HIDDEN_SIZE,"layers":$LAYERS,"condition_tokens":$CONDITION_TOKENS,"topology_loss_weight":$TOPOLOGY_LOSS_WEIGHT,"edge_action_loss_weight":$EDGE_ACTION_LOSS_WEIGHT,"edge_choice_loss_weight":$EDGE_CHOICE_LOSS_WEIGHT,"corner_closure_presence_loss_weight":$CORNER_CLOSURE_PRESENCE_LOSS_WEIGHT,"edge_choice_candidates":$EDGE_CHOICE_CANDIDATES,"run_free_run_eval":$RUN_FREE_RUN_EVAL,"variants":"$VARIANTS"}
JSON
log "launched gate sweep root=$REMOTE_ROOT b2=$B2_RUN_PREFIX"
echo "$OUT_DIR"
