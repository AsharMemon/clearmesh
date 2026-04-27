# ClearMesh GPU Launch Runbook

**One page.** Copy each block verbatim, paste output as instructed. If any
step fails, stop and report — don't "fix it and continue."

**Prereqs (assumed already done on the host):**
- `scripts/setup/setup_all.sh` has run successfully
- `/workspace/TRELLIS.2/`, `/workspace/UltraShape-1.0/`, and
  `/workspace/checkpoints/ultrashape_v1.pt` exist
- `conda activate clearmesh`
- `cd ~/clearmesh` (or wherever the repo lives)

**Deliverables to send back:**
1. `/tmp/env_fingerprint.txt` (step 1)
2. `/tmp/clearmesh_e2e_report.json` + `/tmp/clearmesh_e2e_out.glb` (step 2)
3. `/tmp/easy3e_checks.txt` (step 3)
4. `/tmp/preempt_drill.txt` + `ls` output of checkpoint dir (step 4)
5. Your go/no-go verdict from step 5

---

## Step 1 — Preflight env fingerprint

```bash
{
  echo "=== nvidia-smi ==="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo
  echo "=== python / torch / trellis2 ==="
  python -c "
import torch, sys
print(f'python={sys.version.split()[0]}')
print(f'torch={torch.__version__}  cuda={torch.version.cuda}  avail={torch.cuda.is_available()}')
try:
    import trellis2
    print(f'trellis2={getattr(trellis2, \"__version__\", \"unknown\")} @ {trellis2.__file__}')
except Exception as e:
    print(f'trellis2 IMPORT FAILED: {e}')
"
  echo
  echo "=== commits ==="
  echo "clearmesh: $(git rev-parse HEAD)"
  echo "trellis2:  $(git -C /workspace/TRELLIS.2 rev-parse HEAD)"
  echo
  echo "=== key package versions ==="
  pip freeze | grep -iE '^(torch|trellis|flash|xformers|trimesh|rembg|diffusers)==' | sort
  echo
  echo "=== checkpoints ==="
  ls -lh /workspace/checkpoints/ultrashape_v1.pt 2>&1
  ls -d /workspace/models/trellis2-4b 2>&1
} | tee /tmp/env_fingerprint.txt
```

**Pass if:** `cuda=True`, `trellis2` imports, both commits are hashes (not "unknown"), `ultrashape_v1.pt` is >100 MB.
**Fail if:** any import error, or a commit missing. → STOP. Report `/tmp/env_fingerprint.txt`.

---

## Step 2 — End-to-end smoke

```bash
bash scripts/run_e2e_gpu.sh 2>&1 | tee /tmp/e2e_console.log
```

Runtime: ~90–180s on A100. Emits `/tmp/clearmesh_e2e_out.glb` and `/tmp/clearmesh_e2e_report.json`.

**Pass if:** `"overall_pass": true` in the JSON and last line reads `[e2e_smoke] PASS`.
**Fail if:** `overall_pass: false`. → STOP. Report the JSON (it includes stage-level reason + traceback).

---

## Step 3 — Easy3E three-check

Validates the unblocks landed in `clearmesh/editing/STATUS.md`. Prints return values, not pass/fail.

```bash
{
python - <<'PY'
import torch
from clearmesh.editing.easy3e import Easy3EEditor
from clearmesh.editing.slat_encoder import SLATEncoder

MODEL_DIR = "/workspace/models/trellis2-4b"
TRELLIS2_DIR = "/workspace/TRELLIS.2"

# --- 3a. Flow model loads directly via trellis2.models.from_pretrained ---
editor = Easy3EEditor(
    trellis2_dir=TRELLIS2_DIR,
    model_dir=MODEL_DIR,
)
print("=== 3a. _try_load_raw_model('sparse_structure_flow_model') ===")
ss_flow = editor._try_load_raw_model("sparse_structure_flow_model")
print(f"  loaded: {type(ss_flow).__name__ if ss_flow else None}")
print(f"  fingerprint: {editor._build_fingerprint()}")

# --- 3b. SS encoder probe ---
print()
print("=== 3b. SLATEncoder._try_load_ss_encoder() ===")
enc = SLATEncoder(model_dir=MODEL_DIR, trellis2_dir=TRELLIS2_DIR)
ss_enc = enc._try_load_ss_encoder()
print(f"  loaded: {type(ss_enc).__name__ if ss_enc else None}")

# --- 3c. Flow sig cache contents after probe ---
print()
print("=== 3c. ~/.cache/clearmesh/flow_sig.json after first real call ===")
import os, json
cache = os.path.expanduser("~/.cache/clearmesh/flow_sig.json")
if os.path.exists(cache):
    with open(cache) as f:
        history = json.load(f)
    for entry in history[-3:]:
        print(f"  {entry}")
else:
    print("  (not yet written — _flow_call has not run against real data)")
PY
} | tee /tmp/easy3e_checks.txt
```

**Pass if:**
- 3a prints `loaded: SparseStructureFlowModel` (or similar — just not `None`)
- 3b prints a non-None class name (informational — None is acceptable here, just tells us SS encoder stays aliased; report the actual value)
- 3c either shows a recent entry OR the "not yet written" line (fine — this file only grows when Easy3E actually runs an edit)

**Fail if:** 3a returns `None` — this breaks Easy3E voxel flow editing. Report the error message printed just before it.

---

## Step 4 — Preemption + resume drill

Validates that SIGUSR1 → emergency save → auto-resume works **before** committing 85K steps to it.

```bash
# Runs training briefly, sends SIGUSR1, then restarts and confirms
# auto-resume from checkpoint_latest.pt. Requires SLAT pairs already
# generated under data_dir (see configs/train_stage2_slat.yaml).

CKPT_DIR=/tmp/clearmesh_preempt_drill
rm -rf $CKPT_DIR && mkdir -p $CKPT_DIR
: > /tmp/preempt_drill.txt

# --- First run: train for 2 minutes (well into triple-digit steps on
# any A100/H100), then SIGUSR1. Fixed wait is simpler than parsing tqdm.
python -m clearmesh.stage2.train \
    --config configs/train_stage2_slat.yaml \
    --output_dir $CKPT_DIR \
    >> /tmp/preempt_drill.txt 2>&1 &
PID=$!
echo "Training PID=$PID; sleeping 120s before preempting..."
sleep 120

if ! kill -0 $PID 2>/dev/null; then
  echo "ERROR: training died before preempt window — tail of log:" >&2
  tail -40 /tmp/preempt_drill.txt
  exit 1
fi

echo "=== sending SIGUSR1 to $PID ==="
kill -USR1 $PID
wait $PID 2>/dev/null

echo
echo "=== first run exited; checkpoints on disk: ==="
ls -lh $CKPT_DIR/ | tee -a /tmp/preempt_drill.txt
EMERGENCY_CKPT=$(ls $CKPT_DIR/emergency_*.pt 2>/dev/null | head -1)
if [[ -z "$EMERGENCY_CKPT" ]]; then
  echo "ERROR: no emergency checkpoint was written — SIGUSR1 not handled" >&2
  tail -40 /tmp/preempt_drill.txt
  exit 1
fi
echo "Emergency checkpoint: $EMERGENCY_CKPT"

# --- Second run: no --resume_from needed; load_checkpoint() auto-picks
# checkpoint_latest.pt (which save_checkpoint always mirrors).
echo
echo "=== resuming (60s timeout — we only need to see 'Resumed at step N') ==="
timeout 60 python -m clearmesh.stage2.train \
    --config configs/train_stage2_slat.yaml \
    --output_dir $CKPT_DIR \
    2>&1 | tee -a /tmp/preempt_drill.txt | grep -E 'Resumed at step|Emergency|starting from scratch' | head -5
```

**Pass if the log contains:**
- `!!! PREEMPTION ... saving emergency checkpoint !!!`
- `Emergency checkpoint saved. Exiting.`
- `checkpoint_latest.pt` and `emergency_*.pt` in the ckpt dir
- On resume: `Resumed at step {N}` where `N ≥ 40`

**Fail if:** no emergency line, no checkpoint written, or resume says "starting from scratch." → STOP. Report `/tmp/preempt_drill.txt` and `ls -la $CKPT_DIR`.

---

## Step 5 — Go / no-go decision

| Step | Pass condition | If fails |
|------|----------------|----------|
| 1    | GPU, torch, trellis2 all green | Can't proceed — fix env |
| 2    | `overall_pass: true` in JSON | Report; don't train |
| 3a   | SS flow model loads | Report; don't train |
| 3b   | (informational) | Fine either way — report value |
| 4    | Preempt + resume works | Report; don't train on spot |

**All pass →** launch the real 85K run:

```bash
# Start preemption handler in background (sends SIGUSR1 on GCP spot warn)
./scripts/utils/preemption_handler.sh &

# Launch training
python -m clearmesh.stage2.train --config configs/train_stage2_slat.yaml

# Monitor (separate terminal). Path matches configs/train_stage2_slat.yaml:output_dir.
python scripts/utils/monitor_training.py \
    --checkpoint_dir /workspace/checkpoints/clearmesh_stage2_slat --watch
```

Expected: 120–200 GPU-hours @ A100 spot; checkpoint every 2000 steps; ~43 checkpoints total.

**Any fail →** stop, report the deliverables list at the top of this doc, wait for a fix before burning compute.

---

## Reporting template

Paste this and fill in:

```
clearmesh commit:   <from step 1>
trellis2  commit:   <from step 1>
GPU:                <from step 1>

Step 1 env:         PASS / FAIL  (attach /tmp/env_fingerprint.txt)
Step 2 e2e:         PASS / FAIL  (attach /tmp/clearmesh_e2e_report.json)
Step 3a ss_flow:    <class name or None>
Step 3b ss_enc:     <class name or None>
Step 3c flow_sig:   <last entry>
Step 4 preempt:     PASS / FAIL  (attach /tmp/preempt_drill.txt)

Verdict:            GO / NO-GO for 85K training run
Blocker (if any):   <one sentence>
```
