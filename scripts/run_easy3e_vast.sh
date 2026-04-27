#!/usr/bin/env bash
# Drive the Easy3E demo on a Vast.ai instance from your LOCAL machine.
#
# Runs end-to-end with one command: starts/resumes the pod, clones
# clearmesh on it, uploads the two demo scripts (easy3e_demo.py +
# run_easy3e_demo.sh — they're not pushed to the remote yet), runs
# setup_all.sh, runs the demo, pulls artifacts back to your laptop, and
# tells you the command to destroy the instance.
#
# Why this lives locally: my sandbox can't SSH to Vast high-ports.
# You run this; I read the output you paste back.
#
# Usage:
#   ./scripts/run_easy3e_vast.sh <INSTANCE_ID> ["INSTRUCTION"]
#
# Example:
#   ./scripts/run_easy3e_vast.sh 35076678
#   ./scripts/run_easy3e_vast.sh 35076678 "give it a matte black ceramic finish"
#
# Environment overrides:
#   VASTAI       — path to the vastai CLI (default: auto-detect)
#   SKIP_SETUP=1 — assume setup_all.sh already ran on the pod (iteration mode)
#   LOCAL_OUT    — where to put downloaded artifacts (default: /tmp/clearmesh_easy3e_demo_local)
#
# Expected runtime on first run: 35-50 min.
# With SKIP_SETUP=1 (iteration on an already-set-up pod): 5-7 min.

set -euo pipefail

INSTANCE_ID="${1:?Usage: $0 <INSTANCE_ID> [\"INSTRUCTION\"]}"
INSTRUCTION="${2:-paint the mug bright red with a glossy finish}"

# Auto-detect vastai if not specified.
VASTAI="${VASTAI:-$(command -v vastai 2>/dev/null || echo /Users/Ashar/Library/Python/3.14/bin/vastai)}"
if [[ ! -x "$VASTAI" ]]; then
    echo "ERROR: vastai CLI not found at $VASTAI. Install with: pip install vastai" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_OUT="${LOCAL_OUT:-/tmp/clearmesh_easy3e_demo_local}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

echo "=== Easy3E Vast Driver ==="
echo "  Instance:    $INSTANCE_ID"
echo "  Instruction: $INSTRUCTION"
echo "  Repo root:   $REPO_ROOT"
echo "  vastai:      $VASTAI"
echo "  Balance:     \$$("$VASTAI" show user --raw | python3 -c 'import json,sys; print(f"{json.load(sys.stdin)[\"credit\"]:.2f}")')"
echo ""

# ── Step 0: Ensure the instance is running ───────────────────────────
STATUS=$("$VASTAI" show instance "$INSTANCE_ID" --raw 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("actual_status","?"))')
echo "→ Step 0: instance status = $STATUS"
if [[ "$STATUS" != "running" ]]; then
    echo "  starting instance..."
    "$VASTAI" start instance "$INSTANCE_ID"
    # Wait for running
    for i in {1..30}; do
        STATUS=$("$VASTAI" show instance "$INSTANCE_ID" --raw 2>/dev/null \
            | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("actual_status","?"))')
        echo "  [$(date +%H:%M:%S)] attempt=$i  status=$STATUS"
        [[ "$STATUS" == "running" ]] && break
        sleep 10
    done
fi
if [[ "$STATUS" != "running" ]]; then
    echo "ERROR: instance never reached 'running' state." >&2
    exit 1
fi

# ── Step 1: Wait for SSH to be reachable ─────────────────────────────
echo ""
echo "→ Step 1: waiting for SSH..."
SSH_URL=$("$VASTAI" ssh-url "$INSTANCE_ID" 2>/dev/null | tail -1)
SSH_HOST=$(echo "$SSH_URL" | sed -E 's#ssh://[^@]+@([^:]+):.*#\1#')
SSH_PORT=$(echo "$SSH_URL" | sed -E 's#ssh://[^@]+@[^:]+:([0-9]+)#\1#')
echo "  SSH URL: root@$SSH_HOST:$SSH_PORT"

SSH="ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST"
SCP_P="scp $SSH_OPTS -P $SSH_PORT"

for i in {1..30}; do
    if $SSH "true" 2>/dev/null; then
        echo "  SSH OK on attempt $i"
        break
    fi
    echo "  [$(date +%H:%M:%S)] SSH not ready (attempt $i/30)..."
    sleep 10
done
if ! $SSH "true" 2>/dev/null; then
    echo "ERROR: SSH never came up." >&2
    exit 1
fi

# Show we have a GPU + disk before we commit to a 40-min setup.
$SSH "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv && df -h /workspace 2>/dev/null || df -h /"

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
    # ── Step 2: Clone repo + upload unpushed scripts ────────────────
    echo ""
    echo "→ Step 2: cloning clearmesh on pod + uploading demo scripts..."
    $SSH "apt-get update -qq && apt-get install -y -qq git curl jq 2>&1 | tail -3 \
          && mkdir -p /workspace && cd /workspace \
          && if [[ ! -d clearmesh ]]; then git clone https://github.com/AsharMemon/clearmesh.git; else cd clearmesh && git pull; fi \
          && cd /workspace/clearmesh && echo clearmesh_commit=\$(git rev-parse --short HEAD)"

    # These two are authored locally and not yet in the upstream repo.
    # Overwrite whatever's on the pod with the local versions.
    $SCP_P "$REPO_ROOT/scripts/easy3e_demo.py" "root@$SSH_HOST:/workspace/clearmesh/scripts/easy3e_demo.py"
    $SCP_P "$REPO_ROOT/scripts/run_easy3e_demo.sh" "root@$SSH_HOST:/workspace/clearmesh/scripts/run_easy3e_demo.sh"
    $SSH "chmod +x /workspace/clearmesh/scripts/*.sh /workspace/clearmesh/scripts/setup/*.sh /workspace/clearmesh/scripts/gcp/*.sh 2>/dev/null || true"

    # ── Step 3: Run setup_all.sh (long — 25-40 min) ─────────────────
    echo ""
    echo "→ Step 3: running setup_all.sh in background on pod (25-40 min)..."
    echo "          This is the longest phase — model weights + CUDA builds."
    $SSH "cd /workspace/clearmesh \
          && nohup bash scripts/setup/setup_all.sh /workspace --skip-rigging --skip-optional > /tmp/setup.log 2>&1 & \
          echo setup_pid=\$!"

    # Poll until setup finishes or times out.
    START=$(date +%s)
    TIMEOUT=4200   # 70 min hard cap
    while true; do
        sleep 30
        STATE=$($SSH "if pgrep -f 'setup_all.sh' > /dev/null 2>&1; then \
                         echo RUNNING; \
                       elif grep -q 'Setup Complete' /tmp/setup.log 2>/dev/null; then \
                         echo DONE; \
                       else \
                         echo FAILED; \
                       fi" 2>/dev/null || echo UNREACHABLE)
        ELAPSED=$(( $(date +%s) - START ))
        # Pull the latest setup marker line for progress visibility.
        LAST=$($SSH "tail -1 /tmp/setup.log 2>/dev/null | cut -c1-100" 2>/dev/null || echo "")
        printf "  [t+%dm%02ds] state=%s | %s\n" $((ELAPSED/60)) $((ELAPSED%60)) "$STATE" "$LAST"
        case "$STATE" in
            DONE) break ;;
            FAILED)
                echo "ERROR: setup failed. Tail of /tmp/setup.log:" >&2
                $SSH "tail -60 /tmp/setup.log" >&2
                exit 1
                ;;
            UNREACHABLE)
                echo "  SSH hiccup — retrying..." >&2
                ;;
        esac
        if (( ELAPSED > TIMEOUT )); then
            echo "ERROR: setup exceeded ${TIMEOUT}s timeout." >&2
            $SSH "tail -40 /tmp/setup.log" >&2
            exit 1
        fi
    done
    echo "  setup complete in ${ELAPSED}s"
else
    echo ""
    echo "→ Step 3: SKIP_SETUP=1 — assuming setup_all.sh already ran."
    # Still re-upload the scripts so we always run the latest local version.
    $SCP_P "$REPO_ROOT/scripts/easy3e_demo.py" "root@$SSH_HOST:/workspace/clearmesh/scripts/easy3e_demo.py"
    $SCP_P "$REPO_ROOT/scripts/run_easy3e_demo.sh" "root@$SSH_HOST:/workspace/clearmesh/scripts/run_easy3e_demo.sh"
    $SSH "chmod +x /workspace/clearmesh/scripts/run_easy3e_demo.sh"
fi

# ── Step 4: Run the demo ────────────────────────────────────────────
# Source conda from both likely install locations (pytorch/pytorch image
# usually has it at /opt/conda; setup_all.sh's install_miniconda would
# put it at /root/miniconda3).
echo ""
echo "→ Step 4: running Easy3E demo on pod..."
$SSH "bash -lc '
    set -e
    for cp in /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh; do
        if [[ -f \$cp ]]; then source \$cp; break; fi
    done
    conda activate clearmesh
    cd /workspace/clearmesh
    export INSTRUCTION=\"$INSTRUCTION\"
    bash scripts/run_easy3e_demo.sh
'" 2>&1 | tee "$LOCAL_OUT-remote-run.log" || {
    echo "ERROR: demo run failed. Tail of local log:" >&2
    tail -30 "$LOCAL_OUT-remote-run.log" >&2
    exit 1
}

# ── Step 5: Pull artifacts back ─────────────────────────────────────
echo ""
echo "→ Step 5: pulling artifacts to $LOCAL_OUT..."
mkdir -p "$LOCAL_OUT"
$SCP_P -r "root@$SSH_HOST:/tmp/clearmesh_easy3e_demo/" "$LOCAL_OUT/"

echo ""
echo "=== Done ==="
echo "  Local artifacts: $LOCAL_OUT/clearmesh_easy3e_demo/"
ls -lh "$LOCAL_OUT/clearmesh_easy3e_demo/" 2>/dev/null || true

echo ""
echo "  Balance now: \$$("$VASTAI" show user --raw | python3 -c 'import json,sys; print(f"{json.load(sys.stdin)[\"credit\"]:.2f}")')"
echo ""
echo "  When you're done iterating, destroy the instance to stop billing:"
echo "    $VASTAI destroy instance $INSTANCE_ID"
echo ""
echo "  To iterate with a new instruction (fast, uses cached setup):"
echo "    SKIP_SETUP=1 ./scripts/run_easy3e_vast.sh $INSTANCE_ID \"your new instruction\""
