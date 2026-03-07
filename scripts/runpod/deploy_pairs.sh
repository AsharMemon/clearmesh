#!/usr/bin/env bash
# Deploy and run pair generation across 3 pods/shards.
# Usage: ./scripts/runpod/deploy_pairs.sh [extra args for generate_pairs.py]
set -euo pipefail

SSH_KEY="${SSH_KEY:-~/.ssh/id_ed25519}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"

# Pod definitions: port -> shard_id
PODS=(25620 44772 29275)
HOST="root@195.26.233.55"
NUM_SHARDS=${#PODS[@]}

EXTRA_ARGS="${*}"

for i in "${!PODS[@]}"; do
  port="${PODS[$i]}"
  shard_id="$i"

  echo "==> Starting shard ${shard_id}/${NUM_SHARDS} on port ${port}..."

  # shellcheck disable=SC2029
  ssh $SSH_OPTS -i "$SSH_KEY" -p "$port" "$HOST" bash -lc "'
    cd /workspace/clearmesh 2>/dev/null || cd /workspace

    # Run watchdog in a detached tmux session so it persists after SSH disconnect
    if command -v tmux &>/dev/null; then
      tmux kill-session -t pairs_shard_${shard_id} 2>/dev/null || true
      tmux new-session -d -s pairs_shard_${shard_id} \
        \"bash scripts/data/run_pairs_watchdog.sh ${shard_id} ${NUM_SHARDS} ${EXTRA_ARGS} 2>&1 | tee /workspace/pairs_shard_${shard_id}.log\"
      echo \"[shard ${shard_id}] Started in tmux session pairs_shard_${shard_id}\"
      echo \"[shard ${shard_id}] Log: /workspace/pairs_shard_${shard_id}.log\"
      echo \"[shard ${shard_id}] Attach: tmux attach -t pairs_shard_${shard_id}\"
    else
      # Fallback: nohup
      nohup bash scripts/data/run_pairs_watchdog.sh ${shard_id} ${NUM_SHARDS} ${EXTRA_ARGS} \
        > /workspace/pairs_shard_${shard_id}.log 2>&1 &
      echo \"[shard ${shard_id}] Started with nohup (PID: \$!)\"
      echo \"[shard ${shard_id}] Log: /workspace/pairs_shard_${shard_id}.log\"
    fi
  '" &

done

wait
echo ""
echo "==> All ${NUM_SHARDS} shards launched."
echo "    Monitor logs:  ssh <pod> tail -f /workspace/pairs_shard_<N>.log"
echo "    Attach tmux:   ssh <pod> tmux attach -t pairs_shard_<N>"
