#!/usr/bin/env bash
# Launch a bounded FACE-Q teacher-forced closure diagnostic.
#
# This is a circuit test, not a production run. It reuses uploaded Pool A shard
# archives, first runs a model-free teacher-identity target/eval sanity check,
# then trains on a deliberately tiny subset. Passing criteria are:
#   - teacher_identity raw targets are watertight without boundary fill
#   - train teacher-forced token accuracy approaches 1.0
#   - train teacher-forced meshes are watertight without relying on repair
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)_faceq_tf_closure_gate}"

# Use already-preserved shards by default so this gate is quick and reproducible.
export RUN_STAMP
export SHARD_IDS="${SHARD_IDS:-0010 0011 0012 0013 0014}"
export STRICT_LIMIT="${STRICT_LIMIT:-512}"
export TEST_RATIO="${TEST_RATIO:-0.05}"

# Tiny train subset on purpose: if this cannot memorize, larger scale is waste.
export TRAIN_LIMIT="${TRAIN_LIMIT:-32}"
export STEPS="${STEPS:-30000}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-2048}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-384}"
export LAYERS="${LAYERS:-8}"
export HEADS="${HEADS:-8}"
export CONDITION_TOKENS="${CONDITION_TOKENS:-128}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
export LOG_EVERY="${LOG_EVERY:-500}"
export EVAL_LIMIT="${EVAL_LIMIT:-32}"
export PAIR_SAMPLES="${PAIR_SAMPLES:-1024}"

export B2_RUN_PREFIX="${B2_RUN_PREFIX:-face-runs/faceq-tf-closure-gates/$RUN_STAMP}"

exec "$(dirname "${BASH_SOURCE[0]}")/launch_faceq_partial_scale_gate.sh"
