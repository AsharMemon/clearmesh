#!/usr/bin/env bash
# Launch the next strict FACE gate with the current paper-faithful settings.
#
# This wrapper exists to prevent stale ablation knobs from leaking into the next
# expensive Thunder run. Override only the corpus size / GPU / run stamp unless
# you are intentionally running an explicitly named ablation.
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)_strict_tokenconcat}"
RUN_LABEL="${RUN_LABEL:-strict_tokenconcat_legacy_causal_512f_30k}"

export RUN_STAMP
export RUN_LABEL

# Corpus rung defaults: bounded evidence gate, not production scale.
export SELECT_TARGET="${SELECT_TARGET:-10000}"
export SCAN_LIMIT="${SCAN_LIMIT:-300000}"
export CURATION_TARGET="${CURATION_TARGET:-10000}"
export MIN_QUALITY="${MIN_QUALITY:-2}"
export TARGET_FACES="${TARGET_FACES:-512}"
export TOKEN_MAX_FACES="${TOKEN_MAX_FACES:-512}"
export MODEL_MAX_FACES="${MODEL_MAX_FACES:-512}"
export NUM_BINS="${NUM_BINS:-128}"
export PAPER_WITHIN_FACE_ORDER="${PAPER_WITHIN_FACE_ORDER:-rotate_min_zyx}"

# FACE paper-method knobs.
export POINT_SAMPLES="${POINT_SAMPLES:-8192}"
export VECSET_TOKENS="${VECSET_TOKENS:-2048}"
export LATENT_DIM="${LATENT_DIM:-64}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
export ENCODER_HIDDEN_SIZE="${ENCODER_HIDDEN_SIZE:-768}"
export ENCODER_LAYERS="${ENCODER_LAYERS:-6}"
export DECODER_LAYERS="${DECODER_LAYERS:-12}"
export HEADS="${HEADS:-12}"
export OPTIMIZER="${OPTIMIZER:-muon}"
export LR="${LR:-0.0006}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export PRECISION="${PRECISION:-bf16}"
export STEPS="${STEPS:-30000}"
export BATCH_SIZE="${BATCH_SIZE:-2}"

# Smoking-gun strict lane. Do not override in a paper gate.
export CAUSAL_MLP_VARIANT=legacy_concat
export FACE_EMBEDDING_VARIANT=token_concat_project
export ALLOW_DEPRECATED_FACE_EMBEDDING=0
export STRICT_FACE_PAPER_GATE=1
export DECODE_HEAD=causal

# Online paper augmentation is allowed here; CACHE_FPS_INDICES must stay off.
export CACHE_FPS_INDICES=0
export PREFETCH_BATCHES="${PREFETCH_BATCHES:-1}"
export TEACHER_FORCED_LIMIT="${TEACHER_FORCED_LIMIT:-256}"
export AR_LIMIT="${AR_LIMIT:-32}"
export TEACHER_PREFIX_LIMIT="${TEACHER_PREFIX_LIMIT:-16}"
export TEACHER_PREFIX_FACE_COUNTS="${TEACHER_PREFIX_FACE_COUNTS:-1 4 16}"
export PREDICTED_LIMIT="${PREDICTED_LIMIT:-8}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
export SAVE_CURRENT_CHECKPOINT="${SAVE_CURRENT_CHECKPOINT:-1}"
export SELECTION_EVAL_EVERY="${SELECTION_EVAL_EVERY:-1000}"
export SELECTION_EVAL_BATCH_SIZE="${SELECTION_EVAL_BATCH_SIZE:-1}"
export MIN_SCALE_DATASET_SAMPLES="${MIN_SCALE_DATASET_SAMPLES:-1000}"
export FAIL_ON_SCALE_NOT_READY="${FAIL_ON_SCALE_NOT_READY:-0}"
export RUN_REMOTE_TESTS="${RUN_REMOTE_TESTS:-1}"

echo "[clearmesh] next strict FACE gate"
echo "  run_stamp=$RUN_STAMP"
echo "  run_label=$RUN_LABEL"
echo "  face_embedding_variant=$FACE_EMBEDDING_VARIANT"
echo "  causal_mlp_variant=$CAUSAL_MLP_VARIANT"
echo "  allow_deprecated_face_embedding=$ALLOW_DEPRECATED_FACE_EMBEDDING"
echo "  strict_face_paper_gate=$STRICT_FACE_PAPER_GATE"
echo "  select_target=$SELECT_TARGET steps=$STEPS model_max_faces=$MODEL_MAX_FACES"

scripts/thunder/launch_face_paper_corpus_gate_instance.sh "$@"
