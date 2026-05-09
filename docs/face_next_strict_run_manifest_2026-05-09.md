# FACE Next Strict Run Manifest - 2026-05-09

## Why This Exists

The completed `20260508_a100_objpp10k_rotmin_cleanfilters_rung1b` run is not a
strict FACE evidence run. It used:

```text
FACE_EMBEDDING_VARIANT=continuous_mlp
ALLOW_DEPRECATED_FACE_EMBEDDING=1
```

That bypassed the smoking-gun FACE insight: previous faces should be embedded by
preserving the ordered nine quantized coordinate-token identities and projecting
them into one face token. In this repo that strict lane is:

```text
FACE_EMBEDDING_VARIANT=token_concat_project
```

The next run must use this manifest or an equivalent command that passes the
same preflight guards.

## Required Strict Settings

```text
STRICT_FACE_PAPER_GATE=1
FACE_EMBEDDING_VARIANT=token_concat_project
ALLOW_DEPRECATED_FACE_EMBEDDING=0
CAUSAL_MLP_VARIANT=legacy_concat
DECODE_HEAD=causal

NUM_BINS=128
POINT_SAMPLES=8192
VECSET_TOKENS=2048
LATENT_DIM=64
OPTIMIZER=muon
LR=0.0006
WEIGHT_DECAY=0.1
PRECISION=bf16

PAPER_WITHIN_FACE_ORDER=rotate_min_zyx
CACHE_FPS_INDICES=0
```

## Recommended Next Command

```bash
RUN_STAMP=20260509_strict_tokenconcat_10k \
scripts/thunder/launch_face_next_strict_gate.sh
```

The wrapper hard-sets the smoking-gun settings and delegates to the normal
Thunder corpus-gate launcher.

## Guardrails Added

The following scripts now refuse deprecated face embeddings during strict paper
gates:

```text
scripts/thunder/launch_face_paper_corpus_gate_instance.sh
scripts/thunder/launch_face_paper_existing_split_on_instance.sh
scripts/thunder/face_paper_curated_corpus_gate.sh
scripts/thunder/face_paper_existing_split_gate.sh
scripts/thunder/face_paper_train_eval_job.sh
```

To run `continuous_mlp` or `discrete_sum`, the run must be explicitly labeled as
an ablation and set:

```text
STRICT_FACE_PAPER_GATE=0
```

That output must not be used as primary FACE paper-faithful evidence.

## Acceptance Criteria

Do not promote based on loss alone. Promotion requires:

```text
train teacher-forced accuracy materially higher than the failed run
train free-running AR token accuracy materially higher than the failed run
train AR meshes mostly close / low boundary edges
held-out AR not collapsing on the first face
held-out boundary and edge-pairing metrics materially improve
visual contact sheets are coherent
```

The failed `continuous_mlp` run ended with selection loss near `1.96`, but
free-running AR still collapsed. The next strict run must prove generation
dynamics, not just token loss.
