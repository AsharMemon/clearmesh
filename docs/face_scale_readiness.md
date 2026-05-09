# FACE Scale Readiness

This note tracks the evidence gates for scaling the FACE paper-lane run. The goal is to avoid spending on a 130k+ mesh / 100k-step run until the smaller probes show that the implementation, tokenization, and free-running topology are healthy.

## Current Evidence

- Strict paper knobs are active in the A100 lane: 128 coordinate bins, 8192 point samples, 2048 VecSet tokens, bottleneck dimension 64, Muon, bf16, and online rotation/flipping/axis-scaling augmentation.
- The 8-sample no-augmentation memorization probe reached near-zero teacher-forced loss.
- Teacher-forced memorization decoded watertight meshes: 8/8 watertight, boundary edges 0.
- Full-GT-face autoregressive memorization decoded watertight meshes: 5/5 watertight, boundary edges 0.
- The earlier 128-face AR memorization failure was a truncated-prefix evaluation artifact, not proof that the model could not close topology.
- The strict110 paper-augmentation A100 probe completed and is a clean no-scale signal.
- The strict110 no-augmentation A100 probe completed and is useful as a capacity diagnostic, not paper-faithful scale evidence.
- A bounded 512-candidate Objaverse++ corpus rung is running on A100. It built a
  strict split of 285 passing token samples (228 train / 57 test) and is
  training the strict paper lane with full-face AR eval enabled.

## Latest A100 Probe Results

### Strict110 Paper Augmentation

Artifact root:

```text
.codex_outputs/face_strict110_probe_20260506_074020/clearmesh_face_strict110_probe_20260506_074020
```

Key metrics:

```text
train teacher-forced:
  accuracy: 0.1491
  loss: 2.9754
  watertight: 0/88
  mean boundary edges: 1008.1

train full-face AR:
  accuracy: 0.1374
  loss: 2.9596
  watertight: 0/10
  mean boundary edges: 82.6
  mean edge pairing: 0.0765

test full-face AR:
  accuracy: 0.1322
  loss: 3.0644
  watertight: 0/10
  mean boundary edges: 75.7
  mean edge pairing: 0.0583
```

Interpretation:

```text
No production scaling. The model is underfit under paper augmentation at this
data size, and free-running topology is not coherent.
```

### Strict110 No Augmentation

Artifact root:

```text
.codex_outputs/face_strict110_noaug_probe_20260506_094422/clearmesh_face_strict110_noaug_probe_20260506_094422
```

Key metrics:

```text
train teacher-forced:
  accuracy: 0.99898
  loss: 0.00592
  watertight: 39/88
  mean boundary edges: 10.85
  mean edge pairing: 0.98877

train full-face AR:
  accuracy: 0.99934
  loss: 0.00467
  watertight: 2/10
  mean boundary edges: 14.2
  mean edge pairing: 0.98859

test full-face AR:
  accuracy: 0.0516
  loss: 8.5758
  watertight: 0/10
  mean boundary edges: 983.2
  mean edge pairing: 0.2450
```

Interpretation:

```text
The model has enough capacity to memorize the 88 training meshes without
augmentation, so the tokenizer/model path is not fundamentally dead. But this is
not a paper-faithful scale signal because augmentation is disabled and held-out
generation collapses.
```

## Automated Scale Gate

Future FACE paper-lane train/eval jobs now write:

```text
<run_dir>/scale_readiness.json
```

The gate blocks promotion unless all of the following are true:

- Paper knobs match the strict lane: 8192 points, 2048 VecSet tokens, latent 64, Muon, bf16, online augmentation, Shape2VecSet encoder, causal CausalMLP.
- Eval is full-face, not a truncated prefix.
- Train teacher-forced accuracy/loss show that the run is fitted.
- Train full-face AR is mostly watertight with low boundary edges and high edge pairing.
- Held-out teacher-forced and held-out full-face AR show a real generalization signal.
- Geometry metrics and visual contact sheets are plausible enough to justify the next corpus rung.

## Scale Gate

Do not launch a full production run unless a bounded paper-lane run shows:

- Train and held-out teacher-forced losses trend down without collapse or obvious data corruption.
- Full-face autoregressive train previews are coherent and mostly watertight.
- Full-face autoregressive held-out previews are at least structurally plausible.
- Boundary-edge counts are low enough that failures are model/data quality issues rather than tokenizer/eval artifacts.
- No obvious mismatch remains in FACE tokenization, augmentation, CausalMLP conditioning, or face-count handling.
- Reduced-capacity gates can promote us to the next corpus rung, but not to a
  full paper reproduction claim. The full production profile still needs the
  larger paper-scale model capacity once the smaller gates prove the lane is
  healthy.

## Current Bounded Corpus Rung

Thunder:

```text
instance: 0 / l8eerqv2
remote lab: /tmp/clearmesh_face_paper_corpus_gate_20260506_112711_paper512
run: /tmp/clearmesh_face_paper_corpus_gate_20260506_112711_paper512/runs/paper_corpus_gate_128_vec2048_muon_aug
```

Corpus:

```text
downloaded candidates: 512
curated meshes: 423
tokenized meshes: 417
strict passing token samples: 285
train/test split: 228 / 57
```

Early training signal:

```text
step 1 selection loss: 4.8897
step 1000 selection loss: 4.0988
step 2000 selection loss: 3.4827
step 3000 selection loss: 3.3578
step 4000 selection loss: 3.3224
step 5000 selection loss: 3.2756
step 6000 selection loss: 3.2362
step 7000 selection loss: 3.2021
step 7500 train loss: 4.1590
step 8000 selection loss: 3.1789
step 9000 selection loss: 3.1498
step 10000 selection loss: 3.1267
step 11000 selection loss: 3.1214
step 12000 selection loss: 3.2172
step 13000 selection loss: 3.0761
step 13500 train loss: 3.1624
step 14000 selection loss: 3.0723
step 15000 selection loss: 3.0736
step 15500 train loss: 3.1285
step 16000 selection loss: 3.0596
step 17000 selection loss: 3.0403
step 17500 train loss: 3.3491
step 18000 selection loss: 3.0318
throughput: about 2.3-2.7 steps/sec on A100
```

Interpretation:

```text
Continue bounded run to completion. This is not production-scale evidence yet.
Promotion requires scale_readiness.json plus train/test full-face AR contact
sheets after eval.
```

## Production Run Shape

Future scale attempts should be two-stage:

```text
1. Corpus/cache stage on cheaper hardware:
   scripts/thunder/face_objaversepp_corpus_pilot.sh

2. Package the strict split for durable storage / upload:
   scripts/research/package_face_corpus.py

3. Train/eval stage on A100/H100 from the prepared split:
   scripts/thunder/face_paper_existing_split_gate.sh
```

This keeps the paper lane identical while avoiding A100 time spent on Objaverse
download, target conversion, tokenization, and strict gate filtering. The
train-only gate consumes:

```text
DATA_RUN=<prepared corpus root>
SPLIT_DIR=$DATA_RUN/split_pass
TRAIN_DIR=$SPLIT_DIR/train
TEST_DIR=$SPLIT_DIR/test
```

and writes the same outputs as the full corpus gate:

```text
<run_dir>/summary.json
<run_dir>/scale_readiness.json
<lab_root>/train_ar_contact_sheet.png
<lab_root>/test_ar_contact_sheet.png
<lab_root>.tar.gz
```

Lean corpus packaging command:

```bash
python scripts/research/package_face_corpus.py \
  --data-run /tmp/clearmesh_face_objpp_corpus \
  --output /tmp/clearmesh_face_objpp_corpus_train_split.tar.gz
```

Optional B2 upload:

```bash
B2_BUCKET=clearmesh-pairs \
B2_PREFIX=face-corpora \
scripts/data/upload_face_corpus_b2.sh /tmp/clearmesh_face_objpp_corpus_train_split.tar.gz
```

Download the same prepared corpus on another machine/GPU:

```bash
B2_BUCKET=clearmesh-pairs \
B2_PREFIX=face-corpora \
scripts/data/download_face_corpus_b2.sh face_objpp_corpus_train_split.tar.gz /tmp/clearmesh_face_corpus
```

Train-only launch on a prepared Thunder instance:

```bash
THUNDER_INSTANCE_ID=<id> \
LOCAL_DATA_DIR=/path/to/prepared_corpus_or_extracted_archive \
scripts/thunder/launch_face_paper_existing_split_on_instance.sh
```

One-shot harvest for active/completed runs:

```bash
RUN_INFO=.codex_outputs/face_paper_corpus_gate_setup_20260506_112711_paper512/run_info.json
DELETE_ON_COMPLETE=1 \
DELETE_ON_FAILURE=1 \
scripts/thunder/harvest_face_paper_run.sh "$RUN_INFO"
```

Or, in a single shell-safe line that does not expand `$RUN_INFO` before assigning it:

```bash
RUN_INFO=.codex_outputs/face_paper_corpus_gate_setup_20260506_112711_paper512/run_info.json \
DELETE_ON_COMPLETE=1 \
DELETE_ON_FAILURE=1 \
scripts/thunder/harvest_face_paper_run.sh
```

Combined harvest plus inspection wrapper:

```bash
RUN_INFO=.codex_outputs/face_paper_corpus_gate_setup_20260506_112711_paper512/run_info.json \
DELETE_ON_COMPLETE=1 \
DELETE_ON_FAILURE=1 \
scripts/thunder/harvest_and_inspect_face_paper_run.sh
```

Inspect a fetched/extracted lab root and get a machine-readable next action:

```bash
python scripts/research/inspect_face_paper_gate.py \
  .codex_outputs/face_paper_corpus_gate_setup_20260506_112711_paper512/clearmesh_face_paper_corpus_gate_20260506_112711_paper512
```

Guarded next-rung launcher. This refuses to spend GPU unless the inspection is
scale-ready, train/test AR contact sheets exist, visual review is explicitly
marked as passed, and scale is explicitly confirmed:

```bash
INSPECTION_JSON=.codex_outputs/face_paper_corpus_gate_setup_20260506_112711_paper512/face_gate_inspection.json \
NEXT_SELECT_TARGET=2048 \
VISUAL_REVIEW_PASSED=1 \
CONFIRM_SCALE=1 \
scripts/thunder/promote_face_paper_next_rung.sh
```

## If The Current Paper Gate Is Strong

- Fetch the archive and inspect train/test contact sheets.
- Run a predicted-face-count/EOS eval as a separate production-readiness probe.
- Build a 1k-5k curated watertight mesh corpus and run the same paper-lane profile before jumping to 130k+.
- Only then launch the 130k+ / 100k-step production run.

## If The Current Paper Gate Is Weak

- Already completed: run a 110-sample no-augmentation overfit with full-face AR to separate capacity from augmentation/generalization.
- Next: build a larger curated strict 128-bin corpus before judging paper augmentation again; 88 train meshes is too small for the paper's random rotation/flip/axis-scale regime.
- Next: keep the current strict paper lane as the source of truth; indexed topology, boundary fill, and repair-based promotion remain product experiments, not FACE paper evidence.
- Next: add a predicted-face-count/EOS diagnostic only after GT-face-count train AR is mostly closed.

## Current Decision

```text
Scale to 130k/100k-step production run: NO.
Run current bounded paper-lane corpus rung to eval: YES.
Run another/larger corpus rung: ONLY after inspecting current scale_readiness and galleries.
Keep Thunder instances idle: NO; all instances should be deleted after archive fetch.
```
