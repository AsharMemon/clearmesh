# FACE First-Face / Constrained Decode Probe - 2026-05-11

## Purpose

We tested whether the 10k/100k FACE checkpoint already contains a usable first-face/topology solution that greedy decoding is failing to extract.

The probe keeps the trained model unchanged and uses the same FACE strict-lane path:

- same Shape2VecSet-style encoder
- same causal face decoder
- same `legacy_concat` CausalMLP
- same `token_concat_project` face embedding
- same `rotate_min_zyx` FACE paper ordering
- same 128-bin, 2048-VecSet-token, 64-latent checkpoint

The probe adds only inference-time diagnostics:

- first-face within-face beam search
- first-face teacher-path rank measurement
- geometric rescoring from input point cloud support
- canonical minimum-vertex anchor scoring from FACE ordering logic
- conservative topology-aware rollout scoring

This is not a paper-claimed decoder. It is a hostile diagnostic for whether smarter decoding can rescue the existing checkpoint.

## Artifacts

Remote A100 run root:

```text
/tmp/clearmesh_face_paper_corpus_gate_20260510_rotate10k_100k_preserve_a100_v2
```

Probe outputs:

```text
runs/paper_corpus_gate_128_vec2048_muon_aug/eval/first_face_probe_beam512_train.json
runs/paper_corpus_gate_128_vec2048_muon_aug/eval/first_face_probe_beam512_test.json
runs/paper_corpus_gate_128_vec2048_muon_aug/eval/constrained_decode_probe_test.json
```

Local lightweight copies:

```text
.codex_outputs/face_first_face_probe_20260511/first_face_probe_beam512_train.json
.codex_outputs/face_first_face_probe_20260511/first_face_probe_beam512_test.json
.codex_outputs/face_first_face_probe_20260511/constrained_decode_probe_test.json
```

Implementation:

```text
scripts/research/probe_face_first_face_decode.py
```

## Probe Settings

First-face beam probes:

```text
limit: 32 train / 32 test
beam_width: 512
slot_topk: 16
rollout: disabled
```

Constrained rollout probe:

```text
limit: 16 test
beam_width: 64
slot_topk: 8
rollout_faces: 64
strategies: greedy, hybrid_constrained, topology_constrained, surface_constrained
```

## Result 1: Exact Teacher First Face Is Not In Practical Beam

| Split | Samples | Exact first face in beam | Teacher path max-rank mean | Teacher path mean-rank mean | Teacher-path top-1 slot acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 32 | 0.0% | 33.09 | 9.18 | 24.65% |
| Test | 32 | 0.0% | 37.91 | 10.50 | 25.69% |

Interpretation:

The correct first face is not merely being discarded by greedy top-1. It is usually outside a practical joint face beam. Even when we force the teacher prefix inside the CausalMLP slot chain, at least one of the 9 coordinate slots often has rank above 16.

Teacher max-rank threshold rates:

| Split | <=8 | <=16 | <=32 | <=64 | <=128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 3.1% | 31.2% | 71.9% | 84.4% | 100% |
| Test | 3.1% | 28.1% | 65.6% | 81.2% | 100% |

This means top-k helps only if k is very large, and a large k still needs a very good rescoring signal to choose the exact face from many plausible first triangles.

## Result 2: Geometric Rescoring Helps Slightly, Not Enough

| Split | Best logprob first-face L1 | Best hybrid first-face L1 | Best oracle-in-beam L1 | Best logprob token acc | Best hybrid token acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 10.40 | 8.31 | 6.91 | 21.18% | 25.35% |
| Test | 12.55 | 11.88 | 10.99 | 21.88% | 24.31% |

The hybrid score uses logprob, surface support from the point cloud, the FACE canonical minimum-vertex anchor, and topology validity. It improves the average first-face distance a little, but the selected face is still far from exact.

Conclusion: point-cloud geometry can nudge candidates but cannot reconstruct missing probability mass from weak first-face logits.

## Result 3: Topology-Aware Rollout Does Not Rescue This Checkpoint

Held-out 16-sample, 64-face rollout:

| Strategy | Token acc | Edge F1 | Boundary edges | Edge pairing | Watertight edge graph | Face-0 divergence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy | 11.83% | 0.0104 | 12.06 | 0.634 | 18.75% | 100% |
| Hybrid constrained | 10.38% | 0.0104 | 19.13 | 0.382 | 0.00% | 100% |
| Surface constrained | 12.01% | 0.0052 | 14.69 | 0.341 | 0.00% | 100% |
| Topology constrained | 10.94% | 0.0104 | 19.75 | 0.394 | 0.00% | 100% |

Interpretation:

The topology-aware heuristic is not rescuing the current model. In fact, it often hurts. That is useful: the candidate set at each AR step is not rich enough for local topology rescoring to find the true mesh sequence. The decoder is not just choosing the wrong manifold option; it is not producing enough correct local options.

## Paper / Logic Cross-Check

The FACE paper's core bet is that end-to-end training makes the VecSet latent and face decoder learn a semantically meaningful sequence distribution. The paper does not claim a beam decoder, geometric rescoring, or topology-constrained inference. So this probe is a legitimate diagnostic extension, not a paper-faithfulness requirement.

The probe does reinforce the paper's likely dependency on scale/capacity:

- A first face must be inferred from only `C`, BOS, and position 0.
- The paper-scale model/dataset are much larger than this run.
- Our smaller model has not learned a sharp enough BOS-to-first-face distribution.

But it also tells us not to expect decoding tricks alone to substitute for that learned distribution.

## Diagnosis

The exact failure is now sharper:

```text
The 100k checkpoint does not assign enough probability to the exact canonical first face.
The correct first face is absent from beam512/slot-topk16 for 32/32 train and 32/32 test samples.
Topology/geometric rescoring cannot recover what the candidate distribution does not contain.
```

So the next productive interventions are training/representation interventions, not just inference-time search.

## Recommended Next Actions

1. Run a first-face auxiliary objective or curriculum gate.

Train with explicit monitoring/loss on face 0, or temporarily oversample/weight the first face. This is still paper-compatible if treated as a diagnostic weighting ablation, but not a pure paper reproduction claim.

2. Try paper-scale capacity before more data-only scaling.

Use hidden 1024, decoder 24, heads 16 on the existing 5k-10k usable corpus. If first-face ranks improve materially, scale data next.

3. Add exact first-face validation to scale gates.

Do not use loss alone. Track:

```text
first-face exact rate
teacher-path max rank
teacher-path top-k coverage
beam exact-in-beam rate
prefix1/prefix16 rollout quality
```

4. Consider a representation fix if strict FACE continues to fail.

Coordinate-only FACE tokens make topology implicit and fragile. The indexed/connectivity-token lane remains the more direct route to watertight editable meshes if paper-faithful FACE cannot learn exact welding at practical scale.

## Bottom Line

Top-k/beam/geometric/topology decoding does not rescue the 100k checkpoint. The result is not neutral: it proves the first-face problem is inside the learned distribution, not just in greedy inference.

Scaling may still help, especially with paper-scale capacity, but the next gate should require first-face rank/beam improvement before launching a full 350k/500k production training run.

## Addendum: Claude Opus Review + Anchor-Equivalence Audit

A narrow Claude Opus review agreed with the main diagnosis: this is not primarily a greedy-decoding problem. It ranked the likely failure as:

1. Face-0 gradient starvation: face 0 is a tiny fraction of full-sequence CE, but it is the only step that must localize geometry from `C` without autoregressive context.
2. VecSet-to-anchor localization failure: the decoder is choosing faces from the wrong mesh region at the cold-start position.
3. Face-order degeneracy: real, but not yet the binding constraint.
4. Data-only scaling: premature until train-set face-0 rank improves.

We then audited whether exact row-0 failure was just an arbitrary tie-breaking artifact under `rotate_min_zyx`.

Ordering degeneracy is severe:

| Metric | Value |
| --- | ---: |
| Samples audited | 5,276 |
| Samples with >1 same minimum ZYX anchor | 100% |
| Mean same-anchor group size | 5.66 faces |
| Mean second-min L-infinity gap | 0 bins |
| Mean second-min L1 gap | 0 bins |

But the equivalence probe showed that the existing checkpoint does **not** even recover the same-anchor group:

| Split | Same-anchor group mean | Best logprob same-min vertex | Best hybrid same-min vertex | Best oracle-in-beam same-min vertex |
| --- | ---: | ---: | ---: | ---: |
| Train | 5.59 | 0.0% | 0.0% | 0.0% |
| Test | 5.56 | 0.0% | 0.0% | 0.0% |

This changes the interpretation slightly:

```text
Tie ambiguity makes exact row-0 too strict later.
But this checkpoint currently misses the correct first-anchor region entirely.
So the immediate bottleneck is VecSet/BOS -> first-anchor localization, not arbitrary tie choice.
```

Additional artifacts:

```text
runs/paper_corpus_gate_128_vec2048_muon_aug/eval/face_order_degeneracy_train_test.json
runs/paper_corpus_gate_128_vec2048_muon_aug/eval/first_face_equivalence_train.json
runs/paper_corpus_gate_128_vec2048_muon_aug/eval/first_face_equivalence_test.json
.codex_outputs/claude_consults/face0_equivalence_opus_consult_20260511.md
```

## Current Bounded Intervention

Launched a diagnostic continuation from the 100k checkpoint:

```text
run: first_face_weight8_cont3k_20260511_v2
init: paper_corpus_gate_128_vec2048_muon_aug/checkpoint.pt
steps: 3,000
lr: 1e-4
first_face_loss_weight: 8.0
architecture/data/order: unchanged
```

Pass/fail gate for this diagnostic:

| Metric | Pass | Partial | Fail |
| --- | ---: | ---: | ---: |
| Train teacher max-rank | <=8 | <=16 | >16 |
| Test teacher max-rank | <=12 | <=24 | >24 |
| Train same-min-vertex rate | >50% | >0% | 0% |
| Exact-in-beam at beam512/topk16 | >0/32 | n/a | 0/32 |

If this fails, the next default-off tool is now available:

```text
--loss-face-prefix-count 1
```

That masks coordinate CE to face 0 only while preserving the default paper loss when unset. The intended next bounded test would be a short first-face curriculum, then a full-loss continuation, before any larger data scaling.

## Weight-Only Continuation Result

The `first_face_weight8_cont3k_20260511_v2` diagnostic completed and failed the gate.

| Split | Exact in beam | Teacher max-rank mean | Teacher mean-rank mean | Slot top-1 acc | Same-min vertex rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.0% | 30.25 | 9.22 | 23.26% | 0.0% |
| Test | 0.0% | 37.69 | 11.19 | 27.43% | 0.0% |

Relative to the original checkpoint, train rank improved only slightly and test did not improve. This rules out a tiny post-hoc face-0 weight continuation as sufficient.

Next rung launched:

```text
run: face0_prefix32_2k_20260511
init: paper_corpus_gate_128_vec2048_muon_aug/checkpoint.pt
steps: 2,000
lr: 1e-4
first_face_loss_weight: 32.0
loss_face_prefix_count: 1
eos_loss_weight: 0
```

This is the direct test of whether the existing VecSet can learn the cold-start anchor when gradients are not diluted by the remaining face sequence.

## Prefix Curriculum Result

The online-augmented `face0_prefix32_2k_20260511` run provided a weak positive signal but still failed the gate.

| Split | Exact in beam | Teacher max-rank mean | Teacher mean-rank mean | Slot top-1 acc | Hybrid same-min vertex | Oracle same-min vertex |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 0.0% | 29.78 | 9.58 | 28.13% | 6.25% | 12.50% |
| Test | 0.0% | 37.19 | 10.88 | 30.56% | 3.13% | 3.13% |

Interpretation:

```text
The VecSet/BOS -> first-anchor mapping can be nudged, but a short online-augmented prefix curriculum is not enough.
The first face remains outside beam512/topk16 and mostly outside the correct minimum-vertex region.
```

A stricter isolating diagnostic was launched next:

```text
run: face0_prefix32_noaug_cacheoff_2k_20260511
init: paper_corpus_gate_128_vec2048_muon_aug/checkpoint.pt
steps: 2,000
first_face_loss_weight: 32.0
loss_face_prefix_count: 1
disable_augment: true
cache_fps_indices: false
```

This is not intended as production training. It asks one specific question: can the current representation close canonical train face 0 when augmentation entropy is removed? If no, the next likely intervention is architectural: an auxiliary first-face/start-anchor head or explicit anchor conditioning, rather than more blind data scaling.

## No-Augmentation Face-0 Diagnostic Result

The no-augmentation `face0_prefix32_noaug_cacheoff_2k_20260511` run produced the first strong evidence that the existing representation can learn the cold-start anchor if the target is stable.

Training signal:

```text
initial canonical face-0 loss: 2.42
best canonical face-0 loss: 0.47 at step 1992
```

Beam/equivalence probe:

| Split | Exact in beam | Teacher max-rank mean | Teacher mean-rank mean | Slot top-1 acc | Hybrid L1 | Oracle L1 | Hybrid same-min vertex | Oracle same-min vertex |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 6.25% | 18.09 | 5.08 | 39.24% | 3.52 | 2.45 | 12.50% | 25.00% |
| Test | 6.25% | 25.69 | 6.80 | 36.11% | 6.47 | 3.60 | 9.38% | 12.50% |

Compared with the original 100k checkpoint:

```text
train max rank: 33.09 -> 18.09
test max rank: 37.91 -> 25.69
train exact-in-beam: 0.00% -> 6.25%
test exact-in-beam: 0.00% -> 6.25%
```

Interpretation:

```text
The model is not fundamentally incapable of first-face localization.
The paper-style online augmentation / arbitrary first-face target appears to inject enough cold-start entropy that full-sequence CE does not solve face 0.
```

Scale implication:

```text
More samples can help after we add a first-face curriculum/stabilization stage.
More samples alone are unlikely to fix this, because the 10k/100k run failed on train face 0 too.
```

Next rung launched:

```text
run: face0_prefix32_noaug_cacheoff_cont8k_20260511
init: face0_prefix32_noaug_cacheoff_2k_20260511/checkpoint.pt
steps: 8,000
first_face_loss_weight: 32.0
loss_face_prefix_count: 1
disable_augment: true
```

Gate for continuing toward larger data:

```text
train teacher max-rank <= 8
test teacher max-rank <= 12-16
exact-in-beam materially above zero
same-min vertex rate > 50% train, >25% held-out
```

## Lane Decision: Tie-Marginal Face-0 Loss

Claude Opus hostile-audited the next-lane choice and converged with our local diagnosis: do **Lane A**, not blind scale, not inference-only rescoring, and not a MeshRipple-style pivot yet.

Decision:

```text
Implement and test first-face tie-group / set-marginal loss.
```

Why:

```text
- Every audited sample has multiple faces sharing the first rotate_min_zyx minimum anchor.
- The arbitrary row-0 target is therefore label-noisy.
- No-augmentation prefix curriculum already moved predictions toward the right region.
- Same-min oracle signal means some currently "wrong" predictions are actually valid tie-group starts.
- Scaling arbitrary tie labels is likely to scale contradiction, not solve it.
```

Implementation added:

```text
flag: --first-face-tie-marginal-loss
scope: face 0 only
objective: -logsumexp over unique same-min-anchor candidate face probabilities
baseline: default off, normal FACE CE unchanged
compatibility: requires causal decode head
```

Tests:

```text
- singleton same-min group matches ordinary CE
- same-min non-row0 candidate receives credit
- focused FACE suite: 64 passed
```

Bounded falsification run launched on Thunder A100:

```text
run: face0_tie_marginal_prefix32_noaug_5k_20260511
init: face0_prefix32_noaug_cacheoff_cont8k_20260511/checkpoint.pt
steps: 5,000
first_face_loss_weight: 32.0
loss_face_prefix_count: 1
first_face_tie_marginal_loss: true
disable_augment: true
selection_eval_every: 1,000
```

Pass signal:

```text
train maxrank <= 8
held-out maxrank <= 12-16
same-min oracle/group metrics materially above the no-aug 8k checkpoint
```

If this fails:

```text
The first-face bottleneck is more likely VecSet/BOS/CausalMLP capacity or missing explicit anchor conditioning, not just ordering label ambiguity.
```

## Tie-Marginal 5K Result

Run completed on the A100:

```text
run: face0_tie_marginal_prefix32_noaug_5k_20260511
checkpoint: /tmp/clearmesh_face_paper_corpus_gate_20260510_rotate10k_100k_preserve_a100_v2/runs/face0_tie_marginal_prefix32_noaug_5k_20260511/checkpoint.pt
```

Training/selection objective:

```text
selection loss: 1.7108 -> 1.6327 -> 1.5787 -> 1.5282 -> 1.4704
best step: 5000
```

First-face probe comparison, 32 train / 32 test, beam512, slot-topk16:

| Run | Split | Exact in beam | Max-rank mean | Mean-rank mean | Slot top-1 acc | Hybrid L1 | Oracle L1 | Hybrid same-min | Oracle same-min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 10k/100k | Train | 0.00% | 33.09 | 9.18 | 24.7% | 8.31 | 6.91 | 0.00% | 0.00% |
| Original 10k/100k | Test | 0.00% | 37.91 | 10.50 | 25.7% | 11.88 | 10.99 | 0.00% | 0.00% |
| No-aug prefix 2k | Train | 6.25% | 18.09 | 5.08 | 39.2% | 3.52 | 2.45 | 12.50% | 25.00% |
| No-aug prefix 2k | Test | 6.25% | 25.69 | 6.80 | 36.1% | 6.47 | 3.60 | 9.38% | 12.50% |
| No-aug prefix 8k | Train | 21.88% | 17.50 | 4.75 | 47.9% | 3.49 | 1.93 | 15.62% | 40.62% |
| No-aug prefix 8k | Test | 6.25% | 24.88 | 6.38 | 37.2% | 6.46 | 3.38 | 12.50% | 18.75% |
| Tie-marginal 5k | Train | 25.00% | 16.75 | 4.28 | 50.7% | 2.75 | 1.73 | 31.25% | 50.00% |
| Tie-marginal 5k | Test | 3.12% | 27.19 | 6.68 | 41.0% | 7.24 | 4.05 | 12.50% | 28.12% |

Interpretation:

```text
Tie-marginal loss is a real train-side improvement and improves held-out same-min oracle/top-1 signal.
It does not yet solve held-out exact face-0, and test max-rank worsened relative to the no-aug 8k checkpoint.
This is a positive research signal, not a green light for a 350k/100k production run.
```

Critical next step:

```text
Use tie-marginal as part of the curriculum, but add a held-out-aware or augmentation-ramped stage before scaling:
1. tie-marginal prefix with no aug / z-aug ramp
2. full-sequence continuation with tie-marginal still active
3. then scale data only if test maxrank and same-min metrics improve together
```

## Follow-Up Curriculum Results

The first tie-marginal result was a useful signal, but not enough to scale. Two bounded follow-ups were run to falsify the next obvious hypotheses:

```text
1. Gentle z-rotation ramp from the tie-marginal prefix checkpoint.
2. Full-sequence no-augmentation continuation from the same checkpoint.
```

### Gentle Z-Ramp 5K

Run:

```text
run: face0_tie_marginal_prefix32_zramp_5k_20260512
init: face0_tie_marginal_prefix32_noaug_5k_20260511/checkpoint.pt
steps: 5,000
first_face_loss_weight: 32.0
loss_face_prefix_count: 1
first_face_tie_marginal_loss: true
augment_rotation: z
scale range: 0.9-1.1
flip probability: 0.0
lr: 7e-5
```

Result:

```text
selection loss: 1.7809 final
augmentation: 10,000 successes, 0 fallbacks
```

Probe result, 32 train / 32 test, beam512, slot-topk16:

| Run | Split | Exact in beam | Max-rank mean | Mean-rank mean | Slot top-1 acc | Hybrid L1 | Oracle L1 | Hybrid same-min | Oracle same-min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tie-marginal prefix 5k | Train | 25.00% | 16.75 | 4.28 | 50.7% | 2.75 | 1.73 | 31.25% | 50.00% |
| Tie-marginal prefix 5k | Test | 3.12% | 27.19 | 6.68 | 41.0% | 7.24 | 4.05 | 12.50% | 28.12% |
| Z-ramp 5k | Train | 6.25% | 15.66 | 4.70 | 44.1% | 3.86 | 2.53 | 15.62% | 40.62% |
| Z-ramp 5k | Test | 6.25% | 28.62 | 7.61 | 36.5% | 6.84 | 4.74 | 9.38% | 21.88% |

Interpretation:

```text
The z-ramp is not the next lane.
It slightly improved one exact held-out count but worsened the broader distributional metrics:
top-1 accuracy, mean-rank, oracle L1, and same-min rates.
Do not escalate to full SO3 augmentation until the early-prefix autoregressive problem is stronger.
```

### Full-Sequence No-Augmentation 5K

Run:

```text
run: face0_tie_marginal_fullseq_noaug_5k_20260512
init: face0_tie_marginal_prefix32_noaug_5k_20260511/checkpoint.pt
steps: 5,000
first_face_loss_weight: 16.0
loss_face_prefix_count: 0
first_face_tie_marginal_loss: true
disable_augment: true
lr: 7e-5
```

Result:

```text
selection loss: 0.5836 final
best step: 5000
```

Probe result, 32 train / 32 test, beam512, slot-topk16:

| Run | Split | Exact in beam | Max-rank mean | Mean-rank mean | Slot top-1 acc | Hybrid L1 | Oracle L1 | Hybrid same-min | Oracle same-min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tie-marginal prefix 5k | Train | 25.00% | 16.75 | 4.28 | 50.7% | 2.75 | 1.73 | 31.25% | 50.00% |
| Tie-marginal prefix 5k | Test | 3.12% | 27.19 | 6.68 | 41.0% | 7.24 | 4.05 | 12.50% | 28.12% |
| Full-seq no-aug 5k | Train | 37.50% | 13.25 | 3.62 | 56.6% | 2.48 | 1.48 | 40.62% | 59.38% |
| Full-seq no-aug 5k | Test | 15.62% | 22.69 | 6.09 | 38.5% | 5.85 | 3.80 | 3.12% | 31.25% |

Interpretation:

```text
This is the best first-face checkpoint so far.
Full-sequence no-augmentation did not erase the first-face gains; it improved train exact, held-out exact, max-rank, and oracle distance.
This suggests the model benefits from reconnecting face 0 to the full mesh sequence after a tie-marginal prefix stage.
```

However, short free-running rollout is still weak:

| Split | Strategy | First-face divergence | Boundary edges | Edge pairing | Generated token acc | Watertight edge graph |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | Greedy | 93.75% | 17.81 | 0.625 | 0.157 | 12.50% |
| Train | Hybrid constrained | 93.75% | 26.75 | 0.497 | 0.151 | 6.25% |
| Train | Topology constrained | 93.75% | 25.19 | 0.530 | 0.139 | 0.00% |
| Test | Greedy | 100.00% | 7.62 | 0.627 | 0.184 | 25.00% |
| Test | Hybrid constrained | 100.00% | 22.75 | 0.454 | 0.166 | 0.00% |
| Test | Topology constrained | 100.00% | 17.38 | 0.481 | 0.170 | 0.00% |

Conclusion:

```text
First-face teacher-forced/beam quality is now meaningfully better.
Free-running AR still fails early, and inference-only topology rescoring does not fix it.
The next bounded lane is early-prefix exposure-bias training, not blind 350k scale and not stronger decoding tricks.
```

## Prefix-16 Exposure-Bias Rung

The next experiment is a short prefix-16 no-augmentation continuation from the full-sequence checkpoint:

```text
run: face0_tie_marginal_prefix16_noaug_4k_20260512
init: face0_tie_marginal_fullseq_noaug_5k_20260512/checkpoint.pt
steps: 4,000
first_face_loss_weight: 16.0
loss_face_prefix_count: 16
first_face_tie_marginal_loss: true
disable_augment: true
lr: 7e-5
selection_eval_every: 4,000
```

Purpose:

```text
Test whether emphasizing the first 16 faces improves short free-running rollout without destroying first-face beam quality.
Pass criterion is improved rollout token accuracy / edge pairing, not just lower training loss.
```

If prefix-16 improves first-face metrics but rollout remains brittle, the next prepared diagnostic is a default-off noisy-teacher-prefix stage:

```text
flag: --input-face-token-noise-prob
flag: --input-face-token-noise-max-offset
flag: --input-face-noise-prefix-count
effect: perturb decoder input face tokens only; target faces are unchanged
purpose: train recovery from small early autoregressive errors
default: disabled, so strict paper teacher forcing is preserved unless explicitly enabled
```

Focused tests verify that decoder-input noise is opt-in and does not mutate targets.

### Prefix-16 Noisy Teacher-Input Probe

Run:

```text
run: face0_tie_marginal_prefix16_noisy005_noaug_3k_20260512
init: face0_tie_marginal_prefix16_noaug_4k_20260512/checkpoint.pt
steps: 3,000
first_face_loss_weight: 16.0
loss_face_prefix_count: 16
first_face_tie_marginal_loss: true
input_face_token_noise_prob: 0.05
input_face_token_noise_max_offset: 1
input_face_noise_prefix_count: 16
disable_augment: true
lr: 5e-5
```

Result:

```text
selection loss: 0.8541 final
previous prefix-16 selection loss: 0.9270
```

First-face probe result, 32 train / 32 test, beam512, slot-topk16:

| Run | Split | Exact in beam | Max-rank mean | Slot top-1 acc | Logprob exact row0 | Oracle exact row0 | Logprob same-min | Oracle same-min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefix-16 no-noise | Train | 43.75% | 13.53 | 62.85% | 15.62% | 43.75% | 56.25% | 71.88% |
| Prefix-16 no-noise | Test | 15.62% | 23.62 | 40.28% | 0.00% | 15.62% | 9.38% | 34.38% |
| Prefix-16 noise 0.05 | Train | 56.25% | 11.16 | 65.28% | 21.88% | 56.25% | 59.38% | 75.00% |
| Prefix-16 noise 0.05 | Test | 15.62% | 24.69 | 40.62% | 3.12% | 15.62% | 6.25% | 25.00% |

Greedy rollout64 result, 16 train / 16 test:

| Run | Split | First-face divergence | Boundary edges | Edge pairing | Generated token acc | Watertight edge graph |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Prefix-16 no-noise | Train | 87.50% | 18.25 | 0.598 | 0.181 | 12.50% |
| Prefix-16 no-noise | Test | 100.00% | 14.19 | 0.617 | 0.191 | 25.00% |
| Prefix-16 noise 0.05 | Train | 62.50% | 30.62 | 0.582 | 0.195 | 6.25% |
| Prefix-16 noise 0.05 | Test | 100.00% | 17.50 | 0.521 | 0.200 | 0.00% |

Interpretation:

```text
Noisy teacher input helps train first-face ranking and slightly improves generated token accuracy.
It does not improve held-out first-face exactness, held-out same-min behavior, edge pairing, or watertightness.
The train divergence drop is real, but it comes with worse boundary/edge-pairing topology.
Do not use 0.05 noisy-prefix as a default scale setting.
If used again, try smaller noise such as 0.01-0.02 or anneal it briefly, then consolidate with the normal full-sequence objective.
```
