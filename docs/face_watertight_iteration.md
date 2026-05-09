# FACE Watertightness Iteration Loop

Status update, 2026-05-05: this document is now a ClearMesh production-research
sidecar, not the primary FACE reproduction lane. The user preference is to
return FACE work to strict paper details and fill only unavailable details.
Use `docs/face_strict_paper_recommitment.md` and
`docs/face_paper_faithfulness_checklist.md` as the primary FACE direction.
Do not scale indexed/half-edge experiments as FACE evidence.

This is the fast CPU loop for deciding whether a FACE-style route is worth GPU
training. It is intentionally hostile: if a target cannot survive deterministic
tokenization and decode as a watertight edge graph, no amount of autoregressive
training should be trusted to fix it.

## Current Finding

The paper-style coordinate-token path is not the best production representation
for our watertight/editable target. Explicit indexed topology is materially
better before model training:

| Scan | Family | Bins | Raw watertight | Manifold-repaired watertight |
| --- | --- | ---: | ---: | ---: |
| Strict targets | paper | 128 | 110/156 | 130/156 |
| Strict targets | indexed | 128 | 146/156 | 149/156 |
| Strict targets | paper | 4096 | 145/156 | 150/156 |
| Strict targets | indexed | 4096 | 152/156 | 154/156 |

Two important details:

- Within-face paper ordering did not move watertightness in the scan. The
  bottleneck is graph identity/reuse, not triangle corner permutation.
- Indexed topology with conservative manifold repair reaches 154/156 at both
  512 and 4096 bins. If we also require zero quantization face loss, 4096 keeps
  149/156 while 512 keeps 143/156.

Interpretation: we should not scale the paper coordinate decoder blindly. The
highest-probability production path is FACE-indexed / half-edge constrained
decoding: learn vertices and connectivity explicitly, then enforce graph closure
during sampling.

## Sequence-Order Fix

The first indexed overfit target exposed a second mismatch: indexed faces were
trained in deterministic lexicographic order, while constrained inference grows
from open boundary edges. That means the model could be trained to jump to a new
frontier while the sampler forcibly rejects that jump.

`boundary_growth` indexed ordering fixes this by greedily shelling each connected
component from its boundary. On the 512-face seed target:

| Indexed order | Extra zero-closure jumps after first face |
| --- | ---: |
| `lex` | 16 |
| `boundary_growth` | 0 |

Local MPS overfit smokes after this change:

| Smoke | Samples | Max faces | Free-run watertight | Mean boundary edges | Mean nonmanifold edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| Box | 1 | 12 | 1/1 | 0.0 | 0.0 |
| Mixed primitives | 3 | 32 | 3/3 | 0.0 | 0.0 |
| Mixed + icosphere | 4 | 80 | 4/4 | 0.0 | 0.0 |

These are not production-quality proof yet, but they are a strong circuit-level
signal: with an aligned sequence and constrained decoding, FACE-indexed can
autoregressively produce closed meshes on small watertight targets.

## Real-Mesh Ladder Update

The first 50 real targets built from the strict indexed gate contained three
assets that were token-closed but not decoded-watertight after dequantization.
The stricter curated dataset is therefore:

```text
.codex_outputs/face_indexed_real50_boundary/dataset_4096_decoded_watertight
```

Current curation:

| Gate | Result |
| --- | ---: |
| Token watertight | 47/47 |
| Decoded watertight | 47/47 |
| Extra zero-closure jumps | 0 |
| Max faces | 512 |
| Max vertices | 258 |

A local MPS 8-mesh overfit on this stricter set reached 100% teacher-forced
token accuracy and 8/8 watertight teacher-forced exports after 2K steps. More
importantly, free-running AR passed on the first three real meshes when the
closure bonus was reduced:

| Decode | Closure bonus | Free-run watertight | Boundary edges | Nonmanifold edges |
| --- | ---: | ---: | ---: | ---: |
| `edge_constrained` | 5.0 | 0/1 | 50 | 50 |
| `edge_constrained` | 1.0 | 3/3 | 0.0 | 0.0 |
| `edge_constrained` | 1.0 | 8/8 | 0.0 | 0.0 |

Interpretation: the previous selector was too aggressive. A high closure bonus
can overpower the learned logits late in generation and choose a wrong but more
"closing" face. Keep `CLOSURE_BONUS=1.0` as the default until a sweep proves a
better value.

The raw Chamfer values in the first report looked alarming, but that was a
metric-scale issue: identical generated/teacher GLBs can show large raw Chamfer
when the object is large and the sampled point sets differ. The evaluator now
also reports `chamfer_l2_normalized` and `hausdorff_l2_normalized`, divided by
the reference bounding-box scale. On the 8-mesh local run the free-run GLBs are
vertex/face identical to the teacher GLBs, and normalized Chamfer is tiny:

| Gate | Result |
| --- | ---: |
| Free-run exact GLB topology | 8/8 |
| Mean normalized Chamfer | 0.000052 |
| Max normalized Chamfer | 0.000090 |

This was a green light for the next bounded ladder (`16 -> 32 -> 47` curated
meshes), not for a blind 130K+ production run. The later May 5 A100 ladder did
run and supersedes this local-only verdict: it is not scale-ready yet.

Thunder note: three Thunder instances have shown a GPU in the login banner but
no `/dev/nvidia*` device inside the container. The launch script now requires a
positive `CLEARMESH_GPU_PREFLIGHT_OK` marker after `nvidia-smi`; do not trust
the `tnr connect` exit code alone, because it can return success even when the
remote shell exits after the missing-device check. Do not run expensive setup on
an instance until that marker appears.

## A100 Seed/Edge-Policy Findings

The May 5 A100 probes changed the immediate plan. The constrained decoder can
make watertight outputs, but the free-running shapes are still too spiky/noisy
for production and the decode loop is far too slow. Do not scale to 130K+ meshes
or 100K steps from this branch yet.

Probe artifacts:

| Probe | Artifact root |
| --- | --- |
| Edge-action baseline | `.codex_outputs/face_indexed_edge_action_probe_a1/face_indexed_scale_20260505_edge_action_probe_a1_limit47_steps2500` |
| Edge-choice auxiliary | `.codex_outputs/face_indexed_edge_choice_probe_a1/face_indexed_scale_20260505_edge_choice_probe_a1_limit47_steps2500` |
| Early-face weighting | `.codex_outputs/face_indexed_edge_choice_early_probe_a1/face_indexed_scale_20260505_edge_choice_early_probe_a1_limit47_steps2500` |
| Seed + edge-choice | `.codex_outputs/face_indexed_seed_face_probe_a1/face_indexed_scale_20260505_seed_face_probe_a1_limit47_steps2500` |
| Seed + edge-action | `.codex_outputs/face_indexed_seed_edge_action_probe_a1/face_indexed_scale_20260505_seed_edge_action_probe_a1_limit47_steps2500` |

Key diagnostic:

| Checkpoint | First-face policy | Teacher token acc | Teacher face exact | Free-run watertight |
| --- | ---: | ---: | ---: | ---: |
| Edge-action baseline | corner head 31.9% | 0.9880 | 0.9661 | 12/12 |
| Edge-choice auxiliary | corner head 23.4% | 0.9652 | 0.9041 | 12/12 |
| Early-face weighting | corner head 97.9% | 0.9038 | 0.7472 | 6/6 |
| Seed + edge-choice | seed head 100% | 0.9609 | 0.8911 | 6/6 |
| Seed + edge-action | seed head 100% | 0.9837 | 0.9536 | 6/6 |

Interpretation:

- The seed-face head solved the first-face failure cleanly.
- Early-face weighting was the wrong fix: it fixed the opening face by harming
  later autoregressive learning.
- Edge-choice also appears to compete with the main decoder on this small data
  regime. Keep it off unless a later formulation improves teacher-forced exact.
- The best current research branch is seed-face + edge-action, not edge-choice.
- Even the best branch is not scale-ready: the scale gate still blocks on
  teacher-forced watertightness/exactness and free-run normal consistency.

Latest bounded A100 result, seed-face + edge-action:

| Metric | Result |
| --- | ---: |
| Free-run watertight | 6/6 |
| Free-run boundary edges | 0.0 |
| Free-run nonmanifold edges | 0.0 |
| Mean normalized Chamfer | 0.00372 |
| Median normal consistency | 0.651 |
| Mean decode time | 111.3 s/mesh |
| Teacher token accuracy | 0.9841 |
| Teacher face exact | 0.9539 |
| Scale gate | blocked |

Next research step should not be another blind scale run. The highest-value
tests are:

- Improve teacher-forced exactness without edge-choice, likely with a slightly
  longer seed+edge-action run or a less conflicting auxiliary loss.
- Add exposure-bias diagnostics: generate with a small beam over boundary edges
  and compare against greedy to see whether the learned logits contain a good
  path that the sampler misses.
- Profile/vectorize the boundary decoder. Current free-run decode is CPU-bound
  and around 100 seconds per 512-face mesh.
- Add geometry/normal losses or feature-aware projection only after the AR
  memorization gate approaches `>=0.995` token accuracy and `>=0.99` face exact.

## Scripts

- `scripts/research/face_token_oracle.py`
  runs representation/bin/order scans on token shards or source meshes.
- `scripts/research/select_face_oracle_passes.py`
  turns oracle rows into pass/fail manifests for the next experiment.
- `scripts/research/build_face_token_dataset.py --indexed-face-order boundary_growth`
  writes topology targets in constrained-decoder-friendly shelling order.
- `scripts/thunder/face_micro_overfit_ladder.sh`
  runs the one-sample GPU trust ladder. Do not launch large runs until this can
  autoregressively overfit one topology-safe target.
- `scripts/research/assess_face_indexed_scale_readiness.py`
  reads curation, teacher-forced, and free-run eval reports and blocks scaling
  when topology is closed but geometry fidelity is still weak.
- `scripts/research/refresh_eval_pair_metrics.py`
  recomputes generated-vs-teacher metrics from exported GLBs and adds
  scale-normalized Chamfer/Hausdorff to older eval reports.

## Reproduce The Current Strict-Target Scan

```bash
.venv/bin/python scripts/research/face_token_oracle.py \
  --mesh-dir .codex_outputs/a100_seed_data/extracted/clearmesh_face_objpp200_512_20260504_020301/strict_targets/meshes \
  --output-dir .codex_outputs/face_token_oracle/strict_target_mesh_scan \
  --families paper,indexed \
  --bins 64,128,256,512,1024 \
  --orders preserve,rotate_min_zyx,sort_zyx \
  --workers 6 \
  --max-faces 4096
```

```bash
.venv/bin/python scripts/research/face_token_oracle.py \
  --mesh-dir .codex_outputs/a100_seed_data/extracted/clearmesh_face_objpp200_512_20260504_020301/strict_targets/meshes \
  --output-dir .codex_outputs/face_token_oracle/strict_target_high_bins_scan \
  --families paper,indexed \
  --bins 1024,2048,4096 \
  --orders preserve \
  --workers 6 \
  --max-faces 4096
```

## Select Training-Safe Targets

Indexed 4096, allowing conservative manifold repair:

```bash
.venv/bin/python scripts/research/select_face_oracle_passes.py \
  --oracle-rows .codex_outputs/face_token_oracle/strict_target_high_bins_scan/oracle_rows.jsonl \
  --output-dir .codex_outputs/face_token_oracle/strict_target_high_bins_indexed4096_gate \
  --families indexed \
  --bins 4096 \
  --orders indexed \
  --repair-mode manifold \
  --max-boundary-edges 0 \
  --max-nonmanifold-edges 0 \
  --min-edge-pairing-ratio 1.0 \
  --min-pass-rate 0.98
```

Indexed 4096, strict zero face-loss target gate:

```bash
.venv/bin/python scripts/research/select_face_oracle_passes.py \
  --oracle-rows .codex_outputs/face_token_oracle/strict_target_high_bins_scan/oracle_rows.jsonl \
  --output-dir .codex_outputs/face_token_oracle/strict_target_high_bins_indexed4096_zero_loss_gate \
  --families indexed \
  --bins 4096 \
  --orders indexed \
  --repair-mode manifold \
  --max-quantization-face-loss 0 \
  --max-boundary-edges 0 \
  --max-nonmanifold-edges 0 \
  --min-edge-pairing-ratio 1.0
```

The selector writes:

- `selection_summary.json`
- `pass_manifest.jsonl`
- `fail_manifest.jsonl`
- optional `assets/` when `--copy-mode copy` or `--copy-mode symlink` is used

## GPU Ladder Rule

The next GPU run should be a one-sample indexed topology overfit, not a large
paper-coordinate run.

Use `--indexed-face-order boundary_growth` for the dataset. A lexicographic
indexed target is useful for ablations, but it is the wrong default for the
constrained watertight route.

Success criteria:

- Teacher-forced token accuracy approaches memorization on one selected target.
- Free-running AR reproduces a stable mesh rather than fragments.
- Output passes zero boundary edges and zero non-manifold edges.
- If constrained sampling is enabled, it improves closure without collapsing
  diversity into repeated faces.
- Sweep closure bonuses conservatively. Current best local real-mesh setting is
  `--closure-bonus 1.0`; `5.0` is too strong.

Only after that should we scale to a curated multi-sample run.

Machine gate for the current local 8-mesh run:

```bash
python3 scripts/research/assess_face_indexed_scale_readiness.py \
  --curation-summary .codex_outputs/face_indexed_real50_boundary/dataset_4096_decoded_watertight/curation_summary.json \
  --teacher-eval .codex_outputs/face_indexed_real50_boundary/local_real8_curated_teacher_2k/eval_teacher_forced_normalized.json \
  --free-run-eval .codex_outputs/face_indexed_real50_boundary/local_real8_curated_teacher_2k/eval_free_run_eight_bonus1_normalized.json \
  --output .codex_outputs/face_indexed_real50_boundary/local_real8_curated_teacher_2k/scale_readiness_normalized.json
```

Historical local normalized verdict:

```text
topology_ready: true
geometry_ready: true
scale_ready: true
recommendation: promote to the next larger curated run
```

This verdict only justified the A100 ladder. The current A100 verdict above is
`scale_ready: false`; use that result for production scaling decisions.

## Production Direction

The production-grade topology route should be:

```text
TRELLIS.2 visual proxy
  -> manifoldized / normalized source target
  -> part or chart segmentation
  -> explicit vertex-table + indexed/half-edge topology prior
  -> constrained autoregressive face/patch decoding
  -> deterministic repair + feature-aware projection
  -> Blender gate promotion
```

The moat is not just image-to-3D quality. It is the closed-loop validator:
representation choice, target curation, repair, constrained decoding, projection,
and Blender promotion all measured before we spend large GPU time.

## 2026-05-05 A100 Topology Update

A bounded A100 confirmation on the 47 curated real targets completed and the
Thunder instance was deleted after artifact fetch.

Artifacts:

```text
.codex_outputs/face_indexed_topocausal47_eval_euler_fill_a1/face_indexed_eval_20260505_topocausal47_eval_euler_fill_a1_limit47/
.codex_outputs/face_indexed_topocausal47_eval_euler_fill_a1/local_unpinched/
.codex_outputs/face_indexed_topocausal47_eval_euler_fill_a1/local_unpinched_contact_sheet.png
```

Free-run with centroid boundary fill plus degenerate-face pre-drop plus
Eulerian boundary cycle decomposition:

```text
attempted: 47
watertight: 47 / 47
mean_boundary_edges: 0.0
mean_nonmanifold_edges: 0.0
mean_edge_pairing_ratio: 1.0
mean_chamfer_l2_normalized: 0.00350
median_normal_consistency: 0.67298
median_decode_sec: 6.29 on A100
```

Important correction: edge-watertight is not enough. A stricter vertex-link
manifold audit found bow-tie/pinched vertices in all 47 free-run meshes. The
harness now counts true nonmanifold vertices by checking each vertex one-ring
link, not just vertices touched by nonmanifold edges.

A deterministic unpinch cleanup was added. It duplicates only vertices whose
incident faces form multiple disconnected fans. This preserves geometry while
undoing over-welded bow-tie links.

Local postprocess on the 47 A100 free-run meshes:

```text
watertight: 47 / 47
mean_boundary_edges: 0.0
mean_nonmanifold_edges: 0.0
mean_nonmanifold_vertices: 0.0
mean_split_vertices_added: 87.7
mean_normal_consistency: ~0.686
```

What this means:

- The topology/closure repair path can now produce edge-watertight meshes and
  remove true pinched vertices without changing geometry.
- This is a valid fast production cleanup primitive for over-welded generator
  output.
- It does not solve free-run FACE geometry quality. The current tiny indexed
  decoder still creates too many long cross-object edges and visual blobs.
- A vertex-link constraint during decoding was tested locally and reduced some
  pinches, but it broke closure, so it is not the default path.
- A geometry/local-neighbor candidate prior was tested locally; it preserved
  watertightness after unpinch but did not materially fix visual quality and was
  much slower on CPU.

Do not scale this FACE-lite checkpoint to a 100k-step production run yet. The
next real research fix should target exposure bias and geometry-faithful free-run
connectivity, not just longer training.

Near-term production use:

```text
TRELLIS.2 / UltraShape / proxy mesh
  -> cleanup with split_nonmanifold_vertices=true
  -> feature-aware projection / chart quad sidecar
  -> Blender/editability gate
```

Near-term FACE research use:

```text
curated vertex-table targets
  -> train an edge-action / boundary-completion decoder
  -> local candidate set supervised by target boundary-growth order
  -> scheduled-sampling or beam/repair-aware training
  -> scale only after free-run visual geometry improves, not before
```

## 2026-05-05 Edge-Action Decoder Scaffold

After the unpinch pass, the remaining blocker is visual/geometry quality: the
free-run decoder still selects too many long cross-object triangles. A local
candidate-only prior was not enough because the current model was never trained
to solve boundary completion as an action.

New scaffold added:

```text
hidden AR face context + open boundary edge (a, b)
  -> edge_action_mlp
  -> logits over candidate third vertex indices
```

Training now supports:

```bash
python scripts/research/train_face_indexed_conditioned_tiny.py \
  --edge-action-loss-weight 0.5 \
  --corner-head causal \
  --topology-loss-weight 0.1 \
  ...
```

Evaluation now supports:

```bash
python scripts/research/eval_face_indexed_conditioned_tiny.py \
  --edge-action-bonus 0.1 \
  --local-candidate-neighbors 24 \
  --split-pinched-vertices \
  ...
```

Boundary-action oracle on the 47 curated targets:

```text
total_edge_actions: 21,185
median local rank of true third vertex: 3
p95 local rank: 39
local-24 coverage: 91.47%
local-48 coverage: 96.30%
local-64 coverage: 97.38%
```

This supports the next serious training run: train an edge-action/boundary
completion head, then decode with local candidates plus a global fallback. This
is a better scaling candidate than continuing the current corner-only decoder.

A CPU smoke verified the new edge-action training/eval path:

```text
steps: 8
samples: 2
edge_action_loss logged and decreasing slightly
1-sample eval loaded the checkpoint and produced watertight + unpinched output
```

Production gate was also tightened: `production_max_nonmanifold_vertices`
defaults to `0`, so pinched watertight meshes no longer promote. Cleanup now
splits nonmanifold vertices by default unless metadata explicitly disables it
with `cleanup_split_nonmanifold_vertices=false`.

## 2026-05-05 Cap-192 Seed Curriculum Probe

A larger A100 probe was run with the seed-face head, edge-action auxiliary,
cap-192 hidden size, and a seed-face curriculum that disables the seed loss
after step 1000:

```text
artifact:
.codex_outputs/face_indexed_seed_edge_action_curriculum_cap192_a1/
  face_indexed_scale_20260505_seed_edge_action_curriculum_cap192_a1_limit47_steps6000
```

Training did memorize the tiny curated set:

| Metric | Result |
| --- | ---: |
| Best loss | 0.000786 |
| Best step | 5965 |
| Teacher-forced watertight | 6/6 |
| Teacher-forced token accuracy | 1.0 |
| Teacher-forced face exact | 1.0 |
| Teacher-forced median normal consistency | 0.923 |

Free-run topology also passed, but visual geometry still did not:

| Metric | Result |
| --- | ---: |
| Free-run watertight | 6/6 |
| Free-run boundary edges | 0.0 |
| Free-run nonmanifold edges | 0.0 |
| Free-run nonmanifold vertices | 0.0 |
| Free-run mean normalized Chamfer | 0.00403 |
| Free-run p95 normalized Chamfer | 0.01145 |
| Free-run median normal consistency | 0.624 |
| Mean decode time | 231.5 s/mesh |
| Scale gate | blocked |

Interpretation: this is an important positive signal, but not production-ready.
The model can learn the target sequence under teacher forcing, and the
constrained decoder can keep meshes closed. The remaining failure is
free-running connectivity quality and decode time.

The first contact sheet makes the issue obvious: several outputs are closed but
over-dense/spiky shells. Do not promote this branch to 130K+ meshes or 100K
steps until free-run normal consistency and visuals improve.

## 2026-05-05 Free-Run Divergence Diagnostic

A new diagnostic was added:

```bash
python scripts/research/diagnose_face_indexed_free_run.py \
  --checkpoint <run>/face_indexed_v2_tiny.pt \
  --dataset-dir <run>/uploaded/dataset_4096_decoded_watertight \
  --output <run>/free_run_divergence_diagnostic.json
```

The diagnostic stops at the first topology mismatch and reports whether the
teacher next face was present in the constrained candidate set.

The heavy heuristic decode profile used in the A100 eval diverged extremely
early:

| Profile | All-prefix matched | Mean matched faces | Median matched faces |
| --- | ---: | ---: | ---: |
| Heavy closure/edge-length/topology bonuses | 0/6 | 4.8 | 4.0 |

Removing those bonuses and relying mostly on model score improved prefix
matching sharply:

| Profile | Candidate settings | All-prefix matched | Mean matched faces | Median matched faces |
| --- | --- | ---: | ---: | ---: |
| Model-only-ish | top_k=16, local=24 | 1/6 | 176.8 | 89.5 |
| Model-wide | top_k=32, local=96 | 3/6 | 375.2 | 393.5 |

This is the current smoking gun:

- The model is much better than the heavy-heuristic render implied.
- The old sampler was over-steering the model into wrong but "closed-looking"
  faces.
- After removing heuristic pressure, the main remaining failures are candidate
  pruning misses: the teacher face often is not in the top-k/local candidate
  set.
- A wider candidate set recovers much more of the teacher path, but it is too
  slow to use naively in production.

Next bounded fix:

```text
Train/search objective:
  make boundary-edge third-vertex ranking accurate enough that the true third
  vertex appears in a small candidate set.

Decode objective:
  use model-dominant scoring with only hard manifold constraints, not large
  closure/length/topology bonuses.

Runtime objective:
  vectorize/cache candidate scoring and replace full-sequence Transformer
  recompute with an incremental or chunked causal decode path.
```

Promotion rule remains unchanged: topology-only success is not enough. Promote
only after free-run visuals and normal consistency improve with a decode profile
that can plausibly run in production.

The model-wide profile was then run through the full free-run evaluator. This
is the first FACE-indexed probe that passes the current scale-readiness gate on
the 6-sample tiny curated set:

| Metric | Heavy heuristic profile | Model-wide profile |
| --- | ---: | ---: |
| Watertight | 6/6 | 6/6 |
| Mean normalized Chamfer | 0.00403 | 0.00069 |
| Median normalized Chamfer | 0.00320 | 0.00062 |
| Median normal consistency | 0.624 | 0.903 |
| Mean decode time | 231.5 s/mesh | 155.5 s/mesh |
| Max decode time | 321.0 s/mesh | 381.8 s/mesh |
| Scale gate | blocked | promoted on tiny-set gate |

Artifacts:

```text
<run>/model_wide_eval/eval_report.json
<run>/model_wide_eval/contact_sheet.png
<run>/model_wide_eval/scale_readiness.json
```

Important caveat: this is still an overfit/tiny-set result. It is a green light
for the next bounded run, not proof of generalization. The immediate next check
should be a held-out or larger curated dataset, ideally at least 100-150 strict
targets before committing to a 130K+ production run.

## 2026-05-05 Edge-Action Proposer Decode

The next diagnostic tested whether the edge-action auxiliary head should be
used as a candidate proposer rather than as a score bonus. This matters because
large edge/closure bonuses were overpowering the FACE model score and producing
closed but spiky meshes.

New decoder knob:

```bash
--edge-action-candidate-top-k 16
```

This computes the edge-action head and adds its top third-vertex predictions to
the candidate set, while keeping `--edge-action-bonus 0.0` so it does not
directly change the score.

Prefix diagnostic on the tiny cap-192 checkpoint:

| Profile | Candidate settings | All-prefix matched | Mean matched faces | Median matched faces |
| --- | --- | ---: | ---: | ---: |
| Model-only-ish | top_k=16, local=24 | 1/6 | 176.8 | 89.5 |
| Model-wide | top_k=32, local=96 | 3/6 | 375.2 | 393.5 |
| Edge-action proposer | top_k=16, local=24, edge-action candidate top_k=16 | 4/6 | 435.0 | 474.0 |

Full tiny-set free-run eval with the edge-action proposer:

| Metric | Result |
| --- | ---: |
| Watertight | 6/6 |
| Mean normalized Chamfer | 0.000585 |
| Median normal consistency | 0.923 |
| Mean decode time | 64.98 s/mesh |
| Max decode time | 102.38 s/mesh |
| Scale gate | promoted on tiny-set gate |

Artifacts:

```text
.codex_outputs/face_indexed_seed_edge_action_curriculum_cap192_a1/
  face_indexed_scale_20260505_seed_edge_action_curriculum_cap192_a1_limit47_steps6000/
    edgeaction_proposer_eval/eval_report.json
    edgeaction_proposer_eval/contact_sheet.png
    edgeaction_proposer_eval/scale_readiness.json
    free_run_divergence_edgeaction_proposer_diagnostic.json
```

Interpretation: this is the best decode profile so far. It keeps the
model-dominant scoring that fixed the spiky over-closure failure, recovers more
teacher-prefix paths than model-wide, and cuts tiny-set mean decode time by
about 58% versus model-wide.

## 2026-05-05 Held-Out 144 Probe

A 144-shard strict curated dataset was created from the larger strict target
pool. The first held-out A100 probe trained on the first 120 shards and evaluated
12 held-out shards starting at offset 120:

```text
dataset:
.codex_outputs/face_indexed_strict156_boundary/
  dataset_4096_decoded_watertight.tar.gz

split:
train first 120
eval offset 120, limit 12
```

The model-wide held-out eval was watertight but not production-ready:

| Metric | Result |
| --- | ---: |
| Free-run watertight | 12/12 |
| Free-run mean normalized Chamfer | 0.00410 |
| Free-run median normalized Chamfer | 0.00278 |
| Free-run p95 normalized Chamfer | 0.00933 |
| Free-run median normal consistency | 0.734 |
| Free-run mean decode time | 199.85 s/mesh |
| Free-run median decode time | 283.60 s/mesh |
| Free-run max decode time | 327.94 s/mesh |

The old readiness gate incorrectly treated held-out teacher-forced exact token
accuracy as a memorization blocker. That is valid for train-overfit gates but
not for held-out generalization gates, where exact face order is not the product
objective. The gate now supports:

```bash
--teacher-gate-mode memorization     # strict train/tiny overfit gate
--teacher-gate-mode generalization   # held-out gate; teacher-forced is diagnostic
```

The Thunder scripts default to `generalization` when `EVAL_OFFSET > 0`, and
`memorization` otherwise.

Also fixed: the Thunder smoke script's post-run completion check now uses a
plain sentinel `grep` instead of a nested Python heredoc. The previous heredoc
could fail after a successful remote run and prevent artifact download.

Current verdict:

- Held-out topology closure is promising: 12/12 watertight with zero boundary
  and nonmanifold edges.
- Held-out geometry is not yet strong enough: median normal consistency was
  just below the `0.75` gate.
- Runtime is not acceptable in model-wide mode.
- The next held-out probe must use the edge-action proposer profile and fetch
  visuals before any larger scale-up.

## 2026-05-05 Held-Out Edge-Action Proposer Probe

The fixed Thunder launcher successfully ran and fetched the held-out edge-action
proposer probe:

```text
.codex_outputs/face_indexed_heldout144_edgeprop_a3/
  face_indexed_scale_20260505_face_indexed_heldout144_edgeprop_a3_limit120_steps10000
```

Training reproduced the previous curve exactly:

| Step | Loss | Token loss | Edge-action loss |
| ---: | ---: | ---: | ---: |
| 2000 | 0.6249 | 0.4443 | 0.1586 |
| 4000 | 0.1243 | 0.1061 | 0.0275 |
| 6000 | 0.0119 | 0.00966 | 0.00331 |
| 8000 | 0.00311 | 0.00251 | 0.000923 |
| 10000 | 0.000946 | 0.000754 | 0.000323 |

Held-out free-run metrics improved runtime and slightly improved Chamfer versus
model-wide, but visuals still failed:

| Metric | Model-wide held-out | Edge-action proposer held-out |
| --- | ---: | ---: |
| Watertight after repair | 12/12 | 12/12 |
| Mean normalized Chamfer | 0.00410 | 0.00367 |
| Median normalized Chamfer | 0.00278 | 0.00235 |
| Median normal consistency | 0.734 | 0.734 |
| Mean decode time | 199.85 s/mesh | 64.60 s/mesh |
| Median decode time | 283.60 s/mesh | 92.73 s/mesh |
| Max decode time | 327.94 s/mesh | 105.35 s/mesh |

The contact sheet shows many closed but collapsed/spiky outputs. The metrics are
therefore too forgiving unless they inspect repair burden.

The readiness gate was tightened again to treat boundary-fill dependence as a
topology blocker. It now reads `boundary_fill_report` from each eval item and
requires:

```text
mean raw boundary edges requiring fill <= 0
mean boundary-fill face ratio <= 0
```

On this held-out run, the stricter raw-fill gate reports:

| Raw-fill metric | Result |
| --- | ---: |
| Mean raw boundary edges before fill | 302.7 |
| Median raw boundary edges before fill | 440.0 |
| Max raw boundary edges before fill | 472.0 |
| Mean boundary-fill face ratio | 0.686 |
| Median boundary-fill face ratio | 0.859 |

Updated verdict:

```text
Do not scale.
Topology is not actually green.
The decoder is relying on hole filling to create watertightness.
```

An oracle first-face sanity check was also run on the first four held-out
samples:

```text
<run>/oracle_seed1_eval_limit4
```

It did not fix the problem. Even with the correct first face supplied, the
512-face held-out samples still needed 444-470 raw boundary edges filled.

Interpretation:

- The failure is not only first-face/seed selection.
- The continuation policy does not yet generalize boundary-growth connectivity.
- The next useful experiment is not bigger production training yet. It is a
  bounded architecture/data fix that makes the decoder produce mostly closed
  raw topology before repair.

## 2026-05-05 Edge-Choice Probe Setup

The next bounded test trains the already-present boundary edge-choice head
instead of only training the third-vertex edge-action head:

```text
EDGE_CHOICE_LOSS_WEIGHT=0.5
EDGE_CHOICE_CANDIDATES=64
EDGE_CHOICE_BONUS=1.0
EDGE_ACTION_LOSS_WEIGHT=0.5
EDGE_ACTION_CANDIDATE_TOP_K=16
```

Why this matters: the prior held-out run trained the model to choose a third
vertex for a boundary edge, but the decode still had to choose which boundary
edge to extend through indirect corner logits. If this was the main topology
failure, the new run should reduce raw boundary edges before centroid fill, not
just improve post-repair watertightness.

I also added a no-retrain decode ablation knob:

```text
--edge-choice-candidate-top-k
EDGE_CHOICE_CANDIDATE_TOP_K
```

This mirrors the edge-action proposer. It lets the edge-choice head propose
candidate boundary edges even when its score bonus is weak or being swept. That
is deliberately conservative: it changes decode candidate recall, not the model
weights, so a fetched checkpoint can be re-evaluated immediately if the first
edge-choice run looks close but still overfills holes.

### A4 Result: Edge Choice Helps Geometry, Not Raw Closure

The first held-out edge-choice run completed, but the launcher exited before
fetching artifacts because `tnr connect` returned nonzero after the remote
completion sentinel. The streamed metrics were preserved in `local_launch.log`.

Key free-run metrics:

| Metric | A3 edge-action proposer | A4 edge-choice trained |
| --- | ---: | ---: |
| Watertight after centroid fill | 12/12 | 12/12 |
| Mean normalized Chamfer | 0.00367 | 0.00357 |
| Median normalized Chamfer | 0.00235 | 0.00252 |
| Median normal consistency | 0.734 | 0.806 |
| Mean decode time | 64.6 s | 40.8 s |
| Median decode time | 92.7 s | 58.3 s |
| Mean raw fill edges | 302.7 | 248.8 |
| Mean fill face ratio | 0.686 | 0.514 |

Interpretation:

- Edge choice is a real improvement: faster decode, better median normals, and
  lower repair burden.
- It is still not enough for scale-up: the raw decoder still leaves roughly 249
  boundary edges per held-out mesh before centroid fill.
- The readiness gate correctly blocks promotion even though post-fill
  watertightness and geometry metrics look acceptable.

Launcher fix:

```text
tnr connect may exit nonzero after remote logout.
If the completion sentinel is present, continue to fetch artifacts anyway.
```

The next corrected probe is `A5b`, which repeats the same training setup on the
same 144-shard dataset and enables `EDGE_CHOICE_CANDIDATE_TOP_K=64` during
decode so the edge-choice head can increase boundary-edge candidate recall.

### Geometry-Aware Edge Head Fix

A hostile code audit found an important architecture mismatch in the previous
edge-action and edge-choice heads: they were conditioned on learned vertex index
embeddings, not on the actual quantized vertex-table coordinates. That means
the boundary heads had to learn "complete this open edge" mostly from arbitrary
index IDs plus a pooled shape latent. This is a weak generalization signal for
held-out meshes and is consistent with the observed failure mode: watertight
after centroid fill, but hundreds of raw boundary edges before fill and many
long cross-object triangles.

The indexed decoder now supports:

```text
--edge-head-mode geometry
```

In geometry mode:

- edge-action uses a pointer-style score over vertex-table geometry:
  `hidden + vertex_hidden(edge_a) + vertex_hidden(edge_b) -> query`, then scores
  every candidate vertex against coordinate-aware vertex keys.
- edge-choice scores candidate boundary edges from
  `hidden + vertex_hidden(edge_a) + vertex_hidden(edge_b)`.
- invalid vertices and the two edge endpoints are hard-masked in the action
  logits.
- old checkpoints still load in `index` mode because eval reads
  `edge_head_mode` from checkpoint args and falls back to `index` when absent.

Smoke result:

```text
artifact: .codex_outputs/face_geometry_edge_head_smoke_small
dataset: 3 tiny synthetic boundary-growth meshes
training: 12 steps, geometry edge-action + edge-choice losses logged
eval: 1 free-run synthetic box, watertight after centroid fill, decode path OK
```

This is not quality proof. It is a stronger next architecture candidate for the
held-out raw-closure problem than simply increasing edge-choice candidate count
with the old index-only boundary heads.

Thunder launcher update: A100 and A6000 instances repeatedly reached Thunder
`RUNNING` while SSH stayed unavailable for Thunder's internal 2-minute connect
timeout. The launcher now retries GPU preflight with:

```text
PREFLIGHT_ATTEMPTS
PREFLIGHT_RETRY_SEC
```

so a slow-to-open SSH service is not deleted after one premature attempt. Broken
instances are still deleted if the full retry budget fails.
