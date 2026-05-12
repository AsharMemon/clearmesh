# FACE 10k/100k Full Audit - 2026-05-11

## Scope

This audit covers the completed ClearMesh FACE strict-lane run:

- Thunder instance: `tnr-0`, A100XL, name `1pt9wvts`
- Remote root: `/tmp/clearmesh_face_paper_corpus_gate_20260510_rotate10k_100k_preserve_a100_v2`
- Local lightweight artifacts: `.codex_outputs/face_10k_100k_audit_20260511`
- Run dir: `.codex_outputs/face_10k_100k_audit_20260511/runs/paper_corpus_gate_128_vec2048_muon_aug`
- Visual sheets:
  - `.codex_outputs/face_10k_100k_audit_20260511/train_ar_contact_sheet.png`
  - `.codex_outputs/face_10k_100k_audit_20260511/test_ar_contact_sheet.png`
- Supporting tables:
  - `.codex_outputs/face_10k_100k_audit_20260511/audit_tables.md`
  - `.codex_outputs/face_10k_100k_audit_20260511/audit_tables.json`
  - `.codex_outputs/face_10k_100k_audit_20260511/ar_failure_analysis.json`

The audit answers three questions:

1. What exactly failed?
2. Are the results a positive enough signal to scale?
3. Are our conclusions too strict or unsupported?

## Executive Verdict

This run is a real training improvement, but not a production-scale green light.

The model learned a meaningful teacher-forced coordinate reconstruction distribution, and selection loss improved strongly from `1.9478` at 10k to best `0.7508` at 90k. However, full autoregressive generation is still not stable. The dominant failure is not a single bad mesh or a gallery bug. It is systematic first-face and early-token uncertainty, followed by exposure-bias drift and topology breakage.

The strongest evidence:

- First-face exact prediction is `0 / 32` on train AR samples and `0 / 32` on test AR samples.
- First-face exact prediction is also `0 / 128` on train teacher-forced eval and `0 / 128` on test teacher-forced eval.
- Full AR diverges at face `0` for every train and test sample evaluated.
- Teacher-forced token accuracy is only about `78.6-78.9%`, not the `95%+` needed for exact welded topology.
- Teacher-forced geometry is close by Chamfer, but topology is broken: mean teacher-forced boundary edges are `~907` train and `~806` test.
- Test AR watertight rate is `7 / 32`, but visual inspection shows these are mostly closed wrong sequences, not faithful reconstructions.

The uncomfortable but useful conclusion: `rotate_min_zyx` was a real fix for the loss curve, but it did not solve the central AR/topology problem.

## Run Configuration Audit

The run used the current strict-lane settings we intended:

- `PAPER_WITHIN_FACE_ORDER=rotate_min_zyx`
- `FACE_EMBEDDING_VARIANT=token_concat_project`
- `CAUSAL_MLP_VARIANT=legacy_concat`
- online augmentation enabled
- `CACHE_FPS_INDICES=0`
- 128 coordinate bins
- 8192 point samples
- 2048 VecSet tokens
- latent bottleneck dimension 64
- Shape2VecSet-style encoder
- causal decoder with cross-attention
- Muon optimizer, bf16
- 100k steps
- selection eval every 10k
- strict token-hash deduplication and leakage check

The run is not full paper-scale capacity:

- hidden size: 768, not the stricter paper-scale gate target of 1024+
- encoder layers: 6, not 8+
- decoder layers: 12, not 24+
- heads: 12, not 16+
- usable train split: 4221 samples, not 130k curated samples

This means the run is a useful bounded scale signal, not a final FACE reproduction claim.

## Corpus/Target Integrity

The strict targets are not the immediate problem.

Evidence:

- The strict gate produced watertight target token graphs for evaluated rows.
- In per-sample AR results, teacher token topology is valid: `teacher_token_boundary_edge_count = 0`, `teacher_token_edge_pairing_ratio = 1.0`, `teacher_token_watertight_edge_graph = true`.
- Train/test token-hash leakage was checked before training.

So the failure is not that the evaluation target meshes are open. The failure is that predicted coordinate tokens do not reproduce the exact shared quantized vertices needed to weld edges.

## Training Curve

Selection loss improved strongly:

| Step | Selection Loss | Best Loss | Best Step |
|---:|---:|---:|---:|
| 10000 | 1.9478 | 1.9478 | 10000 |
| 20000 | 1.1231 | 1.1231 | 20000 |
| 30000 | 0.9500 | 0.9500 | 30000 |
| 40000 | 0.8760 | 0.8760 | 40000 |
| 50000 | 0.8213 | 0.8213 | 50000 |
| 60000 | 0.7940 | 0.7940 | 60000 |
| 70000 | 0.7853 | 0.7853 | 70000 |
| 80000 | 0.7552 | 0.7552 | 80000 |
| 90000 | 0.7508 | 0.7508 | 90000 |
| 100000 | 0.7674 | 0.7508 | 90000 |

Interpretation:

- The optimizer/corpus path is alive.
- The model is still learning through 80k-90k.
- The final 100k point regressed slightly, so the best checkpoint is 90k.
- A lower loss alone is not enough for production because coordinate-token CE does not guarantee exact face sequence or edge welding.

A rough intuition: a cross-entropy of `0.75` nats still implies substantial uncertainty. Even if per-token top-1 accuracy is around `0.79`, an exact nine-coordinate face is much harder, and a 512-face exact ordered sequence is much harder again.

## Compact Result Summary

| Eval | Attempted | Watertight | First-Face Divergence | TF Accuracy | AR/Generated Accuracy | Mean Boundary | Edge Pair | Median Chamfer | Median Normal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train teacher-forced | 128 | 0 | 128 | 0.789 | 0.789 | 907.0 | 0.220 | 0.0016 | 0.831 |
| test teacher-forced | 128 | 0 | 128 | 0.786 | 0.786 | 806.4 | 0.222 | 0.0016 | 0.840 |
| train AR | 32 | 1 | 32 | 0.794 | 0.059 | 40.2 | 0.473 | 0.0754 | 0.757 |
| test AR | 32 | 7 | 32 | 0.790 | 0.087 | 16.9 | 0.586 | 0.1762 | 0.736 |
| train AR prefix16 | 16 | 1 | 0 | 0.793 | 0.175 | 52.6 | 0.673 | 0.0206 | 0.762 |
| test AR prefix16 | 16 | 3 | 0 | 0.785 | 0.267 | 57.5 | 0.669 | 0.0213 | 0.788 |
| test predicted-count AR | 8 | 2 | 8 | 0.775 | 0.064 | 20.9 | 0.674 | 0.1736 | 0.724 |

## Failure Mode 1 - Teacher-Forced Underfit

The model has not yet solved the supervised reconstruction task.

Evidence:

- Train teacher-forced accuracy: `0.7886`
- Test teacher-forced accuracy: `0.7863`
- Train teacher-forced loss: `0.7886`
- Test teacher-forced loss: `0.8035`
- Readiness threshold was `>=0.95` accuracy and `<=0.3` loss for train teacher-forced closure.

Why this matters:

FACE-style coordinate reconstruction has a brutal exactness requirement. Each face has 9 coordinate tokens. A small number of wrong coordinate bins can still keep Chamfer low, but it destroys exact shared vertices and therefore breaks topology.

This explains why teacher-forced Chamfer is excellent while watertightness is zero.

## Failure Mode 2 - First-Face Collapse

Full AR generation fails before exposure bias even has a chance to accumulate.

Evidence:

- Train AR first-face divergence: `32 / 32`
- Test AR first-face divergence: `32 / 32`
- Train teacher-forced first-face exact: `0 / 128`
- Test teacher-forced first-face exact: `0 / 128`
- Train AR first-face exact: `0 / 32`
- Test AR first-face exact: `0 / 32`
- First divergent slot is usually one of `z0`, `y0`, or `x0`.

First-face teacher metrics:

- Train first-face slot top-1 accuracy: about `24.7%`
- Test first-face slot top-1 accuracy: about `25.7%`
- Mean first-face target probability: about `0.16-0.17`
- Mean first-face target rank: about `9-10`

Interpretation:

The first generated face is the hardest conditional decision. At position 0 the decoder sees only BOS, positional embedding, and VecSet. It does not yet have a previous face to anchor the mesh sequence. Because the target order starts from a canonical geometric extreme/minimum face, many plausible first faces can be close in the point cloud. Greedy decoding chooses one plausible face, but not the exact target face, so every subsequent exact-token metric collapses.

This is not just a visual artifact. It is visible directly in logits/ranks.

## Failure Mode 3 - Coordinate-Close But Topologically Broken

Teacher-forced outputs are near the surface but fail edge welding.

Evidence:

- Test teacher-forced median normalized Chamfer: `0.00155`
- Test teacher-forced median normal consistency: `0.840`
- Test teacher-forced watertight rate: `0 / 128`
- Test teacher-forced mean boundary edges: `806.4`
- Test teacher-forced mean edge-pairing ratio: `0.222`

Interpretation:

Chamfer is forgiving. Topology is not. If two adjacent faces should share a quantized vertex, all three coordinate bins must match exactly. A one-bin error can create a duplicate nearby vertex instead of a shared vertex. That keeps the surface visually close but opens the edge graph.

This is the core reason FACE-like coordinate tokens can look promising by geometry metrics while failing our production editability gates.

## Failure Mode 4 - Closed Wrong Meshes

Some AR outputs are watertight, but they are not faithful reconstructions.

Evidence:

- Test AR watertight: `7 / 32`
- Those watertight rows still diverged at face 0.
- Many watertight rows have low generated token accuracy, for example `0.003-0.213`.
- Visual sheets show closed primitive-like blobs, shards, folded planes, or degenerate slivers rather than matching teacher meshes.

Interpretation:

Watertightness alone is not enough. The decoder can sometimes emit a self-consistent closed token graph, but it is not conditioned tightly enough on the target shape. This is why the audit separates `topology validity` from `target reconstruction`.

## Failure Mode 5 - Teacher Prefix Helps But Does Not Rescue

Giving the model correct early faces improves token accuracy but does not close the gap.

Evidence:

- Train AR base generated accuracy: `0.059`
- Train prefix16 generated accuracy: `0.175`
- Test AR base generated accuracy: `0.087`
- Test prefix16 generated accuracy: `0.267`
- Test prefix16 watertight rate: `3 / 16`, not better than full test AR in a meaningful way
- Prefix removes first-face divergence by construction, but later rollout still drifts.

Interpretation:

The first face is a major failure, but not the only failure. Even with 16 teacher-provided faces, the model cannot reliably continue the target sequence. That points to both supervised underfit and exposure-bias/conditional-distribution weakness.

## Failure Mode 6 - Predicted Count Is Not Product-Ready

Product inference cannot depend on ground-truth face counts.

Evidence:

- Test predicted-count mean predicted/reference face ratio: `5.45x`
- Test predicted-count AR token accuracy: `0.064`
- EOS did not reliably terminate at the right count.

Likely causes:

- EOS loss weight is low: `0.05`.
- Training/eval still leans heavily on ground-truth face-count reconstruction for the primary gate.
- The model has not learned global sequence termination under free-run conditions.

## Visual Audit

The contact sheets are decisive. They show that the generated AR meshes are not close enough for a production interpretation.

- Train AR sheet: `.codex_outputs/face_10k_100k_audit_20260511/train_ar_contact_sheet.png`
- Test AR sheet: `.codex_outputs/face_10k_100k_audit_20260511/test_ar_contact_sheet.png`

Visual patterns:

- generated outputs are often tiny slivers or folded triangles
- some outputs are closed but unrelated primitive-like shapes
- detailed source shapes frequently collapse into simple sheets/blobs
- one generated GLB had no mesh geometry after degenerate-face cleanup

I patched `scripts/eval/make_mesh_contact_sheet.py` so future contact sheets show a placeholder for invalid generated meshes instead of aborting the whole gallery.

## Code Path Audit

### Things that look correct

- Strict run uses `rotate_min_zyx` target ordering.
- Strict run uses `token_concat_project` face embedding.
- Strict run uses `legacy_concat` CausalMLP, the closest public-code match we selected after prior TreeMeshGPT/CausalMLP review.
- Incremental generation and full causal hidden paths are covered by existing tests.
- The local FACE-focused test subset passes: `52 passed`.
- Target token graphs are watertight before prediction.
- B2 preservation uploaded checkpoints and metadata.

### Things that remain under-specified or risky

1. Exact FACE CausalMLP internals are unpublished.

Our `legacy_concat` is reasoned from the cited public-code direction. It is not guaranteed to match FACE exactly.

2. Exact FACE face embedding is unpublished.

`token_concat_project` is a strong literal interpretation, but the paper does not give enough implementation detail to prove exact parity.

3. Encoder is Shape2VecSet-style but dependency-light.

We match the broad VecSet idea, FPS queries, 2048 tokens, and bottleneck 64, but this is still our implementation, not verified official FACE code.

4. Current capacity is below paper-scale gate.

The readiness gate correctly warns that this run uses smaller width/depth/head count than the stricter full-scale profile.

5. Dataset scale is far below the paper setting.

The paper note we extracted says roughly 130k meshes fewer than 4000 faces and 100k steps. This run used about 5.3k strict token samples and 100k steps.

6. Training targets are strict adapted assets, not necessarily the same distribution as FACE's curated Objaverse subset.

The strict adapter gives legal watertight targets, but it may simplify or convexify assets in a way that creates a different learning problem.

7. Online augmentation retokenizes from dequantized token geometry.

This is reasonable and had zero known fallback issue in previous diagnostics, but it may add quantization/order churn compared with training directly from original high-quality meshes.

## Is This Too Strict Compared With The Paper?

Partly yes, but not in the way that saves the run.

Metrics that are stricter than typical papers:

- exact generated token accuracy
- exact edge-pairing ratio
- watertightness gates
- predicted-count product inference
- Blender/contact-sheet visual sanity

Metrics that papers often emphasize more:

- Chamfer
- Hausdorff
- normal consistency
- qualitative examples

If we judged only teacher-forced Chamfer, this run would look much more positive. But production wants editable meshes, and the visual sheets confirm that the stricter gates are not hallucinating a problem. Full AR outputs are visibly wrong. So the production gate is stricter than the paper, but appropriately stricter for our product goal.

## Would Scaling Improve This?

Likely yes for teacher-forced loss and first-face probability, but scaling alone is not proven sufficient.

Positive scaling signals:

- Selection loss improved cleanly through 90k.
- Train and test teacher-forced metrics are similar, so we are not merely memorizing a tiny train split.
- Prefix16 improves AR token accuracy, so the decoder is learning some conditional structure.
- Current dataset is much smaller than paper scale.
- Current model capacity is smaller than the full-scale target.

Negative scaling signals:

- First-face exact is still zero across all evaluated teacher-forced and AR sets.
- Train AR is worse than test AR on watertightness, indicating watertight outputs are not a faithful reconstruction signal.
- Teacher-forced topology is zero watertight despite low Chamfer.
- Predicted-count inference overruns badly.
- The contact sheets show qualitative failure, not just metric harshness.

My best estimate: scaling to 130k-350k with full capacity would probably improve supervised FACE reconstruction, but we should not expect it to magically produce production-editable meshes unless first-face exactness, topology welding, and EOS termination improve in smaller bounded gates first.

## Specific Root Cause Ranking

1. Supervised coordinate-token underfit.

Confidence: high. Evidence is direct: `~79%` teacher-forced token accuracy, loss `~0.8`, first-face exact `0%`.

2. First-face ambiguity and weak BOS-to-first-face conditioning.

Confidence: high. Evidence is direct: all full AR samples diverge at face 0; first-face target ranks/probs are poor.

3. Coordinate-token topology fragility.

Confidence: high. Evidence is direct: target token graph is watertight, teacher-forced Chamfer is low, teacher-forced predicted topology is open.

4. Exposure bias after early faces.

Confidence: medium-high. Prefix16 improves but does not rescue. This supports, but does not isolate, exposure bias.

5. Capacity/data scale below paper needs.

Confidence: medium-high. The run is smaller than paper-scale, but the exact amount of quality gain from scaling is unknown.

6. CausalMLP/face embedding mismatch with unpublished FACE code.

Confidence: medium. We cannot prove exactness. Tests show internal consistency, not paper identity.

7. Evaluation too strict.

Confidence: medium-low as a primary explanation. Exact metrics are strict, but visual sheets independently show failure.

## Recommended Next Experiments

Do not jump straight to 350k/100k as the next expensive training action.

Run these bounded validations first:

1. First-face closure probe.

Train/evaluate a dedicated first-face diagnostic on the same split:

- report exact first-face rate
- report per-slot first-face accuracy
- report first-face rank/prob distribution
- optionally try top-k/beam first-face selection with geometric rescoring

Promotion condition: first-face exact becomes nonzero and materially improves, not just average CE.

2. Teacher-forced topology closure probe.

Take teacher-forced logits and test whether small decoding changes improve edge welding:

- coordinate beam inside each face
- vertex-snapping/welding constrained decode
- reuse-nearby-vertex constrained decode
- topology-aware post-decode repair

Promotion condition: teacher-forced watertight and edge-pairing improve while Chamfer stays low.

3. Capacity rung before data rung.

Repeat a smaller data run with full paper-scale capacity:

- hidden 1024
- encoder layers 8+
- decoder layers 24
- heads 16
- same 5k-10k data

Promotion condition: teacher-forced accuracy moves toward `>90%`, first-face exact improves, AR visuals improve.

4. High-quality corpus rung.

If the capacity rung improves, launch a larger corpus rung, but use a staged path:

- 20k usable
- 50k usable
- 130k usable
- only then 350k+

5. Product fallback path in parallel.

Do not wait for FACE alone to solve editability. Continue the deterministic production mesh path:

- TRELLIS.2 / source mesh
- manifoldization
- structure/part decomposition
- chart retopo/quad remesh
- feature-aware projection
- Blender gate

FACE can become a topology prior or candidate generator, but current evidence does not make it the only production path.

## Audit Of This Audit

### Claim: First-face collapse is the dominant AR failure.

Evidence strength: high.

Support:

- `first_divergent_face_index = 0` for all train/test full AR evaluated samples.
- First-face exact is zero even under teacher-forced first-face logits.
- Teacher prefix removes first-face divergence by construction and improves generated accuracy.

Counterpoint:

- Prefix16 still fails, so first-face collapse is not the only issue.

Verdict:

True but incomplete. First-face collapse is a major necessary blocker, not the whole disease.

### Claim: Teacher-forced underfit blocks scale readiness.

Evidence strength: high.

Support:

- Train/test teacher-forced accuracy is only around `79%`.
- The readiness threshold is `95%` train accuracy, which is reasonable for exact topology reconstruction.
- First-face exact is zero.

Counterpoint:

- Papers may not require exact token accuracy or watertight topology.
- Chamfer is already low under teacher forcing.

Verdict:

True for production-editable meshes. Possibly too strict for matching a paper's Chamfer-only story.

### Claim: Watertight AR samples are not good enough.

Evidence strength: high.

Support:

- Watertight test AR samples still have low token accuracy.
- Visual sheets show closed wrong shapes.
- Some closed shapes are clearly primitive-like or unrelated.

Counterpoint:

- Visual preview is a lightweight renderer and can hide details.

Verdict:

Still true. The mismatch is large enough that renderer limitations do not explain it.

### Claim: The strict target corpus is not the immediate source of failure.

Evidence strength: medium-high.

Support:

- Evaluated teacher tokens are watertight edge graphs.
- Leakage checks passed.
- Token-hash split was deduped.

Counterpoint:

- The strict adapter may change the data distribution and make the learning problem unlike FACE's curated Objaverse training.

Verdict:

Targets are topologically legal, but corpus distribution may still be a scaling/generalization issue.

### Claim: Scaling alone is not enough to justify a 350k production run.

Evidence strength: medium-high.

Support:

- 100k steps on 5k usable did not solve train AR.
- First-face exact remains zero.
- Capacity is below paper-scale, so scaling data alone would be confounded.

Counterpoint:

- The paper used about 130k meshes, and scaling laws can be nonlinear.
- Selection loss improved steadily, so more/better data and bigger model could help significantly.

Verdict:

Do not launch 350k as a blind production run. It is reasonable to prep the corpus in parallel, but training spend should be staged behind first-face/topology closure gates.

### Claim: Our implementation may differ from FACE.

Evidence strength: medium.

Support:

- Exact CausalMLP and face embedding details are unpublished.
- Our encoder is dependency-light Shape2VecSet-style, not official FACE code.
- The capacity is smaller than paper-scale.

Counterpoint:

- The known public/paper details we have are represented: 2048 VecSet tokens, latent 64, 128 bins, Muon, online augmentation, FACE coordinate tokens, causal decoder, cross-attention.
- Internal consistency tests pass.

Verdict:

We are paper-inspired and strict-lane faithful where specified, but not provably paper-identical.

## Bottom Line

This run is useful and not wasted. It proves the pipeline can train, preserve checkpoints, improve loss, and evaluate full FACE-style metrics at 100k steps. It also proves the current FACE strict lane is not yet producing production-editable meshes.

The next decisive work should be first-face/topology closure, not immediate 350k training. The corpus prep can continue in parallel, but the model training gate should remain staged.
