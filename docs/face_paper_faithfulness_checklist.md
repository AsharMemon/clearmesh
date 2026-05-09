# FACE Paper-Faithfulness Checklist

Date: 2026-05-03
Paper: `/Users/Ashar/Downloads/2603.01515v2.pdf`
Implementation target: FACE ARAE reconstruction path, not image-to-mesh latent diffusion.

## Current Verdict

We are recommitting to the paper-faithful reconstruction ARAE as the primary
FACE lane. The indexed / half-edge / boundary-fill experiments are now
explicitly quarantined as ClearMesh production research, not FACE reproduction
evidence.

The strict paper policy is captured in
`docs/face_strict_paper_recommitment.md`: preserve paper details first, and fill
only details that are unavailable or under-specified.

We are structurally faithful to the paper's reconstruction ARAE, but not yet paper-scale or production-speed.

The implementation now passes the most important pre-scale sanity gates: no-augmentation synthetic overfits, frozen/offline augmented synthetic overfit, and real-mesh overfits all reach effectively exact teacher-forced token accuracy. The strict voxel-shell target run is the cleanest proof so far: seven cleaned real targets pass the strict paper-token dataset gate, then the paper-faithful FACE ARAE reconstructs all seven as watertight teacher-forced GLBs with zero boundary edges.

Two blockers remain before a long production run:

- Free-running generation still needs more optimization and held-out validation for production. The original no-cache AR path took roughly 16-17 seconds per 128-face sample on one A6000; the incremental + one-pass greedy coordinate decoder now generates the full 7-mesh strict overfit set in 125 seconds, with the largest 3016-face mesh taking 46 seconds.
- Held-out generalization is not solved at tiny data scale. A five-train / two-test split of the strict voxel-shell real targets overfits the train split perfectly but gets 0/2 held-out watertight with mean boundary edges 41 and teacher-forced accuracy 0.038. A larger grouped 60-shard strict run proves the grouped split/dataset harness works, but the 3200-step 192-hidden model is undertrained: train AR sample is only 1/12 watertight and held-out is 3/12 watertight while selection loss is still 1.2087 and falling.
- Input cleanliness matters. Several raw real source meshes were already open/fragmented, and quantization can make zero-area faces that must be removed or manifoldized before training/eval. FACE should train on clean, manifoldized triangle targets, not raw arbitrary decimations.
- Target preparation should prefer voxel-shell/manifold targets with simplification disabled or carefully validated. In local probes, the voxel shell was watertight before simplification; aggressive quadric simplification introduced non-manifold edges and forced a convex-hull fallback.

## Equation / Variable Crosswalk

| Paper item | Paper meaning | Code location | Status | Notes |
| --- | --- | --- | --- | --- |
| `P` | input point cloud | `surface_points + surface_normals` in `scripts/research/build_face_token_dataset.py` and `PaperFaceSample.point_features` | Faithful | Paper text says `P in R^{m x 3}`, but implementation details say 8192 points with normals. We use XYZ+normal as 6D features. |
| `Q` | FPS/downsampled query points | `_farthest_point_indices(...)` in `clearmesh/mesh_heads/face_paper.py` | Faithful structure | Uses deterministic FPS over XYZ. Exact FPS kernel may differ from paper implementation. |
| `K_P, V_P` | full point set projections | `point_projection(point_features)` in `VecSetEncoder` | Faithful structure | Current projection shares K/V because PyTorch MHA applies internal K/V projections. |
| Eq. 1 `C' = CrossAttn(Q, K_P, V_P)` | FPS queries attend to full point set | `VecSetEncoder.forward` | Faithful structure | Implemented with `nn.MultiheadAttention`. |
| Eq. 2 `C = TransformerEncoder_LE(C')` | VecSet refinement | `self.encoder(vecset)` in `VecSetEncoder` | Faithful structure | Default paper-scale arguments are 8 encoder layers, 768 hidden, bottleneck 64. |
| `C in R^{k x d_latent}` | compact latent VecSet | `bottleneck(..., latent_dim=64)` | Faithful | Paper uses 2048 tokens and 64 bottleneck. Smoke runs scale `k` down. |
| `F=(f_1,...,f_N)` | ordered face sequence | `paper_tokens` arrays | Faithful | Variable face count is supported in dataset/eval by GT face count. EOS is under-specified in the paper and not implemented yet. |
| Face ordering | lexicographic ZYX by minimum-coordinate vertex | `canonicalize_mesh_faces_paper_zyx` | Faithful with deterministic tie-break | Paper does not specify tie-breaks or within-face rotation. We preserve within-face order and use full-token tie-breaks. |
| `f_i=(v_i^0,v_i^1,v_i^2) in R^9` | one face as one token | `paper_tokens` shape `(F, 9)` | Faithful | Tokens are `z,y,x` per vertex to match Fig. 2. |
| `t_{i-1}=MLP_embed(f_{i-1})` | Face Pooling MLP | `self.face_pooling` | Faithful structure | Exact MLP depth/activation is under-specified. |
| Eq. 3 `H'_l = CausalSelfAttn(H_l)` | causal face-level decoder self-attn | `FaceDecoderBlock.self_attn` with causal mask | Faithful | Uses PyTorch MHA. |
| Eq. 4/5 `H_{l+1}=CrossAttn(Q=H'_l,K=C,V=C)` | decoder cross-attn to VecSet every layer | `FaceDecoderBlock.cross_attn(..., vecset, vecset)` | Faithful | Cross-attention uses `kdim/vdim=latent_dim`. |
| `h_i in R^{d_model}` | latent face vector | `model.hidden(...)[..., i, :]` | Faithful | Decoder hidden size defaults to 1024. |
| `L_{i,1}...L_{i,9}` | logits for nine quantized coordinates | `CausalCoordinateMLP` | Faithful objective | Exact CausalMLP internals are not specified by FACE; implementation follows the cited TreeMeshGPT-style staged coordinate heads. |
| Eq. 6 CE objective | mean coordinate-token CE over faces and coordinates | `F.cross_entropy(...).sum()/valid.sum()` | Faithful | Uses teacher forcing. |
| Quantization `[0,127]` | 128 coordinate bins | `num_bins=128` defaults | Faithful | Higher-res scaling is not yet enabled in run profiles. |
| Train augmentation | random rotation, flipping, independent axis scaling | `_augment_sample(...)` | Faithful intent | Exact ranges/distribution are not specified; online augmentation is unstable at tiny scale because FACE ordering changes with transforms. Frozen/offline augmented variants are the current safe recipe. |
| Optimizer | Muon, lr `6e-4`, wd `0.1` | `_build_optimizer(...)` plus fallback | Faithful family | Native PyTorch Muon used when available; fallback follows documented Muon for 2D params. Non-2D params use AdamW as PyTorch docs recommend. |
| Inference | deterministic top-1 autoregressive sampling | `_generate_tokens(...)` | Faithful | Uses GT face count; EOS/variable termination remains missing because the paper under-specifies it. |

## Known Non-Exact / Risk Items

- Exact 3DShape2VecSet internals are approximated with FPS queries, cross-attention, and TransformerEncoder. We have not imported the original 3DShape2VecSet code.
- Exact FACE CausalMLP architecture is under-specified. We cross-checked the cited TreeMeshGPT code and use staged causal coordinate heads.
- EOS / face-count termination is not implemented. Current evaluation uses ground-truth face count to isolate reconstruction quality.
- Free-running inference now has an incremental decoder and one-pass greedy coordinate decoding. This is correct against the full causal hidden-state path and much faster than the original loop, but high-face-count deployment still needs projected KV caching, blockwise generation, or a non-AR promotion stage.
- We have no Objaverse 130K training subset wired yet. Current GPU runs use synthetic primitives plus a seven-mesh real decimated proof set.
- Raw real meshes need cleanup/manifoldization before tokenization. The tokenizer now drops duplicate and colinear quantized faces, but that can expose holes in dirty sources.
- Smoke runs are far smaller than paper scale: paper reports ~500M params, 8192 points, 2048 VecSet tokens, 4000-face cap, 100K steps, 8x A100 80GB.
- Thunder A6000 currently has PyTorch 2.8 without native Muon, so we use the local fallback unless we install newer PyTorch/CUDA.

## Completed Gates

| Gate | Result |
| --- | --- |
| Tokenizer paper roundtrip tests | Pass: `tests/test_face_tokens.py` |
| Local compile gates | Pass |
| Local tiny train/eval smoke | Pass as plumbing, not quality |
| Augmented mixed synthetic smoke | Trains but poor rollouts: 0/8 watertight, mean teacher-forced accuracy not measured in first artifact, bad contact sheet |
| No-augmentation 8-box overfit | Pass: 8/8 watertight, boundary edges 0, edge pairing 1.0, teacher-forced accuracy 1.0 |
| No-augmentation mixed primitive overfit | Pass: 10/10 watertight, boundary edges 0, edge pairing 1.0, teacher-forced accuracy 1.0 |
| Online mild augmentation mixed primitive probe | Fail: geometry improves but topology/rollout fail at tiny scale; target ordering changes too much per step |
| Frozen/offline augmentation mixed primitive overfit | Pass: 12/12 watertight, boundary edges 0, edge pairing 1.0, teacher-forced accuracy 1.0 |
| Real decimated seven-mesh no-augmentation overfit | Pass for token learning: best loss 0.000131 at step 2496, teacher-forced accuracy 1.0 |
| Real decimated teacher-forced full-sequence export | Pass for reconstruction identity: generated and teacher GLBs have identical vertices/faces in the audit |
| Real decimated capped AR rollout | Pass as a prefix sanity check: first 128 faces on two samples match visually; optimized runtime is 6.9s total for two samples |
| Strict voxel-shell target prep | Pass: seven real proxies -> seven watertight voxel-shell targets, one component, zero non-manifold edges |
| Strict voxel-shell paper-token dataset gate | Pass: 7/7 strict paper-token targets, boundary edges 0, edge pairing 1.0 |
| Strict voxel-shell paper-faithful FACE overfit | Pass: best full-dataset loss 0.0001087 at step 2600; teacher-forced export 7/7 watertight, boundary edges 0, edge pairing 1.0, mean teacher-forced accuracy ~1.0 |
| Strict voxel-shell full-count AR rollout | Pass on training-set overfit: 7/7 watertight, boundary edges 0, edge pairing 1.0; total eval 125s on A6000, 3016-face mesh 46s |
| Strict voxel-shell held-out AR split | Fail as expected at tiny data scale: train 5/5 watertight, held-out 0/2 watertight, mean held-out boundary edges 41 |
| Strict mixed15 aug60 grouped dataset gate | Pass: 60/60 strict paper-token shards, 15 groups, grouped split gives 48 train / 12 held-out shards with no augmented source leakage |
| Strict mixed15 aug60 grouped AR holdout, 3200 steps | Undertrained negative gate: train AR sample 1/12 watertight, held-out 3/12 watertight, best selection loss 1.2087 and still improving |

## Lab-Style Trust Ladder Before A Long Run

1. Paper/equation parity checklist: complete enough to identify remaining unknowns.
2. Tokenizer tests: complete.
3. Tiny synthetic overfit: complete, pass.
4. AR rollout after overfit: complete, pass.
5. Mixed primitive overfit without augmentation: complete, pass.
6. Mixed primitive with mild online augmentation: complete, fail at tiny scale.
7. Frozen/offline augmentation: complete, pass.
8. Small real-mesh overfit: complete, pass for token learning, input cleanup still needed.
9. Cached/incremental AR generation: complete enough for current experiments.
10. Clean manifoldized real-mesh target set, 16-64 meshes: started with grouped mixed15/aug60 corpus; next run must train long enough to fit train AR before judging held-out quality.
11. Small real-mesh validation, 1K-5K meshes: next after the 16-64 cleaned split shows learning curves that improve held-out topology.
12. Scaling probe: fit curves for loss, teacher-forced accuracy, Chamfer/Hausdorff, watertightness versus steps/model/data.
13. Long run only after the curves are sane and checkpoints visually improve every few thousand steps.

## Go / No-Go Criteria For Longer Run

Do not launch a paper-scale run until all are true on a 16-64 mesh real or high-quality synthetic subset:

- Teacher-forced accuracy exceeds 95% on train-overfit.
- Top-1 AR rollout is mostly stable, not spiking or collapsing.
- Boundary edges trend downward with training.
- A held-out mini-validation split improves over checkpoints.
- Visual contact sheets show recognizably correct meshes before full convergence.
- GPU utilization is healthy enough that the run is not CPU/data-loader bound.
- Free-running generation is fast enough for the target profile, either through KV caching, blockwise decoding, or a non-AR production promotion stage.
- Training targets have passed a manifold/nondegenerate mesh gate.

## Latest Artifacts

- Augmented mixed synthetic failure sheet: `artifacts/research_proofs/2026-05-03/face_paper_faithful_v1/face_paper_faithful_v1_contact_sheet.png`
- No-augmentation overfit success sheet: `artifacts/research_proofs/2026-05-03/face_paper_overfit_boxes_v1/face_paper_overfit_boxes_contact_sheet.png`
- Overfit eval report: `artifacts/research_proofs/2026-05-03/face_paper_overfit_boxes_v1/eval_report.json`
- Frozen/offline augmentation success sheet: `artifacts/research_proofs/2026-05-03/face_paper_offline_aug_mixed_v1/face_paper_offline_aug_mixed_contact_sheet.png`
- Real decimated teacher-forced sheet: `artifacts/research_proofs/2026-05-03/face_paper_real_decimated_overfit_v1/face_paper_real_teacher_forced_contact_sheet.png`
- Real decimated capped AR-128 sheet: `artifacts/research_proofs/2026-05-03/face_paper_real_decimated_overfit_v1/face_paper_real_ar128_incremental_greedy_contact_sheet.png`
- Real decimated reports: `artifacts/research_proofs/2026-05-03/face_paper_real_decimated_overfit_v1/eval_report_teacher_forced.json`, `artifacts/research_proofs/2026-05-03/face_paper_real_decimated_overfit_v1/eval_report_ar128_incremental_greedy.json`
- Strict voxel-shell target gallery: `artifacts/research_proofs/2026-05-03/face_strict_voxel16_targets/strict_voxel16_target_gallery.png`
- Strict voxel-shell FACE proof summary: `artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_overfit_v2/PROOF_SUMMARY.md`
- Strict voxel-shell FACE teacher-forced sheet: `artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_overfit_v2/face_paper_strict_voxel16_teacher_forced_contact_sheet.png`
- Strict voxel-shell FACE eval report: `artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_overfit_v2/eval_report.json`
- Strict voxel-shell FACE full AR sheet: `artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_overfit_v2_ar_full/face_paper_strict_voxel16_ar_full_contact_sheet.png`
- Strict voxel-shell FACE full AR eval report: `artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_overfit_v2_ar_full/eval_report_ar_full.json`
- Strict voxel-shell FACE holdout summary: `artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_holdout_v1/HOLDOUT_SUMMARY.md`
- Strict voxel-shell FACE holdout test sheet: `artifacts/research_proofs/2026-05-03/face_paper_strict_voxel16_holdout_v1/face_paper_strict_voxel16_holdout_test_ar_contact_sheet.png`
- Strict mixed15 aug60 source gallery: `artifacts/research_proofs/2026-05-03/face_strict_mixed15_aug60/face_strict_mixed15_aug60_gallery.png`
- Strict mixed15 aug60 grouped holdout summary: `artifacts/research_proofs/2026-05-03/face_paper_strict_mixed15_aug60_holdout_v1/HOLDOUT_SUMMARY.md`
- Strict mixed15 aug60 grouped train AR sheet: `artifacts/research_proofs/2026-05-03/face_paper_strict_mixed15_aug60_holdout_v1/face_paper_strict_mixed15_aug60_train_ar_contact_sheet.png`
- Strict mixed15 aug60 grouped held-out AR sheet: `artifacts/research_proofs/2026-05-03/face_paper_strict_mixed15_aug60_holdout_v1/face_paper_strict_mixed15_aug60_test_ar_contact_sheet.png`

## Implementation Update: Gap Closure Pass

The paper-faithful FACE path now addresses the major non-training-length gaps from the hostile audit:

- Added `encoder_backend=shape2vecset`, a dependency-light implementation of the official 3DShape2VecSet encoder topology.
- Added `causal_mlp_variant=legacy_concat` as the paper-faithful default after checking the cited TreeMeshGPT public code; `paper_chain` remains as an experimental reasoned variant.
- Added an EOS head, EOS training loss, and predicted-count autoregressive evaluation.
- Added paper-scale/smoke/A6000 profiles in `configs/face_paper_profiles.json`.
- Added curated corpus tooling in `scripts/data/build_face_training_corpus.py`.
- Changed paper holdout wrappers to default augmentation on.
- Added `scripts/thunder/face_paper_ab_remote_job.sh` for matched causal-vs-parallel ablations.

The remaining out-of-scope item is the FACE image-to-VecSet DiT; ClearMesh is deliberately using TRELLIS/LATTICE/UltraShape as the image-conditioned geometry source before FACE remeshing.

## Implementation Update: Corpus Pilot Hardening

The real-corpus path now has a production-oriented pilot:

- Added `requirements-data.txt` and `scripts/thunder/install_face_training_env.sh` so Objaverse++, datasets, Hugging Face, and `fast-simplification` installs are reproducible on fresh Thunder instances.
- Added `scripts/thunder/face_objaversepp_corpus_pilot.sh`, an end-to-end Objaverse++ -> strict target -> FACE token gate job.
- Added per-asset mesh inspection timeouts and file-size rejects in `scripts/data/build_face_training_corpus.py`; one pathological GLB now becomes a reject reason instead of freezing the corpus builder.
- Added manifest input support to strict target and token dataset builders, preserving explicit accepted/rejected lineage.
- Added adapter face-budget enforcement with `max_target_face_ratio`; missing decimation now causes a target reject instead of a silent high-face acceptance.
- Changed the FACE dataset gate so empty manifests fail instead of reporting `passes: true`.

First real Thunder cell:

```text
Objaverse++ selected/downloaded assets
  -> one candidate accepted from a 5-asset pilot
  -> voxel-shell target: 512 faces, watertight, 1 component, 0 boundary edges, 0 non-manifold edges
  -> 128-bin paper-token gate: pass, decoded watertight, edge pairing 1.0
```

Important finding: 16-bin quantization failed on the same real target by collapsing vertices and producing artificial boundary edges. Keep 16 bins for toy smoke tests only; real paper-token gates should use 128 bins.

## Objaverse++ 50-Asset Real Pilot Update

A 50-candidate real Objaverse++ cell produced 38 strict voxel-shell targets and 27 strict 128-bin paper-token passes. A 3000-step small FACE run on the 27 passing shards showed memorization but no generalization yet:

```text
train teacher-forced accuracy: 0.7264
train teacher-forced watertight: 1/22
test teacher-forced accuracy: 0.0343
test teacher-forced watertight: 0/5
train AR-128 watertight: 0/5
test AR-128 watertight: 0/5
```

Interpretation: the paper-faithful architecture is alive enough to learn on real shards, but the 27-shard dataset is far too small. The next meaningful gate is a larger 512-bin corpus and longer training curve, not another tiny hyperparameter tweak.

512-bin tokenization materially improved target usability on the same strict meshes:

```text
128 bins: 27/38 strict pass
256 bins: 33/38 strict pass
512 bins: 37/38 strict pass
```

This is a deliberate production deviation from the paper's 128-bin setting; keep 128-bin runs as ablations only.

## Objaverse++ 200-Asset 512-Bin Ladder

Background Thunder run:

```text
instance: 1 / A6000
run_dir: /tmp/clearmesh_face_objpp200_512_20260504_020301
raw downloaded: 200
curated accepted: 156
strict targets: 154/156
512-bin strict paper-token pass: 136/154, pass_rate 0.8831
train/test split: 109 / 27
train strict gate: 109/109
test strict gate: 27/27
```

First training signal on the 109-shard split:

```text
512 bins, model_max_faces=512, point_samples=2048, hidden=256, decoder_layers=6
step 1 selection_loss: 6.2764
step 1000 selection_loss: 5.5761
```

This is not a quality result yet, but it is the first sane learning curve on a larger real gated corpus.

Final result from the same run:

```text
steps: 8000
online augmentation: enabled
train teacher-forced accuracy: 0.0547
test teacher-forced accuracy: 0.0712
train AR-128 watertight: 0/5
test AR-128 watertight: 0/5
```

Hostile-audit interpretation:

```text
This is not a green light to scale. The run combined 512-bin tokens, a small
256-hidden / 6-layer model, only 109 train shards, and full online SO3/flip/
axis-scale augmentation. Our earlier trust ladder already showed that online
augmentation is unstable before a no-augmentation overfit gate is healthy.
```

Code correction after this run:

```text
1. Thunder train/eval jobs now default DISABLE_AUGMENT=1 for ladder runs.
2. Muon optimizer grouping no longer sends embedding tables, positional
   embeddings, norms, biases, BOS, or EOS heads through Muon; those use AdamW.
3. Thunder ladder/smoke profiles now default to AdamW. Keep Muon as an explicit
   paper-scale reproduction setting, but do not let a native Muon implementation
   bottleneck block the practical quality ladder.
```

Next gate:

```text
Rerun the same 109/27 strict 512-bin split with augmentation disabled and the
fixed optimizer grouping / AdamW. Judge train teacher-forced accuracy and train
AR watertightness before launching a larger corpus run.
```

Follow-up Thunder A/B:

```text
run_dir: /tmp/clearmesh_face_objpp200_512_20260504_020301/face_512bins_noaug_adamw_8000_20260504_030910
settings: same 109/27 split, 512 bins, no online augmentation, AdamW
step 1 selection_loss: 6.2391
step 1000 selection_loss: 4.7985
throughput: ~4.86 steps/sec

comparison:
failed augmented run step 1000 selection_loss: 5.5761
```

This is an immediate positive separation. Let the run finish before judging
mesh quality, but the training loop is no longer stuck at the previous loss
regime.

Final no-augmentation / AdamW result:

```text
steps: 8000
best/selection loss at step 8000: 2.5763
train teacher-forced accuracy: 0.4239
test teacher-forced accuracy: 0.0314
train teacher-forced watertight: 0/109
test teacher-forced watertight: 0/27
train AR-128 watertight: 0/5
test AR-128 watertight: 0/5
train AR-128 speed: ~31.9 faces/sec
test AR-128 speed: ~35.2 faces/sec
```

Hostile-audit interpretation:

```text
The correction was necessary and recovered the training curve, but it did not
prove mesh quality. At this data/model scale, FACE is learning the train split
but has not learned a reliable reconstruction manifold, and held-out
generalization is still essentially absent.
```

The full reset audit is now recorded in:

```text
docs/face_full_hostile_audit_2026-05-04.md
```
