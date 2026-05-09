> Superseded note (2026-05-08): after re-reading FACE methods and the cited TreeMeshGPT CausalMLP lineage, the strict lane now defaults back to `legacy_concat`. The `paper_chain` recommendation below is retained as historical Opus-review context and should be treated as an ablation, not the current launch default. See `docs/face_architecture_reaudit_2026-05-08.md`.

# FACE Opus 4.7 Review Consensus - 2026-05-07

## Context

We paused before scaling the FACE reproduction and reviewed the implementation with Claude Opus 4.7 in high-thinking mode. The goal was to check whether our implementation had drifted from the FACE paper and whether the next run should scale data or first close architectural mismatches.

## Consensus

Do not scale to production yet. First run a bounded paper-faithfulness gate that tests whether the corrected architecture improves train autoregressive closure and held-out coherence.

The review agreed that the previous implementation was close in several paper-level knobs, but still had two architecture-level deviations that could plausibly break the paper's central autoregressive behavior:

- The previous-face embedding compressed the 9 coordinate tokens in ways that were not faithful to Figure 2's explicit 9-token face embedding path.
- The CausalMLP prefix path mixed previous coordinate tokens with sum-pooling, losing ordered coordinate-prefix information inside the face.

## Required Implementation Changes

- Use `face_embedding_variant=token_concat_project` by default.
- Use `causal_mlp_variant=paper_chain` by default.
- Preserve previous-face coordinate token order in the face embedding.
- Preserve within-face causal coordinate order in `PaperChainCausalCoordinateMLP`.
- Add diagnostics that expose per-coordinate-slot failure modes.
- Add train/test leakage checking before any serious run.
- Add identity retokenization checking so quantization/tokenizer drift is visible.
- Add predicted-count autoregressive evaluation, not only fixed target-count evaluation.
- Refuse deprecated face embeddings for new paper-faithful training unless explicitly overridden.

## Bounded Gate Settings

The agreed next gate is intentionally not production-scale:

- Dataset: existing strict Objaverse++ split, training limited to 512 samples.
- Coordinates: 128 quantization bins.
- Input point cloud: 8192 points.
- VecSet latent: 2048 tokens, bottleneck dimension 64.
- Model: hidden 1024, encoder hidden 768, encoder layers 6, decoder layers 12, heads 16.
- Optimizer: Muon.
- Precision: bf16.
- Augmentation: paper-style online augmentation enabled.
- Eval: teacher-forced, fixed-count autoregressive, predicted-count autoregressive, topology/editability metrics, slot-level diagnostics.

## Promotion Rule

Only promote to the next corpus rung if the gate shows all of the following:

- Train autoregressive closure is coherent, not merely high teacher-forced accuracy.
- Held-out autoregressive samples show recognizable mesh structure.
- Boundary and non-manifold metrics materially improve.
- Slot-level diagnostics do not show a systematic coordinate-order failure.
- Predicted-count generation does not collapse face count or terminate pathologically.
- `scale_readiness.json` is strong and matches visual review.

If the gate fails, diagnose the most likely bottleneck and run another bounded validation. Do not jump to 2k, 5k, or 350k samples on weak AR evidence.

## Active Run

The current A100 gate launched from this consensus is:

- Thunder instance id: `0`
- Thunder name/uuid: `fmumaekn`
- Local run info: `.codex_outputs/face_paper_existing_split_setup_20260507_opus47_tokenconcat_gate512_a100_rerun1/run_info.json`
- Remote lab root: `/tmp/clearmesh_face_paper_existing_split_gate_20260507_opus47_tokenconcat_gate512_a100_rerun1`
- Remote run dir: `/tmp/clearmesh_face_paper_existing_split_gate_20260507_opus47_tokenconcat_gate512_a100_rerun1/runs/opus47_tokenconcat_gate512_a100_rerun1`

