# Easy3E Editing Module — Implementation Status

Easy3E (arxiv:2602.21499v1) is a training-free 3D editing method built on
TRELLIS.2's latent diffusion backbone. ClearMesh integrates it as an
optional editing stage.

This document tracks which parts are fully wired vs. blocked on deeper
TRELLIS.2 introspection.

## What works end-to-end (requires a loaded `Trellis2ImageTo3DPipeline`)

| Component                      | File                 | Status | Notes |
| ------------------------------ | -------------------- | ------ | ----- |
| `SLATEncoder.encode`           | `slat_encoder.py`    | ✅     | `FlexiDualGridVaeEncoder` + `o_voxel.convert.flexible_dual_grid._C`. Same path as `scripts/data/validate_alignment_noise.py`. |
| `SLATEncoder.decode`           | `slat_encoder.py`    | ✅     | `pipeline.decode_shape_slat(slat, 512)`. Resolution must be 512 (see `infer_slat.py:468`). |
| `SLATEncoder.{save,load}_slat` | `slat_encoder.py`    | ✅     | Tensor-only serialization. |
| `VoxelFlowEdit._encode_image_condition` | `voxel_flowedit.py` | ✅ | `pipeline.preprocess_image` + `pipeline.get_cond`. |
| `VoxelFlowEdit.auto_detect_edit_mask`   | `voxel_flowedit.py` | ✅ | Real 2D→3D projection with explicit camera params. Validated offline (`/tmp/test_mask_projection.py`). |
| `SLATRepainter._generate_features`      | `slat_repaint.py`   | ✅ | `pipeline.sample_shape_slat` with fine (1024) flow model. Matches `generate_slat_pairs.py:185` exactly. |
| `SLATRepainter._replay_source_trajectory` | `slat_repaint.py` | 🟡 | Identity passthrough with shape reconciliation when N_old ≠ N_new. Strict trajectory replay needs per-step flow-model velocity (see blockers below). |
| `SLATRepainter.repaint` end-to-end | `slat_repaint.py` | ✅ | Generate-then-blend with soft boundary mask. |
| `Easy3EEditor.edit` (image-guided, feature repaint only) | `easy3e.py` | ✅ | Works: encode → auto-mask → repaint → decode → repair → export. |
| `Easy3EEditor.edit_from_text`          | `easy3e.py`         | ✅ | Runs InstructPix2Pix on a source render, then `edit()`. |
| `Easy3EEditor.edit_iterative`          | `easy3e.py`         | ✅ | Chains multiple edits. |
| `CtrlAdapter` forward pass (control extraction) | `ctrl_adapter.py` | ✅ | Normal → multi-scale control features + cross-view attention. |

## What is blocked (and why)

### 1. `VoxelFlowEdit._compute_velocity` — full strict Easy3E voxel editing

**Status**: 🟡 **Wired — needs GPU validation.**

**What changed**: `Easy3EEditor._try_load_raw_model("sparse_structure_flow_model")`
reads `pipeline.json` and loads the flow model directly via
`trellis2.models.from_pretrained`, following the exact pattern in
`scripts/data/generate_slat_pairs_fast.py:86-116`. Once loaded, it's
injected into `VoxelFlowEdit(flow_model=ss_flow, pipeline=pipeline)`.

`VoxelFlowEdit._flow_call` probes four signature variants on first call
and caches the winner: positional `(x, t, cond)`, and kwargs `context=`,
`cond=`, `encoder_hidden_states=`. If all four fail it raises with a
clear context string — no silent wrong-keyword fallback.

`Easy3EEditor.edit` now exercises `voxel_flowedit.edit(...)` whenever
`flow_model is not None`, wrapped in a try/except so a bad load falls
back to feature-repaint-only rather than crashing the pipeline.

**Remaining GPU validation**:
1. Run `Easy3EEditor(pipeline=pipeline).edit(source_mesh, edit_image)`
   on RunPod and confirm `_flow_sig` converges within 4 tries.
2. Check the flow model's in/out tensor shape contract matches Easy3E's
   (B, C, R, R, R) assumption — if it expects `SparseTensor`, we need a
   small wrapper in `_compute_velocity` to wrap/unwrap.

### 2. `SLATEncoder._encode_ss_latent` — mesh → Sparse Structure VAE latent

**Status**: 🟡 **Probe added — needs GPU to know if encoder exists.**

**What changed**: `SLATEncoder._try_load_ss_encoder()` now introspects
`trellis2.models` for plausible encoder class names (case-insensitive
match on "sparse" + "encoder"/"enc", excluding "decoder"), plus explicit
name priorities: `SparseStructureEncoder`, `SparseStructureVaeEncoder`,
`SparseStructureVae{/VAE}`. It also scans `{model_dir}/ckpts/` for the
matching config + weights pair (`ss_enc*`, `sparse_structure_enc*`,
`ss_vae*`).

Still not wired into `encode()` — it returns None today, and
`ss_latent` continues to alias `voxel_indices`. That's intentional: we
don't want to silently change `encode()`'s output on existing callers
until GPU introspection confirms what's actually available.

**Remaining GPU validation**:
1. On RunPod: `python -c "from clearmesh.editing.slat_encoder import SLATEncoder; \
   enc = SLATEncoder(model_dir='/workspace/models/trellis2-4b'); \
   print(enc._try_load_ss_encoder())"`
2. If it prints a loaded module, wire it into `encode()` (produces
   a real SS latent) and flip `_ss_aliases_voxels = False` default.
3. If it returns None, fall back to (b): generate an SS latent via
   `pipeline.sample_sparse_structure` with source image conditioning —
   same contract, lower fidelity.

### 3. `CtrlAdapter.generate` — normal-guided texture generation

**Status**: 🔴 **Integration contract documented; bound to ERA3D work.**

**What changed**: `CtrlAdapter.generate`'s docstring now lays out the
full integration contract — what `base_model` must expose (scheduler,
unet with `down_block_additional_residuals` or equivalent, text
encoder, optional vae) for the Ctrl-Adapter signals to flow correctly
into the diffusion loop. The body still raises `NotImplementedError`
because the injection API is ERA3D-specific and getting it wrong
produces plausible-looking but unsupervised outputs.

The adapter `forward()` (control-signal extraction) is real and
tested. Training (`train_ctrl_adapter.py`) is wired. Only the
inference-time loop against a frozen ERA3D is missing.

**Current behavior**: `Easy3EEditor._apply_texture` prints a
"not yet implemented" warning and returns the mesh untouched. Texture
editing still works through the standard PBR texture path in
`clearmesh/textures/` — this is specifically for the Ctrl-Adapter
edit-target-guided variant.

**To unblock**:
1. Pick a concrete base model — ERA3D is the reference, but
   `mv-diffusion`/`MVDream` or similar may be easier.
2. Bind its scheduler + unet.forward contract to the docstring's
   listed kwargs (mostly `down_block_additional_residuals`).
3. Add a smoke test that runs 5 denoising steps and confirms tensor
   shapes through the loop — no need for perceptual quality yet.

## Testing

- Offline: `python3 /tmp/test_mask_projection.py` — validates 2D→3D
  edit-mask projection geometry with no TRELLIS.2 dependency.
- On GPU (RunPod / Vast): running `Easy3EEditor.edit(source_mesh, edit_image)`
  exercises encode → image-cond → repaint → decode end-to-end. Requires
  TRELLIS.2 4B checkpoints at `/workspace/models/trellis2-4b` and the
  shape encoder safetensors (`shape_enc_next_dc_f16c32_fp16.*`).

## Related

- Paper: Easy3E (arxiv:2602.21499v1)
- TRELLIS.2 pipeline reference: `scripts/data/generate_slat_pairs.py`,
  `scripts/data/validate_alignment_noise.py`, `clearmesh/stage2/infer_slat.py`
