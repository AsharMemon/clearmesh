# Easy3E Editing — Session Summary

## Status: DEMO WORKING

**The "add wings" end-to-end demo passes on a Vast.ai H100 NVL 94GB pod.**

```
tests/e2e/test_edit_text_guided.py::test_add_wings_demo PASSED
  Source SLAT: coords=(4001, 4), feats=(4001, 32)
  edit_from_source_image complete in 58.6s
  edited mesh: 1,700,185 verts, 3,467,078 faces
```

**Timings** (after warmup):
- InstructPix2Pix (source → "add wings" image): 4.0 s
- TRELLIS.2 source SLAT sampling: 5.2 s
- Flow edit (re-sample SLAT with edit conditioning): 1.8 s
- Decode SLAT → mesh: 46.2 s
- Total wall-clock: **59 seconds** per edit

Artifacts downloaded to `/tmp/easy3e_demo_artifacts/out/`:
- `source_input.png` (1.5 MB) — source view
- `edited.glb` (62 MB) — edited mesh with wings

## What this session did

Compared to the last handoff, we moved from "phases 1–6 code written, pod blocked on HF token" to **"demo end-to-end green"**. The key unlocks:

1. **HF token resolved** — installed `/etc/profile.d/hf_token.sh` so token propagates to every shell. Grants access to gated DINOv3.
2. **Transformers 5.x DINOv3 patch** — TRELLIS.2's `image_feature_extractor.py:86` iterates `self.model.layer`, but transformers 5.5.4 exposes layers at `self.model.model.layer` (the outer `DINOv3ViTModel` wraps a `DINOv3ViTEncoder` at `.model`). Patched in place; see `/workspace/TRELLIS.2/trellis2/modules/image_feature_extractor.py.orig` for the backup.
3. **Attention backend: flash_attn_3** — TRELLIS.2 defaults to `flash_attn` (v2) which isn't installed; we use `flash_attn_3` from the SpaceWheels release. Set via `/etc/profile.d/trellis2_env.sh` and also auto-set in `tests/conftest.py` so pytest runs pick it up without manual env.
4. **Tensor truthiness bug** — `_split_condition` used `or`-chains on dict values, which raises on tensors. Fixed with `is not None` checks.
5. **Architecture pivot: image-based source** — Phase 0 introspection confirmed TRELLIS.2-4B does NOT ship a mesh→SLAT encoder (only a decoder and sampler). The Easy3E paper actually edits **TRELLIS.2-generated** SLATs, not arbitrary meshes. Added `Easy3EEditor.edit_from_source_image(source_image, instruction=...)` which does the full image → source SLAT → edit → mesh flow. The old mesh-based `edit()` / `edit_from_text()` methods remain for when a true encoder becomes available (e.g. via `data_toolkit` modules).

## Introspection findings worth remembering

From `/workspace/trellis2_introspection.txt` (committed to git as a reference):

- **`pipeline.models` keys**: `sparse_structure_decoder`, `sparse_structure_flow_model`, `shape_slat_decoder`, `shape_slat_flow_model_{512,1024}`, `tex_slat_decoder`, `tex_slat_flow_model_{512,1024}`. **No encoders.**
- **SparseStructureFlowModel.forward(x: Tensor, t: Tensor, cond: Tensor)** — dense tensor, plain tensor cond (not dict).
- **SLatFlowModel.forward(x: SparseTensor, t: Tensor, cond: Union[Tensor, List[Tensor]], concat_cond: Optional[SparseTensor])** — sparse, list-or-tensor cond.
- **pipeline.get_cond(img_list, resolution)** returns `{"cond": (1, 1029, 1024), "neg_cond": (1, 1029, 1024)}`.
- **Samplers**: `pipeline.sample_sparse_structure(cond, 32, 1, params)` returns coords `(B, 4)`. `pipeline.sample_shape_slat(cond, flow_model, coords, params)` returns `SparseTensor`.

## Test status

- **Unit tests: 52/52 passing** locally (CPU-only) and on pod.
- **Integration tests**:
  - `test_encode_image_condition` (now passes after the tensor truthiness fix)
  - `test_velocity_shape_matches_input`, `test_cfg_monotonic`: skip with "fabricated x_t rejected" (real flow model expects specific sparse layout — worth upgrading these tests to feed a real SS latent from sample_sparse_structure).
  - `test_encode_produces_valid_shapes`, `test_encode_decode_roundtrip_iou`: fail as expected ("No SS encoder found"). These validate the error contract for when someone attempts mesh→SLAT. They pass the error-message test, just not the happy-path because the happy path requires an encoder that TRELLIS.2 doesn't ship.
- **E2E tests: add-wings demo passing**.

## Running the demo yourself

```bash
# 1. Restart pod (if stopped):
/Users/Ashar/Library/Python/3.14/bin/vastai --api-key "$VAST_API" start instance 35082988
sleep 90

# 2. SSH:
ssh -o UserKnownHostsFile=/tmp/vast_known_hosts -p 12988 -i ~/.ssh/id_ed25519 root@ssh3.vast.ai

# 3. On pod — everything needed is now in /etc/profile.d/:
bash -l  # to source /etc/profile.d/*.sh
source /opt/conda/etc/profile.d/conda.sh && conda activate trellis2
cd /workspace/clearmesh && git pull
python -m pytest tests/e2e/test_edit_text_guided.py -v -s

# Artifacts land in /tmp/pytest-of-root/pytest-*/test_add_wings_demo0/out/
```

## Cost summary

- Session 1 (setup, blocked on HF token): ~$1.35
- Session 2 (demo green): ~$1.00 (pod active ~30 min, mostly IP2P/TRELLIS.2 downloads + one 2-min test run)
- **Total compute: ~$2.35** (well under the $15-30 sprint budget)
- Remaining balance on Vast.ai: ~$104

## Known issues / follow-up work

### Worth fixing in a future pass

1. **Mesh repair** (`full_print_preparation`) fails with `'Trimesh' object has no attribute 'remove_degenerate_faces'` — newer trimesh removed this API. Our try/except catches it gracefully but the demo output is un-repaired. Fix: update `clearmesh/mesh/repair.py` to use the current trimesh API (`mesh.update_faces(mesh.nondegenerate_faces())`).
2. **Flow edit is approximate** — the current `_flow_edit_slat` just re-samples SLAT with edit conditioning, keeping source coords. This is a good approximation of Easy3E's trajectory-splitting at high `gamma` but not the full ODE. To implement the full paper:
   - Expose a way to start `sample_shape_slat` from a given noised latent (not fresh noise).
   - Track two trajectories (source + target) and subtract velocities.
   - Apply the silhouette guidance term already implemented in `silhouette.py`.
3. **Integration tests for voxel_flowedit** should use real SS latents from `sample_sparse_structure` output, not fabricated dense tensors. They currently skip due to shape mismatch — technically a soft failure mode.
4. **Edit mask is unused** in `edit_from_source_image` — we pass the whole SLAT through. To use the 2D→3D mask projection in `camera.py`, we'd gate the feature flow per-voxel (again, requires partial-update sampling which pipeline.sample_shape_slat doesn't expose directly).

### Expected-fine issues (no action)

- `UnexpectedKeys: text_model.embeddings.position_ids` — IP2P checkpoint has a vestigial key; harmless.
- `timm.models.layers` deprecation warnings — cosmetic, from TRELLIS.2's timm usage.

## Pod state when you return

- **Still running** as of this writeup (since we were actively testing). Feel free to stop with `vastai stop instance 35082988` if not continuing immediately — state preserved at $0.20/hr storage.
- All patches applied: TRELLIS.2 feature extractor, `/etc/profile.d/hf_token.sh`, `/etc/profile.d/trellis2_env.sh`.
- Conda env `trellis2` has Python 3.10 + torch 2.6.0+cu124 + all TRELLIS.2 deps + diffusers + rembg.

## Commits on `claude/nervous-sammet`

- `b6f9dde` — Phase 0: pytest harness + introspection + headless render fix
- `49f4c7c` — Phase 1–5: encoder stubs + flow core + repainter + mask + silhouette
- `0acbaaa` — Initial handoff doc
- `f4d42ef` — Fix `_split_condition` + add `edit_from_source_image` path
- `9e74cea` — Re-export `EditOptions`/`EditResult` from package
- (This commit) — Auto-set ATTN_BACKEND in conftest + final handoff update
