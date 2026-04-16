# Easy3E Editing — Autonomous Session Handoff

## Status summary

**Phases 0–6 implemented and committed. Phase 7 (add-wings demo) blocked on HF token for gated DINOv3.**

- **Commits on `claude/nervous-sammet`**: `b6f9dde` (Phase 0) + `49f4c7c` (Phases 1–6). Pushed to remote.
- **Local tests**: 52 pass, 6 GPU-only integration/e2e tests skip cleanly on macOS.
- **Pod**: Vast.ai H100 NVL 94GB (contract 35082988, ssh3.vast.ai:12988) is **stopped**, charging $0.20/hr for storage only. Deps installed, TRELLIS.2 cloned, model weights partially downloaded (~14GB). Restart and continue when the token is ready.

## The one remaining blocker

TRELLIS.2's image conditioning model (`facebook/dinov3-vitl16-pretrain-lvd1689m`) is a **gated HuggingFace repo**. Downloading it requires an HF token with access granted. Without it:
- Pipeline loading fails partway through
- Introspection cannot complete
- Image conditioning `pipeline.get_cond(...)` cannot produce DINOv3 features
- End-to-end demo cannot run

The previous session's memory notes confirm this: `DINOv3 model is gated - needs HuggingFace token authentication`. You have solved this before.

## To resume (15 minutes of setup, then ~3 minutes for the demo)

1. **Request DINOv3 access if you haven't already** — visit https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m and click "Agree and access". Access is typically granted immediately.

2. **Set HF token in your shell**:
   ```bash
   export HF_TOKEN=hf_...
   # Or persistent:
   echo 'export HF_TOKEN=hf_...' >> ~/.zshrc
   ```

3. **Restart the pod**:
   ```bash
   /Users/Ashar/Library/Python/3.14/bin/vastai --api-key "$VAST_API" start instance 35082988
   # Wait ~1 min for status to become running
   /Users/Ashar/Library/Python/3.14/bin/vastai --api-key "$VAST_API" show instance 35082988 --raw | grep actual_status
   ```

4. **SSH in and set the token**:
   ```bash
   ssh -o UserKnownHostsFile=/tmp/vast_known_hosts -p 12988 -i ~/.ssh/id_ed25519 root@ssh3.vast.ai
   export HF_TOKEN=hf_...
   ```

5. **Pull latest code and run the introspection script first** — its output drives any remaining wrinkles in Phase 1–5:
   ```bash
   source /opt/conda/etc/profile.d/conda.sh && conda activate trellis2
   export PYTHONPATH=/workspace/TRELLIS.2:$PYTHONPATH
   cd /workspace/clearmesh
   git pull origin claude/nervous-sammet
   python scripts/setup/inspect_trellis2.py > /workspace/trellis2_introspection.txt 2>&1
   tail -100 /workspace/trellis2_introspection.txt
   ```

6. **Run tests in order of increasing cost**:
   ```bash
   cd /workspace/clearmesh
   pytest -q tests/unit/                                # CPU-only, all should pass (~2s)
   pytest -q tests/integration/ -m "gpu and trellis2"   # GPU, ~30s
   pytest -q tests/e2e/ -m "slow and gpu and trellis2"  # The demo, ~2-3 min
   ```

7. **Inspect the demo artifacts**:
   ```bash
   ls -la /tmp/pytest-*/out/source.glb /tmp/pytest-*/out/edited.glb
   # Download and open in Blender / online GLB viewer
   ```

## What was implemented

### New files
- `clearmesh/editing/camera.py` — `CanonicalCamera`, `project_voxels_to_pixels`
- `clearmesh/editing/silhouette.py` — target extraction, voxel splat render, BCE grad
- `scripts/setup/inspect_trellis2.py` — Phase 0 introspection (runs on pod)
- `pytest.ini`, `tests/{unit,integration,e2e}/` — full pytest infra with gpu/slow/trellis2 markers
- `tests/unit/test_imports.py` (11 tests)
- `tests/unit/test_flowedit_math.py` (12 tests — forward diffuse, trajectory, masking)
- `tests/unit/test_slat_encoder.py` (10 tests — resolver paths, rasterization)
- `tests/unit/test_camera.py` (11 tests — projection math, mask projection)
- `tests/unit/test_silhouette.py` (8 tests — splat, BCE, gradient)
- `tests/integration/test_slat_encode.py` (roundtrip IoU test)
- `tests/integration/test_flowedit.py` (velocity + CFG tests)
- `tests/e2e/test_edit_text_guided.py` (the "add wings" demo test)

### Modified files
- `clearmesh/editing/slat_encoder.py` — three-path encoder resolution, real decode via pipeline, dense occupancy rasterization.
- `clearmesh/editing/voxel_flowedit.py` — real `_compute_velocity` with manual CFG, `_encode_image_condition` via `pipeline.get_cond`, real `auto_detect_edit_mask` with 2D→3D projection, real `_silhouette_guidance`.
- `clearmesh/editing/slat_repaint.py` — real `_generate_features` via `pipeline.sample_shape_slat`, true source-trajectory replay with pad/truncate fallback.
- `clearmesh/editing/easy3e.py` — shared-pipeline constructor, lazy pipeline property, repair wrapped in try/except.
- `clearmesh/editing/image_edit.py` — `_render_view` now uses pyrender(EGL) → nvdiffrast → pyglet fallback chain.

### Scope decisions worth reviewing when you return

Per your approval during planning, silhouette guidance and auto-mask projection were both implemented rather than deferred. Some specific choices:

1. **Silhouette guidance backend is voxel-soft, not nvdiffrast mesh raster.** The nvdiffrast path is partially written but requires a FlexiCubes extraction each ODE step (~1s × 25 steps). Voxel-soft projects voxels as Gaussian splats using `||x_t||` as occupancy — differentiable, fast, CPU-testable. Good enough for the primary edit signal (which is `v_edit`, the trajectory split). If demo quality suffers, upgrade to nvdiffrast in Phase 5.5.

2. **Camera pose** (`CanonicalCamera.trellis2_default`): assumes 40° yfov, eye at (0, 0, 2), up=+Y. The introspection script will dump what `pipeline.get_cond` actually uses — compare and adjust if needed. If projection test results look off on the pod, this is the first place to look.

3. **Encoder resolution**: Three paths tried in `_resolve_ss_encoder`. Phase 0 introspection tells us which is live. If **none work**, the fallback is the "image-proxy encoder" (render source → run `pipeline.run(img)` to get a SLAT) — this isn't implemented yet because the plan flagged it as a Phase-0-gated decision.

4. **Mesh repair**: wrapped in try/except. Edited meshes from the flow ODE often have holes; PyMeshFix rejects them. We keep the unrepaired mesh with a warning rather than failing the whole edit.

## Cost to date

- Pod active time: ~45 min at $1.789/hr ≈ $1.35
- Pod storage (stopped): ~$0.20/hr, accumulates until you resume or destroy
- Remaining sprint: Phase 7 demo is ~3 min of GPU time = $0.09, plus a few debug iterations if the pipeline doesn't call exactly how we assumed

Total projected sprint cost if no surprises: **$2-3**. With normal debug iteration: **$5-10**.

## If something doesn't fit the plan

- The introspection output is the source of truth. If `pipeline.models` doesn't contain `sparse_structure_flow_model`, the CFG math in `_compute_velocity` needs adjustment. Error message from `_resolve_ss_encoder` lists all three paths checked.
- If `pipeline.sample_shape_slat` returns something other than `{SparseTensor, Tensor}`, `_generate_features` at `slat_repaint.py:213` needs the return-type case added.
- If mesh repair rejects the edited mesh completely, check `enable_repair=False` in the demo options and verify the raw decoded mesh.

## Restarting pod and running demo — copy/paste commands

```bash
# From your laptop:
/Users/Ashar/Library/Python/3.14/bin/vastai --api-key "$VAST_API" start instance 35082988
sleep 90  # wait for boot
ssh -o UserKnownHostsFile=/tmp/vast_known_hosts -p 12988 -i ~/.ssh/id_ed25519 root@ssh3.vast.ai
# On pod:
export HF_TOKEN=hf_YOUR_TOKEN_HERE
source /opt/conda/etc/profile.d/conda.sh && conda activate trellis2
export PYTHONPATH=/workspace/TRELLIS.2:$PYTHONPATH
cd /workspace/clearmesh && git pull
python scripts/setup/inspect_trellis2.py | tee /workspace/introspection.txt
pytest -q tests/ -m "gpu and trellis2"
pytest -q tests/e2e/ -m "slow and gpu and trellis2" -v
```
