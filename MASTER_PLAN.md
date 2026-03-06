# ClearMesh Master Plan & Project Legend

> **Last updated:** Pod migration + persistent /workspace setup
> **RunPod pod:** On-demand A100 80GB (DC: US-MD-1)
> **SSH:** `ssh root@154.54.102.24 -p 17969 -i ~/.ssh/id_ed25519`
> **Activation:** `source /workspace/activate.sh` or `source /workspace/activate.sh trellis2`
> **Setup log:** `tail -20 /workspace/full_setup.log`

---

## 1. What Is ClearMesh?

A unified 3D generation system with four modes:

| Mode | Input | Output | Status |
|------|-------|--------|--------|
| **Image-to-3D** | Photo/render | Print-ready mesh | Pipeline built, Stage 2 untrained |
| **Text-to-3D** | Text prompt | Print-ready mesh | NEW — needs implementation |
| **Image-guided editing** | Mesh + edited image | Edited mesh | NEW — needs implementation |
| **Text-guided editing** | Mesh + text instruction | Edited mesh | NEW — needs implementation |

Core tech: TRELLIS.2 (coarse) + RefinementDiT Stage 2 (fine) + Easy3E editing (training-free geometry) + FlexiCubes + NDC + optional auto-rigging.

---

## 2. Data Flow — All Four Modes

```
IMAGE-TO-3D (existing):
  Image → rembg → TRELLIS.2 → O-Voxel → mesh → Stage 2 → repair → export

TEXT-TO-3D (new):
  Text → FLUX/SDXL → Image → [same as Image-to-3D above]

IMAGE-GUIDED EDITING (new, Easy3E):
  Source mesh + edit image
    → SLAT encode (TRELLIS.2 encoder) → source SLAT
    → Voxel FlowEdit (flow ODE, 25 steps, training-free) → edited SLAT structure
    → SLAT Repainting (preserve unedited, regenerate edited) → full edited SLAT
    → SLAT decode → edited mesh
    → (optional) Ctrl-Adapter texture → textured mesh
    → repair → export

TEXT-GUIDED EDITING (new):
  Source mesh + text instruction
    → render source view → InstructPix2Pix(view, text) → edit image
    → [same as Image-guided editing above]
```

---

## 3. Module Map

### Existing Modules (all implemented, code complete)

| Module | Path | Purpose |
|--------|------|---------|
| Pipeline | `clearmesh/pipeline.py` | 12-stage orchestrator |
| RefinementDiT | `clearmesh/stage2/model.py` | Stage 2 DiT (12 blocks, dim=512) |
| Training Loop | `clearmesh/stage2/train.py` | FlexiCubes-in-loop, spot resilience |
| Losses | `clearmesh/stage2/losses.py` | Chamfer, normal, edge, watertight, SDF |
| Mesh Extraction | `clearmesh/mesh/extraction.py` | NDC, FlexiCubes, MC, O-Voxel |
| Mesh Repair | `clearmesh/mesh/repair.py` | PyMeshFix + Open3D |
| Mesh Export | `clearmesh/mesh/export.py` | STL/GLB/OBJ/FBX |
| Background Removal | `clearmesh/utils/background_removal.py` | rembg wrapper |
| Scale | `clearmesh/utils/scale.py` | Miniature sizing + base + hollowing |
| PBR Textures | `clearmesh/texture/pbr.py` | PBR materials |
| Auto-Rigging | `clearmesh/rigging/auto_rigger.py` | Puppeteer/UniRig |

### Data Scripts (all implemented)

| Script | Path | Needs Modification? |
|--------|------|---------------------|
| Download Objaverse | `scripts/data/download_objaverse.py` | No |
| Filter Dataset | `scripts/data/filter_dataset.py` | No |
| Render Conditioning | `scripts/data/render_conditioning.py` | **YES — add normal maps** |
| Generate Pairs | `scripts/data/generate_pairs.py` | **YES — add SLAT saving** |
| Convert O-Voxel | `scripts/data/convert_ovoxel.py` | No |

### NEW Modules to Build

| Module | Path | Training? | Purpose |
|--------|------|-----------|---------|
| Easy3E Orchestrator | `clearmesh/editing/easy3e.py` | No | Main editing entry point |
| Voxel FlowEdit | `clearmesh/editing/voxel_flowedit.py` | No | Flow-matching ODE for structure editing |
| SLAT Repainting | `clearmesh/editing/slat_repaint.py` | No | Per-voxel feature repainting |
| SLAT Encoder | `clearmesh/editing/slat_encoder.py` | No | TRELLIS.2 SLAT encode/decode access |
| Ctrl-Adapter | `clearmesh/editing/ctrl_adapter.py` | **YES (~20 GPU-hrs)** | Normal-guided texture adapter |
| Ctrl-Adapter Train | `clearmesh/editing/train_ctrl_adapter.py` | N/A | Training loop |
| Image Edit | `clearmesh/editing/image_edit.py` | No | InstructPix2Pix wrapper |
| Text-to-3D | `clearmesh/text_to_3d/generate.py` | No | Text→Image→3D |
| Text-Guided Edit | `clearmesh/text_to_3d/text_edit.py` | No | Text instruction→3D edit |
| 6-View Renderer | `scripts/data/render_ctrl_adapter_data.py` | N/A | 6-view RGB+normal for Ctrl-Adapter |

### NEW Config Files

| Config | Purpose |
|--------|---------|
| `configs/train_ctrl_adapter.yaml` | Ctrl-Adapter training params |
| `configs/editing.yaml` | Easy3E editing defaults |
| `configs/text_to_3d.yaml` | Text-to-3D model + generation settings |

---

## 4. Phase Status & Roadmap

### Phase 1: Environment Setup — 90% COMPLETE

- [x] RunPod A100 80GB pod running
- [x] Conda envs: `trellis2`, `clearmesh`, `rigging`
- [x] TRELLIS.2 + 7 CUDA extensions built
- [x] TRELLIS.2-4B weights + DINOv3 loaded
- [x] BiRefNet patched for transformers 5.2.0
- [x] ClearMesh package installed (Kaolin, FlexiCubes)
- [x] Smoke test: image → 2.1M vertex mesh in 20s
- [ ] Verify NDC compilation (deferred)

### Phase 2: Data Preparation — IN PROGRESS

**Original tasks:**
- [x] 100K Objaverse download (running, ~5.5hr ETA)
- [ ] Filter to ~50K quality meshes
- [ ] Render conditioning views
- [ ] Generate TRELLIS.2 coarse/fine pairs (~60 GPU-hrs)
- [ ] Convert fine meshes to O-Voxel
- [ ] Build train/val manifest

**NEW modifications for editing support:**
- [ ] **2a.** Add normal map rendering to `render_conditioning.py`
- [ ] **2b.** Add SLAT latent saving to `generate_pairs.py` (via TRELLIS.2 data_toolkit `encode_shape_latent.py` / `encode_ss_latent.py`)
- [ ] **2c.** Create `render_ctrl_adapter_data.py` — render 6-view RGB + normals for 5K subset

**Other Claude's recommendation vs reality:**
| Suggestion | Verdict | Reason |
|-----------|---------|--------|
| Render normal maps | ✅ Do it | Needed for Ctrl-Adapter training |
| Save SLAT encoder outputs | ✅ Do it | Needed for editing pipeline |
| Create 5-10K editing pairs with random perturbation | ❌ Skip | Easy3E geometry editing is **training-free** — no paired editing data needed |
| Don't change Stage 2 | ✅ Correct | Easy3E is independent of Stage 2 |

### Phase 3: Stage 2 Training — NOT STARTED

- [ ] Train RefinementDiT on coarse/fine pairs (100K steps, ~150 GPU-hrs)
- [ ] Validate on held-out set
- [ ] Checkpoint selection

### Phase 4: Integration & Testing — NOT STARTED

- [ ] End-to-end Image-to-3D pipeline test
- [ ] Benchmark quality vs TRELLIS.2 alone
- [ ] Optional: PartCrafter, SuperCarver, retopology

### Phase 5: Auto-Rigging — NOT STARTED (optional)

- [ ] Install UniRig/Puppeteer
- [ ] Test rigging pipeline

### Phase 6: Easy3E 3D Editing — NOT STARTED (NEW)

**6a. SLAT Encoder Access** (~1 week)
- [ ] Build `slat_encoder.py` — wrap TRELLIS.2 data_toolkit encoders
- [ ] Test: mesh → SLAT → mesh roundtrip

**6b. Geometry Editing** (~1-2 weeks, training-free)
- [ ] Build `voxel_flowedit.py` — flow-matching ODE with silhouette guidance
- [ ] Build `slat_repaint.py` — feature repainting with edit masks
- [ ] Build `easy3e.py` orchestrator
- [ ] Test: source mesh + edit image → edited mesh

**6c. Ctrl-Adapter Texture** (~1 week + 20 GPU-hrs training)
- [ ] Build `ctrl_adapter.py` model
- [ ] Build `train_ctrl_adapter.py` training loop
- [ ] Train on 5K 6-view + normal data from Phase 2c
- [ ] Integrate into editing pipeline

**6d. Image Edit Utilities** (~2-3 days)
- [ ] Build `image_edit.py` (InstructPix2Pix wrapper)
- [ ] Test: source render + text → edited image

### Phase 7: Text-to-3D — NOT STARTED (NEW)

- [ ] Build `text_to_3d/generate.py` (FLUX/SDXL → Image → pipeline)
- [ ] Build `text_to_3d/text_edit.py` (text → InstructPix2Pix → Easy3E)
- [ ] Update `pipeline.py` with `generate_from_text()`, `edit()`, `edit_from_text()`, `edit_iterative()` methods
- [ ] Update CLI with `text`, `edit` subcommands

---

## 5. Phase 2 Specific Changes (What To Do Now)

### 5a. Modify `render_conditioning.py` — Add Normal Maps

**File:** `scripts/data/render_conditioning.py`
**What:** Add a `render_normal_map()` function that renders view-space normals as RGB colors.

```python
def render_normal_map(mesh, camera_transform, image_size=512) -> Image:
    """Render surface normal map (normals mapped [-1,1] → [0,255] RGB)."""
    mesh_copy = mesh.copy()
    # Transform normals to camera space
    cam_normals = (np.linalg.inv(camera_transform[:3,:3]) @ mesh.vertex_normals.T).T
    # Map to RGB: [-1,1] → [0,255]
    colors = ((cam_normals + 1) * 0.5 * 255).astype(np.uint8)
    mesh_copy.visual.vertex_colors = np.hstack([colors, np.full((len(colors),1), 255, dtype=np.uint8)])
    # Render using same scene setup as RGB render
    ...
```

**Output format:** `{uid}/view_000_normal.png` alongside existing `{uid}/view_000.png`
**CLI:** Add `--render-normals` flag

### 5b. Modify `generate_pairs.py` — Save SLAT Latents

**File:** `scripts/data/generate_pairs.py`
**What:** After running `pipeline.run()`, also encode the coarse mesh to SLAT using TRELLIS.2's data_toolkit.

**Approach:** Use TRELLIS.2's `encode_shape_latent.py` and `encode_ss_latent.py` via the `o_voxel` API and data_toolkit. The coarse mesh already gets converted to O-Voxel (dual grid) format — the SLAT encoding runs on that.

**Output:** `{uid}/slat.pt` containing `{'ss_latent': Tensor, 'shape_latent': Tensor}`

### 5c. New Script: `render_ctrl_adapter_data.py`

**File:** `scripts/data/render_ctrl_adapter_data.py`
**What:** Render 6 canonical views (front/back/left/right/top/bottom) at 512x512 with paired normal maps. Only for 5K subset.

**Output structure:**
```
ctrl_adapter_data/{uid}/rgb_000.png ... rgb_005.png, normal_000.png ... normal_005.png
```

**Disk:** ~12GB for 5K objects

---

## 6. Updated Project Structure

```
clearmesh/
  clearmesh/
    __init__.py
    pipeline.py                  (MODIFY: add edit/text methods)
    stage2/                      (UNCHANGED)
      model.py, train.py, losses.py
    editing/                     (NEW)
      __init__.py
      easy3e.py                  (orchestrator)
      voxel_flowedit.py          (flow ODE, training-free)
      slat_repaint.py            (feature repainting, training-free)
      slat_encoder.py            (TRELLIS.2 SLAT access)
      ctrl_adapter.py            (trainable texture adapter)
      train_ctrl_adapter.py      (training loop)
      image_edit.py              (InstructPix2Pix wrapper)
    text_to_3d/                  (NEW)
      __init__.py
      generate.py                (text → image → 3D)
      text_edit.py               (text instruction → 3D edit)
    mesh/                        (UNCHANGED)
    texture/                     (UNCHANGED)
    rigging/                     (UNCHANGED)
    utils/                       (UNCHANGED)
  scripts/data/
    render_conditioning.py       (MODIFY: add normals)
    generate_pairs.py            (MODIFY: add SLAT saving)
    render_ctrl_adapter_data.py  (NEW)
    ... (rest unchanged)
  configs/
    train_ctrl_adapter.yaml      (NEW)
    editing.yaml                 (NEW)
    text_to_3d.yaml              (NEW)
    ... (rest unchanged)
```

---

## 7. Key Technical Details

### Easy3E Architecture (from paper 2602.21499v1)

**Core insight:** Easy3E edits within TRELLIS's own latent space. The geometry editing is **training-free** — it reuses TRELLIS.2's pretrained flow-matching model. Only the Ctrl-Adapter (texture) needs training.

**SLAT = Sparse Latent = (V, {z_p})**
- `V`: Voxel structure — binary occupancy encoded by 3D VAE into continuous latent
- `{z_p}`: Per-voxel features — fused from multi-view DINOv2 embeddings

**Voxel FlowEdit equation:**
```
dx_t = M_l * v_edit(x_t, t)dt + M_l * (Gamma * xi_traj - eta * G_sil)dt
```
- `v_edit` = velocity difference between target and source flow trajectories
- `G_sil` = silhouette gradient guidance (BCE loss)
- `xi_traj` = trajectory correction (keeps state on manifold)
- `M_l` = edit mask (only modify selected region)

**SLAT Repainting:**
- Edited voxels: regenerate features conditioned on target image
- Unedited voxels: replay source trajectory to preserve identity

### TRELLIS.2 SLAT Access Points

On RunPod at `/workspace/TRELLIS.2/`:
- **`data_toolkit/encode_shape_latent.py`** — encode mesh → shape latent
- **`data_toolkit/encode_ss_latent.py`** — encode → sparse structure latent
- **`data_toolkit/dual_grid.py`** — mesh → O-Voxel (already tested, 0.5s/mesh)
- **`o_voxel.convert.mesh_to_flexible_dual_grid()`** — Python API for O-Voxel conversion
- **`o_voxel.io.write_vxz()` / `read_vxz()`** — I/O for voxel files

### Text-to-3D Approach

Two-step: Text → Image (FLUX.1-schnell, 4 steps) → existing Image-to-3D pipeline.
For text editing: Text + source render → InstructPix2Pix → edit image → Easy3E.

---

## 8. Cost Estimates

| Phase | GPU Hours | Cost |
|-------|-----------|------|
| Phase 2 (original data prep) | ~65 hrs | ~$51 |
| Phase 2a-c (editing data additions) | ~16 hrs | ~$13 |
| Phase 3 (Stage 2 training) | ~150 hrs | ~$119 |
| Phase 4 (integration) | ~10 hrs | ~$8 |
| Phase 6a-b (Easy3E editing, training-free) | ~10 hrs | ~$8 |
| Phase 6c (Ctrl-Adapter training) | ~20 hrs | ~$16 |
| Phase 7 (text-to-3D, no training) | ~5 hrs | ~$4 |
| **Total** | **~276 hrs** | **~$218** |

---

## 9. What To Do Right Now

**Immediate (this session):**
1. Download is running (100K Objaverse, ~5.5hr ETA)
2. While waiting: implement Phase 2a (normal maps in `render_conditioning.py`)
3. While waiting: create `render_ctrl_adapter_data.py` skeleton

**Next session:**
1. Filter downloaded meshes
2. Run conditioning renders (with normals)
3. Begin SLAT encoder investigation on RunPod (inspect TRELLIS.2 internals for `encode_shape_latent.py` API)

---

## 10. Verification Checklist

- [ ] Normal maps render correctly (visual spot-check)
- [ ] SLAT roundtrip: mesh → encode → decode → mesh (geometry preserved)
- [ ] O-Voxel conversion working (already verified: 0.5s/mesh, 58x compression)
- [ ] Ctrl-Adapter training converges (loss decreases)
- [ ] VoxelFlowEdit: source + edit image → plausible edited mesh
- [ ] Text-to-3D: "a red dragon" → generates reasonable 3D model
- [ ] Iterative editing: edit1 → edit2 → edit3 accumulates changes
- [ ] All exports (STL/GLB) valid after editing pipeline
