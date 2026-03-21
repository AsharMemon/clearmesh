# ClearMesh — Technical Learnings & Realizations

> **Last updated:** 2026-03-12 (Stage 2 training run at step ~29K/100K)
> **Pod:** Vast.ai H100 SXM 80GB ($1.10/hr), `ssh -p 17590 root@ssh2.vast.ai`
> **Training log:** `/workspace/train_mlp_head.log`

This document captures hard-won technical insights from building ClearMesh Stage 2 — a DiT-based SDF refinement model that takes TRELLIS 2's coarse SLAT features and produces refined signed distance fields.

---

## 1. Architecture — RefinementDiT

### 1.1 What Works
- **527.5M parameters**: 12 DiT blocks (first 12 of TRELLIS 2's 30), dim=1536, 12 heads
- **Pretrained initialization**: Loading TRELLIS 2's shape transformer weights gives a massive head start. Without it, the model barely learns.
- **Epsilon prediction**: The model predicts noise (epsilon), not x0 or velocity. This is the standard choice for DDPM-family models and works well with the linear schedule.
- **Noise schedule**: `alpha(t) = 1 - t`, so `noisy_sdf = sqrt(1-t) * sdf + sqrt(t) * noise`. Simple and effective.
- **MLP output head**: `nn.Sequential(LayerNorm(1536), Linear(1536, 256), GELU(), Linear(256, 1))` — critical for preventing weight collapse (see §3).

### 1.2 Model Forward Signature
```python
forward(coarse_voxels,    # (B, N, 32) — SLAT features from TRELLIS 2
        positions,         # (B, N, 3)  — voxel coordinates
        timestep,          # (B,)       — diffusion timestep t ∈ [0, 1)
        cond_features,     # (B, M, 1024) or None — DINOv2/v3 image features
        cond_mask,         # (B, M) or None — foreground token mask
        noisy_sdf)         # (B, N, 1)  — noisy SDF at timestep t
→ predicted_noise          # (B, N, 1)
```

### 1.3 Inference Modes
Two inference modes are available:
1. **`refine()`** — Full DDIM sampling (50 steps, t: 0.99→0). Uses x0 clipping. More diverse but slower.
2. **`predict_x0()`** — Single-step direct prediction. Starts from random noise at low t, predicts x0 in one pass. Faster, more stable, less diverse.

---

## 2. DINOv2 / DINOv3 Conditioning

### 2.1 What It Is
TRELLIS 2 uses a DINOv2 (or DINOv3) Vision Transformer to encode the input image into per-patch feature tokens. These are 1024-dimensional vectors (ViT-L), typically 257 tokens (256 patches + 1 CLS token). They capture rich semantic and geometric information about the input image.

### 2.2 How We Use It
- **Cross-attention**: Each DiT block has a cross-attention layer that attends from voxel features to DINOv2 tokens. This lets the model condition its SDF predictions on the input image.
- **Optional**: The model works without conditioning (cond_features=None), but quality improves with it. During training, only pairs that have saved `cond_features.npy` get conditioning.
- **Current gap**: Most of our 34K training pairs do NOT have cond_features saved (only recent pair generation saves them). So the model is mostly learning unconditional refinement. Future data generation should always save these.

### 2.3 DINOv3 vs DINOv2
- TRELLIS 2 uses `dinov3-vitl16` (a gated HuggingFace model, mirror: `tao-hunter/dinov3-vitl16-pretrain-lvd1689m`)
- DINOv3 is trained on LVD-1.689M (a curated 1.7M image dataset) and outputs 1024-dim features, same as DINOv2-ViT-L
- Our `cond_dim=1024` works for both DINOv2 and DINOv3
- The model stored in `pipeline.json` is patched to use a local path during Vast.ai deployment
- **Key realization**: When we scale data with the 50K pilot, we MUST save cond_features for every pair. This is the biggest potential quality unlock — image-conditioned refinement should dramatically outperform unconditional.

---

## 3. Weight Collapse — The Hidden Killer

### 3.1 The Problem
With a single `nn.Linear(1536, 1)` output layer, the model's output weights collapse to near-zero (std ≈ 0.001) during training. The model learns to predict near-constant output regardless of input, because:
- The pretrained backbone produces features with magnitudes around 0.1-1.0
- A single linear layer projecting 1536→1 is overwhelmed by the large input dimension
- Gradient flow through the single weight vector is too constrained
- The model "solves" the loss by outputting a constant (the mean noise, which is ~0)

### 3.2 The Fix — MLP Head
```python
self.out_head = nn.Sequential(
    nn.LayerNorm(1536),        # Normalize backbone features
    nn.Linear(1536, 256),      # Bottleneck
    nn.GELU(),                 # Non-linearity
    nn.Linear(256, 1),         # Final scalar
)
```
With this, `out_head[1].weight` std stabilizes at ~0.021 (not 0.001). The LayerNorm is critical — it normalizes the backbone features before the projection, preventing scale mismatch.

### 3.3 How to Detect
Monitor `out_head` weight std during training:
```python
for i, layer in enumerate(model.out_head):
    if hasattr(layer, 'weight'):
        print(f"out_head[{i}]: std={layer.weight.float().std():.6f}")
```
If std drops below 0.005, weight collapse is happening.

---

## 4. SDF Scale Mismatch

### 4.1 The Problem
Raw SDF values from the fine meshes have tiny magnitudes (typically std ≈ 0.04-0.09). But diffusion noise has std ≈ 1.0. When training:
```
noisy_sdf = sqrt(alpha) * sdf + sqrt(1-alpha) * noise
```
The noise completely dominates the signal. The model learns "just predict the noise, ignore the SDF" because the SDF contribution is 10-25x smaller than the noise.

### 4.2 The Fix
Scale SDF by `sdf_scale=10.0` during data loading:
```python
gt_sdf = torch.from_numpy(np.load("fine_sdf.npy")).float() * self.sdf_scale
```
This puts SDF values into the same magnitude range as the noise, allowing the model to learn the relationship between them.

### 4.3 At Inference
Remember to divide by `sdf_scale` at the end of DDIM sampling:
```python
return sdf / sdf_scale  # Convert back to real SDF units
```

---

## 5. DDIM Sampling Failures — Numerical Divergence

### 5.1 The Problem
Even when the model has excellent noise prediction quality (0.88-0.98 correlation), DDIM sampling can produce completely wrong outputs (correlation ≈ 0, SDF values in ±30 range vs expected ±0.5).

### 5.2 Root Cause — Error Amplification at High t
The x0 reconstruction formula:
```
x0_pred = (x_t - sqrt(1-alpha_t) * eps_pred) / sqrt(alpha_t)
```
At high timestep t (near 1.0), `alpha_t ≈ 0`, so we divide by `sqrt(alpha_t) ≈ 0`. This amplifies ANY error in noise prediction by a factor of `1/sqrt(alpha_t)`:
- At t=0.9: errors amplified ~3x
- At t=0.99: errors amplified ~10x
- At t=0.999: errors amplified ~31x
- At t=0.9999: errors amplified ~100x

These amplified errors feed into the next DDIM step, creating a cascade that blows up the prediction within a few steps.

### 5.3 The Fixes (Applied)
1. **x0 clipping**: Clamp `x0_pred` to `[-5.0, 5.0]` (in scaled SDF space) after each DDIM step. This bounds the error propagation.
2. **Lower t_max**: Start DDIM from `t_max=0.99` instead of `t_max=1.0-1e-4`. Avoids the most numerically unstable region.
3. **`predict_x0()` method**: Bypass DDIM entirely — start from noise at low t (e.g., 0.05), predict x0 in a single forward pass. This gives 0.995 correlation at step 1000 and avoids error accumulation.

### 5.4 Detection
If DDIM output has SDF values > ±5.0 (when expected range is ±0.5), or if correlation between prediction and GT is near 0 or negative, DDIM has diverged. Try:
- Increasing x0_clip from 5.0 to 10.0
- Reducing num_steps from 50 to 20 (fewer steps = less accumulation)
- Using `predict_x0()` instead

### 5.5 Future Consideration — Alternative Noise Schedules
The linear schedule `alpha(t) = 1-t` has a harsh SNR curve (SNR goes to 0 as t→1). A cosine schedule `alpha(t) = cos²(πt/2)` or a shifted schedule could improve DDIM stability. This is a potential future improvement but requires retraining.

---

## 6. Differential Learning Rate

### 6.1 Why It Matters
The model has two types of parameters:
- **Pretrained backbone** (260 tensors, 527M params): Already well-initialized from TRELLIS 2. Needs gentle fine-tuning.
- **Fresh heads** (8 tensors): `sdf_proj` and `out_head` are randomly initialized. Need faster learning.

### 6.2 Configuration
```python
fresh_names = {"sdf_proj", "out_head"}
optimizer = AdamW([
    {"params": pretrained_params, "lr": 5e-5},      # Backbone: slow
    {"params": fresh_params,      "lr": 2.5e-4},    # Heads: 5× faster
], weight_decay=0.01)
```

### 6.3 Critical Bug (Fixed)
When the output layer was renamed from `out_layer` to `out_head` (for the MLP), `fresh_names` in `train.py` still said `{"sdf_proj", "out_layer"}`. This meant `out_head` was trained at the SLOW backbone LR (5e-5) instead of the fast LR (2.5e-4), causing extremely slow convergence. Always verify `fresh_names` matches the actual model attribute names.

---

## 7. Data Quality — Outlier Filtering

### 7.1 The Problem
Of 38,036 generated training pairs, 3,205 (8.4%) had anomalously high SDF standard deviation (std > 0.5). These represent degenerate meshes, failed TRELLIS generations, or misaligned coordinate systems. Training on these corrupts the loss landscape.

### 7.2 The Fix
Filter during dataset loading:
```yaml
max_sdf_std: 0.5  # In train config
```
```python
# In dataset __init__:
if max_sdf_std and sdf_std > max_sdf_std:
    skip_pair()  # 34,831 clean pairs remain
```

### 7.3 Broader Lesson for the Filter Pipeline
The 8-layer filter pipeline (`filter_pipeline.py`) covers pre-pair-gen quality (Layers 0-5) but the most impactful filter turned out to be a simple post-pair-gen SDF std check (effectively part of Layer 7). When scaling to 50K+ candidates:
- **Layer 7 (pair value scoring)** should include SDF std, range, and histogram checks
- Models that produce SDF outliers often have: non-manifold geometry, extreme aspect ratios, or very thin structures that TRELLIS can't handle well
- Consider adding SDF histogram entropy as a quality signal — good pairs have smooth, unimodal SDF distributions; bad ones are often bimodal or heavy-tailed

---

## 8. spconv Key Mismatch — TRELLIS 2 Decoder

### 8.1 The Problem
When loading TRELLIS 2's FlexiDualGridVaeDecoder (shape_dec_next_dc, 474M params), the checkpoint keys don't match the model because spconv wraps convolution layers with an extra `.conv` layer:
- **Checkpoint**: `blocks.0.0.conv.weight`
- **Model expects**: `blocks.0.0.conv.conv.weight`

### 8.2 The Fix
Regex-based key remapping:
```python
remapped = {}
for k, v in state.items():
    new_k = re.sub(r'(\.conv[12]?)\.weight$', r'\1.conv.weight', k)
    new_k = re.sub(r'(\.conv[12]?)\.bias$', r'\1.conv.bias', new_k)
    remapped[new_k] = v
```

### 8.3 Remaining Issue — spconv Upsampling
The decoder loads successfully but fails during forward pass in the upsampling blocks (SparseResBlockC2S3d) with `can't find suitable algorithm for 0` in spconv's implicit GEMM kernel. This happens specifically on the `conv1` in the upsampling block (1024→4096 channels, 3x3x3 kernel). Disabling autocast doesn't help. torchsparse is not installed as an alternative.

**Impact**: We cannot compute coarse_sdf from the decoder for residual prediction. Training uses absolute SDF targets instead. Residual prediction (delta_sdf = fine_sdf - coarse_sdf) is deferred.

---

## 9. Overfit Test — Lessons Learned

### 9.1 Expected Behavior
With only 2 training samples and a 527M model:
- **Training loss drops to near zero** (the model memorizes the specific noise patterns for those 2 samples)
- **But DDIM sampling fails** because the model learned the specific noise vectors, not the denoising function
- **Noise prediction correlation is mediocre** (~0.21-0.56) because the model overfits to specific noise instances, not the noise distribution

### 9.2 What This Tells Us
An overfit test with 2 samples is useful for verifying:
- ✅ Weight collapse is fixed (out_head std stays > 0.01)
- ✅ Training loop mechanics work (loss decreases)
- ✅ Model can fit data (loss → 0)
- ❌ Does NOT validate DDIM quality (need diverse training data)
- ❌ Does NOT validate generalization (by definition)

### 9.3 Full Training Validation
With 34K+ training pairs, noise prediction quality jumps dramatically:
- Step 1000: noise_corr=0.88 at t=0.1 (vs 0.21 with overfit)
- Step 1000: x0_corr=0.995 at t=0.01 (near-perfect reconstruction!)
- This confirms the architecture IS correct; the overfit test's poor DDIM was purely due to memorization.

---

## 10. Progressive Token Schedule

### 10.1 How It Works
```yaml
progressive_schedule:
  - step: 0
    num_tokens: 2048    # Start small, fast iteration
  - step: 30000
    num_tokens: 4096    # Medium resolution
  - step: 70000
    num_tokens: 8192    # Full resolution
```

Each training pair has N voxels (typically 800-8000+). At each step, we randomly subsample to `num_tokens` voxels. This:
- **Speeds up early training** (2048 tokens = 4x faster than 8192)
- **Allows global structure learning first** (sparse sampling sees the whole shape)
- **Then fine-grained detail** (dense sampling captures local variations)

### 10.2 Expected Behavior at Transitions
When num_tokens increases at step 30K (2048→4096):
- Expect a **temporary loss spike** — the model now sees more voxels per step, revealing prediction errors at finer granularity
- Loss should recover within ~2K steps as the model adapts
- This is normal and expected

---

## 11. Training Loss Interpretation

### 11.1 Loss Components
```
total = noise_mse + sdf_loss
```
- **noise_mse**: MSE between predicted and actual noise. This is the primary diffusion loss.
- **sdf_loss**: Direct SDF prediction loss (x0 reconstruction quality). Secondary metric.

### 11.2 Typical Loss Curve (34K pairs, H100)
| Step | noise_mse | total | Notes |
|------|-----------|-------|-------|
| 100 | 0.69 | 0.89 | Random initialization |
| 500 | 0.45 | 0.67 | Rapid initial learning |
| 1000 | 0.06 | 0.06 | Backbone features kicking in |
| 5000 | 0.07 | 0.13 | Noisy but trending down |
| 10000 | 0.05 | 0.08 | Stabilizing |
| 25000 | 0.03 | 0.06 | Good convergence |
| 29000 | 0.015 | 0.02 | Still improving |

### 11.3 Noise in Loss Values
The loss is inherently noisy because:
1. Random timestep t per batch (some t values are harder than others)
2. Small batch size (4)
3. Different pairs have different difficulty levels
4. Random voxel subsampling (progressive schedule)

Don't worry about step-to-step fluctuations. Look at the **trend over 5K-10K steps**.

---

## 12. Infrastructure Notes

### 12.1 Vast.ai Pod Configuration
- **GPU**: H100 SXM 80GB ($1.10/hr) — essential for 527M model with bf16
- **Training speed**: ~6.17 it/s (batch_size=4, num_tokens=2048)
- **100K steps ETA**: ~4.5 hours
- **Checkpoint size**: ~5.5GB per checkpoint (model + optimizer state)
- **Checkpointing**: Every 1000 steps, keep latest + milestone checkpoints

### 12.2 Memory Budget
- Model (bf16): ~1.1GB
- Optimizer states: ~4.4GB (Adam has 2 states per param)
- Activations (with gradient checkpointing): ~3-5GB
- DataLoader workers: ~2GB
- **Total**: ~12-15GB training, ~65GB available for batch/tokens
- **Evaluation during training**: NOT possible (OOM). Must wait for training to finish.

### 12.3 Data Storage
- Training pairs: `/workspace/data/training_pairs/` (34,831 pairs, ~25GB)
- Each pair: `coarse_voxels.npy` (N×32), `positions.npy` (N×3), `fine_sdf.npy` (N×1)
- Optional: `cond_features.npy` (M×1024) — DINOv2/v3 image features
- Sharded into `shard_0/` through `shard_N/` for parallel generation

---

## 13. Catastrophic Failure Modes (Ranked by Severity)

### 13.1 Weight Collapse (CRITICAL — Training Appears to Work But Model is Broken)
- **Symptom**: Training loss decreases. Model outputs near-constant predictions.
- **Detection**: Check `out_head` weight std < 0.005
- **Fix**: MLP head with LayerNorm (§3)
- **Danger**: This is silent — loss still goes down because predicting mean noise is a reasonable strategy. You won't notice until evaluation.

### 13.2 SDF Scale Mismatch (CRITICAL — Model Ignores SDF Signal)
- **Symptom**: Loss plateaus early (~1.0). Noise prediction is random.
- **Detection**: Check if raw SDF std ≪ 1.0. If std < 0.1, scaling is needed.
- **Fix**: `sdf_scale=10.0` (§4)
- **Danger**: Easy to miss if you don't check the raw data statistics.

### 13.3 DDIM Divergence (HIGH — Good Model, Bad Inference)
- **Symptom**: Excellent noise prediction but DDIM produces garbage (SDF values ±30 vs expected ±0.5).
- **Detection**: Compare noise_corr (should be high) vs DDIM output range (should match GT range).
- **Fix**: x0 clipping + lower t_max (§5), or use `predict_x0()` instead.
- **Danger**: You might blame the model when the issue is purely in the sampling procedure.

### 13.4 fresh_names Mismatch (MEDIUM — Subtle Training Slowdown)
- **Symptom**: Training converges but very slowly. Fresh heads aren't getting the higher LR.
- **Detection**: Check that `fresh_names` in train.py matches actual model attribute names.
- **Fix**: Update fresh_names when renaming layers (§6.3)

### 13.5 Outlier Training Data (MEDIUM — Noisy Gradients)
- **Symptom**: Loss is noisy with occasional spikes 10x above mean.
- **Detection**: Check SDF std distribution across pairs. Look for bimodal or heavy-tailed distributions.
- **Fix**: `max_sdf_std=0.5` filter (§7)

### 13.6 Overfit Test Misinterpretation (LOW — Wastes Time)
- **Symptom**: Overfit on 2 samples shows poor DDIM. You think the architecture is broken.
- **Reality**: 527M model memorizes specific noise instances, not the denoising function. DDIM failure on 2 samples is EXPECTED. (§9)
- **Fix**: Don't judge DDIM quality from overfit tests. Use full training for that.

---

## 14. Next Steps & Open Questions

### 14.1 Immediate (Current Training Run)
- [ ] Training completes at ~100K steps (~00:21 UTC)
- [ ] Auto-eval runs: 20-pair evaluation with noise prediction + DDIM (with x0 clip)
- [ ] Cup mesh generated via TRELLIS 2 → Stage 2 → marching cubes

### 14.2 Architecture Experiments (If Current Run Succeeds)
- [ ] **Residual prediction**: Train on `delta_sdf = fine_sdf - coarse_sdf` instead of absolute SDF. Requires solving spconv decoder issue or alternative coarse_sdf computation.
- [ ] **Cosine noise schedule**: `alpha(t) = cos²(πt/2)` may improve DDIM stability without needing x0 clipping.
- [ ] **Velocity prediction** (v-prediction): Instead of predicting noise, predict `v = sqrt(alpha)*noise - sqrt(1-alpha)*x0`. Better behaved at t close to 0 and 1.
- [ ] **Conditioning utilization**: Current training mostly lacks cond_features. Re-generating pairs WITH DINOv2/v3 features should improve quality substantially.

### 14.3 Data Scaling (Phase B)
- [ ] Build 50K candidate pool via `build_candidate_pool.py`
- [ ] Deploy 8 Vast.ai pods for parallel pair generation
- [ ] All pairs MUST save `cond_features.npy`
- [ ] Target: 25K-35K high-quality pairs (50-70% success rate)
- [ ] Re-run training with larger dataset

### 14.4 Open Questions
1. **Is absolute SDF or residual SDF better?** We can't test residual until the spconv decoder is fixed.
2. **How much does DINOv2/v3 conditioning improve quality?** Need pairs with cond_features to test.
3. **What's the optimal training duration?** 100K steps may be overkill or insufficient — depends on eval results.
4. **Should we use classifier-free guidance during inference?** Train with random conditioning dropout, then use guidance scale > 1 at inference for sharper outputs.

---

## 15. Quick Reference — Key Paths

| Item | Path |
|------|------|
| Model code | `clearmesh/stage2/model.py` |
| Training code | `clearmesh/stage2/train.py` |
| E2E inference | `clearmesh/stage2/infer_e2e.py` |
| Training config | `configs/train_stage2_vast.yaml` |
| Training log | `/workspace/train_mlp_head.log` (on pod) |
| Checkpoints | `/workspace/checkpoints/clearmesh_stage2/` (on pod) |
| Training pairs | `/workspace/data/training_pairs/` (on pod) |
| Eval script | `/workspace/eval_full_training.py` (on pod) |
| Auto-eval log | `/workspace/auto_eval.log` (on pod) |
| Cup mesh output | `/workspace/e2e_results_final/` (on pod) |
| TRELLIS 2 model | `/workspace/models/trellis2-4b/` (on pod) |
| Pretrained weights | `slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` |
| Shape decoder | `ckpts/shape_dec_next_dc_f16c32_fp16.safetensors` |

---

## 16. Session History

| Session | Key Achievement |
|---------|----------------|
| 1 | Environment setup, TRELLIS 2 installation |
| 2 | Pair generation pipeline, initial training |
| 3 | SDF scale fix (10x), first successful training |
| 4 | Weight collapse diagnosed, single linear head fails |
| 5 | MLP head implemented, weight collapse solved |
| 6 | Overfit test (2 samples), DDIM instability diagnosed |
| 7 | Full training launched (34K pairs), DDIM x0 clipping fix |
| 8 (current) | Training at 29K/100K, predict_x0() method, auto-eval setup |
