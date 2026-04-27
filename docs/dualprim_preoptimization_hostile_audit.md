# DualPrim pre-optimization hostile audit

This note is intentionally skeptical. The goal is to identify places where the
implementation can look paper-faithful at a glance but still produce softer,
blobbier, or otherwise non-paper behavior before optimization has a fair chance.

Reference paper equations are from §3.1–§3.2 and Figure 4 / Table 1 of
`2603.16133`.

## Executive summary

Current verdict after a hostile read:

- `Eq. 2`: had a real mismatch and is now fixed
- `Eq. 3`: structurally correct
- `Eq. 4/5/6`: algebraically correct
- `Eq. 7`: algebraically close, but still has implementation choices worth distrusting
- `Eq. 8/9/10/11`: broadly faithful, but some practical deviations remain

Most important finding:

- The superquadric implicit in `sq_implicit()` was previously missing the
  paper's outer `^(ε1/2)` term from Eq. 2. That changes the field shape and
  gradient before optimization even begins. This is a real pre-optimization
  error, not just a tuning difference.

## Equation-by-equation

### Eq. 2: primitive implicit surface function

Paper:

`f(p,Q) = [ (|x/a_x|^(2/ε2) + |y/a_y|^(2/ε2))^(ε2/ε1) + |z/a_z|^(2/ε1) ]^(ε1/2) - 1`

Audit result:

- Previously, the implementation omitted the outer `^(ε1/2)` entirely and used
  the inner expression directly.
- That effectively changes the field family except in special cases.
- The analytic gradient implementation was correspondingly derived for the
  wrong field.

Action taken:

- Fixed `sq_implicit()` and `sq_implicit_grad()` in
  [superquadric.py](/Users/Ashar/Documents/GitHub/clearmesh/.claude/worktrees/nervous-sammet/clearmesh/dualprim/superquadric.py)
  to include the full paper form.

Remaining skepticism:

- We still use `_safe_positive_pow()` and explicit clamps for stability.
- That is not paper-literal, but it is a bounded deviation rather than the
  previous structural mismatch.

### Eq. 3: local frame transform

Paper:

`p' = R_Q^-1 (p - T_Q)`

Audit result:

- Structurally correct.
- The code uses a batched Euler-to-matrix helper and applies `R^T` as the
  inverse for orthogonal rotations.

Remaining skepticism:

- The exact Euler convention is an implementation choice, but nothing here
  currently looks obviously wrong relative to the paper text.

### Eq. 4: effectiveness probability `P_E`

Paper:

`P_E = Φ(-f(p,PSQ)/θ - μ) * Φ(-f(p,NSQ)/θ - μ)`

Audit result:

- Algebraically correct in code.
- `θ` is genuinely in the live path and is not missing.

Remaining skepticism:

- `μ` default has often been `0.0`, while the paper describes it as a "small offset".
- Our historical `θ` handling added a floor/curriculum that is not specified in
  the paper and can materially change gate sharpness.
- A new `paper_literal` gate mode now exists to reduce those extra behaviors.

### Eq. 5: combined field

Paper:

`f(p,S) = f(p,PSQ) * (1 - P_E) - f(p,NSQ) * P_E`

Audit result:

- Algebraically correct in code.

Remaining skepticism:

- If Eq. 2 or Eq. 4 are off, Eq. 5 inherits those problems even though its own
  algebra is fine.

### Eq. 6: combined normal

Paper:

`n(p,S) = Normalize(f'(p,PSQ)) * (1 - P_E) - Normalize(f'(p,NSQ)) * P_E`

Audit result:

- Algebraically correct in code.

Remaining skepticism:

- The quality of this term depends heavily on the correctness of the Eq. 2
  gradient and on how sharp `P_E` actually is in practice.

### Eq. 7: density from field

Paper:

`σ_S(p) = max((Φ(f(p+Δp,S)/θ) - Φ(f(p-Δp,S)/θ)) / Φ(f(p+Δp,S)/θ), 0)`

Audit result:

- The formula is implemented in the correct forward/backward-point form.

Remaining skepticism:

- `Δp` is a fixed implementation choice (`0.01`) and not clearly paper-specified.
- We clamp the final density ratio to `[0, 1]`, which is numerically safe but
  stricter than the paper text.
- The same `θ` floor / gate-mode choice also affects Eq. 7 through the CDF.

### Eq. 8: color

Audit result:

- Broadly faithful.
- Basic color is blended by density share and an MLP residual is added.

Remaining skepticism:

- The lighting residual uses detached normals as input, which is a practical
  stability choice rather than a paper-literal requirement.
- Our implementation currently bakes primitive transparency `α_k` into the
  per-primitive density before the color blend:
  `c(p) = Σ_k c_basic,k * (α_k σ_k) / Σ_j (α_j σ_j) + C(p)`.
- The paper notation in Eq. 8 writes the numerator as `σ_{S_k}(p) / σ(p)`
  rather than explicitly `α_k σ_{S_k}(p) / σ(p)`.
- That notation is ambiguous because the text defines point density using
  primitive transparency, but a hostile reading is still fair here:
  alpha-weighted color blending may let high-alpha primitives dominate local
  appearance and hide sharper local alternatives.

### Eq. 9 / Eq. 10: volumetric mask and normal accumulation

Audit result:

- Broadly faithful alpha compositing.

Remaining skepticism:

- The rendering chain includes a few stability-oriented clamps and `nan_to_num`
  guards that are reasonable in practice but not paper-literal.
- Our final rendered normal is explicitly normalized after volumetric
  accumulation. Eq. 10 itself only writes an accumulated sum
  `Σ_i T_i α_i n_i` and does not explicitly include a post-normalization step.
- That normalization is numerically convenient, but it can also suppress the
  magnitude information that indicates uncertain or mixed normals, which may
  soften supervision and hide sharp seams.

### Eq. 11: point normal aggregation across primitives

Audit result:

- Broadly faithful.

Remaining skepticism:

- The numerator effectively uses alpha-weighted primitive density in practice.
- That is physically reasonable, but the notation in the paper is compact enough
  that exact literal correspondence is somewhat ambiguous.
- More concretely, the current implementation computes point normals as
  `Σ_k n_k * (α_k σ_k) / Σ_j (α_j σ_j)`.
- A more literal reading of Eq. 11 is
  `Σ_k n_k * σ_k / Σ_j σ_j`, with primitive transparency already handled in the
  separate point-density / transmittance chain.
- If alpha weighting is too aggressive here, it can blur local normals toward
  large saturated primitives and make the rendered supervision less sharp than
  the paper intends.

## Highest-priority remaining suspects

After fixing Eq. 2, the main pre-optimization suspects are:

1. `θ` handling around Eq. 4 and Eq. 7
2. `μ` choice in Eq. 4
3. `Δp` choice in Eq. 7
4. density clamp and other safety behavior in the renderer
5. view-dependent filtering aggressiveness before/around pruning
6. alpha-weighted blending in Eq. 8 / Eq. 11
7. forced post-normalization after Eq. 10

## New adversarial note on Eq. 7

The current renderer historically used a fixed `Δp = 0.01` for every ray sample.

That is suspicious because:

- the renderer already knows the local sample spacing `δ_i`
- the paper writes Eq. 7 in terms of `p ± Δp`, not in terms of one global
  scene-wide constant
- a fixed step can mis-calibrate density when the actual ray spacing changes

Action taken:

- added a `delta_p_mode` switch with:
  - `fixed`: legacy behavior
  - `half_delta`: use half the local ray spacing per sample

This is the next clean pre-optimization probe after the Eq. 2 fix.

## New adversarial note on Eq. 8 / Eq. 10 / Eq. 11

The current renderer makes three practical choices that are plausible, but not
obviously paper-literal:

1. `Eq. 8` color blending uses alpha-weighted primitive density.
2. `Eq. 11` point-normal blending uses alpha-weighted primitive density.
3. `Eq. 10` final rendered normals are normalized after volumetric accumulation.

All three choices can bias the render toward smoother, more saturated dominant
primitives. That is exactly the kind of subtle pre-optimization behavior that
could preserve coarse structure but wash out local sharpness.

Action taken:

- added hostile-audit switches for:
  - `color_weight_mode = alpha_density | density_only`
  - `point_normal_weight_mode = alpha_density | density_only`
  - `final_normal_normalize = true | false`

These are intended for short controlled probes from the same resume base as the
Eq. 2 / Eq. 7 audit.

## Working hypothesis

The blobbiness is probably not caused by one missing algebraic line anymore.

The most likely remaining explanation is a combination of:

- the historically incorrect Eq. 2 field shape,
- the gate sharpness regime (`θ`, `μ`, floor/curriculum),
- and the fact that plain superquadrics are still limited on local sharp detail.

## Post-render hostile audit: Eq. 12–18

This section is intentionally adversarial about the *loss* side too, because
even with correct geometry/render equations, a slightly off objective can keep
the model in a softer basin than the paper.

### Eq. 12: total objective

Paper:

`L = L_rgb + λ_mask L_mask + λ_sp L_sp + λ_e L_e + λ_max L_max + λ_norm_reg L_norm_reg`

Audit result:

- Structurally present in
  [losses.py](/Users/Ashar/Documents/GitHub/clearmesh/.claude/worktrees/nervous-sammet/clearmesh/dualprim/losses.py).
- The paper terms are all there.

Remaining skepticism:

- The actual `λ` values are still implementation choices rather than paper
  values. That means Eq. 12 is algebraically faithful but not numerically
  paper-locked.
- We also optionally add `loss_open_ray`, which is not in the paper. It is
  usually disabled for paper-mode runs, but it exists in the same total-loss
  path.

### Eq. 13: RGB reconstruction

Paper:

`L_rgb = Σ ||I_render − I_gt|| · M_gt`

Audit result:

- Implemented as masked L1.

Remaining skepticism:

- The current implementation uses `.mean()` over all sampled rays after masking,
  not an explicit normalization by foreground-pixel count.
- That is a subtle but real scaling choice. If foreground occupancy changes,
  the effective strength of Eq. 13 relative to Eq. 14 / Eq. 18 also changes.

### Eq. 14: mask loss

Paper:

`L_mask = BCE(M_render, M_gt)`

Audit result:

- Present.

Remaining skepticism:

- The implementation uses an epsilon-clamped BCE for stability.
- That is practical, but it means the gradient scale can become extremely large
  near `M_render ≈ 1` on background rays.
- We also maintain an `mse` fallback path, which is explicitly not paper
  behavior and should be treated as such in any audit or reproduction claim.

### Eq. 15 / Eq. 16 / Eq. 17: primitive-alpha regularizers

Paper:

- `L_sp = (1/K) Σ α(p)`
- `L_e  = -(1/K) Σ [α log α + (1-α) log(1-α)]`
- `L_max = (1/K) Σ ReLU(α-1)`

Audit result:

- All three are present and act on per-primitive alpha, which is correct.

Remaining skepticism:

- The implementation averages over the alive set rather than the fixed initial
  `K`.
- That is defensible in practice, but it is not a literal reading of the paper.
- This changes the incentive after pruning: regularization pressure stays strong
  on the surviving set instead of naturally diluting as primitives die.

### Eq. 18: normal regularization

Paper:

`L_norm_reg = Σ ||N_render − N_pred|| · M_gt`

Audit result:

- Present as masked L1 on normals.

Remaining skepticism:

- As with Eq. 13, the implementation uses `.mean()` over all rays after
  masking, not an explicit normalization by foreground count.
- The exact meaning of `N_render` depends on the renderer-side choices from
  Eq. 10 / Eq. 11:
  - alpha-weighted vs density-only point-normal blending
  - optional post-normalization of rendered normals
- That means Eq. 18 can look paper-faithful while still training against a
  softer normal target than the paper intended.

## Highest-priority remaining suspects after Eq. 12–18 read

1. `Eq. 10 / Eq. 11` renderer normal path still appears smoothing-sensitive
2. `Eq. 13 / Eq. 18` use masked means rather than an explicit foreground-normalized sum
3. `Eq. 15 / Eq. 16 / Eq. 17` average over alive primitives instead of fixed `K`
4. `Eq. 14` epsilon-clamped BCE may be materially changing gradient scale

## Status

- Eq. 2 fix: implemented
- Gate-regime sweep: already showed `paper_literal` helps modestly
- Next honest step: rerun from the best paper-literal gate setting and measure
  whether the corrected Eq. 2 field gives another tangible jump
