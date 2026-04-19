# DualPrim-to-Product Unified Plan

**Status:** Gate 0 in flight (Phase 2 running on pod 35076678).
**Goal:** ship paper-quality structured-mesh refinement with consumer unit economics.
**Guiding principle (user mandate):** maximize quality. Don't compromise iters/resolution below what the paper uses without empirical evidence that a cut is safe.

---

## Product taxonomy

Three distinct components, routinely conflated:

| Component | What it is | Generalization | Speed |
|---|---|---|---|
| **DualPrim solver** | Per-scene optimizer. Paper algorithm. | Fully general (any mesh). | 90 min–4 hrs / mesh |
| **Neural warm-start head** | Learned mesh → initial primitive params. | Bounded by training data distribution. | 1–5 sec |
| **Short refiner** | ≤1000 DualPrim iters from warm start. | Inherits full generality from solver. | ~60–120 sec |

Product path: `mesh → head (fast) → short refiner (general) → primitives`.

---

## Gates (must pass in order; no gate-skipping)

### Gate 0 — DualPrim itself works

**Measurable:** Phase 2 (K=100, 15k iters, hole + window_box, lambda_mask=3, fg_bias=0.7, independent NSQ init, θ curriculum) produces:
- `through-hole-open %` > 0 on the hole canary
- Mean mask IoU ≥ 0.80 on hole + window_box
- Chamfer ×1000 ≤ 30 on both (vs paper's ~7.94; factor-of-4 close is acceptable at this K)

**If this fails:** no further work proceeds. The supervision/optimization is still broken and every downstream gate is a waste.

**Timeline:** Phase 2 completes in ~3 hrs from launch. Currently ~30 min in.

### Gate 0.1 — Iter count and resolution are evidence-based

**Measurable:** controlled comparison on a single canary (`hole`) at:
- `iters ∈ {5000, 10000, 15000, 30000}` at fixed `res=192`
- `res ∈ {128, 192, 256}` at fixed `iters=15000`

Metric: hole_metric + CD×1000 + wall time. Pick the (iters, res) that matches paper-level quality at minimum runtime.

**Why:** we've been at 15k/192 without proving it matches 30k/256. User's quality mandate requires we prove we can't do worse without hurting quality — or we pay the 30k/256 cost.

**Timeline:** 1 GPU-day. Runs in parallel with Phase 2.

### Gate 0.5 — Warm-start hypothesis is empirically sound

**Measurable:** using `--resume-primitives` (shipped in commit `17d8502`), measure:
- Given a trajectory snapshot at step 10,000, how many additional iters bring the scene to the same quality as a full 30k run from scratch?
- Does the answer hold across ≥5 different meshes and ≥2 categories?
- Is the `number_of_refinement_iters × per_iter_time` within the product budget (~60-120s end-to-end)?

**Why:** the entire Tier B business case is "warm start reduces iters 30×." If the true reduction is 3×, the unit economics don't work.

**Timeline:** 1-2 GPU-days, runs after Gate 0 passes.

### Gate 1 — Teacher corpus collected

**Measurable:** `N` diverse meshes, 2 seeds each, 5 trajectory snapshots per seed, all with `metrics.json` passing a quality filter (CD ≤ threshold, hole_metric ≥ threshold, train_ok).

**Scope decisions pending Gate 0.1 result:**
- N = 300 meshes if iters=15k (budget ~$600, 5 days on 8 pods)
- N = 200 meshes if iters=30k (budget ~$1600, 12 days on 8 pods)

Source: Objaverse-LVIS, filtered to hard-surface compositional categories (`furniture, container, tool, vehicle, utensil, appliance, electronics, sport`). Sampler committed in `scripts/dualprim/objaverse_sampler.py`.

### Gate 2 — Warm-start head trains to useful quality

**Measurable:**
- Head outputs predicted primitives with shape-matching loss ≤ 2× teacher's.
- When warm-started and refined 1000 iters, final mesh has hole_metric + CD within 10% of teacher-level.
- Holds on held-out LVIS meshes (train/test split).

**Timeline:** 1-2 weeks after Gate 1.

### Gate 3 — End-to-end in budget

**Measurable:**
- p50 wall time from TRELLIS.2 output → refined mesh ≤ 120s
- p95 wall time ≤ 300s
- Measured success rate (producing a mesh that beats the TRELLIS.2 baseline on hole_metric) ≥ 80% on routed inputs
- Measured compute cost ≤ $0.10/mesh including all operational overhead (queue, storage, bandwidth, retries)

**Timeline:** 2-3 weeks after Gate 2.

### Gate 3.5 — Router works

**Measurable:** a lightweight classifier (face count + category tag + surface smoothness heuristic, initially rule-based, upgradeable to learned) correctly routes:
- ≥90% of hard-surface / compositional inputs → refine
- ≥90% of organic / smooth / degenerate inputs → skip
- False-accept rate (sending a bad-fit mesh to refine) ≤ 10%

**Why:** without routing, ~30-50% of refine jobs produce worse output than baseline, which kills consumer trust.

**Timeline:** 1 week, can run in parallel with Gate 3.

---

## What ships and when

### Week 0-1: Infrastructure + Gate 0/0.1

- Phase 2 completes (Gate 0 signal)
- Iter/resolution sweep (Gate 0.1)
- Objaverse sampler shakedown on ~20 meshes
- Basic async job scaffolding (queue, worker pool, completion notification). NOT production-ready, NOT user-facing.
- Data policy drafted: opt-in consent, 12-month retention, deletion path, commercial reuse language. Legal review required before any user data is collected.

**No user-facing shipping this week.**

### Week 2-3: Gate 0.5 + Gate 1

- Warm-start feasibility test (Gate 0.5)
- Teacher corpus collection on 8 parallel pods
- In parallel: warm-start head architecture (SLAT encoder + set transformer)
- Cost telemetry wired: every run logs `{iters, wall_time_s, gpu_type, compute_cost_usd, metrics}` to a central store

### Week 4-5: Gate 2

- Train warm-start head on teacher corpus
- Evaluate on held-out LVIS subset AND held-out TRELLIS.2-generated meshes
- Iterate until head-predict + 1000-iter refine matches teacher within 10%

### Week 6-7: Gate 3, Gate 3.5, closed beta

- End-to-end pipeline benchmarked on ≥500 diverse inputs, full telemetry
- Router MVP (rule-based, upgradable)
- **Closed beta:** invite 50-200 trusted users. Free. Opt-in consent. Collect:
  - success rate by category
  - average cost per successful refine
  - p95 wall time
  - fraction routed-to-skip

### Week 8+: Public launch gated by beta telemetry

- If success rate ≥ 80% on routed inputs AND cost/mesh ≤ $0.10 AND p95 ≤ 5min: launch consumer tiers
- If any metric misses: iterate, do not launch

**Pricing model when it launches (pending beta data):**

| Tier | Price | Quota shape | Rationale |
|---|---|---|---|
| Free | $0/mo | ~5 refines/mo, low priority queue | Acquisition; generous enough to build habit, low enough to not burn cash |
| Pro | $10-15/mo | **Credit-based, not flat quota.** Starts at ~80-100 refines' worth of credits. | Avoids the full-utilization margin collapse friend flagged |
| Studio | $30-50/mo | ~500 refines, priority queue | Power users; still pay-per-use in effect |
| API / PAYG | ~$0.10-0.20/mesh | pay as you go | Direct cost-revenue tie; no quota risk |

**Not shipping consumer tiers before Gate 3.5 beta telemetry.** Period.

---

## Risks and open questions

1. **Gate 0 may fail.** Phase 2 may show through-hole-open still near zero even with the supervision tuning. Mitigation: if it fails, spend another 3-5 days on supervision (more aggressive mask weight, hole-axis-biased view sampling, curriculum tuning). If still failing after that, revisit whether this method can produce paper-quality at all.

2. **Warm-start hypothesis may be wrong.** Gate 0.5 might reveal that even a good warm start needs 5k-10k refinement steps to recover, not 1k. That breaks the 1-min target. Mitigation: at that point, either push harder on kernel-level speedups (accept longer timeline) or accept a 3-5 min product UX.

3. **Router might not be separable.** Some meshes are borderline. Mitigation: start rule-based + ML-augmented; accept some false-accepts, gate on measured success rate not predicted.

4. **Data policy delays.** If legal review on consent + retention takes 2-4 weeks, teacher data collection from paying users is blocked. Mitigation: we own the Objaverse teacher corpus outright (no user consent needed for public-domain data), so beta can ship even if user-data policy isn't settled.

5. **Paper quality claim may not match reality.** The paper reports CD ×1000 ≈ 7.94 averaged over 180 ShapeNet objects. We've been ≈194 on the hole canary. Factor 25× gap. Phase 2's supervision tuning should close most of this, but if it lands at factor 3-5×, we're not at paper quality and the "identical or superior" goal slips.

---

## Key metrics to start measuring NOW

Friend was right that these matter more than theoretical "2-4 cents endpoint":

- **Success rate by category** — logged per run, rolled up by LVIS bucket
- **Average cost per successful refine** — includes retries, failed jobs, storage, bandwidth
- **p95 wall time** — end-to-end, not training-loop
- **Router skip rate** — fraction of inputs routed away from refine
- **Per-mesh compute cost** — measured, not modeled

Infrastructure to capture these goes into the week-0 async job scaffolding. Metrics.json I already added covers per-run stats; needs a central aggregator.

---

## Decision log

- **Reject:** "ship Premium Refine at $5-15/mesh in week 1." Consumer economics don't support it. Retained as "Studio / API PAYG" option only.
- **Reject:** "use 15k iters everywhere" without proving it. Run Gate 0.1 comparison first.
- **Reject:** "1-2 min is default sync UX." It's async with notification, at minimum.
- **Accept:** user's quality-first mandate. Paper-identical or superior, measured honestly.
- **Accept:** friend's "routing before pricing page."
- **Accept:** friend's Gate 0.5 addition.
- **Accept:** friend's "data policy before consent collection."

---

*Last updated: 2026-04-19. Update this doc when gates pass or plans shift.*
