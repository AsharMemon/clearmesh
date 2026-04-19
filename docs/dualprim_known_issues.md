# DualPrim — known issues / bugs / weirdness

Living list of things we've seen but not fixed. Add when observed,
mark when fixed.

## Active

### 1. window_box hang with independent NSQ init

**Observed:** Phase 2 window_box (K=100, 15k iters, independent NSQ
init, lambda_mask=3, fg_bias=0.7) hung after iter 0.

**Symptoms:**
- Main thread in `S` (sleeping) state, 130 threads, 100% GPU util
- `train.log` and `logs.json` frozen at iter=0 for 17+ minutes
- Iter 0 logged successfully; no NaN-skip events; no exceptions
- Process actively launching CUDA kernels (100% util steady)

**Not reproduced:** Phase 2 hole completed fine with the same config
(just different reference mesh). So it's geometry-specific.

**Hypotheses:**
- Pathological primitive → ray intersection causing sq_implicit to
  produce values that hit the FCLAMP=1e6 saturation on many iters,
  which may interact badly with autograd
- Gradient explosion that clipping doesn't catch (but NaN-skip would
  — so maybe not NaN but very-small-but-nonzero)
- Some synchronization issue between autograd threads and CUDA

**Mitigations to add later:**
- Per-iter wall-clock timeout (if any iter > 30 seconds, abort
  and log the current state of all parameters for post-mortem)
- Heartbeat log line every 30 seconds independent of iter count,
  showing current iter + any suspicious stats
- Add α, θ gradient magnitude to diagnostics so we can spot
  saturation before it causes a hang

**Status:** Killed window_box manually. Does not block round 3 since
round 3 uses coupled init (independent init is the only config seen
to hang).

### 2. α distribution collapse in phase 2 hole

**Observed:** Phase 2 hole's final α values concentrated in
[0.41, 0.62] with median 0.50. All 100 primitives alive, none
below the α ≥ 0.02 prune threshold.

Phase 1 hole (coupled init, K=30): α ranged [-0.01, 1.00], median
0.64. Proper bimodal-ish distribution.

**Hypothesis:** independent init gives all primitives equally weak
initial signal → Adam's gradient + loss clamps + hard [0,1] clamp
drives them all to the sigmoid midpoint. Without a clear differentiator
(some primitives overlapping GT, others far), α stays ambiguous.

**Status:** Evidence the independent init is broken. Coupled init's
α distribution is healthy. Won't recur if we stay on coupled init.
If we ever revive independent init, this is the smoke signal to
watch for.

### 3. Legacy primitives.json schema on older runs

**Observed:** Phase 1 + Phase 2 primitives.json files use the old
schema `{primitives, training_s}` without `iteration`/`num_alive`/
`num_slots` fields. Created before commit 17d8502.

**Impact:** Some tools (notably `load_scene_from_json`) still work
because they reach inside `.primitives` key. But any code assuming
the new schema fields (e.g. dataset loader later) would need a
migration path.

**Fix:** trivial — add a backfill script that reads old JSONs and
writes them out in the new schema. Run once on the existing corpus
when we start using it for training.

## Resolved

*(nothing yet)*
