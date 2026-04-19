# DualPrim Status — live

TL;DR of where we are. Updated whenever something changes. For the
full plan see [dualprim_plan.md](dualprim_plan.md).

## Right now

**Gate 0 status:** Phase 2 hole FAILED definitively. Root cause diagnosed:
the paper-faithful "independent" NSQ init didn't work for us. Phase 1
(coupled init) produced 14 carving primitives / 24 alive; Phase 2
(independent init) produced 0 carvers / 55 alive.

**In flight:**
- Phase 2 `window_box` finishing (should be done ~17:30 pod-time)
- Round 3 `hole` queued — reverts to `coupled` NSQ init, otherwise
  identical to Phase 2. Tests the single-variable hypothesis.

**Next automated decision** (scheduled wakeup at 17:57):
- If round 3 passes Gate 0 → fire `gate05_warmstart.py` for Gate 0.5
- If round 3 fails → draft round-4 (double views, λ_mask=5,
  pruning_interval=2000)

## Headline numbers

| Run | Mask IoU | Through-hole open | Hole recall | Carvers |
|---|---|---|---|---|
| Phase 1 hole (baseline) | 0.60 | 0.0% (0/2 views) | 0.00 | 14 |
| Phase 2 hole (tuned, indep. init) | 0.66 | 0.0% (0/24 views) | 0.00 | **0** |
| Round 3 hole (coupled init) | ⏳ pending | ⏳ | ⏳ | ⏳ |
| Paper (reference) | — | — | — | claim "works" |

## What I built this session

- `clearmesh/dualprim/io.py` — portable primitives JSON save/load
- `clearmesh/dualprim/optimize_scene.py` — trajectory snapshot hooks
- `scripts/dualprim/run_canary.py` — `--resume-primitives`,
  `--trajectory-dir`, `--fg-bias`, `--lambda-mask`, `--pruning-interval`
- `scripts/dualprim/autonomous_runner.py` — `--seeds`, `--trajectory`,
  `--mesh-list` (for Objaverse corpus)
- `scripts/dualprim/objaverse_sampler.py` — Objaverse-LVIS subset
  downloader with category filter + mesh validation
- `scripts/dualprim/gate01_sweep.py` — iter×resolution sweep for
  measured config selection
- `scripts/dualprim/gate05_warmstart.py` — warm-start refinement
  feasibility experiment (the Tier B validation)
- `scripts/dualprim/diagnose_primitives.py` — post-hoc primitive
  positioning diagnostic (found the Phase 2 failure mode)
- `scripts/dualprim/hole_metric.py` (upgraded) — hole-axis-biased
  ring views + auto-detection, made the measurement reliable at
  n=24 instead of n=2 views
- `docs/dualprim_plan.md` — unified gated plan post friend's review

## Pod state

- Pod 35076678 (ssh6.vast.ai:36678), A100_SXM4, ~$14 credit remaining
- phase 2 autonomous_runner: PID 7009, window_box run_canary: PID 8188
- round3 queue watcher: PID 9270, watching for phase2 exit

## Critical open questions

1. **Does coupled init alone fix Gate 0?** Round 3 tests this.
2. **Is phase 2 α-distribution collapse (all ~0.5) a separate issue?**
   The α values in phase 2 primitives.json are narrowly centered,
   suggesting a gradient pathology I'd want to investigate if round 3
   still fails.
3. **If round 3 fails, is round 4 enough or do we need deeper
   changes** (explicit NSQ-in-PSQ loss, different supervision)?

## Key decisions so far

- **Rejected:** ship Premium Refine at $5-15/mesh — consumer economics
  don't support it
- **Rejected:** use 15k iters everywhere without evidence → Gate 0.1
  sweep added
- **Rejected:** paper-faithful "independent" NSQ init for our setup →
  coupled init works empirically
- **Accepted:** Objaverse-LVIS as primary teacher data source (vs
  synthetic canary variants)
- **Accepted:** async UX (not sync 1-2 min), credit-based pricing
  (not flat quotas), router before launch
- **Accepted:** Gate 0.5 feasibility check before investing in
  Tier B neural head

## Git

All work on branch `claude/nervous-sammet`. Recent commits (reverse chronological):

```
b6bbd85 dualprim: --lambda-mask / --pruning-interval CLI + round-4 recipe
4f9e22c dualprim: gate05_warmstart — feasibility experiment for Tier B
acb220b dualprim plan: phase 2 Gate 0 failure analysis + round 3 proposal
2b840ed dualprim: diagnose_primitives script
36ae70b hole_metric: hole-axis-biased views + auto-detection
abd4354 dualprim: --mesh-list on runner + Gate 0.1 sweep script
bd17639 dualprim: unified plan + Objaverse-LVIS sampler
4f9e22c dualprim: gate05_warmstart (feasibility)
17d8502 dualprim: trajectory snapshots + multi-seed + resume-from-primitives
801a64b dualprim: expose --fg-bias CLI flag
934ffe8 dualprim: supervision tuning pass
```
