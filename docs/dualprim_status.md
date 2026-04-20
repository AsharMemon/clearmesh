# DualPrim Status — live

TL;DR of where we are. Updated whenever something changes. For the
full plan see [dualprim_plan.md](dualprim_plan.md).

## ⛔ STOPPED: credit exhausted

**Vast.ai balance: -$0.18** (as of end-of-session). Both pods auto-exited:
- 35076678 (A100 DualPrim work) — exited mid-round-6
- 35082988 (H100 easy3e-sprint, idle at 0% util) — also exited

**User action required:** add credit to resume. Pods preserved disk
state; `/workspace/dualprim_round{3,4,5,6}/` artifacts still exist and
can be evaluated once pods restart.

**Credit burn analysis:**
- A100 at $0.89/hr × 9 hrs ≈ $8 this session (expected)
- H100 at $1.79/hr × 9 hrs idle ≈ $16 unexpected drain
- The H100 is the main culprit. I flagged it in status early but
  didn't destroy it autonomously (not my call on user's other work).

**When you come back:**
1. Top up credit (~$20 recommended — enough for round 6 rerun + Gate
   0.5 + a few more experiments)
2. Restart pod 35076678 — round 6 partial outputs on disk may be
   useful; check `/workspace/dualprim_round6/hole/`
3. Destroy pod 35082988 if easy3e-sprint isn't currently active
4. Resume work following the round-6 decision tree in this doc

---

## Right now

**Gate 0 status:** Phase 2 hole FAILED definitively. Root cause diagnosed:
the paper-faithful "independent" NSQ init didn't work for us. Phase 1
(coupled init) produced 14 carving primitives / 24 alive; Phase 2
(independent init) produced 0 carvers / 55 alive.

**Phase 2 window_box: KILLED** — hung at iter 0 for 17 min (100% GPU,
130 threads, no log progress). See
[dualprim_known_issues.md](dualprim_known_issues.md) §1.

**Round 3 (coupled init) HUNG + killed.** Same hang pattern as
window_box — log frozen past iter ~200, process at 100% GPU but
no observable progress for 1+ hour. Watchdog caught it after I
killed manually.

**Round 3 iter-1000 snapshot DID produce real data** (rendered via
`export_from_snapshot.py`):
- 54 carvers (vs phase 2's 0) ✓
- 57 / 57 NSQs-inside-PSQs (100% retention) ✓
- **3 carvers positioned near hole axis**
- BUT: those 3 carvers at Y = {-0.85, +0.34, +0.55} with NSQ scale
  ~0.25 (diameter 0.5) — GAPS between them larger than carving
  reach. No through-hole formed.
- Mask IoU 0.66, through-hole open **0%**.

**Root cause of hole failure now understood:**
- Coupled init correctly pairs NSQ↔PSQ (supervision structure OK)
- But only 2/26 training views look down the hole axis, so only
  ~3-5% of carvers land near it, and they're spherical not
  elongated. Can't chain into a through-hole.
- Fix: more training views + stronger mask weight + later pruning.

**Round 4 HUNG at iter 200** — watchdog killed at 10 min stall. No
usable output (no trajectory snapshot reached).

**Round 5 (K=50 coupled 5k iters) HUNG at iter 200** — same pattern.
No useful output.

**Pattern diagnosed:**
- K=30 coupled lambda=1: works (phase 1)
- K=30+ coupled lambda=3+: HANGS at iter ~200-1000
- K=100 independent: works (phase 2 hole)
- Changed between phase 1 and later runs:
  - lambda_mask default 1 → 3
  - fg_bias added (0.7)
  - theta curriculum added
  - render cache fix
  - Friend's supervision tuning pass

Something in this list introduced the hang. We haven't bisected.

**Round 6 was running** (K=30 coupled lambda=1 + `--add-hole-axis-views`,
closest possible replication of phase 1 but with stronger hole
supervision) when pod died. Partial outputs possibly on disk at
`/workspace/dualprim_round6/hole/`.

## Headline numbers

| Run | K | iters | Mask IoU | Through-hole open | Carvers | NSQs-inside-PSQ |
|---|---|---|---|---|---|---|
| Phase 1 hole | 30 | 5k | 0.60 | 0.0% (n=2) | 14 | 16/24 (67%) |
| Phase 2 hole (indep. init) | 100 | 15k | 0.66 | 0.0% (n=24 ring) | 0 | 1/55 (2%) |
| Round 3 hole @ iter 1000 (coupled) | 100 | 15k→hung | 0.66 | 0.0% (n=24) | **54** | **57/57 (100%)** |
| Round 4 hole (52 views, λ=5) | 100 | 15k→? | ⏳ | ⏳ | ⏳ | ⏳ |
| Paper | 100 | 30k | — | "works" | — | — |

**Interpretation:**
- Coupled init is **structurally correct** (54 carvers, 100%
  NSQ-inside-PSQ retention). Independent init is broken at current
  supervision level (2% retention).
- Carver count alone doesn't give through-hole. Need carvers
  positioned *along the hole axis*. Round 3 had 3/54. Even phase 1
  with its "best" carver produced only a spherical pocket, not a
  cylinder.
- Training hang at iter ~1000-2000 is a reproducible bug (round 3
  + window_box both hit it). Round 4 has a watchdog mitigation.

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

- Pod 35076678 (ssh6.vast.ai:36678), A100_SXM4, $0.89/hr, ~97% util,
  **~$13 credit remaining**
- Round 3 run_canary: PID 9816, actively training
- Round 3 queue watcher: PID 9270, will trigger hole_metric +
  diagnose_primitives when run_canary exits

### ⚠️ Attention: second pod burning money idle

- Pod **35082988** (ssh3.vast.ai:12988), H100_NVL, **$1.79/hr**
- Label: `easy3e-sprint`
- **0% GPU util**, 5.6 hours uptime = ~$10 burned idle already
- Not used by this DualPrim work; belongs to the Easy3E editing plan
- **Decision needed when user returns:** keep running for Easy3E
  sprint, or destroy and recreate when needed. Idle cost = $43/day
  if left running.
- I am NOT touching it autonomously — destroying someone else's pod
  is not my call.

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
