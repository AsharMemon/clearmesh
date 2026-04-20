# DualPrim Status — live

TL;DR of where we are. Updated whenever something changes. For the
full plan see [dualprim_plan.md](dualprim_plan.md).

## ✅ PIVOTED: now on Thunder Compute

- Thunder instance 0 (A100-80GB) running, SSH alias `tnr-0`
- clearmesh cloned at `/home/ubuntu/clearmesh` (symlinked to
  `/workspace/clearmesh`)
- Canaries regenerated via `scripts/dualprim/make_canaries.py` (5 GLBs
  in `/workspace/`)
- **Round 6 in flight on Thunder** — phase-1 reproducer (K=30,
  coupled, λ=1) + hole-axis training views
- Monitor task `bm6os5ba2` tracks terminal state
- Old vast.ai pods remain exited — disk state preserved but not in use

---

## ⛔ HISTORY: Vast.ai credit exhausted earlier

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

## ⚠️ CRITICAL: Rounds 6-11 produce visually unusable output

Rendered the refit meshes side-by-side with references. All 3 tested
canaries (hole, stool, dumbbell) produce **scattered blob piles**, not
coherent structure:
- hole ref: clean 1×0.6×0.6 box with Y-axis hole. pred: pile of
  random blobs, no hole, no box structure.
- stool ref: cylindrical top + 4 legs. pred: blob pile, no legs
  visible, no clear seat.
- dumbbell ref: 2 spheres + connecting rod. pred: 3 disconnected
  blob clusters floating in space.

**Chamfer numbers already told us this (CD×1000 = 250-280) but visual
confirmation is definitive.** Our DualPrim implementation at K=30/5k
iters simply does not produce usable output, regardless of NaN-grad
fixes, loss type (BCE vs MSE), or init strategy.

**Honest root cause**: we're running at ~5% of paper compute (K=30
vs 100, 5k iters vs 30k) AND our numerical stability is worse (90%
NaN-skip in most rounds). Primitives end up satisfying individual
silhouette rays without forming coherent 3D structure — a fundamental
optimization pathology at under-scaled compute.

## Paths forward (ranked)

1. **Scale to paper config** (K=100, 30k iters, 256px res). With
   heartbeat active, K=100 runs were likely being killed by stale
   watchdog — rerun with corrected threshold. ~4-6 hr per canary.
2. **Consider alternative structurization methods** — PartCrafter
   does part segmentation directly from mesh features and may give
   cleaner part decomposition without the optimization pathology.
3. **Give up on structured refinement for now** and ship Stage 2
   (TRELLIS.2 + RefinementDiT) alone as the v1 product.

## Rounds 6-8 stuck in same local minimum

Three different supervision strategies at K=30 coupled + hole-axis views
all converge to essentially the same state:

| | Mask IoU | Through-hole | Carvers | Near-axis |
|---|---|---|---|---|
| Round 6 (baseline + hole-axis views) | 0.64 | 0% | 25/28 | 1 |
| Round 7 (open-ray loss) | NaN-grad cascade | — | — | — |
| Round 8 (30% hole-ray oversample) | 0.63 | 0% | 26/27 | 1 |

**Key finding:** BCE mask loss has a **fundamental ambiguity** for
hole pixels. `pred_mask → 0` can be satisfied two ways:
  (a) PSQ shrinks / avoids the pixel region
  (b) NSQ carves the region
Both get equal credit. Supervision weight adjustments (views, loss,
sampling) don't distinguish them — the optimizer picks (a) because
it's locally easier. Topology is not preserved regardless.

**Known NaN-grad pitfall discovered:** heavy supervision on hole
rays (either via open-ray loss OR oversampling) pushes BCE gradients
toward -1/(1-p) ≈ -1e6 per ray. At 230 hole rays/batch × 1e6,
sq_implicit's FD-grad chain overflows. Round 8 had 91% iters
NaN-skipped.

## Next directions (ranked)

1. **K=100 coupled + heartbeat** (same as round 3/4 but with
   observability proven). Round 3 iter-1000 had 54 carvers / 3 near
   axis vs K=30's 1 near axis. Primitive density may itself solve
   the local-minimum problem even without loss changes. ~1.5 hr.
2. **NSQ-axial-elongation init**: initialize NSQs with random axis
   elongated 3x (e.g. scale=(0.1, 0.3, 0.1) random axis). Forces
   NSQs to be shaped for through-carving from the start.
3. **Volume-conservation loss**: penalize PSQ-union-volume deviation
   from ref volume. Forces optimizer to CARVE (keep mass, cut holes)
   rather than SHRINK (remove mass).

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
