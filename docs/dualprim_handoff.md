# DualPrim implementation — honest handoff

*Written 2026-04-20. Strategic decision needed from user.*

## TL;DR

After **12 experimental rounds** across 2 cloud providers (vast.ai, Thunder), our DualPrim implementation **does not produce usable output on any canary shape**. All outputs are scattered blob piles, not coherent primitive decompositions. This is not a tuning issue we can solve with more of the same — it needs a strategic decision.

## Evidence

Rendered ref vs pred for all 5 canaries at K=30, 5k iters, coupled init, MSE/BCE mask loss:

| Canary | Reference | Prediction | CD × 1000 |
|---|---|---|---|
| hole (box + cylinder hole) | clean box with hole | scattered blobs | — (through-hole 0%) |
| stool (cyl top + 4 legs) | clean stool | blob pile | 276 |
| dumbbell (2 spheres + rod) | clean dumbbell | 3 disconnected clusters | 252 |
| camera (box + cylinder lens) | clean camera | blob pile | 172 |
| window_box (box + rect cutout) | clean box w/ cutout | blob pile | 128 |

Paper baseline: CD × 1000 ≈ 7.94 averaged over 180 ShapeNet objects.
**Our worst canary is 35× worse, our best is 16× worse.**

Visual PNGs at `/tmp/compare/` on Thunder instance tnr-0, and in
local `/tmp/compare_local/`.

## Why this is probably fundamental, not tuning

1. **20× less compute than paper.** We ran K=30, 5k iters. Paper uses K=100, 30k iters. That alone is a huge quality gap.
2. **NaN-grad cascade at K≥50.** Scaling up triggers numerical instability we've partially but not fully fixed. Root cause narrowed to sq_implicit FD-grad → render backward chain overflow; full fix requires autograd-level sq_implicit instead of FD.
3. **Loss landscape + multi-view optimization is hard.** Primitives satisfy individual silhouette views without forming coherent 3D structure. Classic "multi-view without 3D consistency" failure mode.

## What we DID build (all committed, reusable)

### Infrastructure
- `clearmesh/dualprim/io.py` — portable primitives JSON save/load
- Trajectory snapshots every ~log-spaced iter
- `--resume-primitives` warm-start
- Heartbeat observability (every 60s)
- NaN-grad skip logging (every occurrence)
- Watchdog pattern (log-size stall detection)
- Multi-seed support via `--seeds`
- `--mesh-list` for Objaverse manifest
- `--lambda-open-ray`, `--hole-ray-oversample`, `--lambda-mask`, `--pruning-interval`, `--fg-bias` CLI flags
- MSE mask loss as NaN-safe alternative to BCE
- FD-grad clamping in sq_implicit
- `coupled_axial` NSQ init option

### Diagnostics
- `scripts/dualprim/diagnose_primitives.py` — primitive positioning analysis
- `scripts/dualprim/hole_metric.py` — hole-axis-biased ring view sampling, through-hole open %
- `scripts/dualprim/export_from_snapshot.py` — render mid-training state
- `scripts/dualprim/gate01_sweep.py` — iter × res sweep (untested)
- `scripts/dualprim/gate05_warmstart.py` — warm-start feasibility (needs Gate 0 teacher to run)
- `scripts/dualprim/make_canaries.py` — regenerate test meshes anywhere

### Training infrastructure
- `scripts/dualprim/objaverse_sampler.py` — LVIS subset sampler, tested end-to-end
- `scripts/dualprim/autonomous_runner.py` — multi-experiment orchestrator, `--mesh-list` + `--seeds` + `--trajectory` + `--fg-bias` + `--union-export` + `--nsq-init` + `--lambda-open-ray` + `--hole-ray-oversample`
- `clearmesh/dualprim/warmstart.py` — Tier B neural head skeleton (PointEncoder + SetTransformer + Hungarian loss), smoke-tested

## Strategic decision needed from you

### Option A: Scale to paper config
- K=100, 30k iters, 256px — one canary per 4-6 hrs
- Requires: fix K=100 NaN-grad (sq_implicit needs autograd, not FD-grad)
- Estimated effort: 3-5 days engineering + 1 week compute
- Risk: still may not reach paper quality if paper has undocumented tricks
- **Good fit if:** you want to continue pursuing "paper-identical" quality

### Option B: Try PartCrafter (already in your codebase)
- `clearmesh/partcrafter/` exists but I haven't touched it
- Mesh → part segmentation directly, no optimization loop
- May bypass the optimization pathology entirely
- Estimated effort: 1-2 days to evaluate, ~1 week to integrate
- **Good fit if:** the GOAL is compact structured output, not specifically DualPrim

### Option C: Ship Stage 2 without structured refinement
- TRELLIS.2 + RefinementDiT produces millions-of-triangles meshes
- These are usable for rendering/viewing but NOT for editing/CAD
- Skip DualPrim for v1 product
- Estimated effort: zero — you already have Stage 2 pipeline
- **Good fit if:** you want to launch NOW and add structure refinement in v2

### Option D: Pause DualPrim work, pivot to Easy3E editing (different thread)
- `/Users/Ashar/.claude/plans/drifting-herding-robin.md` has Easy3E editing plan
- Different product angle: image/text-guided mesh editing
- Uses TRELLIS.2 + FlowEdit, separate from DualPrim
- **Good fit if:** structured mesh is less important than editability right now

## My recommendation (ranked)

1. **Option B + C combined**: evaluate PartCrafter this week while shipping Stage 2 as v1. If PartCrafter works, it replaces DualPrim as the Stage 4 structure layer. If not, we have time to revisit DualPrim later.

2. **Option A solo** if you want paper-identical DualPrim specifically: budget 2 weeks for the fix + scale run. I can fix sq_implicit's autograd path with 1-2 days focused work, then one canary per 4-6 hrs at paper scale.

3. **NOT Option C alone**: shipping without structure refinement defeats ClearMesh's core differentiator. But it's the shortest path to any product.

## What I did NOT check (worth considering)

- **Paper reproduction via official code** — the paper may have released reference code that works where ours doesn't. Check arxiv / github.
- **Different primitive parameterization** — maybe superquadrics aren't the right primitive for this task. Neural parts / blob trees / CSG primitives could be alternatives.
- **Direct SDF supervision** — instead of view-based rendering, supervise on mesh TSDF directly (bypasses multi-view-consistency issue). The `mesh_fit` mode exists in our code but was never made to work properly.

## Compute state

- Thunder instance tnr-0 A100-80GB currently idle
- Has restarted twice during session (unreliable for long jobs)
- Vast.ai credit: still -$0.18 (not replenished)

## Commit hashes for reference

All on branch `claude/nervous-sammet`:
- `f9cb2a4` status: visual inspection finding
- `1e66616` mask loss BCE→MSE
- `a13cf30` FD-grad clamp
- `cec3125` coupled_axial init
- `c8c4451` hole-ray oversample
- `618fa04` L1 open-ray loss
- `c9834bb` NaN-grad logging fix
- `80ef921` hole-axis training views
- `ad96df0` heartbeat observability
- `17d8502` trajectory snapshots + resume
- Many more — see `git log --oneline`

## Recommendation for next session

**Stop and decide before running more experiments.** Every additional round at K=30/5k will produce the same blob piles. This is a strategic decision point, not a tuning problem.
