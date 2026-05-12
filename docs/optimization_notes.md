# Production Optimization Notes

## Does 5k MeshRipple Mean An 85-120 Minute Wait?

With the current whole-object MeshRipple autoregressive path on one Thunder RTX A6000: yes, that is the conservative expectation. The 512 profile took 9.0 minutes and the 1k profile took 21.2 minutes on the same proxy. A bounded 5k profile is expected to be much slower because decode length dominates runtime.

That should not become the normal user experience.

## New Default: One Adaptive Pathway

ClearMesh should present one pathway:

```text
TRELLIS.2 visual mesh
  -> mesh passport
  -> normalized control surface
  -> preview publish
  -> part-aware/control-surface refinement
  -> high-resolution publish
  -> optional Easy3E edits and optional autorigging
```

There are no user-facing "routes." The pipeline can internally choose whether a
learned refiner is safe, but every job goes through the same contract.

## Smoking-Gun Optimizations

These are the highest-leverage paths before touching model internals:

| optimization | quality impact | latency impact | production stance |
| --- | --- | --- | --- |
| Reuse/cache TRELLIS.2 proxy | neutral | saves TRELLIS time on retries/edits | do now |
| Mesh passport before refinement | prevents expensive bad runs | saves failed MeshRipple hours | do now |
| Surface normalization | better conditioning, all vertex counts | seconds-minutes versus AR decode | do now |
| Preview-first publish | neutral | users see progress quickly | product default |
| Retopology planning graph | chooses template/cross-field/fallback per chart | seconds | do now |
| Targeted mesh heads only | higher success rate | avoids universal 5k waits | do now |
| Optional constrained shrink-wrap | can improve control-surface fit | fast if sampled projection is enough | benchmark before default |
| OBJ vertex weld after quad remesh | turns duplicate visual quads into connected topology | seconds | do for all automatic quad baselines |
| Part-aware generation | better locality/editability | serial is slower, parallel can match one-part latency | build now |
| Parallel per-part refinement | neutral/better | converts N parts from serial to wall-clock max(part) | needs GPU queue fanout |
| UltraShape-style learned refinement | can add fine geometry | likely minutes, not AR-hours | evaluate as control-surface enhancer |

## Things That Are Not Yet Proven

- A simple face-count increase does not automatically improve topology. In the 512/1k sweep, 1k improved Chamfer but produced more raw fragments.
- Flash-attn is already part of the Thunder path where applicable; the current bottleneck appears decode-token dominated.
- Full/default MeshRipple is not a user-facing tier until quality/cost proves it deserves to exist.

## Recommended UX

```text
Preview:
  TRELLIS.2 visual mesh + normalized control surface.
  Target: visible in minutes, not after autoregressive decode.

Standard:
  high-resolution normalized mesh plus eligible targeted refinement.

High:
  larger control surface, detail projection/rebake, optional targeted learned refinement.

MeshRipple:
  hidden refinement operator, not the default user-facing wait.
```

## 2k Measurement

2k completed in 2291.7 seconds / 38.2 minutes MeshRipple-only on the same Thunder RTX A6000 proxy. It improved Chamfer to 0.0587 raw, but raw connected components increased to 458 and non-manifold edges to 651.

This makes the answer to “can we get 5k quality without waiting 85-120 minutes?” clearer:

```text
Not by simply raising whole-object MeshRipple face count.
Yes, by changing the work shape:
- cache TRELLIS.2 proxy outputs
- show the visual/control preview first
- normalize the conditioning surface
- split into semantic parts
- run targeted refinement only when metrics predict success
- reserve MeshRipple for selected parts or premium/admin reruns
```

## Why This Is Not A Setback

The production insight is that MeshRipple, Mesh Silksong, OmniPart, and
UltraShape all encode structure before detail. Our previous whole-object
MeshRipple experiment skipped that structural normalization and paid for it with
fragments. The new design makes structure the shared substrate:

```text
visual mesh for fidelity
control surface for topology/editability
passport for safe compute decisions
```

## Thunder Surface Normalization Result

On the real shredded TRELLIS proxy, the fixed Poisson control-surface profile ran
in 11 seconds and improved topology substantially:

```text
components: 12,907 -> 259
boundary loops: 13,139 -> 106
non-manifold edges: 168,875 -> 2,911
```

MeshRipple preprocessing on that control surface still produced 83 components,
so this is a strong preview/control-mesh win but not enough to make whole-object
MeshRipple the default. The next quality bet is part-aware quad/cage layout plus
projection/detail baking.

## Thunder Quad Baseline Result

Installing `pyinstantmeshes` on Thunder took only a normal pip install and the
5k pyinstantmeshes run was quick, but the whole-object result was not production
quality. ClearMesh's vertex weld postprocess reduced the worst artifact:

```text
components: 17,953 -> 910
boundary loops: 17,707 -> 111
non-manifold edges: 70,982 -> 4,924
quad ratio: 1.0 -> 1.0
```

That is a good optimization, not a product solution. The remaining topology
damage means generic whole-object quad remeshing should stay a sidecar benchmark
while we build semantic part templates and feature-constrained stitching.
