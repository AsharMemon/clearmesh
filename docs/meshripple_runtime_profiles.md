# MeshRipple Runtime Profiles

These measurements are for the Thunder RTX A6000 lane using an existing TRELLIS.2 proxy mesh and the MeshRipple adapter. They isolate the artist-mesh head and do not include fresh TRELLIS.2 generation unless noted.

## Profiles

| profile | config | target faces | max decode length | intended use |
| --- | --- | ---: | ---: | --- |
| smoke | `configs/meshripple.thunder.smoke.json` | 128 | 128 | command/env validation only |
| 512 | `configs/meshripple.thunder.quality.json` | 512 | 512 | quick topology diagnostics |
| 1k | `configs/meshripple.thunder.1k.json` | 1,024 | 1,024 | first bake-off candidate, not production default yet |
| 2k | `configs/meshripple.thunder.2k.json` | 2,048 | 2,048 | measured production-candidate profile, but topology is still poor whole-object |
| 5k | `configs/meshripple.thunder.5k.json` | 5,000 | 4,096 | high-cost candidate; do not expose broadly yet |
| full | `configs/meshripple.thunder.example.json` | repo default | repo default | near-default generation; can take hours |

## Current Runtime Expectations

| profile | MeshRipple-only runtime | end-to-end with existing proxy | end-to-end with new TRELLIS.2 proxy | confidence |
| --- | ---: | ---: | ---: | --- |
| smoke | 1-3 min | 2-5 min | 8-18 min | medium |
| 512 | 542.3s / 9.0 min observed | 10-12 min | 18-30 min | high |
| 1k | 1272.5s / 21.2 min observed | 22-25 min | 30-50 min | high |
| 2k | 2291.7s / 38.2 min observed | 39-42 min | 50-70 min | high |
| 5k | ~85-120 min estimated, bounded by `max_len=4096` behavior | 90-130 min | 100-155 min | medium |
| full | ~2.5-3h projected from an earlier unbounded run | ~2.5-3h+ | ~3h+ | medium |

## Latest 512/1k/2k Sweep

Artifacts:

```text
eval_results/thunder/meshripple_quality_sweep_512_1k.json
eval_results/thunder/meshripple_quality_sweep_2k.json
eval_results/thunder/previews/
```

| profile | mesh min | raw faces | raw components | tiny | boundary loops | non-manifold | Chamfer L2 | cleaned faces | cleaned components | cleaned non-manifold | cleaned Chamfer L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 9.0 | 497 | 62 | 27 | 23 | 138 | 0.1206 | 371 | 28 | 58 | 0.1205 |
| 1k | 21.2 | 933 | 208 | 194 | 35 | 274 | 0.0693 | 396 | 30 | 27 | 0.0977 |
| 2k | 38.2 | 1842 | 458 | 451 | 101 | 651 | 0.0587 | 544 | 44 | 60 | 0.0631 |

## Interpretation

The 512 profile remains diagnostic only. It is fast enough to expose failures, but the mesh is fragmented and non-watertight.

The 1k and 2k profiles improve proxy distance, but raw topology gets worse as face count rises on this case. That is the key production finding: whole-object MeshRipple face-count scaling is not enough. We should treat 5k as a high-cost experiment, not the default route to quality.

The likely production stance is:

```text
MeshRipple is a targeted refinement operator, not the default visible wait.
Every job should first publish a TRELLIS.2 visual mesh plus normalized control surface.
MeshRipple should only run after the mesh passport says the control surface is coherent.
```

For v1 product UX, keep these queue labels conservative:

```text
Default Preview: TRELLIS.2 visual mesh + normalized control surface
Standard: high-resolution normalized mesh plus targeted refinement if eligible
High: larger control surface/detail projection; MeshRipple only on selected parts
Experimental Full: repo-default MeshRipple, hidden/admin until cost is proven
```

## Cost Notes

- Point-cloud sampling and cleanup/eval are tiny compared with MeshRipple decoding; sweeps sampled 40,960 points in about 0.5-0.6s.
- Reusing a TRELLIS.2 proxy saves a meaningful chunk of time during mesh-head bake-offs.
- Runtime is decode-token dominated, not strictly face-count linear; measure every profile before exposing user-facing ETAs.
- Per-part generation can multiply runtime by part count unless parts are processed in parallel across GPUs.
