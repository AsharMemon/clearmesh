# Surface Normalization Audit

## Thunder Result

Test mesh:

```text
/tmp/clearmesh_batch2_artifacts/projects/trellis2_meshripple_batch/jobs/job_334383d7ebee43fdbc59d18a716aecf2/trellis_proxy/trellis_proxy.glb
```

Thunder profile:

```text
sample_points=40000
poisson_depth=7
density_quantile=0.04
target_faces=20000
orient_normals=false
```

Runtime:

```text
11 seconds on Thunder RTX A6000 host
```

## Metrics

| mesh | faces | vertices | components | tiny components | boundary loops | non-manifold edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw TRELLIS proxy | 232,851 | 213,247 | 12,907 | 12,907 | 13,139 | 168,875 |
| Poisson control surface | 153,092 | 76,711 | 259 | 258 | 106 | 2,911 |

## MeshRipple Preprocess Result

Using the MeshRipple official dense adapter config with `dec_to_facenum=5000`:

| input | raw faces | faces after discrete clean | final faces | final vertices | components after MeshRipple preprocess |
| --- | ---: | ---: | ---: | ---: | ---: |
| Poisson control surface | 153,092 | 101,550 | 6,340 | 2,019 | 83 |

## Interpretation

Surface normalization is valuable for production preview/control output:

```text
12,907 components -> 259 components
13,139 boundary loops -> 106 boundary loops
168,875 non-manifold edges -> 2,911 non-manifold edges
```

But the control surface is still not clean enough to make whole-object
MeshRipple the default:

```text
MeshRipple preprocess still sees 83 components
demo meshes were closer to 1-15 components
```

Conclusion:

```text
Use Poisson/SDF normalization for fast preview and control mesh.
Do not use it alone as a guarantee for MeshRipple eligibility.
Move part-aware generation and quad/cage layout higher than whole-object MeshRipple.
```

## Quad Baseline Follow-Up

The generic retopology planner ran on the same Poisson control surface in about
6 seconds:

```text
charts: 6,383
feature edges: 48,356
boundary edges: 2,658
feature-curve components: 3,493
risk: high
```

The plan recommended `cleanup_or_merge` for 6,292 small-fragment charts and
`cross_field_with_feature_constraints` for 91 feature-rich charts. This is the
desired behavior: it catches the fact that the control surface is still too
fragmented for whole-object quad promotion.

pyinstantmeshes installed cleanly on the Thunder MeshRipple venv and ran quickly
on the same Poisson control surface. It produced pure quads, but generic
whole-object quadrangulation still failed topology gates:

| pass | quad faces | quad ratio | components | boundary loops | non-manifold edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| pyinstantmeshes raw OBJ | 17,971 | 1.0 | 17,953 | 17,707 | 70,982 |
| ClearMesh welded OBJ | 18,105 | 1.0 | 910 | 111 | 4,924 |

The weld pass is worthwhile because many automatic remeshers duplicate vertices
along visual seams. It is not enough by itself. The result strengthens the
decision to move toward part-parametric quad/cage layouts rather than trusting a
single global quad remesh over a damaged whole-object control surface.

## Implementation Notes

The first normalization attempt stalled because generic metric evaluation and
Open3D normal orientation were unsafe on heavily fragmented meshes. The current
implementation now:

```text
uses sparse face-adjacency passport metrics
keeps normal orientation off by default
avoids fragile Open3D simplification in the critical path
uses simplification only as best-effort
```
