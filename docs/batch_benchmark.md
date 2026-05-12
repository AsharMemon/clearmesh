# Batch Benchmark Path

ClearMesh now has a repeatable benchmark lane for the production pipeline:

```text
CSV manifest
  -> create local jobs
  -> TRELLIS.2 proxy generation
  -> point-cloud bridge
  -> optional OmniPart-style part manifest
  -> MeshRipple artist mesh
  -> repair/eval report
  -> export package
```

## Local Job Creation

```bash
python scripts/product/create_batch_jobs.py \
  --manifest manifests/trellis2_meshripple_benchmark.example.csv \
  --state-root .clearmesh_state_batch \
  --artifact-root artifacts_batch \
  --grant-credits 100
```

The manifest supports trusted worker metadata such as `trellis_command`, `part_structure_command`, `easy3e_command`, and `autorigging_command`. Public API requests intentionally strip these executable metadata keys by default.

## Thunder Batch

```bash
THUNDER_INSTANCE_ID=0 \
LIMIT=6 \
CONFIG_JSON=/home/ubuntu/clearmesh/configs/meshripple.thunder.example.json \
scripts/thunder/pipeline_trellis2_meshripple_batch.sh
```

Use `LIMIT=2` with the smoke MeshRipple config for plumbing checks, then raise it for quality runs.

## Metrics To Watch

- `mesh_metrics.connected_components`
- `mesh_metrics.boundary_loop_count`
- `mesh_metrics.nonmanifold_edge_count`
- `mesh_metrics.aspect_ratio_p95`
- `pair_metrics.chamfer_l2`
- `pair_metrics.normal_consistency`

The early goal is not a single perfect number. It is to find which cases consistently break topology or editability.

## MeshRipple Config Profiles

- `configs/meshripple.thunder.smoke.json`: tiny topology plumbing check only.
- `configs/meshripple.thunder.quality.json`: bounded quality pass for batch iteration.
- `configs/meshripple.thunder.example.json`: near-repo-default full generation; this can run for hours on complex TRELLIS meshes.

## Latest Thunder Batch Result

A 2-case bounded run completed successfully on Thunder with:

```bash
LIMIT=2 \
CONFIG_JSON=/home/ubuntu/clearmesh/configs/meshripple.thunder.quality.json \
STATE_ROOT=/tmp/clearmesh_batch2_state \
ARTIFACT_ROOT=/tmp/clearmesh_batch2_artifacts \
scripts/thunder/pipeline_trellis2_meshripple_batch.sh
```

Both cases reached `succeeded` through:

```text
input_validation -> trellis_proxy -> point_cloud_bridge -> mesh_head -> repair_validation -> export_package
```

Early bounded-profile metrics show the pipeline works, but the generated 512-face meshes are not production quality:

| case | face count | components | boundary loops | non-manifold edges | Chamfer L2 | normal consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trellis2_case_00 | 490 | 85 | 29 | 186 | 0.1168 | 0.6681 |
| trellis2_case_01 | 490 | 170 | 18 | 312 | 0.2137 | 0.5618 |

Interpretation: this profile is useful for queue/batch/eval plumbing and rough failure-mode discovery. Final quality comparison needs a higher face target, better component/repair strategy, or per-part generation.

## Cleanup Comparison

A conservative cleanup stage now runs after `mesh_head` and before `repair_validation` when `cleanup_enabled` is true. It keeps the raw `artist_mesh` asset and registers a separate `cleaned_mesh` asset for evaluation/export.

On the two bounded MeshRipple batch outputs, `min_component_faces=8` produced:

| case | components raw -> clean | tiny raw -> clean | boundary loops raw -> clean | non-manifold edges raw -> clean | faces raw -> clean | Chamfer L2 raw -> clean | normal consistency raw -> clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trellis2_case_00 | 85 -> 25 | 54 -> 1 | 29 -> 19 | 186 -> 83 | 490 -> 339 | 0.1168 -> 0.1498 | 0.6681 -> 0.6936 |
| trellis2_case_01 | 170 -> 14 | 133 -> 0 | 18 -> 5 | 312 -> 33 | 490 -> 149 | 0.2137 -> 0.2152 | 0.5618 -> 0.5617 |

Interpretation: cleanup is doing the right kind of topological triage, but at the bounded 512-face profile it can remove too much surface. This is acceptable for diagnostics; final production settings should use a higher MeshRipple face target and possibly part-aware cleanup thresholds.

The worker path was also verified on Thunder with an existing artist mesh: `artist_mesh` -> `cleaned_mesh` -> eval -> export succeeded.

## Blender Preview Renders

Numeric metrics catch topology failures; preview renders help humans triage cases quickly. If Blender is installed on the worker:

```bash
python scripts/eval/render_mesh_preview.py \
  --mesh output.obj \
  --output preview.png \
  --blender blender
```

The existing evaluator already supports Blender roundtrip checks via `--blender`.

## Wire Preview Fallback

When Blender is unavailable, use the lightweight wire preview renderer:

```bash
python scripts/eval/render_mesh_preview_wire.py \
  --mesh eval_results/thunder/meshripple_sweep_meshes/1k/raw.obj \
  --output eval_results/thunder/previews/1k_raw.png
```

This does not replace Blender roundtrip/subdivision validation, but it gives quick visual triage images for fragmented meshes.
