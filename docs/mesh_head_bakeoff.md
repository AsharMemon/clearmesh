# Mesh Head Bake-Off

## Goal

Select one artist-mesh head for ClearMesh v1 by measuring topology, editability, and production behavior on fixed TRELLIS.2 outputs.

## Candidates

```text
1. MeshRipple
2. Mesh Silksong
3. DeepMesh
4. MeshMosaic
5. TreeMeshGPT
6. FastMesh, if inference and weights are usable
```

## Experiment Matrix

For every fixed test case:

```text
TRELLIS.2 raw mesh
TRELLIS.2 -> 16k point cloud -> mesh head
TRELLIS.2 -> 40k point cloud -> mesh head
TRELLIS.2 -> 100k point cloud -> mesh head
OmniPart-style part clouds -> winning whole-object or per-part head
```

## Harness Command

```bash
python scripts/eval/evaluate_meshes.py \
  --manifest manifests/mesh_bakeoff.csv \
  --output eval_results/mesh_bakeoff_report.json \
  --samples 20000
```

Optional Blender roundtrip:

```bash
python scripts/eval/evaluate_meshes.py \
  --manifest manifests/mesh_bakeoff.csv \
  --output eval_results/mesh_bakeoff_report.json \
  --blender blender
```

## CSV Manifest

```csv
case_id,method,mesh_path,reference_path
mug_handle,trellis2,outputs/mug_trellis.glb,refs/mug_proxy.glb
mug_handle,meshripple_40k,outputs/mug_meshripple_40k.obj,refs/mug_proxy.glb
```

## Pass/Fail Signals

Prefer a method that:

```text
- keeps handles, holes, fingers, cables, and branches connected
- avoids tiny floating components
- reduces boundary loops and non-manifold edges
- has sane valence and triangle aspect distributions
- survives subdivision smoke tests
- imports and exports through Blender
- preserves separable semantic parts once OmniPart-style structure is available
```

Reject or deprioritize a method that:

```text
- produces pretty but fragmented meshes
- hides holes in screenshots
- needs unreleased training code for basic inference reproducibility
- collapses thin topology under 16k/40k/100k point budgets
- produces meshes artists cannot edit without immediate cleanup
```

## Public Repo Adapter Layer

ClearMesh now keeps mesh-head repos out of core code and calls them through adapters:

```bash
MESH_HEAD_ROOT=/workspace/mesh-heads INSTALL_ENV=1 bash scripts/setup/install_mesh_heads.sh

python scripts/product/run_mesh_head.py \
  --head meshripple \
  --case-id mug_handle_40k \
  --point-cloud pointclouds/mug_handle_40960.ply \
  --output-dir artifacts/mug_handle/meshripple \
  --config-json configs/meshripple.gpu.example.json
```

The adapter writes stdout/stderr logs and discovers generated mesh files. MeshRipple is wired first because it has public inference commands and checkpoints; other candidates should be added as `GenericCommandMeshHead` configs once their exact inference CLIs are validated on GPU.

## Staged Product Worker

Use the staged worker for end-to-end job bookkeeping:

```bash
python scripts/product/create_local_job.py \
  --input-uri local:///workspace/inputs/mug.png \
  --proxy-mesh-path /workspace/proxies/mug_trellis2.glb \
  --grant-credits 20

python scripts/product/run_pipeline_worker.py \
  --state-root .clearmesh_state \
  --artifact-root artifacts \
  --execute-heavy \
  --mesh-head meshripple \
  --mesh-head-config-json configs/meshripple.gpu.example.json \
  --once
```

Without `--execute-heavy`, GPU-dependent steps are skipped but the job state, billing reservation, artifact directories, repair/export placeholders, and optional autorigging bookkeeping still run.

## MeshRipple Quality Sweep

Use the quality sweep to find the first MeshRipple setting that produces usable topology without unacceptable runtime:

```bash
THUNDER_INSTANCE_ID=0 \
CONFIG_SPECS=512=/home/ubuntu/clearmesh/configs/meshripple.thunder.quality.json,1k=/home/ubuntu/clearmesh/configs/meshripple.thunder.1k.json \
scripts/thunder/meshripple_quality_sweep.sh
```

Profiles now available:

```text
512: configs/meshripple.thunder.quality.json
1k:  configs/meshripple.thunder.1k.json
2k:  configs/meshripple.thunder.2k.json
5k:  configs/meshripple.thunder.5k.json
```

The sweep reuses one TRELLIS proxy, samples a point cloud once, runs each MeshRipple config, cleans each mesh, evaluates raw and cleaned meshes, and writes `quality_sweep.json`.

## Latest MeshRipple Runtime Finding

The Thunder 512/1k sweep artifact is stored at:

```text
eval_results/thunder/meshripple_quality_sweep_512_1k.json
```

Summary on one fixed TRELLIS.2 proxy:

| profile | MeshRipple time | raw faces | raw components | raw non-manifold edges | raw Chamfer L2 | cleaned components | cleaned non-manifold edges | cleaned Chamfer L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 9.0 min | 497 | 62 | 138 | 0.1206 | 28 | 58 | 0.1205 |
| 1k | 21.2 min | 933 | 208 | 274 | 0.0693 | 30 | 27 | 0.0977 |

Conclusion: 1k improves geometric fit but not raw topology on this case. Do not promote 1k to a user-facing production default until we test 2k/5k and add part-aware generation or stronger topology repair.

## Part-Aware Mesh Head Path

Whole-object MeshRipple is not enough if higher face counts mainly create more islands. The worker now supports an internal opt-in path:

```json
{
  "enable_parts": true,
  "metadata": {
    "part_mesh_generation": true,
    "part_mesh_max_parts": 8
  }
}
```

When `part_structure` emits `parts_manifest.json`, the mesh-head step runs MeshRipple per part and records `part_artist_mesh` assets plus a `part_mesh_manifest`. This should remain internal until OmniPart/TRELLIS.2 part manifests are reliable and per-part jobs can be parallelized across GPUs.

## 2k Sweep Update

The 2k profile completed on the same Thunder A6000 proxy:

| profile | mesh min | raw faces | raw components | tiny components | boundary loops | non-manifold edges | Chamfer L2 | cleaned faces | cleaned components | cleaned non-manifold edges | cleaned Chamfer L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2k | 38.2 | 1842 | 458 | 451 | 101 | 651 | 0.0587 | 544 | 44 | 60 | 0.0631 |

Conclusion: geometric fit continues to improve, but fragmentation scales with face count in the whole-object bridge. The next bake-off should prioritize part-aware conditioning and/or alternative mesh heads over pushing MeshRipple whole-object to 5k as the default.
