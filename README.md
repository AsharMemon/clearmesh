# ClearMesh

ClearMesh is a 3D generation research stack pivoting toward one practical goal:
turn an image or prompt into an editable, production-survivable triangle mesh with semantic parts, with Easy3E editing preserved as an optional product mode.

## Current V1 Plan

```text
Image / prompt
  -> TRELLIS.2 visual asset / proxy
  -> evaluation harness
  -> OmniPart-style semantic parts
  -> one winning artist-mesh head
  -> repair + validation + export
```

The project is intentionally not building every possible output branch in v1. DualPrim, B-Rep/STEP, NURBS, quad-native generation, custom LATO/SATO reimplementation, and a custom SLAT-conditioned mesh transformer are parked until the harness proves they are necessary.

## Current Priorities

| Priority | Work | Status |
|---|---|---|
| 1 | Mesh evaluation harness | Starter implemented |
| 2 | TRELLIS.2 point-cloud bridge at 16k / 40k / 100k samples | Next |
| 3 | OmniPart-style part structure | Next integration target |
| 4 | MeshRipple bake-off | First mesh-head candidate |
| 5 | Mesh Silksong / DeepMesh / MeshMosaic / TreeMeshGPT baselines | Bake off after harness is useful |

## Local Production Scaffold

Create and advance a local async job without running the API server:

```bash
python scripts/product/create_local_job.py --input-uri local://example.png --grant-credits 10
python scripts/product/run_local_worker.py --once
```

Run the API scaffold:

```bash
export CLEAR_MESH_API_KEYS="dev_key:dev_secret:user_dev:team_dev"
export CLEARMESH_ADMIN_KEY="admin_secret"
uvicorn clearmesh.api.server:app --reload
```

See [docs/api_scaffold.md](docs/api_scaffold.md).

Run the staged product worker, which can become GPU-active with `--execute-heavy`:

```bash
python scripts/product/run_pipeline_worker.py --once
```

Open the static product UI prototype at [dashboard/product.html](dashboard/product.html).

## Evaluation Harness

Run the starter harness with:

```bash
python scripts/eval/evaluate_meshes.py \
    --manifest manifests/mesh_bakeoff.example.csv \
    --output eval_results/mesh_bakeoff_report.json
```

Optional Blender roundtrip check:

```bash
python scripts/eval/evaluate_meshes.py \
    --manifest manifests/mesh_bakeoff.example.csv \
    --output eval_results/mesh_bakeoff_report.json \
    --blender blender
```

Manifest format:

```csv
case_id,method,mesh_path,reference_path
mug_handle,trellis2,outputs/mug_trellis.glb,refs/mug_proxy.glb
mug_handle,meshripple_40k,outputs/mug_meshripple_40k.obj,refs/mug_proxy.glb
```

The harness currently reports geometry, topology, editability, and production-adjacent metrics, including Chamfer/Hausdorff when a reference mesh is supplied, watertightness, boundary loops, non-manifold edges, connected components, valence histogram, triangle aspect ratios, subdivision smoke tests, and optional Blender import/export.

## Mesh-Head Bake-Off

Current candidate order:

```text
1. MeshRipple
2. Mesh Silksong
3. DeepMesh
4. MeshMosaic
5. TreeMeshGPT
6. FastMesh, if inference and weights run cleanly
```

Selection rule:

```text
Pick the method that wins on topology and editability failures, not the method with the prettiest screenshot.
```

Install public mesh-head repos on a GPU host with:

```bash
MESH_HEAD_ROOT=/workspace/mesh-heads INSTALL_ENV=1 bash scripts/setup/install_mesh_heads.sh
```

Run MeshRipple through the adapter with `scripts/product/run_mesh_head.py` once checkpoints and point clouds are available.

See [docs/architecture.md](docs/architecture.md), [docs/production_roadmap.md](docs/production_roadmap.md), [docs/thunder_runbook.md](docs/thunder_runbook.md), [PIPELINE_PLAN.md](PIPELINE_PLAN.md), [MASTER_PLAN.md](MASTER_PLAN.md), and [docs/mesh_head_bakeoff.md](docs/mesh_head_bakeoff.md) for the detailed roadmap.

## Existing Assets Worth Keeping

```text
TRELLIS.2 setup and generation scripts
Stage 2 refinement code and technical learnings
mesh repair/export utilities
text/image entry-point scaffolding
PBR/export/optional autorigging modules as downstream tools
RunPod/Vast.ai operational scripts
```

## Project Structure

```text
clearmesh/
  clearmesh/
    api/               FastAPI production scaffold
    product/           job, billing, auth, worker, and artifact scaffolding
    mesh_heads/        external public repo adapters
    eval/              mesh evaluation harness
    pointcloud.py      TRELLIS.2 mesh -> point-cloud bridge
    mesh/              extraction, repair, export utilities
    stage2/            TRELLIS.2-aligned refinement research
    editing/           Easy3E editing scaffolds kept intact
    text_to_3d/        text/image entry-point scaffolds
    texture/           PBR texture utilities
    rigging/           optional downstream rigging
  scripts/
    product/           local job and worker scaffold CLIs
    eval/              batch harness CLI
    data/              dataset and TRELLIS.2 data scripts
    setup/             environment setup scripts
    runpod/            GPU pod helpers
  docs/
    mesh_head_bakeoff.md
    archive/           parked/old strategy docs
```

## Environment

The GPU project environment should install `requirements.txt` plus the TRELLIS.2-specific CUDA extensions and model dependencies described in the setup scripts. The local macOS shell may not have mesh dependencies like `trimesh`; run full harness jobs in the `clearmesh` environment or on the GPU pod.

## License

MIT
