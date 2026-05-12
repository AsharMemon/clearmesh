# MeshRipple Hostile Audit

## Current Hypothesis

The public MeshRipple repo is probably not the primary failure. Our ClearMesh invocation was off-distribution in two important ways:

1. `dec_to_facenum` is part of MeshRipple input preprocessing, not merely an output face target.
2. The TRELLIS.2 proxy we fed MeshRipple remains extremely fragmented after MeshRipple preprocessing, unlike the repo demo meshes.

## Evidence

MeshRipple public inference loads meshes from `data.eval_dataset_path`, not our external PLY. The dataset path does:

```text
load mesh
if y_up: reorder vertices
normalize to roughly [-0.5, 0.5]
discretize + clean duplicate/degenerate faces
if dec_to_facenum != -1: Open3D quadric decimation
sample 40,960 points, then choose 16,384 conditioning points
```

Our earlier profiles changed multiple paper/demo assumptions at once:

| setting | official 10k dense demo | our previous 512/1k/2k sweeps |
| --- | ---: | ---: |
| `dec_to_facenum` | `5000` | `512 / 1024 / 2048` |
| `max_len` | `10000` | `512 / 1024 / 2048` |
| `min_len` | `500` | `1` |
| `top_k` | `50` | `20` |
| `top_p` | `0.95` | `0.9` |
| `temperature` | `0.9` | inherited/partially implicit |
| `eos_aug` | `true` | inherited unless overridden |
| `wr_fix` | `true` | inherited unless overridden |

## Preprocessing Comparison

Using MeshRipple's own preprocessing code with the 10k dense config:

| input | dec target | faces after | components after |
| --- | ---: | ---: | ---: |
| demo `1(255).glb` | 5000 | 5000 | 2 |
| demo `1(37).glb` | 5000 | 4999 | 2 |
| demo `1(66).glb` | 5000 | 5000 | 15 |
| demo `1(6).glb` | 5000 | 4999 | 3 |
| ClearMesh TRELLIS proxy | 5000 | 99512 | 10913 |

That last row is the smoking gun. The TRELLIS proxy is so fragmented/nonstandard that Open3D decimation does not produce a clean 5k conditioning mesh. MeshRipple then samples from a shredded surface, so generation also shreds.

## Control Runs

Started on Thunder:

```text
official-single demo:
  input: MeshRipple demo_data/demo/1(255).glb
  config: config_10k_full_dense_mesh.yaml
  changes: batch_size=1, absolute checkpoint path, output path only
```

Next control:

```text
official-like ClearMesh:
  input: TRELLIS.2 proxy
  config: configs/meshripple.thunder.official_dense.json
```

If the official demo is coherent but the TRELLIS proxy run is fragmented, our bridge/preprocessing is wrong. If the official demo is also fragmented, then repo/env/checkpoint is suspect.

## Corrective Direction

Do not expose low `dec_to_facenum` profiles as quality settings. They degrade the conditioning mesh.

Instead:

```text
1. Preserve official MeshRipple settings for correctness baselines.
2. Build a TRELLIS proxy sanitizer before MeshRipple.
3. Convert TRELLIS.2 output into a clean conditioning mesh or clean point cloud that matches MeshRipple's distribution.
4. Only optimize runtime after the official-like baseline works.
```
