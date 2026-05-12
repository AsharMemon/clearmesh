# OmniPart Integration

OmniPart is integrated as a worker hook, not yet as a hard dependency. That keeps the production pipeline stable while we solve the practical mask-generation step.

Official OmniPart CLI usage currently requires:

```bash
python -m scripts.inference_omnipart --image_input IMAGE_PATH --mask_input MASK_PATH
```

ClearMesh wrapper:

```bash
python scripts/product/run_omnipart_structure.py \
  --image input.png \
  --mask part_mask.exr \
  --output-dir artifacts/parts \
  --config-json configs/omnipart.thunder.example.json
```

The wrapper writes `parts_manifest.json`, which the worker records as a `part_manifest` asset when `metadata.part_structure_command` is supplied.

## Next Decisions

- Generate masks from a user UI pass, segmentation model, or TRELLIS/OmniPart preprocessing.
- Compare whole-object MeshRipple against per-part MeshRipple once part manifests are stable.
- Keep MeshMosaic as the stronger candidate when connected components are already available.

## Temporary Single-Part Mask Helper

For plumbing tests before semantic segmentation is available:

```bash
python scripts/product/create_single_part_mask.py \
  --image input.png \
  --output /tmp/omnipart_mask.exr
```

If local OpenCV lacks EXR support, the helper writes a `.npy` fallback and a JSON report. The fallback is useful for validating our own bookkeeping, but OmniPart CLI should receive an actual EXR once the GPU image stack supports it.
