#!/usr/bin/env python3
"""TRELLIS.2 Introspection — runs on the pod, dumps model layout details.

This script is the Phase 0 gating artifact for the Easy3E implementation.
It answers questions that cannot be guessed from local code:

  1. What keys exist in `pipeline.models` and what classes are they?
  2. What is the exact forward signature of SparseStructureFlowModel?
  3. Does TRELLIS.2 expose a mesh→SS-latent encoder, or only a sampler?
  4. Is CFG inside the model or inside the sampler loop?
  5. What SparseTensor type does SLAT use, and what are its attributes?
  6. What camera pose does `pipeline.get_cond` assume?
  7. What do sampler signatures look like (sample_sparse_structure, sample_shape_slat)?

The script is strictly read-only — it imports, instantiates, introspects,
and prints. It performs one optional 1-step forward pass with a dummy
image to dump intermediate tensor shapes.

Output goes to stdout and to {output_path} if provided.

Usage (on pod):
    python scripts/setup/inspect_trellis2.py \
        --trellis2-dir /workspace/TRELLIS.2 \
        --model-dir /workspace/models/trellis2-4b \
        --output /workspace/trellis2_introspection.txt

Exit codes:
    0  — introspection complete, file written
    1  — TRELLIS.2 not importable; check TRELLIS2_DIR and install
    2  — model load failed; check model dir / HF token / disk
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

class Tee:
    """Write to stdout and to a file simultaneously."""

    def __init__(self, path: Path | None):
        self.path = path
        self._fh = open(path, "w") if path else None

    def __call__(self, *args, **kwargs):
        print(*args, **kwargs)
        if self._fh:
            print(*args, **kwargs, file=self._fh)
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


def header(tee, title: str, char: str = "="):
    tee()
    tee(char * 70)
    tee(f" {title}")
    tee(char * 70)


def subheader(tee, title: str):
    tee()
    tee(f"--- {title} ---")


# ---------------------------------------------------------------------------
# Introspection utilities
# ---------------------------------------------------------------------------

def dump_signature(obj: Any, method_name: str) -> str:
    """Return a string describing obj.method_name's signature, or an error."""
    try:
        method = getattr(obj, method_name, None)
        if method is None:
            return f"  [no .{method_name} attribute]"
        sig = inspect.signature(method)
        return f"  {method_name}{sig}"
    except (ValueError, TypeError) as e:
        return f"  {method_name}(<signature unavailable: {e}>)"


def dump_source_location(cls: type) -> str:
    try:
        path = inspect.getsourcefile(cls)
        lineno = inspect.getsourcelines(cls)[1]
        return f"  defined at: {path}:{lineno}"
    except (OSError, TypeError):
        return "  defined at: <unavailable>"


def tensor_summary(t) -> str:
    """Describe a tensor / sparse tensor / None without loading its values."""
    if t is None:
        return "None"
    try:
        import torch
        if isinstance(t, torch.Tensor):
            return f"Tensor(shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device})"
    except ImportError:
        pass
    # SparseTensor duck-typing
    if hasattr(t, "feats") and hasattr(t, "coords"):
        feats_shape = tuple(t.feats.shape) if hasattr(t.feats, "shape") else "?"
        coords_shape = tuple(t.coords.shape) if hasattr(t.coords, "shape") else "?"
        return f"SparseTensor(feats={feats_shape}, coords={coords_shape}, cls={type(t).__name__})"
    if isinstance(t, dict):
        return f"dict(keys={sorted(t.keys())})"
    if isinstance(t, (list, tuple)):
        return f"{type(t).__name__}(len={len(t)})"
    return f"<{type(t).__name__}>"


# ---------------------------------------------------------------------------
# Main introspection routine
# ---------------------------------------------------------------------------

def inspect_pipeline(pipeline, tee):
    """Dump everything we care about from the loaded pipeline."""

    # 1. pipeline-level structure
    header(tee, "1. Pipeline top-level attributes")
    public_attrs = [a for a in dir(pipeline) if not a.startswith("_")]
    tee(f"  public attrs: {public_attrs}")
    tee(f"  pipeline class: {type(pipeline).__name__}")
    tee(f"  pipeline module: {type(pipeline).__module__}")
    tee(f"  pipeline source: {inspect.getsourcefile(type(pipeline))}")

    # 2. pipeline.models
    header(tee, "2. pipeline.models — what flows do we have?")
    if not hasattr(pipeline, "models"):
        tee("  [no .models attribute on pipeline]")
    else:
        models = pipeline.models
        if isinstance(models, dict):
            tee(f"  model keys: {sorted(models.keys())}")
            for key in sorted(models.keys()):
                m = models[key]
                tee(f"\n  models['{key}']:")
                tee(f"    class:  {type(m).__name__}")
                tee(f"    module: {type(m).__module__}")
                try:
                    tee(f"    source: {inspect.getsourcefile(type(m))}")
                except (TypeError, OSError):
                    pass
                # Parameter count
                try:
                    import torch
                    if isinstance(m, torch.nn.Module):
                        n_params = sum(p.numel() for p in m.parameters())
                        tee(f"    n_params: {n_params:,}")
                        tee(dump_signature(m, "forward"))
                except Exception as e:
                    tee(f"    [params/forward not introspectable: {e}]")
        else:
            tee(f"  .models is not a dict: {type(models).__name__}")

    # 3. pipeline methods we plan to call
    header(tee, "3. Pipeline API surface — methods Easy3E will invoke")
    methods_of_interest = [
        "preprocess_image",
        "get_cond",
        "get_cond_image",
        "sample_sparse_structure",
        "sample_shape_slat",
        "sample_shape_slat_cascade",
        "decode_shape_slat",
        "decode",
        "run",
    ]
    for meth in methods_of_interest:
        tee(dump_signature(pipeline, meth))

    # 4. Encoder discovery (critical for Phase 1)
    header(tee, "4. Mesh-to-latent ENCODER discovery (Phase 1 gate)")
    tee("  Looking for any attribute/method containing 'encode' or 'encoder':")
    for attr in dir(pipeline):
        lower = attr.lower()
        if "encode" in lower or "encoder" in lower:
            val = getattr(pipeline, attr, None)
            tee(f"    pipeline.{attr}  ->  {type(val).__name__}")

    if hasattr(pipeline, "models") and isinstance(pipeline.models, dict):
        for key in pipeline.models:
            if "encoder" in key.lower() or "encode" in key.lower() or "vae" in key.lower():
                tee(f"    models['{key}']  ->  {type(pipeline.models[key]).__name__}")

    # 5. Data toolkit (the other encoder path)
    header(tee, "5. trellis2.data_toolkit — alternative encoder source")
    try:
        import trellis2.data_toolkit as dt
        tee(f"  data_toolkit path: {dt.__file__}")
        members = [m for m in dir(dt) if not m.startswith("_")]
        tee(f"  top-level: {members}")

        for mod_name in ["encode_shape_latent", "encode_ss_latent", "dual_grid"]:
            try:
                mod = __import__(f"trellis2.data_toolkit.{mod_name}", fromlist=[mod_name])
                tee(f"\n  {mod_name}.py located at: {mod.__file__}")
                fns = [f for f in dir(mod) if callable(getattr(mod, f)) and not f.startswith("_")]
                tee(f"    callables: {fns}")
                for fn_name in fns[:10]:
                    fn = getattr(mod, fn_name)
                    try:
                        sig = inspect.signature(fn)
                        tee(f"    {fn_name}{sig}")
                    except (ValueError, TypeError):
                        pass
            except Exception as e:
                tee(f"  [failed to import {mod_name}: {e}]")
    except Exception as e:
        tee(f"  [data_toolkit import failed: {e}]")

    # 6. Image conditioning (for Phase 2)
    header(tee, "6. Image conditioning internals")
    if hasattr(pipeline, "image_cond_model"):
        m = pipeline.image_cond_model
        tee(f"  pipeline.image_cond_model: {type(m).__name__}")
    if hasattr(pipeline, "models") and isinstance(pipeline.models, dict):
        for key in pipeline.models:
            if "cond" in key.lower() or "dino" in key.lower() or "image" in key.lower():
                tee(f"  models['{key}']  ->  {type(pipeline.models[key]).__name__}")

    # 7. Sampler source — where does CFG live?
    header(tee, "7. Sampler code — where does CFG live?")
    for meth in ["sample_sparse_structure", "sample_shape_slat"]:
        try:
            fn = getattr(pipeline, meth, None)
            if fn is None:
                continue
            src_path = inspect.getsourcefile(fn)
            src_line = inspect.getsourcelines(fn)[1]
            tee(f"  {meth} source: {src_path}:{src_line}")
            # Grab first 80 lines of source for context
            src_lines = inspect.getsourcelines(fn)[0]
            tee("  --- first 80 lines ---")
            for i, line in enumerate(src_lines[:80]):
                tee(f"    {src_line + i:>4} | {line.rstrip()}")
            tee("  --- (truncated) ---")
        except Exception as e:
            tee(f"  [failed to introspect {meth}: {e}]")

    # 8. One dummy forward to dump tensor shapes
    header(tee, "8. Dummy forward pass — shape dump")
    try:
        import numpy as np
        import torch
        from PIL import Image

        tee("  Creating 512x512 dummy RGB image (red circle on grey)...")
        arr = np.full((512, 512, 3), 128, dtype=np.uint8)
        Y, X = np.ogrid[:512, :512]
        mask = (X - 256) ** 2 + (Y - 256) ** 2 < 100 ** 2
        arr[mask] = (255, 0, 0)
        img = Image.fromarray(arr)

        if hasattr(pipeline, "preprocess_image"):
            proc = pipeline.preprocess_image(img)
            tee(f"  preprocess_image -> {tensor_summary(proc)}")
        else:
            proc = img

        if hasattr(pipeline, "get_cond"):
            cond = pipeline.get_cond([proc], 512)
            tee(f"  get_cond([proc], 512) -> {tensor_summary(cond)}")

            # Dive one level
            if isinstance(cond, dict):
                for k, v in cond.items():
                    tee(f"    cond['{k}'] -> {tensor_summary(v)}")
        else:
            tee("  [pipeline has no get_cond]")

    except Exception as e:
        tee(f"  [dummy forward failed: {e}]")
        tee(traceback.format_exc())

    # 9. o_voxel API
    header(tee, "9. o_voxel API surface")
    try:
        import o_voxel
        tee(f"  o_voxel path: {o_voxel.__file__ if hasattr(o_voxel, '__file__') else '<built-in>'}")
        tee(f"  top-level: {[m for m in dir(o_voxel) if not m.startswith('_')]}")
        if hasattr(o_voxel, "convert"):
            tee(f"  o_voxel.convert: {[m for m in dir(o_voxel.convert) if not m.startswith('_')]}")
            if hasattr(o_voxel.convert, "mesh_to_flexible_dual_grid"):
                sig = inspect.signature(o_voxel.convert.mesh_to_flexible_dual_grid)
                tee(f"    mesh_to_flexible_dual_grid{sig}")
    except Exception as e:
        tee(f"  [o_voxel introspection failed: {e}]")

    header(tee, "DONE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trellis2-dir", default="/workspace/TRELLIS.2")
    p.add_argument("--model-dir", default="/workspace/models/trellis2-4b")
    p.add_argument("--output", default="/workspace/trellis2_introspection.txt",
                   help="File to mirror stdout to. Set to empty to disable.")
    args = p.parse_args()

    output_path = Path(args.output) if args.output else None
    tee = Tee(output_path)

    # Ensure TRELLIS.2 is on path
    if Path(args.trellis2_dir).exists() and args.trellis2_dir not in sys.path:
        sys.path.insert(0, args.trellis2_dir)

    header(tee, "TRELLIS.2 Introspection", char="#")
    tee(f"  trellis2_dir: {args.trellis2_dir}")
    tee(f"  model_dir:    {args.model_dir}")
    tee(f"  output:       {output_path}")

    # Environment
    subheader(tee, "Environment")
    try:
        import torch
        tee(f"  torch version: {torch.__version__}")
        tee(f"  cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            tee(f"  cuda device: {torch.cuda.get_device_name(0)}")
            tee(f"  cuda mem (GB): {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
    except Exception as e:
        tee(f"  [torch env failed: {e}]")
        sys.exit(1)

    # Import TRELLIS.2
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
    except Exception as e:
        tee(f"\n[FATAL] trellis2.pipelines import failed: {e}")
        tee(traceback.format_exc())
        tee.close()
        sys.exit(1)

    # Load pipeline
    tee()
    tee(f"Loading Trellis2ImageTo3DPipeline from {args.model_dir}...")
    try:
        if Path(args.model_dir).exists():
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_dir)
        else:
            tee(f"  [local model dir missing; falling back to HuggingFace]")
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        pipeline.to("cuda")
    except Exception as e:
        tee(f"\n[FATAL] pipeline load failed: {e}")
        tee(traceback.format_exc())
        tee.close()
        sys.exit(2)

    tee("  Pipeline loaded OK.")

    try:
        inspect_pipeline(pipeline, tee)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
