#!/usr/bin/env python3
"""Extract frozen FACE/FACE-Q condition latents for image-conditioned prior training.

The FACE paper trains an image-conditioned DiT to generate the latent VecSet,
then reuses the pretrained autoregressive decoder. This utility materializes the
same kind of training target from an existing FACE-Q checkpoint: for every
indexed FACE sample, run the checkpoint's condition encoder and save the latent
context that the decoder cross-attends to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_indexed_face_decoder


def _load_checkpoint(path: Path, torch):  # type: ignore[no-untyped-def]
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility.
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _sample_paths(dataset_dir: Path, limit: int) -> list[Path]:
    paths = sorted(path for path in dataset_dir.glob("*.npz") if not path.name.startswith("._"))
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no .npz FACE-Q samples found in {dataset_dir}")
    return paths


def _load_sample(path: Path, *, max_vertices: int, point_samples: int | None):
    with np.load(path) as data:
        required = {"surface_points", "surface_normals", "indexed_vertices", "num_bins"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        points = np.asarray(data["surface_points"], dtype=np.float32)
        normals = np.asarray(data["surface_normals"], dtype=np.float32)
        vertices = np.asarray(data["indexed_vertices"], dtype=np.int64)
        num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
    if point_samples is not None and point_samples > 0:
        if len(points) >= point_samples:
            points = points[:point_samples]
            normals = normals[:point_samples]
        else:
            repeat = int(np.ceil(point_samples / max(1, len(points))))
            points = np.tile(points, (repeat, 1))[:point_samples]
            normals = np.tile(normals, (repeat, 1))[:point_samples]
    point_features = np.concatenate([points, normals], axis=1).astype(np.float32)
    vertex_table = np.full((max_vertices, 3), -1, dtype=np.int64)
    vertex_table[: min(len(vertices), max_vertices)] = vertices[:max_vertices]
    return point_features, vertex_table, num_bins, int(len(vertices))


def _build_model(checkpoint: dict, device, torch):  # type: ignore[no-untyped-def]
    train_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(train_args, dict):
        train_args = {}
    num_bins = int(checkpoint.get("num_bins", train_args.get("num_bins", 1024)))
    max_vertices = int(checkpoint.get("max_vertices", train_args.get("max_vertices", 4096)))
    max_faces = int(checkpoint.get("max_faces", train_args.get("max_faces", 4096)))
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=num_bins,
        max_vertices=max_vertices,
        max_faces=max_faces,
        point_feature_dim=6,
        hidden_size=int(train_args.get("hidden_size", 192)),
        layers=int(train_args.get("layers", 4)),
        heads=int(train_args.get("heads", 6)),
        condition_tokens=int(train_args.get("condition_tokens", 8)),
        edge_head_mode=str(train_args.get("edge_head_mode", "index")),
        condition_backend=str(train_args.get("condition_backend", "pooled")),
        decoder_backend=str(train_args.get("decoder_backend", "prefix")),
        encoder_layers=int(train_args.get("encoder_layers", 4)),
        latent_dim=int(train_args.get("latent_dim", 64)),
        face_output_mode=str(train_args.get("face_output_mode", "linear")),
        voxset_resolution=int(train_args.get("voxset_resolution", 16)),
        spatial_gate_sigma=float(train_args.get("spatial_gate_sigma", 0.35)),
        spatial_gate_top_k=int(train_args.get("spatial_gate_top_k", 0)),
    ).to(device)
    state = checkpoint.get("model_state") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise SystemExit("checkpoint does not contain a model_state dictionary")
    load_result = model.load_state_dict(state, strict=False)
    model.eval()
    return model, train_args, num_bins, max_vertices, max_faces, load_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--point-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, torch)
    model, train_args, checkpoint_num_bins, max_vertices, _max_faces, load_result = _build_model(checkpoint, device, torch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = _sample_paths(args.dataset_dir, args.limit)
    point_samples = int(args.point_samples) if args.point_samples and args.point_samples > 0 else None
    manifest_rows = []
    failures = []
    with torch.no_grad():
        for index, path in enumerate(paths):
            out_path = args.output_dir / f"{path.stem}.latent.npz"
            if out_path.exists() and not args.overwrite:
                manifest_rows.append({"source": str(path), "latent": str(out_path), "skipped_existing": True})
                continue
            try:
                point_features_np, vertex_table_np, sample_num_bins, vertex_count = _load_sample(
                    path,
                    max_vertices=max_vertices,
                    point_samples=point_samples,
                )
                if sample_num_bins != checkpoint_num_bins:
                    raise ValueError(f"num_bins mismatch: sample={sample_num_bins}, checkpoint={checkpoint_num_bins}")
                point_features = torch.as_tensor(point_features_np, dtype=torch.float32, device=device).unsqueeze(0)
                vertex_table = torch.as_tensor(vertex_table_np, dtype=torch.long, device=device).unsqueeze(0)
                encoded = model._condition_with_positions(point_features, vertex_table)  # noqa: SLF001 - research utility.
                if isinstance(encoded, tuple):
                    latents, positions = encoded
                else:
                    latents, positions = encoded, None
                payload = {
                    "condition_latents": latents.detach().cpu().numpy().astype(np.float32)[0],
                    "source_path": np.asarray(str(path)),
                    "checkpoint_path": np.asarray(str(args.checkpoint)),
                    "condition_backend": np.asarray(str(getattr(model, "condition_backend", "unknown"))),
                    "decoder_backend": np.asarray(str(getattr(model, "decoder_backend", "unknown"))),
                    "num_bins": np.asarray(checkpoint_num_bins, dtype=np.int64),
                    "vertex_count": np.asarray(vertex_count, dtype=np.int64),
                }
                if positions is not None:
                    payload["condition_positions"] = positions.detach().cpu().numpy().astype(np.float32)[0]
                np.savez_compressed(out_path, **payload)
                manifest_rows.append({"source": str(path), "latent": str(out_path), "vertex_count": vertex_count})
            except Exception as exc:  # noqa: BLE001 - keep extraction moving on bad samples.
                if len(failures) < 32:
                    failures.append({"source": str(path), "error": f"{type(exc).__name__}: {exc}"})
            if (index + 1) % 100 == 0:
                print(json.dumps({"processed": index + 1, "written_or_seen": len(manifest_rows), "failures": len(failures)}), flush=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "count": len(manifest_rows),
        "failures": failures,
        "train_args": train_args,
        "load_missing_keys": sorted(getattr(load_result, "missing_keys", [])),
        "load_unexpected_keys": sorted(getattr(load_result, "unexpected_keys", [])),
        "rows": manifest_rows,
    }
    manifest_path = args.output_dir / "latent_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"latent_manifest": str(manifest_path), "count": len(manifest_rows), "failures": len(failures)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
