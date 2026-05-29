#!/usr/bin/env python3
"""Train a tiny FACE-Q decoder directly from packed localizable patch shards."""

from __future__ import annotations

import argparse
import copy
import functools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_indexed_face_decoder
from clearmesh.utils.checkpoint import args_to_json_safe
from scripts.research.train_face_indexed_conditioned_tiny import FaceIndexedSample, _make_batch


@dataclass(frozen=True)
class PackedPatchRef:
    path: Path
    patch_index: int


@dataclass(frozen=True)
class PackedPatchMeta:
    refs: list[PackedPatchRef]
    num_bins: int
    max_vertices: int
    max_faces: int
    max_points: int


def _manifest_paths(manifest: Path, limit_sources: int = 0) -> list[Path]:
    paths: list[Path] = []
    root = manifest.parent
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = Path(str(row["path"]))
        if not path.is_absolute() and not path.exists():
            path = root / path
        paths.append(path)
        if limit_sources and len(paths) >= int(limit_sources):
            break
    return paths


def _scan_packed_patches(manifest: Path, *, limit_sources: int = 0, limit_patches: int = 0) -> PackedPatchMeta:
    refs: list[PackedPatchRef] = []
    num_bins: int | None = None
    max_vertices = 0
    max_faces = 0
    max_points = 0
    for path in _manifest_paths(manifest, limit_sources=limit_sources):
        with np.load(path) as data:
            sample_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
            if num_bins is None:
                num_bins = sample_bins
            elif num_bins != sample_bins:
                raise ValueError(f"mixed num_bins in packed patch manifest: {num_bins} and {sample_bins}")
            vertex_offsets = np.asarray(data["patch_vertex_offsets"], dtype=np.int64)
            face_offsets = np.asarray(data["patch_face_offsets"], dtype=np.int64)
            point_offsets = np.asarray(data["patch_point_offsets"], dtype=np.int64)
            patch_count = int(len(face_offsets) - 1)
            if patch_count <= 0:
                continue
            max_vertices = max(max_vertices, int(np.max(np.diff(vertex_offsets))))
            max_faces = max(max_faces, int(np.max(np.diff(face_offsets))))
            max_points = max(max_points, int(np.max(np.diff(point_offsets))))
            for patch_index in range(patch_count):
                refs.append(PackedPatchRef(path=path, patch_index=patch_index))
                if limit_patches and len(refs) >= int(limit_patches):
                    break
        if limit_patches and len(refs) >= int(limit_patches):
            break
    if not refs or num_bins is None:
        raise SystemExit(f"No packed localizable FACE patches found in {manifest}")
    return PackedPatchMeta(
        refs=refs,
        num_bins=int(num_bins),
        max_vertices=int(max_vertices),
        max_faces=int(max_faces),
        max_points=int(max_points),
    )


@functools.lru_cache(maxsize=32)
def _load_packed_file(path_text: str) -> dict[str, np.ndarray]:
    with np.load(Path(path_text)) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _load_patch_sample(ref: PackedPatchRef) -> FaceIndexedSample:
    data = _load_packed_file(str(ref.path))
    idx = int(ref.patch_index)
    vertex_offsets = np.asarray(data["patch_vertex_offsets"], dtype=np.int64)
    face_offsets = np.asarray(data["patch_face_offsets"], dtype=np.int64)
    point_offsets = np.asarray(data["patch_point_offsets"], dtype=np.int64)
    v0, v1 = int(vertex_offsets[idx]), int(vertex_offsets[idx + 1])
    f0, f1 = int(face_offsets[idx]), int(face_offsets[idx + 1])
    p0, p1 = int(point_offsets[idx]), int(point_offsets[idx + 1])
    vertices = np.asarray(data["patch_vertices_flat"][v0:v1], dtype=np.int64)
    faces = np.asarray(data["patch_faces_flat"][f0:f1], dtype=np.int64)
    points = np.asarray(data["patch_points_flat"][p0:p1], dtype=np.float32)
    normals = np.asarray(data["patch_normals_flat"][p0:p1], dtype=np.float32)
    anchors = (
        np.asarray(data["anchor_coords"], dtype=np.float32)
        if "anchor_coords" in data
        else np.zeros((0, 3), dtype=np.float32)
    )
    if anchors.ndim == 2 and idx < len(anchors):
        # Make the voxel anchor a first-class conditioning token. This is the
        # minimal LATTICE-style locality signal: known test-time support goes
        # into the same point/normal channel without changing the decoder API.
        if "voxel_resolution" in data:
            resolution = max(1.0, float(np.asarray(data["voxel_resolution"]).reshape(-1)[0]))
        else:
            resolution = max(1.0, float(np.max(anchors) + 1.0))
        anchor_point = ((anchors[idx : idx + 1] + 0.5) / resolution) * 2.0 - 1.0
        points = np.concatenate([anchor_point.astype(np.float32), points], axis=0)
        normals = np.concatenate([np.zeros((1, 3), dtype=np.float32), normals], axis=0)
    if len(points) == 0:
        points = np.zeros((1, 3), dtype=np.float32)
        normals = np.zeros((1, 3), dtype=np.float32)
    if len(normals) != len(points):
        normals = np.zeros((len(points), 3), dtype=np.float32)
    return FaceIndexedSample(
        path=ref.path,
        point_features=np.concatenate([points, normals], axis=1).astype(np.float32),
        vertices=vertices,
        faces=faces,
    )


def _sample_batch_refs(refs: list[PackedPatchRef], batch_size: int) -> list[FaceIndexedSample]:
    return [_load_patch_sample(random.choice(refs)) for _ in range(int(batch_size))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-sources", type=int, default=0)
    parser.add_argument("--limit-patches", type=int, default=0)
    parser.add_argument("--point-samples", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--condition-tokens", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--count-loss-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    meta = _scan_packed_patches(
        args.manifest,
        limit_sources=args.limit_sources,
        limit_patches=args.limit_patches,
    )
    summary = {
        "packed_sources": len({str(ref.path) for ref in meta.refs}),
        "patches": len(meta.refs),
        "num_bins": meta.num_bins,
        "max_vertices": meta.max_vertices,
        "max_faces": meta.max_faces,
        "max_points": meta.max_points,
        "point_samples": int(args.point_samples),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    import torch
    import torch.nn.functional as F

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=meta.num_bins,
        max_vertices=meta.max_vertices,
        max_faces=meta.max_faces,
        point_feature_dim=6,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        condition_tokens=args.condition_tokens,
        condition_backend="pooled",
        decoder_backend="prefix",
        face_output_mode="geometry",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    losses: list[float] = []
    best_loss = float("inf")
    best_step = 0
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    for step in range(1, int(args.steps) + 1):
        samples = _sample_batch_refs(meta.refs, args.batch_size)
        (
            point_features,
            vertex_table,
            input_faces,
            target_faces,
            target_weights,
            *_rest,
            face_counts,
        ) = _make_batch(
            samples,
            max_vertices=meta.max_vertices,
            max_faces=meta.max_faces,
            device=device,
            point_samples=args.point_samples,
        )
        logits = model(point_features, vertex_table, input_faces)
        token_losses = F.cross_entropy(
            logits.reshape(-1, meta.max_vertices),
            target_faces.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(target_weights)
        valid = target_faces.ge(0)
        token_loss = (token_losses * target_weights * valid.to(token_losses.dtype)).sum() / torch.clamp(
            (target_weights * valid.to(target_weights.dtype)).sum(),
            min=1.0,
        )
        count_logits = model.predict_face_count_logits(point_features, vertex_table)
        count_loss = F.cross_entropy(count_logits, face_counts)
        loss = token_loss + float(args.count_loss_weight) * count_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if losses[-1] < best_loss:
            best_loss = losses[-1]
            best_step = step
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if step == 1 or step == int(args.steps) or (args.log_every and step % int(args.log_every) == 0):
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": losses[-1],
                        "token_loss": float(token_loss.detach().cpu()),
                        "count_loss": float(count_loss.detach().cpu()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "args": args_to_json_safe(args),
            "summary": summary,
            "losses": losses,
            "best_loss": best_loss,
            "best_step": best_step,
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "final_loss": losses[-1],
                "best_loss": best_loss,
                "best_step": best_step,
                "device": str(device),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
