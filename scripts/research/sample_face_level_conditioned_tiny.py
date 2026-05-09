#!/usr/bin/env python3
"""Sample a tiny point-conditioned, face-level FACE checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_face_level_decoder
from clearmesh.mesh_heads.face_tokens import (
    FaceTokenSequence,
    FaceTokenTransform,
    decode_face_tokens_to_mesh,
)


def _load_conditioning(path: Path, point_samples: int | None = None) -> tuple[np.ndarray, FaceTokenTransform, int, np.ndarray]:
    data = np.load(path)
    points = np.asarray(data["surface_points"], dtype=np.float32)
    normals = np.asarray(data["surface_normals"], dtype=np.float32)
    if point_samples is not None and point_samples > 0:
        if len(points) >= point_samples:
            points = points[:point_samples]
            normals = normals[:point_samples]
        else:
            repeat = int(np.ceil(point_samples / len(points)))
            points = np.tile(points, (repeat, 1))[:point_samples]
            normals = np.tile(normals, (repeat, 1))[:point_samples]
    transform = FaceTokenTransform(
        center=tuple(np.asarray(data["center"], dtype=np.float64).reshape(3).tolist()),
        scale=float(np.asarray(data["scale"], dtype=np.float64).reshape(-1)[0]),
    )
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
    gt_tokens = np.asarray(data["tokens"], dtype=np.int64)
    return np.concatenate([points, normals], axis=1), transform, num_bins, gt_tokens


def _load_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        reason = str(exc).splitlines()[0]
        print(
            f"Falling back to trusted checkpoint load for legacy metadata: {reason}",
            file=sys.stderr,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, default=None)
    parser.add_argument("--face-count", type=int, default=0)
    parser.add_argument("--point-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    import torch

    checkpoint = _load_checkpoint(args.checkpoint)
    train_args = checkpoint.get("args", {})
    point_features, transform, sample_bins, gt_tokens = _load_conditioning(
        args.sample,
        point_samples=args.point_samples or train_args.get("point_samples") or None,
    )
    num_bins = int(checkpoint.get("num_bins", sample_bins))
    max_faces = int(checkpoint.get("max_faces", len(gt_tokens)))
    face_count = int(args.face_count or max_faces)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = build_tiny_point_conditioned_face_level_decoder(
        num_bins=num_bins,
        max_faces=max_faces,
        point_feature_dim=6,
        hidden_size=int(train_args.get("hidden_size", 192)),
        layers=int(train_args.get("layers", 4)),
        heads=int(train_args.get("heads", 6)),
        condition_tokens=int(train_args.get("condition_tokens", 8)),
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    if args.face_count <= 0 and checkpoint.get("has_count_head"):
        point_preview = torch.as_tensor(point_features, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            predicted = int(torch.argmax(model.predict_face_count_logits(point_preview), dim=-1).item())
        face_count = max(1, min(max_faces, predicted))

    point_tensor = torch.as_tensor(point_features, dtype=torch.float32, device=device).unsqueeze(0)
    input_faces = torch.zeros((1, 1, 9), dtype=torch.long, device=device)
    generated: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(face_count):
            logits = model(point_tensor, input_faces)[:, -1, :, :]
            if args.temperature and args.temperature > 0.0:
                probs = torch.softmax(logits / args.temperature, dim=-1)
                next_face = torch.multinomial(probs.reshape(-1, num_bins), num_samples=1).reshape(1, 9)
            else:
                next_face = torch.argmax(logits, dim=-1)
            generated.append(next_face.squeeze(0).detach().cpu().numpy().astype(np.int64))
            if len(generated) < face_count:
                input_faces = torch.cat([input_faces, next_face.unsqueeze(1)], dim=1)

    tokens = np.stack(generated, axis=0)
    mesh = decode_face_tokens_to_mesh(FaceTokenSequence(tokens=tokens, num_bins=num_bins, transform=transform))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)

    teacher_path = None
    if args.teacher_output is not None:
        args.teacher_output.parent.mkdir(parents=True, exist_ok=True)
        teacher = decode_face_tokens_to_mesh(FaceTokenSequence(gt_tokens, num_bins=num_bins, transform=transform))
        teacher.export(args.teacher_output)
        teacher_path = str(args.teacher_output)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "teacher_output": teacher_path,
                "generated_faces": int(len(mesh.faces)),
                "generated_vertices": int(len(mesh.vertices)),
                "is_watertight": bool(mesh.is_watertight),
                "requested_face_count": int(face_count),
                "missing_checkpoint_keys": list(incompatible.missing_keys),
                "unexpected_checkpoint_keys": list(incompatible.unexpected_keys),
                "device": str(device),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
