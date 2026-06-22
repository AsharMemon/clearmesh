"""FACE-paper mesh head: point cloud -> autoregressive triangle mesh.

Wraps the trained FACE autoregressive autoencoder (``build_paper_face_arae``)
as a ClearMesh mesh-head adapter so the product pipeline can call it the same
way it calls every other head: ``build_mesh_head("face_paper", cfg).run(input)``.

Design mirrors the other external heads (MeshRipple/DeepMesh): the adapter is a
thin *subprocess* wrapper. It shells out to this module's own CLI
(``python -m clearmesh.mesh_heads.face_paper_head``) so the heavy Torch / model
dependencies stay out of the pipeline-worker process. Importing this module for
the adapter alone stays light -- Torch is imported lazily, only inside the
inference functions that the CLI runs.

The faithful FACE model has *no* EOS head (the paper is silent on stopping, and
our paper-faithful run disabled it), so generation needs a target face count.
Resolution order: explicit ``face_count`` config -> ``metadata["face_count"]``
hint on the input -> ``default_face_count``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from .base import (
    MeshHeadError,
    MeshHeadInput,
    MeshHeadResult,
    discover_mesh_outputs,
    run_external_command,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Adapter (subprocess wrapper). Light imports only -- no torch at module load.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FacePaperConfig:
    """Configuration for the FACE-paper mesh head."""

    checkpoint: Path
    python: str = sys.executable
    point_samples: int = 16384
    face_count: int = 0           # explicit override; 0 => derive (see _resolve_face_count)
    default_face_count: int = 800
    min_face_count: int = 32      # floor when deriving from the proxy mesh
    max_face_count: int = 2048    # ceiling when deriving from the proxy mesh
    device: str = "auto"          # auto|cpu|cuda|mps
    timeout_seconds: int | None = 1800
    env: Mapping[str, str] = field(default_factory=dict)


class FacePaperMeshHead:
    """Adapter: a point cloud (+ normals) -> a FACE-generated triangle mesh."""

    requires_point_cloud = True
    adapter_name = "face_paper"

    def __init__(self, config: FacePaperConfig) -> None:
        self.config = config

    def _resolve_face_count(self, mesh_input: MeshHeadInput) -> int:
        """Pick the AR generation length (the faithful model has no EOS head).

        Order: explicit config > ``metadata["face_count"]`` hint > the proxy
        mesh's face count (the coarse mesh we are re-meshing, clamped to
        [min, max]) > ``default_face_count``.
        """
        if self.config.face_count and self.config.face_count > 0:
            return int(self.config.face_count)
        hint = (mesh_input.metadata or {}).get("face_count")
        try:
            if hint and int(hint) > 0:
                return int(hint)
        except (TypeError, ValueError):
            pass
        proxy = mesh_input.proxy_mesh_path
        if proxy is not None and Path(proxy).exists():
            try:
                import trimesh

                proxy_mesh = trimesh.load(proxy, process=False, force="mesh")
                proxy_faces = int(len(proxy_mesh.faces))
                if proxy_faces > 0:
                    return max(self.config.min_face_count, min(proxy_faces, self.config.max_face_count))
            except Exception:
                pass
        return int(self.config.default_face_count)

    def build_command(self, mesh_input: MeshHeadInput) -> list[str]:
        cfg = self.config
        return [
            str(cfg.python),
            "-m",
            "clearmesh.mesh_heads.face_paper_head",
            "--checkpoint", str(cfg.checkpoint),
            "--point-cloud", str(mesh_input.point_cloud_path),
            "--output-dir", str(mesh_input.output_dir),
            "--case-id", str(mesh_input.case_id),
            "--point-samples", str(cfg.point_samples),
            "--face-count", str(self._resolve_face_count(mesh_input)),
            "--device", str(cfg.device),
        ]

    def run(self, mesh_input: MeshHeadInput) -> MeshHeadResult:
        if mesh_input.point_cloud_path is None or not Path(mesh_input.point_cloud_path).exists():
            raise MeshHeadError(
                f"FACE head requires an existing point cloud; got {mesh_input.point_cloud_path!r}"
            )
        if not Path(self.config.checkpoint).exists():
            raise MeshHeadError(f"FACE checkpoint not found: {self.config.checkpoint}")

        output_dir = Path(mesh_input.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / f"{mesh_input.case_id}_face_paper.stdout.log"
        stderr_path = output_dir / f"{mesh_input.case_id}_face_paper.stderr.log"

        since = time.time()
        command = self.build_command(mesh_input)
        run_external_command(
            command,
            cwd=_REPO_ROOT,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            env=self.config.env,
            timeout_seconds=self.config.timeout_seconds,
        )
        meshes = discover_mesh_outputs(output_dir, since_mtime=since)
        if not meshes:
            raise MeshHeadError(
                f"FACE head completed but produced no mesh in {output_dir}. "
                f"See {stdout_path} and {stderr_path}."
            )
        return MeshHeadResult(
            mesh_path=meshes[0],
            adapter_name=self.adapter_name,
            command=command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata={"point_cloud_path": str(mesh_input.point_cloud_path)},
        )


# --------------------------------------------------------------------------- #
# Inference CLI. Torch is imported lazily inside these functions only.
# --------------------------------------------------------------------------- #
def _resolve_device(name: str):  # type: ignore[no-untyped-def]
    import torch

    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_face_paper_model(checkpoint_path: Path, device):  # type: ignore[no-untyped-def]
    """Rebuild the FACE model purely from checkpoint metadata (mirrors eval)."""
    import torch

    from .face_paper import build_paper_face_arae

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    train_args = checkpoint.get("args", {}) or {}
    num_bins = int(checkpoint["num_bins"])
    max_faces = int(checkpoint["max_faces"])
    decode_head = str(checkpoint.get("decode_head") or train_args.get("decode_head") or "causal")
    encoder_backend = str(
        checkpoint.get("encoder_backend") or train_args.get("encoder_backend") or "shape2vecset_normals_concat"
    )
    causal_mlp_variant = str(
        checkpoint.get("causal_mlp_variant") or train_args.get("causal_mlp_variant") or "legacy_concat"
    )
    face_embedding_variant = str(
        checkpoint.get("face_embedding_variant") or train_args.get("face_embedding_variant") or "supplement_concat"
    )
    first_face_head = str(checkpoint.get("first_face_head") or train_args.get("first_face_head") or "none")
    condition_prefix_tokens = int(
        checkpoint.get("condition_prefix_tokens", train_args.get("condition_prefix_tokens", 0))
    )
    condition_prefix_mode = str(
        checkpoint.get("condition_prefix_mode") or train_args.get("condition_prefix_mode") or "learned_queries"
    )
    decoder_cross_attention = bool(
        checkpoint.get(
            "decoder_cross_attention",
            checkpoint.get("has_decoder_cross_attention", train_args.get("decoder_cross_attention", True)),
        )
    )
    has_eos_head = bool(checkpoint.get("has_eos_head", False))

    model = build_paper_face_arae(
        num_bins=num_bins,
        max_faces=max_faces,
        point_feature_dim=6,
        hidden_size=int(train_args.get("hidden_size", 256)),
        encoder_hidden_size=int(train_args.get("encoder_hidden_size", train_args.get("hidden_size", 256))),
        encoder_layers=int(train_args.get("encoder_layers", 4)),
        decoder_layers=int(train_args.get("decoder_layers", 4)),
        heads=int(train_args.get("heads", 8)),
        vecset_tokens=int(train_args.get("vecset_tokens", 128)),
        latent_dim=int(train_args.get("latent_dim", 64)),
        face_token_embed_dim=int(train_args.get("face_token_embed_dim", 0)) or None,
        encoder_drop_path_rate=float(train_args.get("encoder_drop_path_rate", 0.1)),
        encoder_backend=encoder_backend,
        causal_mlp_variant=causal_mlp_variant,
        face_embedding_variant=face_embedding_variant,
        first_face_head=first_face_head,
        condition_prefix_tokens=condition_prefix_tokens,
        condition_prefix_mode=condition_prefix_mode,
        decoder_cross_attention=decoder_cross_attention,
        enable_eos_head=has_eos_head,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, num_bins, max_faces, decode_head, has_eos_head


def _estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Local-PCA normals, oriented outward from the centroid (fallback only)."""
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    centroid = pts.mean(axis=0)
    normals = np.zeros_like(pts)
    for i in range(len(pts)):
        nb = pts[idx[i]]
        cov = np.cov((nb - nb.mean(axis=0)).T)
        _, vecs = np.linalg.eigh(cov)
        n = vecs[:, 0]
        if np.dot(n, pts[i] - centroid) < 0:
            n = -n
        normals[i] = n
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return normals / lengths


def _load_point_cloud(path: Path, point_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Load points (+ normals) from .npz / .npy / mesh / point-cloud files."""
    path = Path(path)
    suffix = path.suffix.lower()
    normals: np.ndarray | None = None

    if suffix == ".npz":
        data = np.load(path)
        keys = set(data.files)
        pk = next((k for k in ("surface_points", "points", "vertices") if k in keys), None)
        nk = next((k for k in ("surface_normals", "normals", "vertex_normals") if k in keys), None)
        if pk is None:
            raise MeshHeadError(f"NPZ {path} has no point array (keys={sorted(keys)})")
        points = np.asarray(data[pk], dtype=np.float64)
        if nk is not None:
            normals = np.asarray(data[nk], dtype=np.float64)
    elif suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float64)
        points = arr[:, :3]
        if arr.shape[1] >= 6:
            normals = arr[:, 3:6]
    else:
        import trimesh

        obj = trimesh.load(path, process=False)
        if isinstance(obj, trimesh.Scene):
            geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
            obj = trimesh.util.concatenate(geoms) if geoms else obj
        if isinstance(obj, trimesh.Trimesh) and len(obj.faces) > 0:
            pts, fidx = trimesh.sample.sample_surface(obj, max(int(point_samples), 1))
            points = np.asarray(pts, dtype=np.float64)
            normals = np.asarray(obj.face_normals[fidx], dtype=np.float64)
        else:
            verts = np.asarray(getattr(obj, "vertices", obj), dtype=np.float64)
            points = verts
            meta = getattr(obj, "metadata", {}) or {}
            if isinstance(meta, dict) and "vertex_normals" in meta:
                normals = np.asarray(meta["vertex_normals"], dtype=np.float64)

    if normals is None or len(normals) != len(points):
        normals = _estimate_normals(points)
    return points, normals


def _decode_next_face(model, hidden, num_bins: int, device, decode_head: str):  # type: ignore[no-untyped-def]
    """Greedy top-1 decode of one face from the current hidden state (eval-faithful)."""
    import torch

    if decode_head == "parallel":
        logits = model.parallel_head(hidden).reshape(hidden.shape[0], hidden.shape[1], 9, num_bins)[:, -1, :, :]
        return torch.argmax(logits[:, :, :num_bins], dim=-1)
    if hasattr(model, "greedy_face_from_hidden"):
        return model.greedy_face_from_hidden(hidden, limit_bins=num_bins)
    prefix = torch.full((hidden.shape[0], 9), -1, dtype=torch.long, device=device)
    for coord in range(9):
        logits = model._causal_logits_from_hidden(hidden, prefix.reshape(hidden.shape[0], 1, 9))[:, 0, :, :]
        prefix[:, coord] = torch.argmax(logits[:, coord, :num_bins], dim=-1)
    return prefix


def run_inference(
    *,
    checkpoint: Path,
    point_cloud: Path,
    output_dir: Path,
    case_id: str,
    point_samples: int,
    face_count: int,
    device_name: str,
) -> Path:
    """End-to-end: point cloud -> normalize -> encode -> greedy AR -> mesh GLB."""
    import torch

    from .face_tokens import FaceTokenSequence, decode_paper_face_tokens_to_mesh, fit_face_token_transform

    t0 = time.time()
    device = _resolve_device(device_name)
    model, num_bins, max_faces, decode_head, has_eos = _load_face_paper_model(Path(checkpoint), device)

    points, normals = _load_point_cloud(Path(point_cloud), point_samples)
    n = len(points)
    if point_samples and point_samples > 0:
        if n >= point_samples:
            points = points[:point_samples]
            normals = normals[:point_samples]
        else:
            rep = int(np.ceil(point_samples / max(n, 1)))
            points = np.tile(points, (rep, 1))[:point_samples]
            normals = np.tile(normals, (rep, 1))[:point_samples]

    transform = fit_face_token_transform(points)
    points_norm = transform.normalize(points)
    point_features = np.concatenate([points_norm, normals], axis=1).astype(np.float32)

    target_faces = int(face_count) if face_count and face_count > 0 else min(max_faces, 800)
    target_faces = max(1, min(target_faces, max_faces))

    point_tensor = torch.as_tensor(point_features, dtype=torch.float32, device=device).unsqueeze(0)
    generated: list[np.ndarray] = []
    with torch.no_grad():
        cache = model.init_incremental_cache(point_tensor)
        previous_face = torch.full((1, 9), -1, dtype=torch.long, device=device)
        for position in range(target_faces):
            hidden = model.incremental_hidden_step(previous_face, position, cache)
            next_face = _decode_next_face(model, hidden, num_bins, device, decode_head)
            generated.append(next_face.squeeze(0).detach().cpu().numpy().astype(np.int64))
            previous_face = next_face
    tokens = np.stack(generated, axis=0) if generated else np.zeros((1, 9), dtype=np.int64)

    sequence = FaceTokenSequence(tokens=tokens, num_bins=num_bins, transform=transform)
    mesh = decode_paper_face_tokens_to_mesh(sequence, denormalize=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{case_id}.glb"
    mesh.export(out_path)
    print(
        f"[face_paper] {out_path.name}: faces={len(mesh.faces)} verts={len(mesh.vertices)} "
        f"watertight={mesh.is_watertight} | device={device} num_bins={num_bins} "
        f"gen_faces={target_faces} eos_head={has_eos} in {time.time() - t0:.1f}s",
        flush=True,
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FACE-paper point-cloud -> mesh inference")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--point-cloud", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-id", default="face_paper")
    parser.add_argument("--point-samples", type=int, default=16384)
    parser.add_argument("--face-count", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    run_inference(
        checkpoint=args.checkpoint,
        point_cloud=args.point_cloud,
        output_dir=args.output_dir,
        case_id=args.case_id,
        point_samples=args.point_samples,
        face_count=args.face_count,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
