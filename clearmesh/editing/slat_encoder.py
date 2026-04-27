#!/usr/bin/env python3
"""SLAT Encoder — Access TRELLIS.2's Sparse Latent representation.

SLAT (Sparse Latent) = (V, {z_p}) where:
  V: Voxel structure — sparse 3D occupancy (coords in a 32^3 grid)
  {z_p}: Per-voxel features — 32-dim shape latent per occupied voxel

This module wraps the real TRELLIS.2 APIs used throughout the repo:
  - Mesh → O-Voxel: `o_voxel.convert.mesh_to_flexible_dual_grid` (CPU) or
    the sparse CPU kernel used by `scripts/data/validate_alignment_noise.py`
  - O-Voxel → shape SLAT: `FlexiDualGridVaeEncoder` loaded from
    `{model_dir}/ckpts/shape_enc_next_dc_f16c32_fp16.{json,safetensors}`
  - Shape SLAT → mesh: `pipeline.decode_shape_slat(slat, 512) -> (meshes, _)`

What is NOT provided (by design):
  - **Sparse Structure (SS) encoding** — a dense 3D VAE that compresses
    voxel occupancy into a compact latent for the SS flow model. No public
    TRELLIS.2 pipeline method exposes this, and it is not imported anywhere
    in the existing ClearMesh scripts (only `sparse_structure_decoder` and
    `sparse_structure_flow_model` are used). `_encode_ss_latent` therefore
    remains blocked: Easy3E voxel-flow editing can't be bootstrapped from
    a mesh without it. See STATUS.md for details.

  - SS latent in SLATRepresentation is therefore stored as the **integer
    voxel coords** (sparse structure = occupancy at those positions), not a
    compressed SS VAE latent. That's the same structure used by UltraShape
    and Stage 2 throughout ClearMesh.

Usage:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("/workspace/models/trellis2-4b").cuda()

    encoder = SLATEncoder(pipeline=pipeline, model_dir="/workspace/models/trellis2-4b")
    slat = encoder.encode(mesh_path="model.glb")
    # slat.shape_latent: (N, 32) — per-voxel shape features
    # slat.voxel_indices: (N, 3) — integer voxel coords
    # slat.ss_latent: same as voxel_indices (see note above)

    mesh = encoder.decode(slat)  # roundtrip
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh


@dataclass
class SLATRepresentation:
    """Sparse Latent representation as used inside TRELLIS.2.

    This is the core intermediate that Easy3E edits. Fields match the
    SparseTensor contract that `pipeline.decode_shape_slat` expects.
    """

    # Per-voxel 32-dim shape features (the "{z_p}" in SLAT notation).
    shape_latent: torch.Tensor  # (N, 32)

    # Occupied voxel integer indices in [0, grid_size).
    voxel_indices: torch.Tensor  # (N, 3) int32

    # Sparse-structure "latent" — in practice identical to voxel_indices
    # because TRELLIS.2's public pipeline does not expose an SS VAE
    # encoder (see module docstring + STATUS.md). Kept as a separate
    # field so downstream code can pivot if a true SS latent becomes
    # available later.
    ss_latent: torch.Tensor  # (N, 3) int32 — alias for voxel_indices

    # Original TRELLIS.2 SparseTensor if produced via the public pipeline.
    # None when encoding from a mesh directly (encoder returns its own
    # SparseTensor via `shape_slat_obj`).
    shape_slat_obj: object | None = None

    # Per-voxel dual-grid vertex positions and edge-intersection flags
    # (outputs of `mesh_to_flexible_dual_grid`). Retained for advanced
    # decoding paths; the default `decode()` does not require them.
    dual_vertices: torch.Tensor | None = None
    intersected: torch.Tensor | None = None

    grid_size: int = 256


class SLATEncoder:
    """Encode/decode meshes to/from TRELLIS.2's SLAT representation.

    Wraps the real TRELLIS.2 APIs. Two construction modes:

      1. Pass a loaded ``pipeline`` (``Trellis2ImageTo3DPipeline``). Used
         for ``decode()``. If you only need encoding, ``pipeline`` can be
         None.

      2. Pass ``model_dir`` so the shape encoder can be loaded from
         ``{model_dir}/ckpts/shape_enc_next_dc_f16c32_fp16.*``.
    """

    def __init__(
        self,
        pipeline=None,
        trellis2_dir: str = "/workspace/TRELLIS.2",
        model_dir: str = "/workspace/models/trellis2-4b",
        device: str | None = None,
        grid_size: int = 256,
    ):
        self.pipeline = pipeline
        self.trellis2_dir = Path(trellis2_dir)
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_size = grid_size

        self._shape_encoder = None
        self._o_voxel = None

        # Make TRELLIS.2 importable (matches setup_trellis2 in infer_slat.py).
        trellis2_str = str(self.trellis2_dir)
        if trellis2_str not in sys.path:
            sys.path.insert(0, trellis2_str)

    # ── Lazy loaders ────────────────────────────────────────────────────

    @property
    def o_voxel(self):
        """Lazy-load o_voxel module (TRELLIS.2 dep)."""
        if self._o_voxel is None:
            import o_voxel  # noqa: F401

            self._o_voxel = o_voxel
        return self._o_voxel

    def _load_shape_encoder(self):
        """Load FlexiDualGridVaeEncoder from checkpoint.

        Mirrors the pattern in
        ``scripts/data/validate_alignment_noise.py`` which works end-to-end
        on TRELLIS.2 4B.
        """
        if self._shape_encoder is not None:
            return self._shape_encoder

        import json

        from safetensors.torch import load_file
        from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeEncoder

        enc_cfg_path = self.model_dir / "ckpts" / "shape_enc_next_dc_f16c32_fp16.json"
        enc_wt_path = self.model_dir / "ckpts" / "shape_enc_next_dc_f16c32_fp16.safetensors"

        if not enc_cfg_path.exists() or not enc_wt_path.exists():
            raise FileNotFoundError(
                "Shape encoder checkpoint not found at "
                f"{enc_cfg_path.parent}. Expected "
                "shape_enc_next_dc_f16c32_fp16.{json,safetensors}."
            )

        with open(enc_cfg_path) as f:
            enc_cfg = json.load(f)
        encoder = FlexiDualGridVaeEncoder(**enc_cfg["args"])
        encoder.load_state_dict(load_file(str(enc_wt_path)))
        encoder = encoder.to(self.device).eval()
        self._shape_encoder = encoder
        return encoder

    def _try_load_ss_encoder(self):
        """Best-effort discovery of a Sparse-Structure VAE encoder.

        TRELLIS.2's public pipeline exposes an SS *decoder* and an SS *flow
        model*, but no SS encoder — we need one to turn a source mesh's
        voxel occupancy into an SS latent for voxel-flow editing (Easy3E).

        This helper introspects ``trellis2.models`` for plausible encoder
        class names, and also checks ``{model_dir}/ckpts/`` for a config +
        weights pair matching common naming conventions. If both succeed,
        returns an eval-mode encoder; otherwise returns None so the caller
        can fall back (we currently alias ``ss_latent`` to ``voxel_indices``).

        Naming scanned (case-insensitive substring match on trellis2.models):
          - SparseStructureEncoder
          - SparseStructureVae / SparseStructureVAE / SparseStructureVaeEncoder
          - Anything containing both 'sparse' and 'encoder' (or 'enc')
            and does NOT already contain 'decoder'

        Ckpt file patterns (under ``{model_dir}/ckpts/``):
          - ss_enc*.{json,safetensors}
          - sparse_structure_enc*.{json,safetensors}
          - ss_vae*.{json,safetensors}  (only when a matching class is
            also available in trellis2.models)

        Returns:
            Loaded encoder module in eval mode, or None.
        """
        import importlib
        import json
        from safetensors.torch import load_file

        # 1. Find an encoder class.
        try:
            trellis_models = importlib.import_module("trellis2.models")
        except ImportError:
            return None

        candidates: list[tuple[str, type]] = []
        for attr in dir(trellis_models):
            low = attr.lower()
            if "decoder" in low:
                continue
            if "sparse" in low and ("encoder" in low or low.endswith("enc")):
                cls = getattr(trellis_models, attr)
                if isinstance(cls, type):
                    candidates.append((attr, cls))
        for preferred in (
            "SparseStructureEncoder",
            "SparseStructureVaeEncoder",
            "SparseStructureVae",
            "SparseStructureVAE",
        ):
            cls = getattr(trellis_models, preferred, None)
            if isinstance(cls, type):
                candidates.insert(0, (preferred, cls))

        if not candidates:
            return None

        # 2. Look for a matching checkpoint.
        ckpt_dir = self.model_dir / "ckpts"
        if not ckpt_dir.exists():
            return None

        patterns = ("ss_enc*", "sparse_structure_enc*", "ss_vae*", "sparse_structure_vae*")
        cfg_path = wts_path = None
        for pat in patterns:
            cfgs = list(ckpt_dir.glob(f"{pat}.json"))
            for c in cfgs:
                w = c.with_suffix(".safetensors")
                if w.exists():
                    cfg_path, wts_path = c, w
                    break
            if cfg_path:
                break

        if not cfg_path:
            return None

        # 3. Try each candidate class with the first matching ckpt.
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg_args = cfg.get("args", cfg)

        for name, cls in candidates:
            try:
                enc = cls(**cfg_args)
                enc.load_state_dict(load_file(str(wts_path)))
                enc = enc.to(self.device).eval()
                print(
                    f"  [SLATEncoder] Loaded SS encoder {name} "
                    f"from {wts_path.name}"
                )
                return enc
            except Exception:
                continue
        return None

    # ── Mesh → O-Voxel ──────────────────────────────────────────────────

    def mesh_to_ovoxel(
        self,
        mesh_path: str | Path | trimesh.Trimesh,
        grid_size: int | None = None,
    ) -> dict:
        """Convert a mesh to O-Voxel (flexible dual grid) representation.

        Uses the CPU path ``_C.mesh_to_flexible_dual_grid_cpu`` from
        ``o_voxel.convert.flexible_dual_grid`` — matches the working impl
        in ``encode_mesh_to_slat`` in ``validate_alignment_noise.py``.

        Args:
            mesh_path: Path to mesh file OR a trimesh.Trimesh instance.
            grid_size: Voxel grid resolution (default: self.grid_size).

        Returns:
            Dict with ``coords`` (N,3 int32), ``dual_vertices`` (N,D float32),
            ``intersected`` (N,D float32), ``grid_size`` (int).
        """
        grid_size = grid_size or self.grid_size

        # Accept either a path or a pre-loaded trimesh.
        if isinstance(mesh_path, trimesh.Trimesh):
            mesh = mesh_path
        else:
            mesh = trimesh.load(str(mesh_path), force="mesh")

        verts = torch.from_numpy(mesh.vertices.astype("float32"))
        faces = torch.from_numpy(mesh.faces.astype("int32"))

        # Same padding/aabb logic as validate_alignment_noise.py:67.
        min_xyz = verts.min(dim=0).values
        max_xyz = verts.max(dim=0).values
        gs = torch.tensor([grid_size] * 3, dtype=torch.int32)
        padding = (max_xyz - min_xyz) / (gs.float() - 1)
        min_xyz = min_xyz - padding * 0.5
        max_xyz = max_xyz + padding * 0.5
        aabb = torch.stack([min_xyz, max_xyz], dim=0).float()
        voxel_size = (aabb[1] - aabb[0]) / gs.float()

        vertices_shifted = verts - aabb[0].reshape(1, 3)
        grid_range = torch.stack([torch.zeros_like(gs), gs], dim=0).int()

        from o_voxel.convert.flexible_dual_grid import _C

        coords, dual_verts, intersected = _C.mesh_to_flexible_dual_grid_cpu(
            vertices_shifted,
            faces,
            voxel_size,
            grid_range,
            1.0,  # face_weight
            1.0,  # boundary_weight
            0.1,  # regularization_weight
            False,
        )

        return {
            "coords": coords.int(),
            "dual_vertices": dual_verts.float(),
            "intersected": intersected.float(),
            "grid_size": grid_size,
            "aabb": aabb,
        }

    # ── Full encode / decode ────────────────────────────────────────────

    def encode(
        self,
        mesh_path: str | Path | trimesh.Trimesh,
        grid_size: int | None = None,
    ) -> SLATRepresentation:
        """Encode a mesh to a SLAT representation.

        Runs the real pipeline used in
        ``scripts/data/validate_alignment_noise.py``::

            mesh → O-Voxel (CPU) → FlexiDualGridVaeEncoder → SLAT

        Args:
            mesh_path: Path to mesh file OR a trimesh.Trimesh.
            grid_size: Voxel grid resolution.

        Returns:
            SLATRepresentation with ``shape_latent`` and ``voxel_indices``
            populated. ``ss_latent`` is aliased to ``voxel_indices`` (see
            module docstring).
        """
        grid_size = grid_size or self.grid_size
        ovox = self.mesh_to_ovoxel(mesh_path, grid_size)

        n_voxels = ovox["coords"].shape[0]
        if n_voxels == 0:
            raise RuntimeError(
                "mesh_to_flexible_dual_grid returned 0 voxels. "
                "Is the mesh empty / degenerate?"
            )

        # Build SparseTensors the encoder expects. Same construction as
        # encode_mesh_to_slat() in validate_alignment_noise.py:95.
        from trellis2.modules import sparse as sp

        batch_idx = torch.zeros(n_voxels, 1, dtype=torch.int32)
        coords_4d = torch.cat([batch_idx, ovox["coords"]], dim=1).to(self.device)
        vst = sp.SparseTensor(
            feats=ovox["dual_vertices"].to(self.device),
            coords=coords_4d,
        )
        ist = sp.SparseTensor(
            feats=ovox["intersected"].to(self.device),
            coords=coords_4d,
        )

        encoder = self._load_shape_encoder()
        with torch.no_grad():
            z = encoder(vst, ist, sample_posterior=False)

        shape_latent = z.feats.float()           # (N', 32)
        voxel_indices = z.coords[:, 1:].int()    # (N', 3)

        return SLATRepresentation(
            shape_latent=shape_latent,
            voxel_indices=voxel_indices,
            ss_latent=voxel_indices,  # alias
            shape_slat_obj=z,
            dual_vertices=ovox["dual_vertices"],
            intersected=ovox["intersected"],
            grid_size=grid_size,
        )

    def decode(self, slat: SLATRepresentation, resolution: int = 512) -> trimesh.Trimesh:
        """Decode a SLAT representation back to a mesh.

        Calls ``pipeline.decode_shape_slat(slat_sparse_tensor, resolution)``,
        which internally runs the FlexiDualGridVaeDecoder. Resolution MUST
        be 512 — passing a smaller grid_size corrupts the hashmap in mesh
        extraction (documented in ``clearmesh/stage2/infer_slat.py:468``).

        Args:
            slat: SLAT representation to decode.
            resolution: Decode grid resolution. Keep at 512.

        Returns:
            Decoded trimesh.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "SLATEncoder.decode requires a loaded Trellis2ImageTo3DPipeline. "
                "Pass pipeline=... to the constructor."
            )

        # Prefer the stored SparseTensor if we have it (feature edits can
        # replace .feats in-place). Otherwise rebuild from shape_latent +
        # voxel_indices.
        if slat.shape_slat_obj is not None and hasattr(slat.shape_slat_obj, "feats"):
            st = slat.shape_slat_obj
            st.feats = slat.shape_latent.to(st.feats.device).to(st.feats.dtype)
        else:
            from trellis2.modules import sparse as sp

            n = slat.voxel_indices.shape[0]
            batch_idx = torch.zeros(n, 1, dtype=torch.int32, device=slat.shape_latent.device)
            coords_4d = torch.cat(
                [batch_idx, slat.voxel_indices.int().to(slat.shape_latent.device)], dim=1
            )
            st = sp.SparseTensor(feats=slat.shape_latent, coords=coords_4d)

        with torch.no_grad():
            meshes, _ = self.pipeline.decode_shape_slat(st, resolution)

        mesh_obj = meshes[0]
        v = mesh_obj.vertices.detach().cpu().float().numpy()
        f = mesh_obj.faces.detach().cpu().numpy()
        return trimesh.Trimesh(vertices=v, faces=f)

    # ── Save / load ─────────────────────────────────────────────────────

    def save_slat(self, slat: SLATRepresentation, path: str | Path) -> None:
        """Save SLAT to disk (tensors only — ``shape_slat_obj`` skipped)."""
        torch.save(
            {
                "shape_latent": slat.shape_latent.cpu(),
                "voxel_indices": slat.voxel_indices.cpu(),
                "ss_latent": slat.ss_latent.cpu() if slat.ss_latent is not None else None,
                "dual_vertices": (
                    slat.dual_vertices.cpu() if slat.dual_vertices is not None else None
                ),
                "intersected": (
                    slat.intersected.cpu() if slat.intersected is not None else None
                ),
                "grid_size": slat.grid_size,
            },
            str(path),
        )

    def load_slat(self, path: str | Path) -> SLATRepresentation:
        """Load SLAT from disk.

        Note: ``shape_slat_obj`` is not serialized — ``decode()`` will
        rebuild a SparseTensor from ``voxel_indices`` + ``shape_latent``.
        """
        data = torch.load(str(path), map_location=self.device, weights_only=True)
        vox = data["voxel_indices"]
        return SLATRepresentation(
            shape_latent=data["shape_latent"].to(self.device),
            voxel_indices=vox.to(self.device),
            ss_latent=(data.get("ss_latent") if data.get("ss_latent") is not None else vox).to(
                self.device
            ),
            dual_vertices=(
                data["dual_vertices"].to(self.device) if data.get("dual_vertices") is not None else None
            ),
            intersected=(
                data["intersected"].to(self.device) if data.get("intersected") is not None else None
            ),
            grid_size=data.get("grid_size", 256),
        )
