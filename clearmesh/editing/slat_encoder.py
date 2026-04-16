#!/usr/bin/env python3
"""SLAT Encoder — Access TRELLIS.2's Sparse Latent representation.

SLAT (Sparse Latent) = (V, {z_p}) where:
  V: Voxel structure — binary occupancy encoded by 3D VAE into continuous latent
  {z_p}: Per-voxel features — fused from multi-view DINOv2 embeddings

This module wraps TRELLIS.2's data_toolkit encoders:
  - encode_shape_latent.py → shape latent (3D VAE encoding of voxel structure)
  - encode_ss_latent.py → sparse structure latent (for flow-matching)
  - dual_grid.py → O-Voxel conversion (mesh → flexible dual grid)

Usage:
    encoder = SLATEncoder(trellis2_dir="/workspace/TRELLIS.2")
    slat = encoder.encode(mesh_path="model.glb")
    # slat.ss_latent: Tensor — sparse structure latent
    # slat.shape_latent: Tensor — shape latent
    # slat.voxel_indices: Tensor — occupied voxel positions
    # slat.dual_vertices: Tensor — dual grid vertex positions

    # Roundtrip test
    mesh = encoder.decode(slat)
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
    """Sparse Latent representation from TRELLIS.2.

    This is the core intermediate representation that Easy3E edits.
    """

    ss_latent: torch.Tensor  # Sparse structure latent (B, N, D_ss)
    shape_latent: torch.Tensor  # Shape latent from 3D VAE (B, N, D_shape)
    voxel_indices: torch.Tensor  # Occupied voxel indices (N, 3)
    dual_vertices: torch.Tensor  # Dual grid vertex positions (M, 3)
    intersected: torch.Tensor | None = None  # Edge intersection flags
    grid_size: int = 256  # Resolution of the voxel grid


class SLATEncoder:
    """Encode/decode meshes to/from TRELLIS.2's SLAT representation.

    Wraps the TRELLIS.2 data_toolkit for:
      1. Mesh → O-Voxel (dual grid) conversion
      2. O-Voxel → sparse structure latent encoding
      3. O-Voxel → shape latent encoding
      4. Latent → mesh decoding (via TRELLIS.2 decoder)
    """

    def __init__(
        self,
        trellis2_dir: str = "/workspace/TRELLIS.2",
        model_dir: str = "/workspace/models/trellis2-4b",
        device: str | None = None,
        grid_size: int = 256,
        pipeline=None,
    ):
        """
        Args:
            trellis2_dir: Root of the TRELLIS.2 git checkout (must contain
                the `trellis2/` package and `trellis2/data_toolkit/`).
            model_dir: Directory containing TRELLIS.2-4B weights. Not strictly
                required if ``pipeline`` is passed in.
            device: Compute device.
            grid_size: O-Voxel grid resolution.
            pipeline: Optional pre-loaded ``Trellis2ImageTo3DPipeline``.
                When passed in, the SS encoder / decoder / shape encoder
                are pulled out of ``pipeline.models`` rather than being
                re-loaded from disk. This is the intended path when the
                editor and the SLAT encoder share one pipeline instance
                (saves ~20s and ~8 GB of GPU memory per editor session).
        """
        self.trellis2_dir = Path(trellis2_dir)
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_size = grid_size
        self._pipeline = pipeline

        # Lazy-loaded components
        self._o_voxel = None
        self._ss_encoder = None
        self._shape_encoder = None
        self._decoder = None

        # Add TRELLIS.2 to path if needed
        trellis2_str = str(self.trellis2_dir)
        if trellis2_str not in sys.path:
            sys.path.insert(0, trellis2_str)

    @property
    def o_voxel(self):
        """Lazy-load o_voxel module."""
        if self._o_voxel is None:
            import o_voxel

            self._o_voxel = o_voxel
        return self._o_voxel

    def mesh_to_ovoxel(
        self,
        mesh_path: str | Path,
        grid_size: int | None = None,
    ) -> dict:
        """Convert a mesh to O-Voxel (flexible dual grid) representation.

        Args:
            mesh_path: Path to mesh file (GLB/OBJ/PLY).
            grid_size: Voxel grid resolution (default: self.grid_size).

        Returns:
            Dictionary with voxel_indices, dual_vertices, intersected tensors.
        """
        grid_size = grid_size or self.grid_size

        # Load and normalize mesh
        mesh = trimesh.load(str(mesh_path), force="mesh")
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.int64)

        # Center and scale to [-0.5, 0.5]
        v_min = vertices.min(dim=0)[0]
        v_max = vertices.max(dim=0)[0]
        center = (v_min + v_max) / 2
        scale = 0.99999 / (v_max - v_min).max()
        vertices = (vertices - center) * scale

        # Convert to O-Voxel
        voxel_indices, dual_vertices, intersected = (
            self.o_voxel.convert.mesh_to_flexible_dual_grid(
                vertices=vertices,
                faces=faces,
                grid_size=grid_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                face_weight=1.0,
                boundary_weight=0.2,
                regularization_weight=1e-2,
            )
        )

        return {
            "voxel_indices": voxel_indices,
            "dual_vertices": dual_vertices,
            "intersected": intersected,
            "grid_size": grid_size,
        }

    def encode(
        self,
        mesh_path: str | Path,
        grid_size: int | None = None,
    ) -> SLATRepresentation:
        """Encode a mesh to full SLAT representation.

        This runs the full encoding pipeline:
          mesh → O-Voxel → (ss_latent, shape_latent)

        Args:
            mesh_path: Path to mesh file.
            grid_size: Voxel grid resolution.

        Returns:
            SLATRepresentation with all latent tensors.
        """
        grid_size = grid_size or self.grid_size
        ovoxel = self.mesh_to_ovoxel(mesh_path, grid_size)

        # Encode sparse structure latent
        ss_latent = self._encode_ss_latent(ovoxel)

        # Encode shape latent
        shape_latent = self._encode_shape_latent(ovoxel)

        return SLATRepresentation(
            ss_latent=ss_latent,
            shape_latent=shape_latent,
            voxel_indices=ovoxel["voxel_indices"],
            dual_vertices=ovoxel["dual_vertices"],
            intersected=ovoxel["intersected"],
            grid_size=grid_size,
        )

    # ------------------------------------------------------------------
    # Encoder resolution — find the right module / weights for each sub-encoder
    # ------------------------------------------------------------------

    def _resolve_ss_encoder(self):
        """Return a callable that maps a dense occupancy grid (B, 1, R, R, R)
        to a sparse-structure latent.

        Resolution order (first that works wins):
          1. ``pipeline.models["sparse_structure_encoder"]`` if the pipeline
             exposes one. TRELLIS.2-4B's pipeline.json currently only lists
             ``sparse_structure_decoder`` under ``models``, so this is the
             uncertain path — included in case future TRELLIS.2 pipelines
             bundle the encoder too.
          2. ``trellis2.data_toolkit.encode_ss_latent`` — the standalone
             encoder script shipped with TRELLIS.2 for dataset preparation.
             Its internal ``encode`` function is the canonical encoder.
          3. Construct a ``SparseStructureEncoder`` module and load the
             companion checkpoint next to the decoder (common HF layout
             is ``ss_enc_conv3d_*`` mirroring ``ss_dec_conv3d_*``).

        Raises RuntimeError with a clear diagnostic if none of the three
        paths resolve. The introspection output from Phase 0 is the
        authoritative source of truth about which path is actually live.
        """
        if self._ss_encoder is not None:
            return self._ss_encoder

        # Path 1: pipeline.models
        if self._pipeline is not None and hasattr(self._pipeline, "models"):
            for key in ("sparse_structure_encoder", "ss_encoder"):
                if key in getattr(self._pipeline, "models", {}):
                    self._ss_encoder = self._pipeline.models[key]
                    return self._ss_encoder

        # Path 2: data_toolkit function
        try:
            from trellis2.data_toolkit import encode_ss_latent as dt_ss
            fn = getattr(dt_ss, "encode", None) or getattr(dt_ss, "encode_ss_latent", None)
            if fn is not None:
                self._ss_encoder = fn
                return self._ss_encoder
        except Exception:
            pass

        # Path 3: from-scratch encoder module + checkpoint
        try:
            from trellis2.models.sparse_structure_vae import SparseStructureEncoder
            ckpt_candidates = sorted(self.model_dir.glob("**/ss_enc_conv3d_*.safetensors"))
            ckpt_candidates += sorted(self.model_dir.glob("**/ss_enc_conv3d_*.pt"))
            if ckpt_candidates:
                enc = SparseStructureEncoder()
                # load state dict (format depends on file extension)
                import safetensors.torch as st
                if ckpt_candidates[0].suffix == ".safetensors":
                    sd = st.load_file(str(ckpt_candidates[0]))
                else:
                    sd = torch.load(ckpt_candidates[0], map_location="cpu")
                enc.load_state_dict(sd, strict=False)
                enc.to(self.device).eval()
                self._ss_encoder = enc
                return self._ss_encoder
        except Exception:
            pass

        raise RuntimeError(
            "No SS encoder found. Tried:\n"
            "  1. pipeline.models['sparse_structure_encoder']\n"
            "  2. trellis2.data_toolkit.encode_ss_latent.encode()\n"
            f"  3. SparseStructureEncoder + checkpoint in {self.model_dir}\n"
            "Run scripts/setup/inspect_trellis2.py on the pod to confirm which path is live. "
            "If no encoder exists, use the 'image-proxy' path: render the source mesh "
            "and invoke pipeline.run(img) to obtain a source SS latent."
        )

    def _resolve_shape_encoder(self):
        """Return a callable that maps O-Voxel dual-grid data to a shape latent."""
        if self._shape_encoder is not None:
            return self._shape_encoder

        if self._pipeline is not None and hasattr(self._pipeline, "models"):
            for key in ("shape_slat_encoder", "shape_encoder"):
                if key in getattr(self._pipeline, "models", {}):
                    self._shape_encoder = self._pipeline.models[key]
                    return self._shape_encoder

        try:
            from trellis2.data_toolkit import encode_shape_latent as dt_shape
            fn = getattr(dt_shape, "encode", None) or getattr(dt_shape, "encode_shape_latent", None)
            if fn is not None:
                self._shape_encoder = fn
                return self._shape_encoder
        except Exception:
            pass

        raise RuntimeError(
            "No shape encoder found. Tried:\n"
            "  1. pipeline.models['shape_slat_encoder']\n"
            "  2. trellis2.data_toolkit.encode_shape_latent.encode()\n"
            "Run scripts/setup/inspect_trellis2.py on the pod to confirm."
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _ovoxel_to_dense_occupancy(self, ovoxel: dict) -> torch.Tensor:
        """Rasterize the O-Voxel indices to a dense (1, 1, R, R, R) occupancy
        grid suitable for feeding into a 3D conv encoder.

        The ss_encoder expects dense occupancy (binary, then downsampled
        through conv blocks), not a sparse voxel list.
        """
        grid_size = ovoxel.get("grid_size", self.grid_size)
        indices = ovoxel["voxel_indices"]  # (N, 3), integer voxel coords
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        indices = indices.to(self.device).long()

        dense = torch.zeros(
            (1, 1, grid_size, grid_size, grid_size),
            dtype=torch.float32,
            device=self.device,
        )
        # Clip any out-of-range indices defensively
        in_range = ((indices >= 0) & (indices < grid_size)).all(dim=1)
        ix = indices[in_range]
        dense[0, 0, ix[:, 0], ix[:, 1], ix[:, 2]] = 1.0
        return dense

    def _encode_ss_latent(self, ovoxel: dict) -> torch.Tensor:
        """Encode O-Voxel to sparse structure latent via the SS encoder.

        Two calling conventions are supported depending on what path
        ``_resolve_ss_encoder`` returns:
          - ``torch.nn.Module`` (encoder net) — we feed dense occupancy.
          - Callable from ``data_toolkit`` — we feed the ovoxel dict.
        """
        encoder = self._resolve_ss_encoder()
        if isinstance(encoder, torch.nn.Module):
            with torch.inference_mode():
                dense = self._ovoxel_to_dense_occupancy(ovoxel)
                out = encoder(dense)
                # Many TRELLIS.2 VAE encoders return (mean, logvar) or a tuple
                if isinstance(out, tuple):
                    out = out[0]
                return out
        # data_toolkit callable path — interface is module-specific;
        # pass the ovoxel dict and let the module extract what it needs.
        return encoder(ovoxel)

    def _encode_shape_latent(self, ovoxel: dict) -> torch.Tensor:
        """Encode O-Voxel dual-grid vertices to per-voxel shape latent."""
        encoder = self._resolve_shape_encoder()
        if isinstance(encoder, torch.nn.Module):
            with torch.inference_mode():
                # Shape encoder expects dual vertices per-voxel;
                # exact API is module-specific. Pass raw tensors and hope
                # for keyword-based forward signature.
                return encoder(
                    voxel_indices=ovoxel["voxel_indices"].to(self.device),
                    dual_vertices=ovoxel["dual_vertices"].to(self.device),
                    intersected=ovoxel.get("intersected"),
                )
        # data_toolkit callable
        return encoder(ovoxel)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, slat: SLATRepresentation, resolution: int | None = None) -> trimesh.Trimesh:
        """Decode a SLAT representation back to a mesh.

        Uses the pipeline's ``decode_shape_slat`` method, which is the
        same one ``generate_pairs.py`` calls at line ~1027 after sampling.
        The SS decoder and shape SLAT decoder both live under
        ``pipeline.models`` and are invoked inside ``decode_shape_slat``.

        Args:
            slat: SLAT representation to decode.
            resolution: Optional resolution override (default: slat.grid_size).
                Must match one of the resolutions supported by the loaded
                pipeline (typically 512 or 1024 for TRELLIS.2-4B).

        Returns:
            Decoded trimesh mesh.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "decode() requires a TRELLIS.2 pipeline instance. Pass one in "
                "via the SLATEncoder constructor (pipeline=...) or call "
                "Easy3EEditor with a shared pipeline."
            )

        res = resolution or slat.grid_size
        # TRELLIS.2-4B only supports 512 and 1024 — clamp silently
        if res not in (512, 1024):
            res = 512 if res < 768 else 1024

        # Reconstruct the SparseTensor the pipeline expects.
        # The exact type depends on the pipeline implementation; duck-typing
        # here covers the two common cases:
        #   (a) pipeline exposes a SparseTensor class we can instantiate
        #   (b) decode_shape_slat accepts a dict of (coords, feats)
        shape_slat_arg = self._make_decoder_input(slat)

        with torch.inference_mode():
            result = self._pipeline.decode_shape_slat(shape_slat_arg, res)

        # Common return shapes: list[trimesh], tuple (list, ...), single trimesh
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, list):
            result = result[0]
        if hasattr(result, "vertices") and not isinstance(result, trimesh.Trimesh):
            # TRELLIS.2 may return its own mesh type; convert to trimesh
            import numpy as np
            v = result.vertices.detach().cpu().numpy() if hasattr(result.vertices, "detach") else np.asarray(result.vertices)
            f = result.faces.detach().cpu().numpy() if hasattr(result.faces, "detach") else np.asarray(result.faces)
            result = trimesh.Trimesh(vertices=v, faces=f)
        return result

    def _make_decoder_input(self, slat: SLATRepresentation):
        """Wrap a SLATRepresentation in whatever container ``decode_shape_slat``
        expects. Tries SparseTensor first, falls back to dict."""
        # Try TRELLIS.2's SparseTensor first
        try:
            from trellis2.modules.sparse import SparseTensor  # type: ignore
            # SparseTensor typically takes (feats, coords) where coords has
            # a leading batch column (B, N, 4).
            feats = slat.shape_latent
            coords = slat.voxel_indices
            if coords.dim() == 2 and coords.shape[-1] == 3:
                batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)
                coords = torch.cat([batch_col, coords], dim=-1)
            feats_2d = feats.squeeze(0) if feats.dim() == 3 else feats
            return SparseTensor(feats=feats_2d.to(self.device), coords=coords.to(self.device))
        except Exception:
            pass
        # Fallback: dict form
        return {
            "feats": slat.shape_latent,
            "coords": slat.voxel_indices,
            "intersected": slat.intersected,
            "dual_vertices": slat.dual_vertices,
        }

    def save_slat(self, slat: SLATRepresentation, path: str | Path) -> None:
        """Save SLAT representation to disk.

        Args:
            slat: SLAT representation to save.
            path: Output .pt file path.
        """
        torch.save(
            {
                "ss_latent": slat.ss_latent,
                "shape_latent": slat.shape_latent,
                "voxel_indices": slat.voxel_indices,
                "dual_vertices": slat.dual_vertices,
                "intersected": slat.intersected,
                "grid_size": slat.grid_size,
            },
            str(path),
        )

    def load_slat(self, path: str | Path) -> SLATRepresentation:
        """Load SLAT representation from disk.

        Args:
            path: Path to .pt file.

        Returns:
            SLATRepresentation.
        """
        data = torch.load(str(path), map_location=self.device, weights_only=True)
        return SLATRepresentation(
            ss_latent=data["ss_latent"],
            shape_latent=data["shape_latent"],
            voxel_indices=data["voxel_indices"],
            dual_vertices=data["dual_vertices"],
            intersected=data.get("intersected"),
            grid_size=data.get("grid_size", 256),
        )
