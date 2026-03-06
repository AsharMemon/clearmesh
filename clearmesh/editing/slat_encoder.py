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
    ):
        self.trellis2_dir = Path(trellis2_dir)
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_size = grid_size

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

    def _encode_ss_latent(self, ovoxel: dict) -> torch.Tensor:
        """Encode O-Voxel to sparse structure latent.

        Uses TRELLIS.2's SS (Sparse Structure) encoder — a 3D VAE that
        encodes binary voxel occupancy into a continuous latent.

        Args:
            ovoxel: O-Voxel dict from mesh_to_ovoxel().

        Returns:
            Sparse structure latent tensor.
        """
        # TODO: Load TRELLIS.2 SS encoder from model weights
        # The encoder is part of the SparseStructureEncoder in TRELLIS.2
        # Path: {model_dir}/ss_encoder/ or within the main model weights
        raise NotImplementedError(
            "SS latent encoding requires TRELLIS.2 encoder weights. "
            "Investigate model structure at: "
            f"{self.model_dir} and {self.trellis2_dir}/trellis2/"
        )

    def _encode_shape_latent(self, ovoxel: dict) -> torch.Tensor:
        """Encode O-Voxel to shape latent.

        Uses TRELLIS.2's shape encoder (3D VAE) to encode the
        dual grid vertex positions into a per-voxel latent.

        Args:
            ovoxel: O-Voxel dict from mesh_to_ovoxel().

        Returns:
            Shape latent tensor.
        """
        # TODO: Load TRELLIS.2 shape encoder
        # This encodes dual_vertices per-voxel into a latent vector
        raise NotImplementedError(
            "Shape latent encoding requires TRELLIS.2 encoder weights. "
            "Investigate TRELLIS.2 data_toolkit/encode_shape_latent.py"
        )

    def decode(self, slat: SLATRepresentation) -> trimesh.Trimesh:
        """Decode a SLAT representation back to a mesh.

        Args:
            slat: SLAT representation to decode.

        Returns:
            Decoded trimesh mesh.
        """
        # TODO: Use TRELLIS.2 decoder to go from SLAT → mesh
        # This reverses the encoding: latent → O-Voxel → mesh
        raise NotImplementedError(
            "SLAT decoding requires TRELLIS.2 decoder. "
            "Investigate TRELLIS.2 pipeline decode methods."
        )

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
