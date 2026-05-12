"""LATTICE reproduction utilities.

These modules are intentionally small and dependency-light. They give us the
paper-critical geometry contracts locally before we spend GPU time on larger
models: active voxel/query construction, vector-displacement field samples,
edge supervision candidates, and irregular point patches.
"""

from .irregular_patches import IrregularPatches, build_irregular_patches
from .queries import Bounds3D, jitter_queries, quantize_points_to_indices, voxel_indices_to_centers
from .topology_vdf import EdgeCandidates, VDFSamples, sample_edge_candidates, sample_vdf, unique_mesh_edges
from .voxelize import ActiveVoxelOptions, ActiveVoxelSet, extract_active_surface_voxels

__all__ = [
    "ActiveVoxelOptions",
    "ActiveVoxelSet",
    "Bounds3D",
    "EdgeCandidates",
    "IrregularPatches",
    "VDFSamples",
    "build_irregular_patches",
    "extract_active_surface_voxels",
    "jitter_queries",
    "quantize_points_to_indices",
    "sample_edge_candidates",
    "sample_vdf",
    "unique_mesh_edges",
    "voxel_indices_to_centers",
]
