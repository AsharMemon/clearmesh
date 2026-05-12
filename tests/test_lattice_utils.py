import numpy as np
import trimesh

from clearmesh.lattice import (
    ActiveVoxelOptions,
    build_irregular_patches,
    extract_active_surface_voxels,
    jitter_queries,
    sample_edge_candidates,
    sample_vdf,
    unique_mesh_edges,
)
from scripts.research.train_lattice_vdf_tiny import _mesh_from_vdf


def test_jitter_queries_stays_within_half_voxel():
    queries = np.zeros((16, 3), dtype=float)
    jittered = jitter_queries(queries, resolution=8, seed=123)
    assert jittered.shape == queries.shape
    assert np.all(np.abs(jittered) <= 1.0 / 8.0)


def test_extract_active_surface_voxels_is_sparse_and_deterministic():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    options = ActiveVoxelOptions(resolution=16, surface_samples=256, seed=7)

    first = extract_active_surface_voxels(mesh, options)
    second = extract_active_surface_voxels(mesh, options)

    assert first.count > 0
    assert first.count < 16**3
    np.testing.assert_array_equal(first.indices, second.indices)
    assert first.centers.shape == first.indices.shape


def test_vdf_displacements_reconstruct_sampled_face_vertices():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    samples = sample_vdf(mesh, count=32, seed=5)

    reconstructed = samples.points[:, None, :] + samples.vertex_displacements
    np.testing.assert_allclose(reconstructed, samples.face_vertices)
    assert samples.features.shape == (32, 15)


def test_edge_candidates_have_no_positive_negative_overlap():
    mesh = trimesh.creation.box()
    positives = unique_mesh_edges(mesh)
    candidates = sample_edge_candidates(mesh, random_negative_count=8, seed=11)

    positive_set = {tuple(edge) for edge in candidates.positive_edges.tolist()}
    negative_set = {tuple(edge) for edge in candidates.negative_edges.tolist()}
    assert len(positives) == candidates.positive_count
    assert candidates.negative_count == 8
    assert positive_set.isdisjoint(negative_set)


def test_irregular_patches_include_anchor_as_nearest_neighbor():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(128, 3))
    patches = build_irregular_patches(points, anchor_count=12, patch_size=9, seed=2)

    assert patches.anchor_indices.shape == (12,)
    assert patches.patch_indices.shape == (12, 9)
    assert np.all(patches.patch_indices[:, 0] == patches.anchor_indices)


def test_vdf_mesh_export_needs_tolerant_welding_for_learned_offsets():
    mesh = trimesh.creation.box()
    points = mesh.triangles_center
    targets = mesh.triangles - points[:, None, :]
    rng = np.random.default_rng(4)
    prediction = np.concatenate(
        [
            (targets + rng.normal(scale=2.0e-4, size=targets.shape)).reshape(len(points), 9),
            mesh.face_normals,
        ],
        axis=1,
    )

    strict = _mesh_from_vdf(points, prediction, weld_digits=4)
    tolerant = _mesh_from_vdf(points, prediction, weld_digits=3)

    assert len(strict.vertices) > len(mesh.vertices)
    assert tolerant.is_watertight
    assert len(tolerant.vertices) == len(mesh.vertices)
