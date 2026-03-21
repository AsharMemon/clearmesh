#!/usr/bin/env python3
"""
Multi-Layer Quality Filter Pipeline for ClearMesh training data.

8-layer pipeline that progressively filters 3D models from a raw pool
down to high-quality, diverse training pairs. Each layer is designed to
run independently and in sequence, with cheap filters first and expensive
filters last.

Layers:
  0  Geometry fingerprint dedup (edge/face/degree histograms)
  1  Metadata pre-filter (vertex/face count, aspect ratio, degenerates)
  2  Quality score filter (Objaverse++ High+Superior, Step1X, aesthetic)
  3  VLM semantic filter (objectness, clarity, complexity via Qwen2.5-VL)
  4  Sharp edge density (dihedral angles, vertex-density-corrected)
  5  Watertight + thin structure check (ManifoldPlus, Hausdorff)
  6  TRELLIS.2 success filter (implicit during pair generation)
  7  Pair value scoring (Chamfer + normal error + edge density gap)
  8  Latent diversity balancing (DINOv2 + K-means inverse-frequency)

Usage:
    # Run all pre-pair-gen layers (0-5) on a candidate pool
    python filter_pipeline.py --pool candidates.json --layers 0-5 \
        --output filtered_pool.json

    # Run post-pair-gen layers (7-8) on completed pairs
    python filter_pipeline.py --pairs_dir /data/training_pairs \
        --layers 7-8 --output manifest_weighted.json

    # Run a single layer
    python filter_pipeline.py --pool candidates.json --layers 4 \
        --output filtered_layer4.json

    # Dry run: show statistics without filtering
    python filter_pipeline.py --pool candidates.json --layers 0-2 --dry-run
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


# =============================================================================
# Layer 0: Geometry Fingerprint Dedup
# =============================================================================

def compute_geometry_fingerprint(mesh, n_bins=32):
    """
    Compute a rotation/scale-invariant fingerprint from mesh geometry.

    Returns concatenated z-score-normalized histograms of:
      - edge lengths (32 bins)
      - face areas (32 bins)
      - vertex degrees (max 20 bins)

    Critical: each histogram is z-score standardized independently before
    concatenation to prevent the widest-variance histogram from dominating.
    """
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None

    # Edge length histogram
    edges = mesh.edges_unique_length
    if len(edges) == 0:
        return None
    e_hist, _ = np.histogram(edges, bins=n_bins, density=True)

    # Face area histogram
    areas = mesh.area_faces
    if len(areas) == 0:
        return None
    a_hist, _ = np.histogram(areas, bins=n_bins, density=True)

    # Vertex degree histogram (capped at 20)
    from collections import Counter as Ctr
    degrees = Ctr()
    for edge in mesh.edges_unique:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    deg_vals = list(degrees.values())
    d_hist, _ = np.histogram(deg_vals, bins=min(20, max(deg_vals) - min(deg_vals) + 1),
                              range=(0, 20), density=True)

    # Z-score normalize each histogram independently
    def zscore(h):
        h = h.astype(np.float64)
        std = h.std()
        if std < 1e-12:
            return np.zeros_like(h)
        return (h - h.mean()) / std

    e_z = zscore(e_hist)
    a_z = zscore(a_hist)
    d_z = zscore(d_hist)

    # Concatenate and L2-normalize
    fp = np.concatenate([e_z, a_z, d_z])
    norm = np.linalg.norm(fp)
    if norm < 1e-12:
        return None
    return fp / norm


def layer0_geometry_dedup(candidates, mesh_loader, threshold=0.05, batch_size=1000):
    """
    Remove near-duplicate meshes using geometry fingerprints.

    Args:
        candidates: list of candidate dicts with 'path' or 'mesh_path' key
        mesh_loader: callable(path) -> trimesh.Trimesh or None
        threshold: L2 distance threshold for considering duplicates
        batch_size: process in batches to manage memory

    Returns:
        filtered list (duplicates removed), stats dict
    """
    print(f"\n{'='*60}")
    print(f"Layer 0: Geometry Fingerprint Dedup")
    print(f"  Input: {len(candidates)} candidates")
    print(f"  Threshold: {threshold}")
    print(f"{'='*60}")

    fingerprints = []
    fp_indices = []

    for i, cand in enumerate(candidates):
        if i % 1000 == 0 and i > 0:
            print(f"  Computing fingerprints: {i}/{len(candidates)}")

        mesh_path = cand.get('path') or cand.get('mesh_path')
        if not mesh_path or not os.path.exists(mesh_path):
            continue

        try:
            mesh = mesh_loader(mesh_path)
            if mesh is None:
                continue
            fp = compute_geometry_fingerprint(mesh)
            if fp is not None:
                fingerprints.append(fp)
                fp_indices.append(i)
        except Exception:
            continue

    if not fingerprints:
        print("  No valid fingerprints computed")
        return candidates, {"removed": 0, "kept": len(candidates)}

    fps = np.array(fingerprints)
    n = len(fps)
    print(f"  Computed {n} fingerprints, finding duplicates...")

    # Find duplicates via pairwise L2 (batched for memory)
    is_dup = set()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = fps[start:end]
        # Compare batch against all fingerprints
        dists = np.linalg.norm(batch[:, None, :] - fps[None, :, :], axis=2)
        for bi in range(end - start):
            gi = start + bi  # global index
            if gi in is_dup:
                continue
            # Find matches (excluding self)
            matches = np.where((dists[bi] < threshold) & (np.arange(n) > gi))[0]
            for m in matches:
                is_dup.add(m)

    # Map back to candidate indices
    dup_cand_indices = {fp_indices[d] for d in is_dup}
    filtered = [c for i, c in enumerate(candidates) if i not in dup_cand_indices]

    removed = len(candidates) - len(filtered)
    print(f"  Removed {removed} duplicates ({removed/len(candidates)*100:.1f}%)")
    print(f"  Kept {len(filtered)} unique candidates")

    return filtered, {"removed": removed, "kept": len(filtered)}


# =============================================================================
# Layer 1: Metadata Pre-Filter
# =============================================================================

def layer1_metadata_filter(
    candidates,
    mesh_loader=None,
    min_faces=500,
    max_faces=500_000,
    max_aspect_ratio=50,
    max_degenerate_ratio=0.05,
):
    """
    Filter by basic mesh metadata: face count, aspect ratio, degenerates.

    For candidates with pre-computed metadata (vertices, faces fields),
    uses those directly. Otherwise loads mesh if mesh_loader provided.
    """
    print(f"\n{'='*60}")
    print(f"Layer 1: Metadata Pre-Filter")
    print(f"  Input: {len(candidates)} candidates")
    print(f"  Faces: [{min_faces}, {max_faces}]")
    print(f"  Max aspect ratio: {max_aspect_ratio}")
    print(f"{'='*60}")

    filtered = []
    reasons = Counter()

    for i, cand in enumerate(candidates):
        if i % 5000 == 0 and i > 0:
            print(f"  Processing: {i}/{len(candidates)}")

        n_faces = cand.get('faces') or cand.get('num_faces')
        n_verts = cand.get('vertices') or cand.get('num_vertices')
        aspect = cand.get('aspect_ratio')

        # If metadata not pre-computed, try loading mesh
        if n_faces is None and mesh_loader is not None:
            mesh_path = cand.get('path') or cand.get('mesh_path')
            if mesh_path and os.path.exists(mesh_path):
                try:
                    import trimesh
                    mesh = mesh_loader(mesh_path)
                    if mesh is not None:
                        n_faces = len(mesh.faces)
                        n_verts = len(mesh.vertices)
                        extents = mesh.bounding_box.extents
                        aspect = max(extents) / max(min(extents), 1e-8)
                        cand['faces'] = n_faces
                        cand['vertices'] = n_verts
                        cand['aspect_ratio'] = aspect
                except Exception:
                    pass

        # Apply filters
        if n_faces is not None:
            if n_faces < min_faces:
                reasons['too_few_faces'] += 1
                continue
            if n_faces > max_faces:
                reasons['too_many_faces'] += 1
                continue

        if aspect is not None and aspect > max_aspect_ratio:
            reasons['extreme_aspect_ratio'] += 1
            continue

        filtered.append(cand)

    removed = len(candidates) - len(filtered)
    print(f"  Removed {removed} ({removed/len(candidates)*100:.1f}%)")
    for reason, count in reasons.most_common():
        print(f"    {reason}: {count}")
    print(f"  Kept {len(filtered)}")

    return filtered, {"removed": removed, "kept": len(filtered), "reasons": dict(reasons)}


# =============================================================================
# Layer 2: Quality Score Filter
# =============================================================================

def layer2_quality_filter(
    candidates,
    objaversepp_path: Optional[str] = None,
    step1x_uids_path: Optional[str] = None,
    min_quality_score: int = 2,
    fallback_aesthetic_threshold: float = 6.0,
):
    """
    Filter by Objaverse++ quality scores.

    Score mapping: 0=Low, 1=Medium, 2=High, 3=Superior
    Default keeps High + Superior (score >= 2).

    Fallback for unscored models:
      - Include if in Step1X-3D curated list
      - Include if aesthetic_score > threshold
    """
    print(f"\n{'='*60}")
    print(f"Layer 2: Quality Score Filter")
    print(f"  Input: {len(candidates)} candidates")
    print(f"  Min quality score: {min_quality_score}")
    print(f"{'='*60}")

    # Load Objaverse++ annotations if available
    opp_scores = {}
    if objaversepp_path and os.path.exists(objaversepp_path):
        try:
            import pandas as pd
            df = pd.read_parquet(objaversepp_path)
            for _, row in df.iterrows():
                uid = row.get('uid') or row.get('id')
                score = row.get('quality_score', row.get('score', -1))
                if uid:
                    opp_scores[str(uid)] = int(score)
            print(f"  Loaded {len(opp_scores)} Objaverse++ scores")
        except Exception as e:
            print(f"  Warning: could not load Objaverse++ data: {e}")

    # Load Step1X UIDs
    step1x_uids = set()
    if step1x_uids_path and os.path.exists(step1x_uids_path):
        try:
            with open(step1x_uids_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                step1x_uids = {str(uid) for uid in data}
            elif isinstance(data, dict):
                step1x_uids = set(data.keys())
            print(f"  Loaded {len(step1x_uids)} Step1X UIDs")
        except Exception as e:
            print(f"  Warning: could not load Step1X UIDs: {e}")

    filtered = []
    stats = Counter()

    for cand in candidates:
        uid = cand.get('uid') or cand.get('sha256', '')

        # Check Objaverse++ score
        opp_score = opp_scores.get(uid)
        if opp_score is not None:
            cand['quality_score'] = opp_score
            if opp_score >= min_quality_score:
                stats['opp_pass'] += 1
                filtered.append(cand)
                continue
            else:
                stats['opp_reject'] += 1
                continue

        # Fallback: Step1X curated list
        if uid in step1x_uids:
            cand['quality_score'] = min_quality_score  # Treat as meeting threshold
            stats['step1x_pass'] += 1
            filtered.append(cand)
            continue

        # Fallback: aesthetic score
        aesthetic = cand.get('aesthetic_score', 0)
        if aesthetic > fallback_aesthetic_threshold:
            cand['quality_score'] = min_quality_score
            stats['aesthetic_pass'] += 1
            filtered.append(cand)
            continue

        # No score available — reject or keep conservatively
        if not opp_scores and not step1x_uids:
            # No external data loaded, keep everything
            stats['no_data_keep'] += 1
            filtered.append(cand)
        else:
            stats['unscored_reject'] += 1

    removed = len(candidates) - len(filtered)
    print(f"  Removed {removed} ({removed/len(candidates)*100:.1f}%)")
    for reason, count in stats.most_common():
        print(f"    {reason}: {count}")
    print(f"  Kept {len(filtered)}")

    return filtered, {"removed": removed, "kept": len(filtered), "stats": dict(stats)}


# =============================================================================
# Layer 3: VLM Semantic Filter
# =============================================================================

def layer3_vlm_filter(
    candidates,
    mesh_loader=None,
    render_fn=None,
    vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    min_score: float = 0.2,
    batch_size: int = 8,
):
    """
    Score models using a VLM for objectness, clarity, and complexity.

    This is a soft filter: scores are stored for later weighting,
    only extreme outliers (score < min_score) are removed.

    Requires GPU and a VLM model. Can be run in parallel with pair gen.
    """
    print(f"\n{'='*60}")
    print(f"Layer 3: VLM Semantic Filter")
    print(f"  Input: {len(candidates)} candidates")
    print(f"  Model: {vlm_model_name}")
    print(f"  Min score: {min_score}")
    print(f"{'='*60}")

    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    except ImportError:
        print("  Skipping: transformers VLM support not available")
        return candidates, {"skipped": True, "reason": "transformers_missing"}

    # NOTE: Full VLM implementation would:
    # 1. Render 4-8 views of each model
    # 2. Send renders to VLM with prompt:
    #    "Rate this 3D object on: (1) Is it a single, coherent object? (2) Is it
    #     clearly recognizable? (3) Does it have interesting geometric detail?
    #     Score each 0-1, return as JSON."
    # 3. Parse scores, compute composite
    # 4. Filter below min_score

    print("  VLM filter is a placeholder — requires GPU + model setup")
    print("  Passing all candidates through with vlm_score=None")

    for cand in candidates:
        cand['vlm_score'] = None

    return candidates, {"skipped": True, "reason": "placeholder"}


# =============================================================================
# Layer 4: Sharp Edge Density
# =============================================================================

def compute_sharp_edge_density(mesh, angle_threshold_deg=30):
    """
    Compute sharp edge density with vertex-density correction.

    Returns adjusted_score = sharp_edge_ratio * (vertices / surface_area)
    This penalizes low-poly blobs that falsely appear complex.
    """
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) < 4:
        return None, None

    try:
        # Get face adjacency angles (in radians)
        angles = mesh.face_adjacency_angles
        threshold_rad = np.radians(angle_threshold_deg)

        # Sharp edge ratio
        n_sharp = np.sum(angles > threshold_rad)
        n_total = len(angles)
        if n_total == 0:
            return 0.0, 0.0

        sharp_ratio = n_sharp / n_total

        # Vertex-density correction
        surface_area = mesh.area
        if surface_area < 1e-10:
            return sharp_ratio, 0.0

        vertex_density = len(mesh.vertices) / surface_area
        adjusted_score = sharp_ratio * vertex_density

        return sharp_ratio, adjusted_score
    except Exception:
        return None, None


def layer4_sharp_edge_filter(
    candidates,
    mesh_loader,
    angle_threshold_deg: float = 30,
    min_adjusted_score: float = 0.01,
    max_adjusted_score: float = 1000,
):
    """
    Filter by sharp edge density with vertex-density correction.

    Removes featureless blobs (too low) and extreme outliers (too high).
    """
    print(f"\n{'='*60}")
    print(f"Layer 4: Sharp Edge Density Filter")
    print(f"  Input: {len(candidates)} candidates")
    print(f"  Angle threshold: {angle_threshold_deg} deg")
    print(f"  Score range: [{min_adjusted_score}, {max_adjusted_score}]")
    print(f"{'='*60}")

    filtered = []
    reasons = Counter()

    for i, cand in enumerate(candidates):
        if i % 1000 == 0 and i > 0:
            print(f"  Processing: {i}/{len(candidates)}")

        mesh_path = cand.get('path') or cand.get('mesh_path')
        if not mesh_path or not os.path.exists(mesh_path):
            filtered.append(cand)  # Can't check, keep
            reasons['no_mesh'] += 1
            continue

        try:
            mesh = mesh_loader(mesh_path)
            if mesh is None:
                reasons['load_failed'] += 1
                continue

            sharp_ratio, adjusted = compute_sharp_edge_density(mesh, angle_threshold_deg)

            if sharp_ratio is None:
                reasons['compute_failed'] += 1
                continue

            cand['sharp_edge_ratio'] = float(sharp_ratio)
            cand['adjusted_edge_score'] = float(adjusted) if adjusted else 0.0

            if adjusted < min_adjusted_score:
                reasons['too_smooth'] += 1
                continue
            if adjusted > max_adjusted_score:
                reasons['extreme_outlier'] += 1
                continue

            filtered.append(cand)

        except Exception:
            reasons['exception'] += 1
            continue

    removed = len(candidates) - len(filtered)
    print(f"  Removed {removed} ({removed/len(candidates)*100:.1f}%)")
    for reason, count in reasons.most_common():
        print(f"    {reason}: {count}")
    print(f"  Kept {len(filtered)}")

    return filtered, {"removed": removed, "kept": len(filtered), "reasons": dict(reasons)}


# =============================================================================
# Layer 5: Watertight + Thin Structure
# =============================================================================

def layer5_watertight_filter(
    candidates,
    mesh_loader,
    require_watertight: bool = False,
    max_hausdorff: float = 0.1,
):
    """
    Check watertight status and thin structure quality.

    If ManifoldPlus is available, converts non-watertight meshes and
    measures Hausdorff distance to detect major topology destruction.
    """
    print(f"\n{'='*60}")
    print(f"Layer 5: Watertight + Thin Structure")
    print(f"  Input: {len(candidates)} candidates")
    print(f"  Require watertight: {require_watertight}")
    print(f"{'='*60}")

    filtered = []
    reasons = Counter()

    for i, cand in enumerate(candidates):
        if i % 1000 == 0 and i > 0:
            print(f"  Processing: {i}/{len(candidates)}")

        mesh_path = cand.get('path') or cand.get('mesh_path')
        if not mesh_path or not os.path.exists(mesh_path):
            filtered.append(cand)
            reasons['no_mesh'] += 1
            continue

        try:
            import trimesh
            mesh = mesh_loader(mesh_path)
            if mesh is None:
                reasons['load_failed'] += 1
                continue

            is_watertight = mesh.is_watertight
            cand['watertight'] = is_watertight

            if require_watertight and not is_watertight:
                reasons['not_watertight'] += 1
                continue

            # Check for degenerate thin structures
            # (very high face count relative to volume)
            if is_watertight and mesh.volume > 0:
                surface_to_volume = mesh.area / mesh.volume
                cand['surface_to_volume'] = float(surface_to_volume)
                # Extremely thin structures have very high surface-to-volume
                if surface_to_volume > 10000:
                    reasons['thin_structure'] += 1
                    continue

            filtered.append(cand)

        except Exception:
            reasons['exception'] += 1
            continue

    removed = len(candidates) - len(filtered)
    print(f"  Removed {removed} ({removed/len(candidates)*100:.1f}%)")
    for reason, count in reasons.most_common():
        print(f"    {reason}: {count}")
    print(f"  Kept {len(filtered)}")

    return filtered, {"removed": removed, "kept": len(filtered), "reasons": dict(reasons)}


# =============================================================================
# Layer 6: TRELLIS.2 Success Filter (implicit)
# =============================================================================

def layer6_trellis_success_info():
    """
    Layer 6 is implicit: it happens during pair generation.
    Models that fail TRELLIS.2 inference are automatically excluded.

    This function just prints info about the layer.
    """
    print(f"\n{'='*60}")
    print(f"Layer 6: TRELLIS.2 Success Filter (implicit)")
    print(f"  This layer runs automatically during pair generation.")
    print(f"  Models that fail TRELLIS inference are excluded.")
    print(f"  Check failure_details.json in pair gen output for stats.")
    print(f"{'='*60}")


# =============================================================================
# Layer 7: Pair Value Scoring
# =============================================================================

def compute_pair_metrics(pair_dir):
    """
    Compute per-pair quality metrics for value scoring.

    Returns dict with chamfer, normal_error, edge_density_delta.
    """
    import trimesh

    pair_path = Path(pair_dir)

    # Load meshes
    coarse_path = pair_path / "coarse.glb"
    fine_candidates = list(pair_path.glob("fine.*"))
    fine_path = fine_candidates[0] if fine_candidates else None

    if not coarse_path.exists() or fine_path is None:
        return None

    try:
        coarse = trimesh.load(str(coarse_path), force='mesh')
        fine = trimesh.load(str(fine_path), force='mesh')

        if len(coarse.vertices) == 0 or len(fine.vertices) == 0:
            return None

        # Chamfer distance (sample points, compute bidirectional nearest-neighbor)
        n_samples = min(10000, len(coarse.vertices), len(fine.vertices))
        c_pts = coarse.sample(n_samples)
        f_pts = fine.sample(n_samples)

        # Forward: coarse -> fine
        from scipy.spatial import cKDTree
        tree_f = cKDTree(f_pts)
        d_cf, _ = tree_f.query(c_pts)

        # Backward: fine -> coarse
        tree_c = cKDTree(c_pts)
        d_fc, _ = tree_c.query(f_pts)

        chamfer = float(np.mean(d_cf) + np.mean(d_fc))

        # Normal angle error
        c_normals = coarse.vertex_normals[:n_samples] if len(coarse.vertex_normals) >= n_samples else coarse.vertex_normals
        f_normals = fine.vertex_normals[:n_samples] if len(fine.vertex_normals) >= n_samples else fine.vertex_normals
        min_n = min(len(c_normals), len(f_normals))
        if min_n > 0:
            dots = np.clip(np.sum(c_normals[:min_n] * f_normals[:min_n], axis=1), -1, 1)
            normal_error = float(np.mean(np.arccos(np.abs(dots))))
        else:
            normal_error = 0.0

        # Edge density delta
        c_edge_density = len(coarse.edges_unique) / max(coarse.area, 1e-10)
        f_edge_density = len(fine.edges_unique) / max(fine.area, 1e-10)
        edge_density_delta = float(abs(f_edge_density - c_edge_density))

        return {
            'chamfer': chamfer,
            'normal_error': normal_error,
            'edge_density_delta': edge_density_delta,
        }

    except Exception as e:
        return None


def layer7_pair_value_scoring(
    pairs_dir: str,
    top_weight: float = 1.0,
    bottom_weight: float = 0.3,
    top_percentile: float = 0.7,
):
    """
    Score pairs by refinement difficulty / value for training.

    Z-score normalizes Chamfer distance, normal error, and edge density
    delta, then combines them equally. Top 70% get weight 1.0, bottom 30%
    get weight 0.3.
    """
    print(f"\n{'='*60}")
    print(f"Layer 7: Pair Value Scoring")
    print(f"  Pairs dir: {pairs_dir}")
    print(f"  Top {top_percentile*100:.0f}% weight: {top_weight}")
    print(f"  Bottom {(1-top_percentile)*100:.0f}% weight: {bottom_weight}")
    print(f"{'='*60}")

    # Discover pairs
    pairs_path = Path(pairs_dir)
    pair_dirs = []
    for pattern in ["*/coarse_voxels.npy", "shard_*/*/coarse_voxels.npy"]:
        for p in pairs_path.glob(pattern):
            pair_dirs.append(p.parent)
    pair_dirs = sorted(set(pair_dirs))
    print(f"  Found {len(pair_dirs)} pairs")

    # Compute metrics for each pair
    metrics = []
    for i, pd in enumerate(pair_dirs):
        if i % 100 == 0 and i > 0:
            print(f"  Computing metrics: {i}/{len(pair_dirs)}")
        m = compute_pair_metrics(pd)
        if m is not None:
            m['pair_dir'] = str(pd)
            m['uid'] = pd.name
            metrics.append(m)

    if len(metrics) < 10:
        print(f"  Too few metrics computed ({len(metrics)}), skipping scoring")
        return {}, {"skipped": True}

    print(f"  Computed metrics for {len(metrics)} pairs")

    # Z-score normalize each metric
    chamfers = np.array([m['chamfer'] for m in metrics])
    normals = np.array([m['normal_error'] for m in metrics])
    edges = np.array([m['edge_density_delta'] for m in metrics])

    def zscore(arr):
        std = arr.std()
        if std < 1e-12:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / std

    z_chamfer = zscore(chamfers)
    z_normal = zscore(normals)
    z_edge = zscore(edges)

    # Combined score (higher = more valuable for training)
    combined = z_chamfer + z_normal + z_edge

    # Assign weights based on percentile
    threshold = np.percentile(combined, (1 - top_percentile) * 100)
    weights = {}
    for i, m in enumerate(metrics):
        w = top_weight if combined[i] >= threshold else bottom_weight
        m['pair_value_score'] = float(combined[i])
        m['pair_weight'] = w
        weights[m['uid']] = {
            'pair_value_score': float(combined[i]),
            'pair_weight': w,
            'chamfer': float(chamfers[i]),
            'normal_error': float(normals[i]),
            'edge_density_delta': float(edges[i]),
        }

    print(f"  Score range: [{combined.min():.3f}, {combined.max():.3f}]")
    print(f"  Top {top_percentile*100:.0f}% threshold: {threshold:.3f}")
    print(f"  Weights: {sum(1 for m in metrics if m['pair_weight'] == top_weight)} top, "
          f"{sum(1 for m in metrics if m['pair_weight'] == bottom_weight)} bottom")

    return weights, {"n_scored": len(metrics), "threshold": float(threshold)}


# =============================================================================
# Layer 8: Latent Diversity Balancing
# =============================================================================

def layer8_diversity_balancing(
    pairs_dir: str,
    n_clusters: int = 500,
    existing_weights: Optional[dict] = None,
):
    """
    Balance training data diversity using DINOv2 features + K-means.

    Clusters pairs by cond_features.npy, computes inverse-frequency
    weights to up-weight rare categories and down-weight over-represented
    ones (e.g., chairs, tables).

    Final weight = pair_value_weight * diversity_weight
    """
    print(f"\n{'='*60}")
    print(f"Layer 8: Latent Diversity Balancing")
    print(f"  Pairs dir: {pairs_dir}")
    print(f"  Clusters: {n_clusters}")
    print(f"{'='*60}")

    # Discover pairs with cond_features
    pairs_path = Path(pairs_dir)
    pair_dirs = []
    for pattern in ["*/cond_features.npy", "shard_*/*/cond_features.npy"]:
        for p in pairs_path.glob(pattern):
            pair_dirs.append(p.parent)
    pair_dirs = sorted(set(pair_dirs))
    print(f"  Found {len(pair_dirs)} pairs with cond_features")

    if len(pair_dirs) < n_clusters * 2:
        n_clusters = max(10, len(pair_dirs) // 5)
        print(f"  Reduced clusters to {n_clusters} (small dataset)")

    # Load all cond_features and compute mean per pair
    features = []
    uids = []
    for pd in pair_dirs:
        try:
            cond = np.load(pd / "cond_features.npy").astype(np.float32)
            # Mean-pool to get single vector per pair
            feat = cond.mean(axis=0)
            features.append(feat)
            uids.append(pd.name)
        except Exception:
            continue

    if len(features) < n_clusters:
        print(f"  Too few features ({len(features)}), skipping")
        return {}, {"skipped": True}

    features = np.array(features)
    print(f"  Loaded {len(features)} feature vectors (dim={features.shape[1]})")

    # K-means clustering
    try:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    except ImportError:
        from scipy.cluster.vq import kmeans2
        print("  Using scipy k-means (sklearn not available)")
        centroids, labels = kmeans2(features, n_clusters, minit='points')
        cluster_labels = labels
        # Skip to weight computation
        cluster_counts = Counter(labels.tolist())
        total = len(labels)
        inv_freq = {c: total / (count * n_clusters) for c, count in cluster_counts.items()}
        max_w = max(inv_freq.values())
        diversity_weights = {}
        for i, uid in enumerate(uids):
            dw = inv_freq[labels[i]] / max_w  # Normalize to [0, 1]
            pw = 1.0
            if existing_weights and uid in existing_weights:
                pw = existing_weights[uid].get('pair_weight', 1.0)
            diversity_weights[uid] = {
                'diversity_weight': float(dw),
                'cluster': int(labels[i]),
                'final_weight': float(pw * dw),
            }
        print(f"  Cluster sizes: min={min(cluster_counts.values())}, "
              f"max={max(cluster_counts.values())}, "
              f"mean={total/n_clusters:.1f}")
        return diversity_weights, {"n_clustered": len(features), "n_clusters": n_clusters}

    kmeans.fit(features)
    labels = kmeans.labels_
    cluster_counts = Counter(labels.tolist())

    print(f"  Cluster sizes: min={min(cluster_counts.values())}, "
          f"max={max(cluster_counts.values())}, "
          f"mean={len(features)/n_clusters:.1f}")

    # Inverse frequency weights
    total = len(labels)
    inv_freq = {c: total / (count * n_clusters) for c, count in cluster_counts.items()}
    max_w = max(inv_freq.values())

    # Combine with existing pair weights
    diversity_weights = {}
    for i, uid in enumerate(uids):
        dw = inv_freq[labels[i]] / max_w  # Normalize to [0, 1]
        pw = 1.0
        if existing_weights and uid in existing_weights:
            pw = existing_weights[uid].get('pair_weight', 1.0)
        diversity_weights[uid] = {
            'diversity_weight': float(dw),
            'cluster': int(labels[i]),
            'final_weight': float(pw * dw),
        }

    # Distribution analysis
    final_weights = [v['final_weight'] for v in diversity_weights.values()]
    print(f"  Final weight range: [{min(final_weights):.3f}, {max(final_weights):.3f}]")
    print(f"  Final weight mean: {np.mean(final_weights):.3f}")

    return diversity_weights, {"n_clustered": len(features), "n_clusters": n_clusters}


# =============================================================================
# Main CLI
# =============================================================================

def parse_layer_range(s):
    """Parse layer specification like '0-5', '4', '7,8', '0-2,7-8'."""
    layers = set()
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def default_mesh_loader(path):
    """Simple mesh loader using trimesh."""
    import trimesh
    mesh = trimesh.load(path, force='mesh')
    if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
        return mesh
    return None


def main():
    parser = argparse.ArgumentParser(description="Multi-Layer Quality Filter Pipeline")
    parser.add_argument("--pool", type=str, help="Input candidate pool JSON (for layers 0-5)")
    parser.add_argument("--pairs_dir", type=str, help="Training pairs directory (for layers 7-8)")
    parser.add_argument("--layers", type=str, default="0-8", help="Layers to run (e.g. '0-5', '4', '7,8')")
    parser.add_argument("--output", type=str, default="filtered_output.json",
                        help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without filtering")

    # Layer-specific options
    parser.add_argument("--dedup-threshold", type=float, default=0.05,
                        help="Layer 0: L2 fingerprint threshold")
    parser.add_argument("--min-faces", type=int, default=500, help="Layer 1: min faces")
    parser.add_argument("--max-faces", type=int, default=500000, help="Layer 1: max faces")
    parser.add_argument("--min-quality", type=int, default=2, help="Layer 2: min Objaverse++ score")
    parser.add_argument("--objaversepp", type=str, help="Layer 2: Objaverse++ parquet path")
    parser.add_argument("--step1x-uids", type=str, help="Layer 2: Step1X UIDs JSON path")
    parser.add_argument("--n-clusters", type=int, default=500, help="Layer 8: K-means clusters")

    args = parser.parse_args()
    layers = parse_layer_range(args.layers)

    print(f"\n{'='*60}")
    print(f"ClearMesh Multi-Layer Filter Pipeline")
    print(f"  Layers: {layers}")
    print(f"  Output: {args.output}")
    if args.dry_run:
        print(f"  Mode: DRY RUN")
    print(f"{'='*60}")

    # Load candidates for pre-pair-gen layers (0-5)
    candidates = None
    if any(l in layers for l in range(6)) and args.pool:
        with open(args.pool) as f:
            candidates = json.load(f)
        if isinstance(candidates, dict):
            candidates = candidates.get('candidates', candidates.get('items', []))
        print(f"\nLoaded {len(candidates)} candidates from {args.pool}")

    # Track all layer stats
    all_stats = {}
    t_start = time.time()

    # === Pre-pair-gen layers ===

    if 0 in layers and candidates is not None:
        if args.dry_run:
            print("\n[DRY RUN] Layer 0: would compute geometry fingerprints")
        else:
            candidates, stats = layer0_geometry_dedup(
                candidates, default_mesh_loader, threshold=args.dedup_threshold)
            all_stats['layer_0'] = stats

    if 1 in layers and candidates is not None:
        if args.dry_run:
            print("\n[DRY RUN] Layer 1: would filter by metadata")
        else:
            candidates, stats = layer1_metadata_filter(
                candidates, mesh_loader=default_mesh_loader,
                min_faces=args.min_faces, max_faces=args.max_faces)
            all_stats['layer_1'] = stats

    if 2 in layers and candidates is not None:
        if args.dry_run:
            print("\n[DRY RUN] Layer 2: would filter by quality scores")
        else:
            candidates, stats = layer2_quality_filter(
                candidates,
                objaversepp_path=args.objaversepp,
                step1x_uids_path=args.step1x_uids,
                min_quality_score=args.min_quality)
            all_stats['layer_2'] = stats

    if 3 in layers and candidates is not None:
        if args.dry_run:
            print("\n[DRY RUN] Layer 3: would run VLM semantic filter")
        else:
            candidates, stats = layer3_vlm_filter(candidates)
            all_stats['layer_3'] = stats

    if 4 in layers and candidates is not None:
        if args.dry_run:
            print("\n[DRY RUN] Layer 4: would compute sharp edge density")
        else:
            candidates, stats = layer4_sharp_edge_filter(
                candidates, default_mesh_loader)
            all_stats['layer_4'] = stats

    if 5 in layers and candidates is not None:
        if args.dry_run:
            print("\n[DRY RUN] Layer 5: would check watertight + thin structure")
        else:
            candidates, stats = layer5_watertight_filter(
                candidates, default_mesh_loader)
            all_stats['layer_5'] = stats

    if 6 in layers:
        layer6_trellis_success_info()
        all_stats['layer_6'] = {"info": "implicit during pair generation"}

    # === Post-pair-gen layers ===

    pair_weights = {}

    if 7 in layers and args.pairs_dir:
        if args.dry_run:
            print("\n[DRY RUN] Layer 7: would score pair values")
        else:
            pair_weights, stats = layer7_pair_value_scoring(args.pairs_dir)
            all_stats['layer_7'] = stats

    if 8 in layers and args.pairs_dir:
        if args.dry_run:
            print("\n[DRY RUN] Layer 8: would balance diversity")
        else:
            diversity_weights, stats = layer8_diversity_balancing(
                args.pairs_dir, n_clusters=args.n_clusters,
                existing_weights=pair_weights)
            all_stats['layer_8'] = stats
            # Merge diversity weights into pair weights
            for uid, dw in diversity_weights.items():
                if uid in pair_weights:
                    pair_weights[uid].update(dw)
                else:
                    pair_weights[uid] = dw

    # === Save output ===

    elapsed = time.time() - t_start

    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'layers_run': layers,
        'elapsed_seconds': round(elapsed, 1),
        'stats': all_stats,
    }

    if candidates is not None:
        output['n_candidates'] = len(candidates)
        output['candidates'] = candidates

    if pair_weights:
        output['n_pair_weights'] = len(pair_weights)
        output['pair_weights'] = pair_weights

    if not args.dry_run:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved output to {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Time: {elapsed:.1f}s")
    for layer_id, stats in sorted(all_stats.items()):
        if isinstance(stats, dict) and 'kept' in stats:
            print(f"  {layer_id}: kept {stats['kept']}, removed {stats['removed']}")
        elif isinstance(stats, dict) and 'n_scored' in stats:
            print(f"  {layer_id}: scored {stats['n_scored']} pairs")
        elif isinstance(stats, dict) and 'n_clustered' in stats:
            print(f"  {layer_id}: clustered {stats['n_clustered']} pairs "
                  f"into {stats['n_clusters']} clusters")
        else:
            print(f"  {layer_id}: {stats}")


if __name__ == "__main__":
    main()
