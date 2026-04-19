"""Configuration defaults for DualPrim reproduction.

Several hyperparameters are NOT fully specified in the paper (arXiv
2603.16133). Where the paper gives values, we follow them exactly.
Where it doesn't, we pick sensible defaults and expose them so they
can be swept. Flagged with "NOT SPECIFIED IN PAPER" comments.

Paper citations refer to sections/equations in the v1 PDF:
  §3 Method
  §4 Optimization
  §5 Experimental Setup
  Table 1 (per-primitive parameter ranges)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DualPrimConfig:
    """All knobs for a DualPrim per-scene optimization."""

    # ================================================================
    # Scene initialization (§4.2 Adaptive Control, §5.1 Implementation)
    # ================================================================
    num_primitives_init: int = 100           # K, paper: "initialize K=100"
    init_space: Tuple[float, float] = (-1.0, 1.0)  # paper: "[-1,1] space"
    seed: int = 0

    # NSQ initialisation strategy:
    #   "coupled"     — NSQ starts at PSQ position with smaller scale
    #                   (more samples per dual-primitive contribute to
    #                   carving early; my original approach)
    #   "independent" — NSQ randomly placed in [-1,1] like PSQ, scale
    #                   independent. Paper-faithful: paper says "K=100
    #                   randomly in [-1,1]^3" without coupling. Most
    #                   pairs won't overlap initially; pruning kills
    #                   the unhelpful ones.
    nsq_init_strategy: str = "coupled"

    # ================================================================
    # Per-primitive parameter RANGES (Table 1)
    # Hard-clipped during optimization.
    # ================================================================
    scale_range: Tuple[float, float] = (0.02, 1.0)      # Table 1 row 1
    shape_range: Tuple[float, float] = (0.05, 2.0)      # Table 1 row 2
    alpha_range: Tuple[float, float] = (0.0, 1.0)       # Table 1 row 3
    sharpness_range: Tuple[float, float] = (0.0, 1.0)   # Table 1 row 4 (θ)
    translation_range: Tuple[float, float] = (-1.0, 1.0)  # Table 1 row 5
    rotation_range_deg: Tuple[float, float] = (-180.0, 180.0)  # Table 1 row 6
    color_range: Tuple[float, float] = (0.0, 1.0)       # Table 1 row 7

    # ================================================================
    # Effectiveness probability gate (Eq 4)
    # P_E(p, S) = Φ(-f(p,PSQ)/θ_S - μ) · Φ(-f(p,NSQ)/θ_S - μ)
    # Φ is sigmoid. μ shifts the gate; the paper notes μ is "a small
    # offset ensuring that zero-crossings are preserved" but DOES NOT
    # give a value. Sensible default below; needs sweep.
    # ================================================================
    mu_gate_offset: float = 0.0    # NOT SPECIFIED IN PAPER
    theta_min: float = 0.01         # floor on per-primitive θ to avoid div-by-zero

    # θ curriculum — a review-driven fix for the P_E-gate collapse mode
    # observed on the hole canary. Without this, θ drifts to its floor
    # on every primitive, the gate becomes razor-sharp, and NSQs that
    # drift outside their PSQs have no gradient pull back in.
    #
    # Schedule: effective theta_min starts at theta_curriculum_start and
    # anneals LINEARLY to config.theta_min over the first
    # theta_curriculum_fraction of training.
    theta_curriculum_start: float = 0.2
    theta_curriculum_fraction: float = 0.5    # first half of training

    # ================================================================
    # Volumetric rendering (§3.2 Renderer, Eq 1, 7-11)
    # Same formulation as NeuS.
    # ================================================================
    num_samples_per_ray: int = 64          # N in Eq 1. NOT SPECIFIED IN PAPER
    near_plane: float = 0.1                 # NOT SPECIFIED
    far_plane: float = 4.0                  # NOT SPECIFIED
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # white

    # Lighting MLP per §5.1: "4 layers with Xavier initialization"
    lighting_mlp_hidden: int = 128          # NOT SPECIFIED IN PAPER
    lighting_mlp_layers: int = 4            # paper-specified

    # ================================================================
    # Loss weights (Eq 12 total loss)
    # L = L_rgb + λ_mask·L_mask + λ_sparse·L_sp + λ_e·L_e
    #        + λ_max·L_max + λ_norm_reg·L_norm_reg
    # Paper doesn't give λ values — these are educated defaults.
    # ================================================================
    lambda_mask: float = 3.0                # Friend's tuning suggestion: bumped
                                             # from 1.0 to put more pressure on
                                             # silhouette/hole boundary, since
                                             # boundaries are what tell DualPrim
                                             # to carve. NOT SPECIFIED IN PAPER.
    lambda_sparse: float = 0.01             # NOT SPECIFIED
    lambda_entropy: float = 0.01            # NOT SPECIFIED
    lambda_max: float = 0.1                 # NOT SPECIFIED
    lambda_norm_reg: float = 0.1            # NOT SPECIFIED

    # ================================================================
    # Adaptive pruning (§4.2)
    # ================================================================
    prune_alpha_threshold: float = 0.02     # paper: "α < 0.02 are discarded"
    prune_scale_threshold: float = 0.01     # paper: "t_a = 0.01"
    export_alpha_threshold: float = 0.5     # paper: "α < T_export = 0.5"
    pruning_interval: int = 1000            # NOT SPECIFIED IN PAPER (in steps)

    # ================================================================
    # Optimizer (NOT SPECIFIED IN PAPER — typical NeuS-family defaults)
    # ================================================================
    learning_rate: float = 5e-4
    lr_scheduler: str = "cosine"            # "cosine" | "constant" | "exponential"
    num_iterations: int = 30_000            # paper quotes per-scene opt with
                                             # NeuS-family scale
    warmup_iterations: int = 1_000

    # ================================================================
    # Export (§3.3 Mesh Exportation)
    # ================================================================
    boolean_backend: str = "manifold3d"     # "manifold3d" | "trimesh" | "pymesh"
    # Marching cubes grid resolution per primitive. The autonomous-runner
    # experiments showed that going from 128 -> 32 -> 16 changes mesh
    # compactness by 16-30x but barely moves Chamfer (194.3 -> 194.0 -> 188.0
    # on the hole canary). Defaulting to 32 since that matches paper-style
    # compactness without quality loss; original 128 default was "more
    # detail = better assumed" — wrong, the limit is the SQ implicit not
    # the tessellation. Keep configurable via DualPrimConfig.
    tessellation_resolution: int = 32

    # ================================================================
    # Supervision mode
    # ================================================================
    # - "paper": source images + StableNormal normals (§5.1)
    # - "mesh_rendered_views": render a ground-truth mesh to 26 views,
    #   use analytic normals. Skips StableNormal noise, cheaper, clean
    #   baseline for parity debugging.
    # - "mesh_fit": no renderer; L2 on mesh TSDF. Sanity path only.
    mode: str = "mesh_rendered_views"

    # Multi-view sampling (§5.1: 24 sphere views + top + bottom = 26)
    num_views: int = 26
    view_resolution: int = 256              # paper: 256x256

    # Normal supervision source
    # - "stablenormal": run StableNormal on rendered RGBs (paper)
    # - "analytic":     compute from mesh (only valid in mesh_* modes)
    normal_source: str = "analytic"

    # ================================================================
    # Logging
    # ================================================================
    log_interval: int = 200
    checkpoint_interval: int = 5_000
    export_interval: int = 5_000             # intermediate mesh snapshots


# Known deviations from the paper when using this config as-is:
#   - lambda_* weights are educated guesses
#   - mu_gate_offset default = 0 (paper says "small offset")
#   - num_samples_per_ray = 64 (not specified)
#   - learning_rate schedule not specified
#   - pruning_interval not specified
#
# When running in "paper" mode, caller should override these from a
# config file once the authors release their code / supplementary.
