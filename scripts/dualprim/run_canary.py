"""One-object DualPrim canary run.

Modes (matches DualPrimConfig.mode):

  mesh_fit
    Just fit dual-primitives to the mesh's TSDF. No renderer, no
    images. Sanity-checks the SQ math, P_E gate, losses wiring. Fast
    (~5 min CPU). Expected: captures geometry but loses parts.

  mesh_rendered_views
    Render our mesh to 26 views with analytic normals, then run the
    full paper objective (RGB + mask + normal + regularizers) with
    that as GT. ~1 hr GPU. Expected: recovers part decomposition.

  paper
    Use the user's source images + StableNormal + paper's multi-view
    objective. Requires multi-view input — typically not available for
    a single-image pipeline.

Usage:
    python scripts/dualprim/run_canary.py \\
        --input /workspace/demo_e2e_compass_v4/03_polished.glb \\
        --out /workspace/dualprim_canary_compass \\
        --mode mesh_rendered_views \\
        --iters 10000

Outputs:
    views/           26-view GT renders (for the _views modes)
    logs.json        per-log-step loss curves
    primitives.json  final primitive parameters
    refit.glb        exported scene mesh
    per_prim_*.glb   each dual-primitive's Boolean difference
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make clearmesh importable on both laptop and pod
_CANDIDATE_ROOTS = [
    "/workspace/clearmesh",
    str(Path(__file__).resolve().parents[2]),
]
for _root in _CANDIDATE_ROOTS:
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)

import numpy as np
import torch
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="path to mesh (GLB/OBJ) for mesh_* modes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="mesh_rendered_views",
                    choices=["mesh_fit", "mesh_rendered_views", "paper"])
    ap.add_argument("--paper-preset", default="off",
                    choices=["off", "parity", "augmented"],
                    help="'parity' forces the paper-stated setup and forbids "
                         "non-paper interventions; --mode paper defaults to "
                         "this. 'augmented' labels paper-driven variants such "
                         "as visual-hull init/birth or NSQ knife pressure.")
    ap.add_argument("--views-dir", default=None,
                    help="Prepared multi-view supervision directory with "
                         "views.json plus NN_rgb.png/NN_mask.png files. "
                         "Required for --mode paper; optional cache override "
                         "for --mode mesh_rendered_views.")
    ap.add_argument("--k", type=int, default=100,
                    help="initial number of dual-primitives")
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--rays", type=int, default=1024)
    ap.add_argument("--resolution", type=int, default=256,
                    help="view render resolution")
    ap.add_argument("--tessellation-resolution", type=int, default=None,
                    help="Override export tessellation resolution")
    ap.add_argument("--num-samples-per-ray", type=int, default=None,
                    help="Override volumetric samples per ray")
    ap.add_argument("--sampling-mode", default=None,
                    choices=["uniform", "hierarchical"],
                    help="Ray sampling strategy. 'hierarchical' runs a "
                         "coarse pass and NeuS/NeRF-style PDF resampling "
                         "near high-density intervals.")
    ap.add_argument("--num-importance-samples-per-ray", type=int, default=None,
                    help="Fine samples per ray for --sampling-mode hierarchical.")
    ap.add_argument("--shape-range-lo", type=float, default=None,
                    help="Override lower bound of the superquadric shape range")
    ap.add_argument("--shape-range-hi", type=float, default=None,
                    help="Override upper bound of the superquadric shape range")
    ap.add_argument("--psq-shape-init", type=float, default=None,
                    help="If set, force all PSQ ε values to this scalar right "
                         "after scene initialization. Useful for reverse-init "
                         "diagnostics (for example, testing whether sharp ε "
                         "holds or drifts back toward rounded solutions).")
    ap.add_argument("--freeze-psq-shape", action="store_true",
                    help="Freeze PSQ ε after initialization by zeroing its "
                         "gradient slice. Diagnostic for testing whether the "
                         "remaining optimizer/render/export path can fit a "
                         "sharp low-ε scaffold when shape is not allowed to "
                         "drift back toward the rounded basin.")
    ap.add_argument("--log-shape-grad-stats", action="store_true",
                    help="Log per-primitive PSQ ε / θ / |grad ε| traces for "
                         "alive primitives at each log step.")
    ap.add_argument("--export-cleanup-min-faces", type=int, default=None,
                    help="Drop tiny disconnected export components below this face count.")
    ap.add_argument("--export-cleanup-min-area-ratio", type=float, default=None,
                    help="Drop export components smaller than this fraction of the "
                         "largest component area.")
    ap.add_argument("--export-smoothing-iters", type=int, default=None,
                    help="Apply Taubin smoothing for this many iterations after export.")
    ap.add_argument("--export-smoothing-lambda", type=float, default=None,
                    help="Taubin smoothing lambda.")
    ap.add_argument("--export-smoothing-nu", type=float, default=None,
                    help="Taubin smoothing nu.")
    ap.add_argument("--normal-source", default=None,
                    choices=["analytic", "stablenormal"],
                    help="Normal supervision source. 'analytic' uses the "
                         "rendered mesh normals; 'stablenormal' runs the "
                         "official StableNormal predictor on the rendered RGBs.")
    ap.add_argument("--stablenormal-turbo", action="store_true",
                    help="Use StableNormal_turbo for faster normal prediction.")
    ap.add_argument("--stablenormal-cache-dir", default=None,
                    help="Optional cache dir passed to StableNormal torch.hub.")
    ap.add_argument("--stablenormal-blend-strength", type=float, default=None,
                    help="When using StableNormal, blend amount vs analytic normals. "
                         "1.0 preserves raw StableNormal targets; lower values keep "
                         "more of the analytic scaffold.")
    ap.add_argument("--stablenormal-agreement-floor", type=float, default=None,
                    help="Cosine-agreement floor for StableNormal/analytic blending.")
    ap.add_argument("--stablenormal-agreement-ceil", type=float, default=None,
                    help="Cosine-agreement ceil for StableNormal/analytic blending.")
    ap.add_argument("--stablenormal-edge-boost", type=float, default=None,
                    help="Extra StableNormal weight near silhouette boundaries.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nsq-init", default="coupled",
                    choices=["coupled", "coupled_axial", "independent"],
                    help="NSQ init strategy: 'coupled' (NSQ near PSQ), "
                         "'coupled_axial' (coupled + 3x elongated on random "
                         "axis, round-10 addition), or "
                         "'independent' (NSQ random, paper-faithful)")
    ap.add_argument("--init-profile", default=None,
                    choices=["biased", "paper_random", "paper_table"],
                    help="Primitive parameter init profile. 'biased' keeps "
                         "our tuned small/boxy init; 'paper_random' samples "
                         "broadly across the paper ranges; 'paper_table' "
                         "uses the fixed scale/epsilon/alpha/theta values "
                         "from the paper init table.")
    ap.add_argument("--union-export", dest="union_export", action="store_true", default=None,
                    help="Boolean-union all primitives at export. "
                    "Slow (~minutes for K>=20) but produces a single "
                    "watertight mesh ~10-100x more compact than concatenate.")
    ap.add_argument("--no-union-export", dest="union_export", action="store_false",
                    help="Preview-only export: concatenate per-primitive meshes. "
                         "Disabled by default for paper/visual-hull runs.")
    ap.add_argument("--allow-preview-export", action="store_true",
                    help="Allow a paper/visual-hull run to finish with a "
                         "non-fused concatenated export. Intended only for "
                         "fast debugging, not paper-quality claims.")
    ap.add_argument("--fg-bias", type=float, default=0.7,
                    help="Fraction of rays drawn from foreground+boundary "
                         "pixels (rest uniform). 0.0 = paper's default "
                         "uniform sampling; 0.7 = friend's recommended "
                         "silhouette-pressure boost.")
    ap.add_argument("--trajectory-dir", default=None,
                    help="If set, write primitives_step_{N}.json snapshots "
                         "at canonical training iters (~log-spaced). Used "
                         "to build a warm-start training corpus: each "
                         "snapshot is one (mesh, seed, step) tuple for the "
                         "downstream feedforward predictor. Default off.")
    ap.add_argument("--resume-primitives", default=None,
                    help="Path to a primitives JSON (endpoint or trajectory "
                         "snapshot) to warm-start from. Skips random init. "
                         "Useful for measuring 'how few iters do I need "
                         "from a good starting point?' — the Gate 3 "
                         "feasibility check.")
    ap.add_argument("--lambda-mask", type=float, default=None,
                    help="Override lambda_mask in config. Friend's initial "
                         "tuning bumped this 1->3; round 4 fallback may "
                         "try 5+ if round 3 still fails.")
    ap.add_argument("--lambda-sparse", type=float, default=None,
                    help="Override lambda_sparse in config.")
    ap.add_argument("--lambda-entropy", type=float, default=None,
                    help="Override lambda_entropy in config.")
    ap.add_argument("--lambda-max", type=float, default=None,
                    help="Override lambda_max in config.")
    ap.add_argument("--lambda-norm-reg", type=float, default=None,
                    help="Override lambda_norm_reg in config.")
    ap.add_argument("--lambda-depth", type=float, default=None,
                    help="Synthetic GT depth supervision weight. Requires "
                         "cached per-view *_depth.npy maps.")
    ap.add_argument("--lambda-edge-mask", type=float, default=None,
                    help="Override extra silhouette-boundary mask loss weight.")
    ap.add_argument("--lambda-shape-box", type=float, default=None,
                    help="Override one-sided PSQ epsilon boxiness prior weight.")
    ap.add_argument("--shape-box-threshold", type=float, default=None,
                    help="Epsilon threshold for --lambda-shape-box.")
    ap.add_argument("--normal-loss-type", default=None,
                    choices=["l1", "angular"],
                    help="Normal-consistency loss: legacy masked L1 or "
                         "angular loss (1 - cos).")
    ap.add_argument("--lambda-norm-reg-final", type=float, default=None,
                    help="Late-stage target for lambda_norm_reg.")
    ap.add_argument("--norm-reg-ramp-start-fraction", type=float, default=None,
                    help="Start fraction for the lambda_norm_reg ramp.")
    ap.add_argument("--norm-reg-ramp-end-fraction", type=float, default=None,
                    help="End fraction for the lambda_norm_reg ramp.")
    ap.add_argument("--masked-loss-norm-mode", default=None,
                    choices=["global_mean", "fg_mean"],
                    help="Hostile-audit switch for Eq. 13 / Eq. 18 masked reduction.")
    ap.add_argument("--primitive-reg-average-mode", default=None,
                    choices=["alive", "fixed_k"],
                    help="Hostile-audit switch for Eq. 15–17 averaging denominator.")
    ap.add_argument("--mask-loss-type", default=None,
                    choices=["bce", "mse"],
                    help="Override the mask loss used in Eq. 14. "
                         "'bce' matches the paper exactly; 'mse' is the "
                         "stability fallback branch we used while tracking "
                         "the NaN issue.")
    ap.add_argument("--prune-alpha-threshold", type=float, default=None,
                    help="Override alpha prune threshold.")
    ap.add_argument("--pruning-interval", type=int, default=None,
                    help="Override pruning_interval (how often to kill "
                         "weak primitives). Default 1000. Increase to "
                         "2000+ to give primitives more time to find "
                         "positions before being pruned.")
    ap.add_argument("--view-prune-weight-threshold", type=float, default=None,
                    help="Override view-dependent prune threshold.")
    ap.add_argument("--view-prune-every-multiplier", type=int, default=None,
                    help="Run view-dependent pruning every N alpha-prune cycles.")
    ap.add_argument("--view-prune-foreground-only", default=None,
                    choices=["true", "false"],
                    help="Whether view-dependent pruning should probe only "
                         "foreground-hit rays or sample uniformly across views.")
    ap.add_argument("--adaptive-prune-target-final", type=int, default=None,
                    help="Late active-set budget for contribution-ranked pruning. "
                         "0 disables this compactness schedule.")
    ap.add_argument("--adaptive-prune-start-fraction", type=float, default=None,
                    help="Training fraction where contribution-budget pruning begins.")
    ap.add_argument("--adaptive-prune-end-fraction", type=float, default=None,
                    help="Training fraction where the final active-set budget is reached.")
    ap.add_argument("--adaptive-prune-min-keep", type=int, default=None,
                    help="Hard lower bound on alive primitives during adaptive pruning.")
    ap.add_argument("--opacity-reset-interval", type=int, default=None,
                    help="Override periodic opacity reset cadence. "
                         "3DGS-inspired: resetting alpha keeps alive "
                         "primitives competing instead of all saturating "
                         "to 1. Set 0 to disable.")
    ap.add_argument("--mu-gate-offset", type=float, default=None,
                    help="Override Eq. 4 gate offset μ.")
    ap.add_argument("--mu-gate-offset-final", type=float, default=None,
                    help="Late-stage target value for μ; ramps from --mu-gate-offset.")
    ap.add_argument("--mu-gate-ramp-start-fraction", type=float, default=None,
                    help="Start fraction for late-stage μ ramp.")
    ap.add_argument("--gate-mode", default=None,
                    choices=["stabilized", "paper_literal"],
                    help="Gate handling mode around Eq. 4 / Eq. 7. "
                         "'stabilized' keeps the current theta-floor "
                         "curriculum; 'paper_literal' disables the "
                         "curriculum and uses only a tiny theta safety eps.")
    ap.add_argument("--theta-min", type=float, default=None,
                    help="Override the minimum effective theta floor.")
    ap.add_argument("--theta-min-nsq", type=float, default=None,
                    help="Separate theta floor for NSQ gate (friend's #1). Higher=softer NSQ gate.")
    ap.add_argument("--paper-literal-theta-eps", type=float, default=None,
                    help="Tiny theta epsilon used only in --gate-mode paper_literal.")
    ap.add_argument("--delta-p-mode", default=None,
                    choices=["fixed", "half_delta"],
                    help="Eq. 7 finite-difference step mode. "
                         "'fixed' uses a global constant; 'half_delta' "
                         "uses half the local ray spacing per sample.")
    ap.add_argument("--delta-p-value", type=float, default=None,
                    help="Fixed Δp used when --delta-p-mode fixed.")
    ap.add_argument("--delta-p-scale", type=float, default=None,
                    help="Multiplier on the local ray spacing when "
                         "--delta-p-mode half_delta.")
    ap.add_argument("--color-weight-mode", default=None,
                    choices=["alpha_density", "density_only"],
                    help="Eq. 8 hostile-audit switch for per-sample color blending.")
    ap.add_argument("--point-normal-weight-mode", default=None,
                    choices=["alpha_density", "density_only"],
                    help="Eq. 11 hostile-audit switch for point-normal blending.")
    ap.add_argument("--final-normal-normalize", default=None,
                    choices=["true", "false"],
                    help="Whether to normalize the final composited normal map after Eq. 10.")
    ap.add_argument("--theta-curriculum-start", type=float, default=None,
                    help="Override the initial theta curriculum floor.")
    ap.add_argument("--theta-curriculum-fraction", type=float, default=None,
                    help="Override the fraction of training used by the theta curriculum.")
    ap.add_argument("--lambda-overlap", type=float, default=None,
                    help="Pairwise PSQ bounding-sphere repulsion (friend's #4 audit fix).")
    ap.add_argument("--lambda-overlap-final", type=float, default=None,
                    help="Optional late-stage target for lambda_overlap. "
                         "Use this to keep early anti-collapse pressure but "
                         "relax final seams/contact.")
    ap.add_argument("--overlap-ramp-start-fraction", type=float, default=None,
                    help="Start fraction for the lambda_overlap ramp/decay.")
    ap.add_argument("--overlap-ramp-end-fraction", type=float, default=None,
                    help="End fraction for the lambda_overlap ramp/decay.")
    ap.add_argument("--lambda-open-ray", type=float, default=None,
                    help="Weight on the open-ray loss (round-7 addition). "
                         "Penalizes predicted mask > 0 on rays passing "
                         "through GT holes, STRONGER than the global BCE "
                         "mask loss. Specifically rewards NSQ carving. "
                         "Default off (0). Round 7 recipe: 5.0. "
                         "NOTE: superseded by --hole-ray-oversample which "
                         "avoids the NaN-grad cascade this loss triggers.")
    ap.add_argument("--lambda-nsq-carve", type=float, default=None,
                    help="Auxiliary Eq-4 knife pressure on false-positive rays. "
                         "Uses only rendered/target masks and calibrated rays, "
                         "but is not part of paper-literal Eq. 12.")
    ap.add_argument("--nsq-carve-samples", type=int, default=None,
                    help="Samples around predicted false-positive depth for --lambda-nsq-carve.")
    ap.add_argument("--nsq-carve-depth-band", type=float, default=None,
                    help="Depth band around predicted false-positive surface for NSQ carve pressure.")
    ap.add_argument("--nsq-carve-residual-threshold", type=float, default=None,
                    help="Minimum rendered-mask false-positive residual used by NSQ carve pressure.")
    ap.add_argument("--hole-ray-oversample", type=float, default=0.0,
                    help="Fraction of rays per batch drawn specifically "
                         "from GT hole pixels. Upweights the existing "
                         "BCE mask loss at hole pixels WITHOUT adding a "
                         "new loss path — avoids round-7's NaN-grad "
                         "cascade. Round 8 recipe: 0.3 (30%% of rays).")
    ap.add_argument("--add-hole-axis-views", action="store_true",
                    help="For mesh_rendered_views, append detected hole-axis "
                         "views to the paper 24+sphere/top/bottom set. Off "
                         "by default because it is not paper parity.")
    ap.add_argument("--n-hole-ring", type=int, default=None,
                    help="Number of views per side of the detected hole axis.")
    ap.add_argument("--hole-tilt-deg", type=float, default=None,
                    help="Cone tilt for --add-hole-axis-views.")
    ap.add_argument("--init-from-visual-hull", action="store_true",
                    help="Initialize primitive regions from calibrated masks "
                         "and views.json instead of blind random placement. "
                         "This is a paper-driven augmented variant, not "
                         "paper-parity random initialization.")
    ap.add_argument("--visual-hull-grid-res", type=int, default=None,
                    help="Voxel resolution for --init-from-visual-hull.")
    ap.add_argument("--visual-hull-scale-margin", type=float, default=None,
                    help="Scale multiplier applied to each visual-hull region box.")
    ap.add_argument("--visual-hull-region-method", default=None,
                    choices=["recursive", "watershed"],
                    help="How to partition the visual hull into primitive supports.")
    ap.add_argument("--visual-hull-active-start", type=int, default=None,
                    help="If >0, only this many visual-hull regions start alive; "
                         "remaining initialized slots are born later.")
    ap.add_argument("--visual-hull-birth-interval", type=int, default=None,
                    help="Activate queued visual-hull primitive slots every N iterations.")
    ap.add_argument("--visual-hull-birth-count", type=int, default=None,
                    help="Number of queued visual-hull slots to activate per birth event.")
    ap.add_argument("--visual-hull-birth-stop-fraction", type=float, default=None,
                    help="Stop visual-hull primitive births after this training fraction.")
    ap.add_argument("--visual-hull-birth-strategy", default=None,
                    choices=["residual", "scheduled", "hybrid"],
                    help="How queued visual-hull primitive slots are selected. "
                         "'residual' activates regions with highest current "
                         "under-coverage; 'scheduled' preserves FIFO behavior; "
                         "'hybrid' falls back to FIFO if residual scores are low.")
    ap.add_argument("--visual-hull-birth-min-score", type=float, default=None,
                    help="Minimum residual score needed to activate a queued region.")
    ap.add_argument("--visual-hull-birth-region-sigma", type=float, default=None,
                    help="Region-support width used when scoring residual rays.")
    ap.add_argument("--visual-hull-nsq-init", default=None,
                    choices=["centered", "knife"],
                    help="NSQ placement for visual-hull init. 'knife' offsets "
                         "NSQs toward region faces so they start as cutters.")
    ap.add_argument("--visual-hull-nsq-offset-fraction", type=float, default=None,
                    help="Offset, in region half-extent units, for visual-hull NSQ knife init.")
    ap.add_argument("--visual-hull-nsq-scale-fraction", type=float, default=None,
                    help="NSQ scale as a fraction of PSQ region scale for visual-hull init.")
    ap.add_argument("--lambda-region-ownership", type=float, default=None,
                    help="Soft local-region ownership prior for visual-hull init. "
                         "Keeps primitives near their mask-derived support cells.")
    ap.add_argument("--lambda-region-ownership-final", type=float, default=None,
                    help="Late-stage target for region ownership. Use 0 to "
                         "release local supports after the assignment phase.")
    ap.add_argument("--region-ownership-ramp-start-fraction", type=float, default=None,
                    help="Start fraction for region-ownership decay/ramp.")
    ap.add_argument("--region-ownership-ramp-end-fraction", type=float, default=None,
                    help="End fraction for region-ownership decay/ramp.")
    ap.add_argument("--region-anchor-margin", type=float, default=None,
                    help="Allowed translation radius in units of each visual-hull "
                         "region half-extent.")
    ap.add_argument("--region-scale-growth", type=float, default=None,
                    help="Allowed PSQ/NSQ scale growth over each visual-hull "
                         "region half-extent before penalty starts.")
    ap.add_argument("--detect-anomaly", action="store_true",
                    help="Enable torch.autograd.detect_anomaly for the "
                         "training loop. Slow but useful for identifying "
                         "the exact backward op that first creates NaN.")
    ap.add_argument("--abort-on-nan-grad", action="store_true",
                    help="Abort immediately on the first non-finite "
                         "gradient instead of continuing with skip logic.")
    ap.add_argument("--prepare-detail-refine", action="store_true",
                    help="After export, prepare narrow-band detail-refinement "
                         "artifacts from the DualPrim mesh and a target mesh. "
                         "This keeps DualPrim as the coarse scaffold and emits "
                         "a constrained manifest for a later detail stage.")
    ap.add_argument("--detail-target-mesh", default=None,
                    help="Reference mesh used to prepare detail-refinement "
                         "artifacts. Required with --prepare-detail-refine.")
    ap.add_argument("--detail-out", default=None,
                    help="Output dir for detail-refinement artifacts. "
                         "Defaults to <out>/detail_refine.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config
    from clearmesh.dualprim import DualPrimConfig, init_scene, train, export_scene

    config = DualPrimConfig(
        num_primitives_init=args.k,
        num_iterations=args.iters,
        view_resolution=args.resolution,
        mode=args.mode,
        seed=args.seed,
        nsq_init_strategy=args.nsq_init,
    )
    paper_preset = args.paper_preset
    if args.mode == "paper" and paper_preset == "off":
        paper_preset = "parity"
    if paper_preset == "parity":
        config.init_profile = "paper_random"
        config.nsq_init_strategy = "independent"
        config.gate_mode = "paper_literal"
        config.theta_min = 0.0
        config.theta_min_nsq = 0.0
        config.theta_curriculum_start = 0.0
        config.theta_curriculum_fraction = 0.0
        config.primitive_reg_average_mode = "fixed_k"
        config.mask_loss_type = "bce"
        config.opacity_reset_interval = 0
        config.lambda_depth = 0.0
        config.lambda_open_ray = 0.0
        config.lambda_nsq_carve = 0.0
        config.lambda_overlap = 0.0
        config.lambda_region_ownership = 0.0
        config.lambda_edge_mask = 0.0
        config.lambda_shape_box = 0.0
        config.init_from_visual_hull = False
        config.visual_hull_birth_interval = 0
        config.adaptive_prune_target_final = 0
    elif paper_preset == "augmented":
        # No silent behavior change here; this preset is primarily a
        # provenance label. Individual augmented mechanisms remain explicit
        # CLI/config choices so ablations are obvious in logs.
        pass
    if args.mode in ("mesh_fit", "mesh_rendered_views") and args.input is None:
        raise ValueError(f"--mode {args.mode} requires --input")
    if args.mode == "paper" and args.views_dir is None:
        raise ValueError("--mode paper requires --views-dir with calibrated RGB/mask views")
    if args.mode == "paper" and args.normal_source is None:
        config.normal_source = "stablenormal"
    if args.init_profile is not None:
        config.init_profile = args.init_profile
    # CLI overrides for round-4+ tuning levers
    if args.lambda_mask is not None:
        config.lambda_mask = args.lambda_mask
    if args.lambda_sparse is not None:
        config.lambda_sparse = args.lambda_sparse
    if args.lambda_entropy is not None:
        config.lambda_entropy = args.lambda_entropy
    if args.lambda_max is not None:
        config.lambda_max = args.lambda_max
    if args.lambda_norm_reg is not None:
        config.lambda_norm_reg = args.lambda_norm_reg
    if args.lambda_depth is not None:
        config.lambda_depth = args.lambda_depth
    if args.lambda_edge_mask is not None:
        config.lambda_edge_mask = args.lambda_edge_mask
    if args.lambda_shape_box is not None:
        config.lambda_shape_box = args.lambda_shape_box
    if args.shape_box_threshold is not None:
        config.shape_box_threshold = args.shape_box_threshold
    if args.normal_loss_type is not None:
        config.normal_loss_type = args.normal_loss_type
    if args.log_shape_grad_stats:
        config.log_shape_grad_stats = True
    if args.lambda_norm_reg_final is not None:
        config.lambda_norm_reg_final = args.lambda_norm_reg_final
    if args.norm_reg_ramp_start_fraction is not None:
        config.norm_reg_ramp_start_fraction = args.norm_reg_ramp_start_fraction
    if args.norm_reg_ramp_end_fraction is not None:
        config.norm_reg_ramp_end_fraction = args.norm_reg_ramp_end_fraction
    if args.masked_loss_norm_mode is not None:
        config.masked_loss_norm_mode = args.masked_loss_norm_mode
    if args.primitive_reg_average_mode is not None:
        config.primitive_reg_average_mode = args.primitive_reg_average_mode
    if args.mask_loss_type is not None:
        config.mask_loss_type = args.mask_loss_type
    if args.tessellation_resolution is not None:
        config.tessellation_resolution = args.tessellation_resolution
    if args.num_samples_per_ray is not None:
        config.num_samples_per_ray = args.num_samples_per_ray
    if args.sampling_mode is not None:
        config.sampling_mode = args.sampling_mode
    if args.num_importance_samples_per_ray is not None:
        config.num_importance_samples_per_ray = args.num_importance_samples_per_ray
    if args.shape_range_lo is not None or args.shape_range_hi is not None:
        lo, hi = config.shape_range
        if args.shape_range_lo is not None:
            lo = args.shape_range_lo
        if args.shape_range_hi is not None:
            hi = args.shape_range_hi
        config.shape_range = (lo, hi)
    if args.export_cleanup_min_faces is not None:
        config.export_cleanup_min_component_faces = args.export_cleanup_min_faces
    if args.export_cleanup_min_area_ratio is not None:
        config.export_cleanup_min_component_area_ratio = args.export_cleanup_min_area_ratio
    if args.export_smoothing_iters is not None:
        config.export_smoothing_iterations = args.export_smoothing_iters
    if args.export_smoothing_lambda is not None:
        config.export_smoothing_lambda = args.export_smoothing_lambda
    if args.export_smoothing_nu is not None:
        config.export_smoothing_nu = args.export_smoothing_nu
    if args.normal_source is not None:
        config.normal_source = args.normal_source
    if args.add_hole_axis_views:
        config.add_hole_axis_views = True
    if args.n_hole_ring is not None:
        config.n_hole_ring = args.n_hole_ring
    if args.hole_tilt_deg is not None:
        config.hole_tilt_deg = args.hole_tilt_deg
    if args.init_from_visual_hull:
        config.init_from_visual_hull = True
    if args.visual_hull_grid_res is not None:
        config.visual_hull_grid_res = args.visual_hull_grid_res
    if args.visual_hull_scale_margin is not None:
        config.visual_hull_scale_margin = args.visual_hull_scale_margin
    if args.visual_hull_region_method is not None:
        config.visual_hull_region_method = args.visual_hull_region_method
    if args.visual_hull_active_start is not None:
        config.visual_hull_active_start = args.visual_hull_active_start
    if args.visual_hull_birth_interval is not None:
        config.visual_hull_birth_interval = args.visual_hull_birth_interval
    if args.visual_hull_birth_count is not None:
        config.visual_hull_birth_count = args.visual_hull_birth_count
    if args.visual_hull_birth_stop_fraction is not None:
        config.visual_hull_birth_stop_fraction = args.visual_hull_birth_stop_fraction
    if args.visual_hull_birth_strategy is not None:
        config.visual_hull_birth_strategy = args.visual_hull_birth_strategy
    if args.visual_hull_birth_min_score is not None:
        config.visual_hull_birth_min_score = args.visual_hull_birth_min_score
    if args.visual_hull_birth_region_sigma is not None:
        config.visual_hull_birth_region_sigma = args.visual_hull_birth_region_sigma
    if args.visual_hull_nsq_init is not None:
        config.visual_hull_nsq_init = args.visual_hull_nsq_init
    if args.visual_hull_nsq_offset_fraction is not None:
        config.visual_hull_nsq_offset_fraction = args.visual_hull_nsq_offset_fraction
    if args.visual_hull_nsq_scale_fraction is not None:
        config.visual_hull_nsq_scale_fraction = args.visual_hull_nsq_scale_fraction
    if args.lambda_region_ownership is not None:
        config.lambda_region_ownership = args.lambda_region_ownership
    if args.lambda_region_ownership_final is not None:
        config.lambda_region_ownership_final = args.lambda_region_ownership_final
    if args.region_ownership_ramp_start_fraction is not None:
        config.region_ownership_ramp_start_fraction = args.region_ownership_ramp_start_fraction
    if args.region_ownership_ramp_end_fraction is not None:
        config.region_ownership_ramp_end_fraction = args.region_ownership_ramp_end_fraction
    if args.region_anchor_margin is not None:
        config.region_anchor_margin = args.region_anchor_margin
    if args.region_scale_growth is not None:
        config.region_scale_growth = args.region_scale_growth
    if config.add_hole_axis_views:
        config.num_views = 26 + 2 * config.n_hole_ring
    if args.stablenormal_turbo:
        config.stablenormal_use_turbo = True
    if args.stablenormal_cache_dir is not None:
        config.stablenormal_cache_dir = args.stablenormal_cache_dir
    if args.stablenormal_blend_strength is not None:
        config.stablenormal_blend_strength = args.stablenormal_blend_strength
    if args.stablenormal_agreement_floor is not None:
        config.stablenormal_agreement_floor = args.stablenormal_agreement_floor
    if args.stablenormal_agreement_ceil is not None:
        config.stablenormal_agreement_ceil = args.stablenormal_agreement_ceil
    if args.stablenormal_edge_boost is not None:
        config.stablenormal_edge_boost = args.stablenormal_edge_boost
    if args.prune_alpha_threshold is not None:
        config.prune_alpha_threshold = args.prune_alpha_threshold
    if args.pruning_interval is not None:
        config.pruning_interval = args.pruning_interval
    if args.view_prune_weight_threshold is not None:
        config.view_prune_weight_threshold = args.view_prune_weight_threshold
    if args.view_prune_every_multiplier is not None:
        config.view_prune_every_multiplier = args.view_prune_every_multiplier
    if args.view_prune_foreground_only is not None:
        config.view_prune_foreground_only = (args.view_prune_foreground_only == "true")
    if args.adaptive_prune_target_final is not None:
        config.adaptive_prune_target_final = args.adaptive_prune_target_final
    if args.adaptive_prune_start_fraction is not None:
        config.adaptive_prune_start_fraction = args.adaptive_prune_start_fraction
    if args.adaptive_prune_end_fraction is not None:
        config.adaptive_prune_end_fraction = args.adaptive_prune_end_fraction
    if args.adaptive_prune_min_keep is not None:
        config.adaptive_prune_min_keep = args.adaptive_prune_min_keep
    if args.opacity_reset_interval is not None:
        config.opacity_reset_interval = args.opacity_reset_interval
    if args.mu_gate_offset is not None:
        config.mu_gate_offset = args.mu_gate_offset
    if args.mu_gate_offset_final is not None:
        config.mu_gate_offset_final = args.mu_gate_offset_final
    if args.mu_gate_ramp_start_fraction is not None:
        config.mu_gate_ramp_start_fraction = args.mu_gate_ramp_start_fraction
    if args.gate_mode is not None:
        config.gate_mode = args.gate_mode
    if args.theta_min is not None:
        config.theta_min = args.theta_min
    if args.theta_min_nsq is not None:
        config.theta_min_nsq = args.theta_min_nsq
    if args.paper_literal_theta_eps is not None:
        config.paper_literal_theta_eps = args.paper_literal_theta_eps
    if args.delta_p_mode is not None:
        config.delta_p_mode = args.delta_p_mode
    if args.delta_p_value is not None:
        config.delta_p_value = args.delta_p_value
    if args.delta_p_scale is not None:
        config.delta_p_scale = args.delta_p_scale
    if args.color_weight_mode is not None:
        config.color_weight_mode = args.color_weight_mode
    if args.point_normal_weight_mode is not None:
        config.point_normal_weight_mode = args.point_normal_weight_mode
    if args.final_normal_normalize is not None:
        config.final_normal_normalize = (args.final_normal_normalize == "true")
    if args.theta_curriculum_start is not None:
        config.theta_curriculum_start = args.theta_curriculum_start
    if args.theta_curriculum_fraction is not None:
        config.theta_curriculum_fraction = args.theta_curriculum_fraction
    if args.lambda_overlap is not None:
        config.lambda_overlap = args.lambda_overlap
    if args.lambda_overlap_final is not None:
        config.lambda_overlap_final = args.lambda_overlap_final
    if args.overlap_ramp_start_fraction is not None:
        config.overlap_ramp_start_fraction = args.overlap_ramp_start_fraction
    if args.overlap_ramp_end_fraction is not None:
        config.overlap_ramp_end_fraction = args.overlap_ramp_end_fraction
    if args.lambda_open_ray is not None:
        config.lambda_open_ray = args.lambda_open_ray
    if args.lambda_nsq_carve is not None:
        config.lambda_nsq_carve = args.lambda_nsq_carve
    if args.nsq_carve_samples is not None:
        config.nsq_carve_samples = args.nsq_carve_samples
    if args.nsq_carve_depth_band is not None:
        config.nsq_carve_depth_band = args.nsq_carve_depth_band
    if args.nsq_carve_residual_threshold is not None:
        config.nsq_carve_residual_threshold = args.nsq_carve_residual_threshold
    if args.views_dir is not None:
        meta_path = Path(args.views_dir) / "views.json"
        if meta_path.exists():
            with open(meta_path) as f:
                config.num_views = len(json.load(f).get("views", []))
    if paper_preset == "parity":
        parity_violations = []
        if config.init_profile != "paper_random":
            parity_violations.append(f"init_profile={config.init_profile}")
        if config.nsq_init_strategy != "independent":
            parity_violations.append(f"nsq_init_strategy={config.nsq_init_strategy}")
        if config.gate_mode != "paper_literal":
            parity_violations.append(f"gate_mode={config.gate_mode}")
        if config.theta_curriculum_start != 0.0 or config.theta_curriculum_fraction != 0.0:
            parity_violations.append("theta_curriculum")
        if config.primitive_reg_average_mode != "fixed_k":
            parity_violations.append(f"primitive_reg_average_mode={config.primitive_reg_average_mode}")
        if config.opacity_reset_interval != 0:
            parity_violations.append(f"opacity_reset_interval={config.opacity_reset_interval}")
        if config.init_from_visual_hull:
            parity_violations.append("init_from_visual_hull")
        if config.visual_hull_birth_interval:
            parity_violations.append(f"visual_hull_birth_interval={config.visual_hull_birth_interval}")
        if config.adaptive_prune_target_final:
            parity_violations.append(f"adaptive_prune_target_final={config.adaptive_prune_target_final}")
        for name in (
            "lambda_depth", "lambda_open_ray", "lambda_nsq_carve",
            "lambda_overlap", "lambda_region_ownership", "lambda_edge_mask",
            "lambda_shape_box",
        ):
            if float(getattr(config, name, 0.0) or 0.0) != 0.0:
                parity_violations.append(f"{name}={getattr(config, name)}")
        if parity_violations:
            raise ValueError(
                "--paper-preset parity forbids non-paper settings: "
                + ", ".join(parity_violations)
                + ". Use --paper-preset augmented for paper-driven variants."
            )
    print(
        "[canary] paper_preset="
        f"{paper_preset} init={config.init_profile} nsq={config.nsq_init_strategy} "
        f"gate={config.gate_mode} reg_avg={config.primitive_reg_average_mode} "
        f"opacity_reset={config.opacity_reset_interval} "
        f"visual_hull={config.init_from_visual_hull} "
        f"aux(depth/open/knife/ov/reg/edge/box)="
        f"{config.lambda_depth}/{config.lambda_open_ray}/{config.lambda_nsq_carve}/"
        f"{config.lambda_overlap}/{config.lambda_region_ownership}/"
        f"{config.lambda_edge_mask}/{config.lambda_shape_box}"
    )
    paperish_export = args.mode == "paper" or config.init_from_visual_hull
    union_export = args.union_export
    if union_export is None:
        union_export = paperish_export
    require_fused_export = paperish_export and not args.allow_preview_export
    if require_fused_export and not union_export:
        raise ValueError(
            "paper/visual-hull runs require fused boolean export. "
            "Use --allow-preview-export --no-union-export only for debugging."
        )
    if require_fused_export and config.boolean_backend == "manifold3d":
        try:
            import manifold3d  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "paper/visual-hull runs now require manifold3d for fused "
                "boolean export. Install requirements.txt or rerun with "
                "--allow-preview-export for debug-only concatenation."
            ) from e
    # Write the effective config for reproducibility
    from dataclasses import asdict
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # ----- Initialize scene -----
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[canary] device={device}")
    print(f"[canary] mode={args.mode}, K={config.num_primitives_init}, "
          f"iters={config.num_iterations}")
    if args.resume_primitives:
        from clearmesh.dualprim.io import load_scene_from_json
        print(f"[canary] WARM START from {args.resume_primitives}")
        scene = load_scene_from_json(
            args.resume_primitives, config,
            device=device, pad_to_K=config.num_primitives_init,
        )
        print(f"[canary] loaded {scene.num_alive}/{scene.K} live primitives")
    else:
        scene = init_scene(config, device=device)
    if args.psq_shape_init is not None:
        from clearmesh.dualprim.types import IDX_PSQ_SHAPE
        with torch.no_grad():
            scene.params[:, IDX_PSQ_SHAPE] = float(args.psq_shape_init)
        print(f"[canary] forced PSQ ε init to {args.psq_shape_init:.4f}")
    if args.freeze_psq_shape:
        from clearmesh.dualprim.types import IDX_PSQ_SHAPE

        def _freeze_psq_shape_grad(grad):
            grad = grad.clone()
            grad[:, IDX_PSQ_SHAPE] = 0.0
            return grad

        scene.params.register_hook(_freeze_psq_shape_grad)
        print("[canary] freezing PSQ ε gradients")

    # ----- Build the ray sampler for this mode -----
    if args.mode == "mesh_fit":
        # mesh_fit skips the renderer entirely. Precompute a TSDF and
        # use the dedicated train_mesh_fit() path below. We bypass the
        # ray sampler / train() flow.
        sampler = None
    elif args.mode == "mesh_rendered_views":
        views_dir = Path(args.views_dir) if args.views_dir else (out_dir / "views")
        if args.views_dir:
            print(f"[canary] using prepared views from {views_dir}")
            _validate_views_dir(
                views_dir,
                require_depth=config.lambda_depth > 0,
                normal_source=config.normal_source,
            )
        else:
            from scripts.dualprim.render_views import render_views
            render_views(
                args.input, str(views_dir),
                resolution=config.view_resolution,
                render_normals=True,
                add_hole_axis_views=config.add_hole_axis_views,
                n_hole_ring=config.n_hole_ring,
                hole_tilt_deg=config.hole_tilt_deg,
            )
        if config.normal_source == "stablenormal":
            print(f"[canary] predicting StableNormal maps in {views_dir}")
            _ensure_stablenormal_views(
                views_dir,
                device=device,
                use_turbo=config.stablenormal_use_turbo,
                cache_dir=config.stablenormal_cache_dir,
            )
        sampler = _build_views_sampler(
            views_dir, device=device,
            fg_bias=args.fg_bias,
            hole_ray_oversample=args.hole_ray_oversample,
            require_depth=config.lambda_depth > 0,
            normal_source=config.normal_source,
            stablenormal_blend_strength=config.stablenormal_blend_strength,
            stablenormal_agreement_floor=config.stablenormal_agreement_floor,
            stablenormal_agreement_ceil=config.stablenormal_agreement_ceil,
            stablenormal_edge_boost=config.stablenormal_edge_boost,
        )
    elif args.mode == "paper":
        views_dir = Path(args.views_dir)
        print(f"[canary] paper mode: using prepared source views from {views_dir}")
        _validate_views_dir(
            views_dir,
            require_depth=config.lambda_depth > 0,
            normal_source=config.normal_source,
        )
        if config.normal_source == "stablenormal":
            print(f"[canary] predicting StableNormal maps in {views_dir}")
            _ensure_stablenormal_views(
                views_dir,
                device=device,
                use_turbo=config.stablenormal_use_turbo,
                cache_dir=config.stablenormal_cache_dir,
            )
        sampler = _build_views_sampler(
            views_dir, device=device,
            fg_bias=args.fg_bias,
            hole_ray_oversample=args.hole_ray_oversample,
            require_depth=config.lambda_depth > 0,
            normal_source=config.normal_source,
            stablenormal_blend_strength=config.stablenormal_blend_strength,
            stablenormal_agreement_floor=config.stablenormal_agreement_floor,
            stablenormal_agreement_ceil=config.stablenormal_agreement_ceil,
            stablenormal_edge_boost=config.stablenormal_edge_boost,
        )

    if config.init_from_visual_hull:
        if args.mode == "mesh_fit":
            raise ValueError("--init-from-visual-hull requires a view-based mode")
        from clearmesh.dualprim.view_init import apply_visual_hull_init
        stats = apply_visual_hull_init(
            scene,
            views_dir,
            config,
            grid_res=config.visual_hull_grid_res,
            scale_margin=config.visual_hull_scale_margin,
            region_method=config.visual_hull_region_method,
        )
        print(f"[canary] visual-hull init: {stats['num_regions']} regions, "
              f"{stats['num_occupied_voxels']:,} occupied voxels "
              f"at grid={stats['grid_res']} method={stats['region_method']} "
              f"initial_alive={stats['num_initially_alive']}")

    # ----- Train -----
    print(f"[canary] training for {config.num_iterations} iters")
    log_path = out_dir / "logs.json"
    log_rows = []

    if args.mode == "mesh_fit":
        # Precompute target TSDF on a grid, use train_mesh_fit
        from clearmesh.dualprim.optimize_scene import train_mesh_fit
        query_points, target_sdf = _build_mesh_fit_tsdf(
            args.input, device=device, resolution=64,
        )
        print(f"[canary] TSDF: {query_points.shape[0]:,} samples, "
              f"target range [{target_sdf.min():.3f}, {target_sdf.max():.3f}]")

        def log_fn(it, parts):
            log_rows.append(parts)
            print(f"[canary] it={it:6d} total={parts['total']:.4f} "
                  f"tsdf={parts['tsdf']:.4f} alive={parts['alive']}")
            with open(log_path, "w") as f:
                json.dump(log_rows, f, indent=2)

        t0 = time.time()
        state = train_mesh_fit(
            scene, query_points, target_sdf, config,
            samples_per_batch=args.rays * 4,   # re-use --rays as batch size
            device=device,
            log_fn=log_fn,
            checkpoint_path=str(out_dir / "checkpoints"),
        )
    else:
        def log_fn(it, parts):
            log_rows.append(parts)
            # Core loss + prune health
            line = (
                f"[canary] it={it:6d} total={parts['total']:.4f} "
                f"rgb={parts['rgb']:.3f} mask={parts['mask']:.3f} "
                f"norm={parts['norm']:.3f} alive={parts['alive']}"
            )
            if parts.get("depth", 0) > 0:
                line += f" depth={parts['depth']:.4f}"
            # Open-ray loss (round 7+): only show if >0
            if parts.get("open", 0) > 0:
                line += f" open={parts['open']:.4f}"
            if parts.get("nsq_carve", 0) > 0:
                line += f" knife={parts['nsq_carve']:.4f}"
            if parts.get("region", 0) > 0:
                line += f" region={parts['region']:.4f}"
            if parts.get("nan_grad_skip"):
                line += " [nan_grad]"
            # NSQ-health diagnostics (added in review round 3)
            if "theta_p50" in parts:
                line += (f"  θ[{parts['theta_p10']:.2f}/{parts['theta_p50']:.2f}/"
                         f"{parts['theta_p90']:.2f}]")
            if "eps_psq_p50" in parts:
                line += (f" εpsq[{parts['eps_psq_p10']:.2f}/{parts['eps_psq_p50']:.2f}/"
                         f"{parts['eps_psq_p90']:.2f}]")
            if "theta_min_eff" in parts:
                line += f" θ_min_eff={parts['theta_min_eff']:.2f}"
            if "mu_gate_eff" in parts:
                line += f" μ={parts['mu_gate_eff']:.2f}"
            if "lambda_norm_eff" in parts:
                line += f" λn={parts['lambda_norm_eff']:.2f}"
            if "lambda_overlap_eff" in parts:
                line += f" λov={parts['lambda_overlap_eff']:.2f}"
            if "lambda_region_eff" in parts and parts["lambda_region_eff"] > 0:
                line += f" λreg={parts['lambda_region_eff']:.3f}"
            if parts.get("birth_count"):
                line += f" birth={parts['birth_count']}:{parts.get('birth_strategy', '?')}"
            if parts.get("adaptive_prune_target"):
                line += (f" prune→{parts['adaptive_prune_target']}"
                         f"(-{parts.get('adaptive_prune_killed', 0)})")
            if "nsq_overlap_pct" in parts:
                line += f" NSQ∩PSQ={parts['nsq_overlap_pct']:.0f}%"
            if "carve_ratio_p50" in parts:
                line += (f" carve[{parts['carve_ratio_p10']:.2f}/"
                         f"{parts['carve_ratio_p50']:.2f}/"
                         f"{parts['carve_ratio_p90']:.2f}]")
            if "nsq_offset_p50" in parts:
                line += (f" off[{parts['nsq_offset_p10']:.2f}/"
                         f"{parts['nsq_offset_p50']:.2f}/"
                         f"{parts['nsq_offset_p90']:.2f}]")
            if "pe_mean_fg" in parts:
                line += f" P_E_fg={parts['pe_mean_fg']:.3f}/{parts['pe_max_fg']:.2f}"
            if "t_render_ms" in parts:
                # Stage timings — per-iter average for this window. Friend's
                # debug recommendation: catches "one iteration takes 30s"
                # cost-cliff failure modes (NaN-skip storms, etc).
                line += (
                    f"  t[r{parts['t_render_ms']:.0f}/l{parts['t_loss_ms']:.0f}"
                    f"/s{parts['t_step_ms']:.0f}/p{parts['t_prune_ms']:.0f}]ms"
                )
            if "nan_grad_summary" in parts:
                line += f" offender={parts['nan_grad_summary']}"
            print(line)
            with open(log_path, "w") as f:
                json.dump(log_rows, f, indent=2)

        t0 = time.time()
        state = train(
            scene, sampler, config,
            rays_per_batch=args.rays,
            device=device,
            log_fn=log_fn,
            checkpoint_path=str(out_dir / "checkpoints"),
            trajectory_dir=args.trajectory_dir,
            detect_anomaly=args.detect_anomaly,
            abort_on_nan_grad=args.abort_on_nan_grad,
        )

    train_dt = time.time() - t0
    print(f"[canary] training done in {train_dt/60:.1f} min "
          f"— {scene.num_alive}/{config.num_primitives_init} alive")

    # ----- Export -----
    export_mode = "union" if union_export else "concat-preview"
    print(f"[canary] exporting (α ≥ {config.export_alpha_threshold}, mode={export_mode})")
    scene_mesh, per_prim = export_scene(
        scene, config,
        union_all=union_export,
        require_union=require_fused_export,
    )
    scene_mesh.export(out_dir / "refit.glb")
    for i, m in enumerate(per_prim):
        m.export(out_dir / f"per_prim_{i:03d}.glb")
    export_summary = _mesh_export_summary(scene_mesh, len(per_prim), union_export)
    with open(out_dir / "export_summary.json", "w") as f:
        json.dump(export_summary, f, indent=2)
    print(
        f"[canary] export: components={export_summary['components']} "
        f"largest_faces={export_summary['largest_component_faces_pct']:.1%} "
        f"watertight={export_summary['watertight']}"
    )

    # Save primitive params (same format as trajectory snapshots for
    # corpus-uniformity — downstream dataset loaders can treat the
    # final primitives.json as just another snapshot keyed at iter==N).
    from clearmesh.dualprim.io import save_scene_json
    save_scene_json(
        scene, out_dir / "primitives.json",
        iteration=config.num_iterations,
        extra_metadata={"training_s": train_dt},
    )

    detail_manifest = None
    if args.prepare_detail_refine:
        if not args.detail_target_mesh:
            raise ValueError("--prepare-detail-refine requires --detail-target-mesh")
        from clearmesh.dualprim import DetailRefineConfig, prepare_detail_refine_artifacts
        detail_out = Path(args.detail_out) if args.detail_out else (out_dir / "detail_refine")
        detail_manifest = prepare_detail_refine_artifacts(
            out_dir / "refit.glb",
            args.detail_target_mesh,
            detail_out,
            DetailRefineConfig(),
        )
        print(f"[canary] detail-refine manifest: {detail_out / 'detail_refine_manifest.json'}")

    print()
    print("=" * 60)
    print(f"CANARY DONE")
    print(f"  mode:     {args.mode}")
    print(f"  K final:  {scene.num_alive}/{config.num_primitives_init}")
    print(f"  iters:    {config.num_iterations}")
    print(f"  train_s:  {train_dt:.1f}")
    print(f"  scene_v:  {len(scene_mesh.vertices):,}")
    print(f"  scene_f:  {len(scene_mesh.faces):,}")
    print(f"  comps:    {export_summary['components']}")
    print(f"  export:   {export_mode}")
    print(f"  out:      {out_dir}")
    if detail_manifest is not None:
        detail = detail_manifest["detail_signal"]
        budget = detail_manifest["budget"]
        print(f"  detail:   band={detail['detail_band_ratio']:.3f} "
              f"chamfer≈{detail['chamfer_proxy']:.4f} "
              f"budget_v≤{budget['max_vertices']:,}")
    print("=" * 60)


# ---------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------

def _mesh_export_summary(mesh: trimesh.Trimesh, num_primitives: int, union_export: bool) -> dict:
    if len(mesh.faces) == 0:
        return {
            "num_primitives": int(num_primitives),
            "union_export": bool(union_export),
            "vertices": 0,
            "faces": 0,
            "components": 0,
            "watertight": False,
            "largest_component_faces": 0,
            "largest_component_faces_pct": 0.0,
            "component_faces_top10": [],
        }
    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        components = [mesh]
    component_faces = [int(len(c.faces)) for c in components]
    largest = max(component_faces, default=0)
    return {
        "num_primitives": int(num_primitives),
        "union_export": bool(union_export),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(components)),
        "watertight": bool(mesh.is_watertight),
        "largest_component_faces": int(largest),
        "largest_component_faces_pct": float(largest / max(len(mesh.faces), 1)),
        "component_faces_top10": sorted(component_faces, reverse=True)[:10],
    }


def _validate_views_dir(
    views_dir: Path,
    *,
    require_depth: bool = False,
    normal_source: str = "analytic",
) -> None:
    meta_path = views_dir / "views.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{views_dir}: missing views.json")
    with open(meta_path) as f:
        meta = json.load(f)
    if "camera" not in meta or "views" not in meta:
        raise ValueError(f"{meta_path}: expected camera + views metadata")
    n_views = len(meta["views"])
    if n_views == 0:
        raise ValueError(f"{meta_path}: no views")
    for i in range(n_views):
        for suffix in ("rgb.png", "mask.png"):
            p = views_dir / f"{i:02d}_{suffix}"
            if not p.exists():
                raise FileNotFoundError(p)
        if normal_source == "analytic":
            p = views_dir / f"{i:02d}_normal.png"
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} is required for --normal-source analytic"
                )
        if require_depth:
            p = views_dir / f"{i:02d}_depth.npy"
            if not p.exists():
                raise FileNotFoundError(p)


def _build_mesh_fit_tsdf(
    mesh_path: str, device: str, resolution: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a TSDF of the reference mesh on a regular grid in [-1, 1]^3.

    Returns (query_points, target_sdf) — both torch tensors on ``device``.
    Used as supervision for the mesh_fit debug path. Tries mesh2sdf
    (fast, CUDA) first and falls back to trimesh contains + EDT if not
    available — same strategy as clearmesh.refit.light_sq.build_tsdf.
    """
    import numpy as np

    mesh = trimesh.load(mesh_path, force="mesh")
    # Normalize to [-1+1/N, 1-1/N]^3 (same frame as init_scene)
    mesh = mesh.copy()
    mesh.vertices -= mesh.centroid
    s = mesh.extents.max()
    if s > 0:
        mesh.vertices *= (2.0 / s) * 0.98    # 2% margin

    lin = np.linspace(-1.0 + 1.0 / resolution, 1.0 - 1.0 / resolution, resolution)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    coords = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)

    try:
        import mesh2sdf
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        # mesh2sdf already returns negative-inside (matches paper).
        # Verified via scripts/dualprim/check_conventions.py group [4]:
        # raw centre-of-sphere = -0.976, raw corner-of-cube = +0.648.
        sdf = mesh2sdf.compute(
            verts, faces, size=resolution,
            fix=False, level=2.0 / resolution, return_mesh=False,
        ).astype(np.float32).reshape(-1)
    except ImportError:
        from scipy.ndimage import distance_transform_edt
        occupied = np.zeros(len(coords), dtype=bool)
        chunk = 100_000
        for i in range(0, len(coords), chunk):
            occupied[i:i + chunk] = mesh.contains(coords[i:i + chunk])
        occ_grid = occupied.reshape(resolution, resolution, resolution)
        voxel = 2.0 / resolution
        d_out = distance_transform_edt(~occ_grid) * voxel
        d_in = distance_transform_edt(occ_grid) * voxel
        sdf = np.where(occ_grid, -d_in, d_out).astype(np.float32).reshape(-1)

    return (
        torch.from_numpy(coords).to(device),
        torch.from_numpy(sdf).to(device),
    )


def _build_views_sampler(views_dir: Path, device: str,
                          fg_bias: float = 0.7,
                          hole_ray_oversample: float = 0.0,
                          require_depth: bool = False,
                          normal_source: str = "analytic",
                          stablenormal_blend_strength: float = 1.0,
                          stablenormal_agreement_floor: float = 0.5,
                          stablenormal_agreement_ceil: float = 0.95,
                          stablenormal_edge_boost: float = 0.0):
    """Sample rays from the rendered views in `views_dir`.

    fg_bias: fraction of rays drawn from "interesting" pixels (mask
    foreground OR mask-boundary OR near-silhouette hole pixels).
    Remaining (1 - fg_bias) drawn uniformly.

    hole_ray_oversample: fraction of rays specifically drawn from
    GT hole pixels (binary_fill_holes(mask) & ~mask). These pixels
    lie inside the projected silhouette but are empty in the ref —
    i.e. where the ref mesh has a hole. Oversampling them
    effectively upweights the existing BCE mask loss at hole pixels
    WITHOUT adding a new loss term (avoids the NaN-grad cascade
    seen when adding loss_open_ray). Friend's point 3: "weight
    hole/boundary rays higher in sampling, not in loss weight."

    Final ratio: n_hole + n_fg + n_uniform = n_rays, where
    n_hole = n_rays * hole_ray_oversample, n_fg = remaining * fg_bias.
    """
    from PIL import Image
    from clearmesh.dualprim import RaySampleBatch

    with open(views_dir / "views.json") as f:
        meta = json.load(f)
    V = len(meta["views"])
    H = W = meta["camera"]["resolution"]

    rgbs = np.zeros((V, H, W, 3), dtype=np.float32)
    masks = np.zeros((V, H, W), dtype=np.float32)
    normals = np.zeros((V, H, W, 3), dtype=np.float32)
    analytic_normals = np.zeros((V, H, W, 3), dtype=np.float32)
    stable_normals = np.zeros((V, H, W, 3), dtype=np.float32)
    depths = np.zeros((V, H, W), dtype=np.float32)
    poses = np.zeros((V, 4, 4), dtype=np.float32)
    has_analytic_normals = all((views_dir / f"{i:02d}_normal.png").exists() for i in range(V))
    for i in range(V):
        rgbs[i] = np.asarray(Image.open(views_dir / f"{i:02d}_rgb.png").convert("RGB")) / 255.0
        m = np.asarray(Image.open(views_dir / f"{i:02d}_mask.png").convert("L"))
        masks[i] = (m > 127).astype(np.float32)
        if has_analytic_normals:
            analytic_n = np.asarray(Image.open(views_dir / f"{i:02d}_normal.png").convert("RGB")) / 255.0
            analytic_normals[i] = analytic_n * 2.0 - 1.0
        elif normal_source == "analytic":
            raise FileNotFoundError(
                f"{views_dir / f'{i:02d}_normal.png'} is required for analytic normal supervision"
            )
        if normal_source == "stablenormal":
            stable_n = np.asarray(Image.open(views_dir / f"{i:02d}_normal_stablenormal.png").convert("RGB")) / 255.0
            stable_normals[i] = stable_n * 2.0 - 1.0
        else:
            normals[i] = analytic_normals[i]
        depth_path = views_dir / f"{i:02d}_depth.npy"
        if depth_path.exists():
            depths[i] = np.load(depth_path).astype(np.float32)
        elif require_depth:
            raise FileNotFoundError(
                f"Depth supervision requested, but missing {depth_path}. "
                "Regenerate views with the depth-enabled render_views.py cache."
            )
        poses[i] = np.asarray(meta["views"][i]["pose_world_from_camera"], dtype=np.float32)

    yfov = meta["camera"]["yfov_rad"]
    # Precompute per-pixel ray directions in camera space
    fx = fy = 0.5 * H / math.tan(yfov / 2.0)
    ys, xs = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    cam_dirs = np.stack([
        (xs - W / 2.0) / fx,
        -(ys - H / 2.0) / fy,   # +Y up
        -np.ones_like(xs),
    ], axis=-1)
    cam_dirs = cam_dirs / np.linalg.norm(cam_dirs, axis=-1, keepdims=True)

    rgbs_t = torch.from_numpy(rgbs).to(device)
    masks_t = torch.from_numpy(masks).to(device)
    depths_t = torch.from_numpy(depths).to(device)
    poses_t = torch.from_numpy(poses).to(device)
    cam_dirs_t = torch.from_numpy(cam_dirs).to(device)

    # Pre-compute per-view HOLE masks = binary_fill_holes(mask) & ~mask.
    # These are the pixels that lie INSIDE the silhouette's convex hull
    # but OUTSIDE the actual mask — i.e. places where the ref mesh has
    # a hole. For round 7's open-ray loss we mark rays hitting these
    # pixels as "hole rays" and reward low predicted opacity on them.
    from scipy import ndimage
    hole_masks = np.zeros((V, H, W), dtype=bool)
    boundary_masks = np.zeros((V, H, W), dtype=bool)
    for v in range(V):
        m = masks[v] > 0.5
        filled = ndimage.binary_fill_holes(m)
        hole_masks[v] = filled & ~m
        eroded = ndimage.binary_erosion(m, iterations=2)
        boundary = m & ~eroded
        outer = ndimage.binary_dilation(m, iterations=2) & ~m
        boundary_masks[v] = boundary | outer

    if normal_source == "stablenormal":
        stable_unit = stable_normals / np.clip(
            np.linalg.norm(stable_normals, axis=-1, keepdims=True), 1e-6, None,
        )
        # StableNormal's object-mode PNGs are not in our world frame.
        # Empirical agreement against the analytic mesh-rendered normals is
        # strongest after flipping the image/camera X axis and rotating by the
        # per-view camera-to-world matrix from views.json. Feeding the direct
        # RGB-decoded vectors as world-space normals makes multi-view normal
        # supervision contradictory.
        stable_unit = stable_unit * np.asarray([-1.0, 1.0, 1.0], dtype=np.float32)
        stable_unit = np.einsum("vij,vhwj->vhwi", poses[:, :3, :3], stable_unit)
        stable_unit = stable_unit / np.clip(
            np.linalg.norm(stable_unit, axis=-1, keepdims=True), 1e-6, None,
        )
        if not has_analytic_normals:
            normals = stable_unit.astype(np.float32)
        elif stablenormal_blend_strength >= 1.0 and stablenormal_edge_boost <= 0.0:
            normals = stable_unit.astype(np.float32)
        else:
            analytic_unit = analytic_normals / np.clip(
                np.linalg.norm(analytic_normals, axis=-1, keepdims=True), 1e-6, None,
            )
            denom = max(stablenormal_agreement_ceil - stablenormal_agreement_floor, 1e-6)
            cosine = np.clip((analytic_unit * stable_unit).sum(axis=-1), -1.0, 1.0)
            agreement = np.clip(
                (cosine - stablenormal_agreement_floor) / denom,
                0.0,
                1.0,
            )
            blend = np.clip(stablenormal_blend_strength, 0.0, 1.0) * agreement
            if stablenormal_edge_boost > 0.0:
                blend = np.clip(
                    blend * (1.0 + stablenormal_edge_boost * boundary_masks.astype(np.float32)),
                    0.0,
                    1.0,
                )
            blended = analytic_unit * (1.0 - blend[..., None]) + stable_unit * blend[..., None]
            normals = blended / np.clip(
                np.linalg.norm(blended, axis=-1, keepdims=True), 1e-6, None,
            )
            normals = normals.astype(np.float32)
    # Build target-normal tensor only after optional StableNormal replacement/blending.
    normals_t = torch.from_numpy(normals).to(device)

    hole_masks_t = torch.from_numpy(hole_masks).to(device)
    edge_weights_t = torch.from_numpy(boundary_masks.astype(np.float32)).to(device)
    n_hole_pixels_total = int(hole_masks.sum())
    print(f"[sampler] total hole pixels across {V} views: {n_hole_pixels_total:,} "
          f"({100.0 * n_hole_pixels_total / (V * H * W):.2f}%)")

    # Pre-compute "interesting" pixel index sets per view: foreground
    # (mask>0.5) PLUS a few-pixel-wide boundary band around the mask
    # (extracted via 1-px erosion XOR mask). Friend's round-7 addition:
    # ALSO include hole pixels in the "interesting" pool so fg-biased
    # sampling actually hits holes with reasonable frequency.
    interesting_per_view = []
    foreground_per_view = []
    for v in range(V):
        m = masks[v] > 0.5
        fg_idx = np.flatnonzero(m.ravel())
        foreground_per_view.append(fg_idx if len(fg_idx) > 0 else np.flatnonzero(hole_masks[v].ravel()))
        interesting = m | boundary_masks[v] | hole_masks[v]
        # Flatten to indices
        idx = np.flatnonzero(interesting.ravel())
        interesting_per_view.append(idx)
    # Pad to same length so we can stack (use max length, sample with
    # replacement if needed — fine since pool is large).
    max_len = max(len(ix) for ix in interesting_per_view)
    interesting_padded = np.zeros((V, max_len), dtype=np.int64)
    for v in range(V):
        ix = interesting_per_view[v]
        # Cycle to fill
        if len(ix) < max_len:
            reps = (max_len + len(ix) - 1) // len(ix)
            ix = np.tile(ix, reps)[:max_len]
        interesting_padded[v] = ix
    interesting_padded_t = torch.from_numpy(interesting_padded).to(device)
    foreground_max_len = max(max(len(ix), 1) for ix in foreground_per_view)
    foreground_padded = np.zeros((V, foreground_max_len), dtype=np.int64)
    for v in range(V):
        ix = foreground_per_view[v]
        if len(ix) == 0:
            ix = np.arange(H * W, dtype=np.int64)
        if len(ix) < foreground_max_len:
            reps = (foreground_max_len + len(ix) - 1) // len(ix)
            ix = np.tile(ix, reps)[:foreground_max_len]
        foreground_padded[v] = ix
    foreground_padded_t = torch.from_numpy(foreground_padded).to(device)

    # Dedicated hole-ray pool: views with ANY hole pixels get a
    # padded index array of those pixels only. Views without hole
    # pixels are excluded from this pool. hole_ray_oversample fraction
    # of rays are drawn from this pool uniformly across valid views.
    hole_views = [v for v in range(V) if hole_masks[v].any()]
    hole_pool_padded = None
    hole_pool_max_len = 0
    hole_views_t = None
    if hole_views and hole_ray_oversample > 0:
        hole_per_view = []
        for v in hole_views:
            idx = np.flatnonzero(hole_masks[v].ravel())
            hole_per_view.append(idx)
        hole_pool_max_len = max(len(ix) for ix in hole_per_view)
        padded = np.zeros((len(hole_views), hole_pool_max_len), dtype=np.int64)
        for i, ix in enumerate(hole_per_view):
            if len(ix) < hole_pool_max_len:
                reps = (hole_pool_max_len + len(ix) - 1) // len(ix)
                ix = np.tile(ix, reps)[:hole_pool_max_len]
            padded[i] = ix
        hole_pool_padded = torch.from_numpy(padded).to(device)
        hole_views_t = torch.tensor(hole_views, dtype=torch.long, device=device)
        print(f"[sampler] {len(hole_views)}/{V} views have hole pixels; "
              f"max pool size {hole_pool_max_len}. "
              f"hole_ray_oversample={hole_ray_oversample:.2f}")

    rng = torch.Generator(device=device).manual_seed(42)

    def sample_view_probe(n_rays: int, foreground_only: bool = True) -> RaySampleBatch:
        rays_per_view = max(1, math.ceil(n_rays / V))
        vi = torch.arange(V, device=device).repeat_interleave(rays_per_view)
        vi = vi[:n_rays]
        if foreground_only:
            col = torch.randint(0, foreground_max_len, (vi.shape[0],), generator=rng, device=device)
            flat = foreground_padded_t[vi, col]
        else:
            flat = torch.randint(0, H * W, (vi.shape[0],), generator=rng, device=device)
        yi = flat // W
        xi = flat % W
        rgb = rgbs_t[vi, yi, xi]
        mask = masks_t[vi, yi, xi]
        normal = normals_t[vi, yi, xi]
        depth = depths_t[vi, yi, xi]
        hole_ray = hole_masks_t[vi, yi, xi]
        edge_weight = edge_weights_t[vi, yi, xi]
        cam_dir = cam_dirs_t[yi, xi]
        rot = poses_t[vi, :3, :3]
        world_dir = torch.einsum("rij,rj->ri", rot, cam_dir)
        origin = poses_t[vi, :3, 3]
        return RaySampleBatch(
            origins=origin, dirs=world_dir,
            rgb_gt=rgb, mask_gt=mask, normals_gt=normal, depth_gt=depth,
            hole_ray_gt=hole_ray, edge_weight_gt=edge_weight, view_idx=vi,
        )

    def sampler(n_rays: int) -> RaySampleBatch:
        # Split ray budget: hole → fg → uniform
        if hole_pool_padded is not None and hole_ray_oversample > 0:
            n_hole = int(n_rays * hole_ray_oversample)
        else:
            n_hole = 0
        remaining = n_rays - n_hole
        n_fg = int(remaining * fg_bias)
        n_uniform = remaining - n_fg

        # Hole rays — oversample from GT hole pixels
        if n_hole > 0:
            vi_h_idx = torch.randint(0, hole_views_t.shape[0], (n_hole,),
                                      generator=rng, device=device)
            vi_h = hole_views_t[vi_h_idx]
            col_h = torch.randint(0, hole_pool_max_len, (n_hole,),
                                   generator=rng, device=device)
            flat_h = hole_pool_padded[vi_h_idx, col_h]
            yi_h = flat_h // W
            xi_h = flat_h % W
        else:
            vi_h = torch.empty(0, dtype=torch.long, device=device)
            yi_h = torch.empty(0, dtype=torch.long, device=device)
            xi_h = torch.empty(0, dtype=torch.long, device=device)

        # Foreground+boundary biased rays
        vi_f = torch.randint(0, V, (n_fg,), generator=rng, device=device)
        col = torch.randint(0, max_len, (n_fg,), generator=rng, device=device)
        flat = interesting_padded_t[vi_f, col]
        yi_f = flat // W
        xi_f = flat % W

        # Uniform rays
        vi_u = torch.randint(0, V, (n_uniform,), generator=rng, device=device)
        yi_u = torch.randint(0, H, (n_uniform,), generator=rng, device=device)
        xi_u = torch.randint(0, W, (n_uniform,), generator=rng, device=device)

        # Concatenate — hole rays first, then fg, then uniform
        vi = torch.cat([vi_h, vi_f, vi_u])
        yi = torch.cat([yi_h, yi_f, yi_u])
        xi = torch.cat([xi_h, xi_f, xi_u])

        rgb = rgbs_t[vi, yi, xi]                              # (R, 3)
        mask = masks_t[vi, yi, xi]                             # (R,)
        normal = normals_t[vi, yi, xi]                         # (R, 3)
        depth = depths_t[vi, yi, xi]                            # (R,)
        hole_ray = hole_masks_t[vi, yi, xi]                    # (R,) bool
        edge_weight = edge_weights_t[vi, yi, xi]                # (R,) float

        cam_dir = cam_dirs_t[yi, xi]
        rot = poses_t[vi, :3, :3]
        world_dir = torch.einsum("rij,rj->ri", rot, cam_dir)
        origin = poses_t[vi, :3, 3]

        return RaySampleBatch(
            origins=origin, dirs=world_dir,
            rgb_gt=rgb, mask_gt=mask, normals_gt=normal, depth_gt=depth,
            hole_ray_gt=hole_ray, edge_weight_gt=edge_weight, view_idx=vi,
        )
    sampler.sample_view_probe = sample_view_probe
    return sampler


def _ensure_stablenormal_views(
    views_dir: Path,
    *,
    device: str,
    use_turbo: bool = False,
    cache_dir: str | None = None,
) -> None:
    """Predict StableNormal normal maps for rendered RGB views.

    Uses the official StableNormal torch.hub entrypoint:
    https://github.com/Stable-X/StableNormal
    """
    from PIL import Image

    with open(views_dir / "views.json") as f:
        meta = json.load(f)
    n_views = len(meta["views"])
    out_paths = [views_dir / f"{i:02d}_normal_stablenormal.png" for i in range(n_views)]
    if all(p.exists() for p in out_paths):
        return

    model_name = "StableNormal_turbo" if use_turbo else "StableNormal"
    kwargs = {"trust_repo": True}
    if cache_dir and Path(cache_dir).exists():
        kwargs["local_cache_dir"] = cache_dir
    predictor = torch.hub.load("Stable-X/StableNormal", model_name, **kwargs)
    if hasattr(predictor, "to"):
        try:
            predictor = predictor.to(device if torch.cuda.is_available() else "cpu")
        except Exception:
            predictor = predictor.to("cpu")
    if hasattr(predictor, "eval"):
        predictor.eval()

    for i, out_path in enumerate(out_paths):
        if out_path.exists():
            continue
        input_image = Image.open(views_dir / f"{i:02d}_rgb.png").convert("RGB")
        pred = predictor(input_image, data_type="object")
        if isinstance(pred, Image.Image):
            normal_image = pred
        elif torch.is_tensor(pred):
            arr = pred.detach().float().cpu()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.permute(1, 2, 0)
            if arr.ndim == 2:
                arr = arr.unsqueeze(-1)
            if arr.min().item() < 0.0:
                arr = (arr + 1.0) * 0.5
            arr = arr.clamp(0.0, 1.0)
            if arr.shape[-1] == 1:
                arr = arr.repeat(1, 1, 3)
            normal_image = Image.fromarray((arr.numpy() * 255.0).astype(np.uint8))
        else:
            arr = np.asarray(pred)
            if arr.ndim == 2:
                arr = arr[..., None]
            if arr.min() < 0:
                arr = (arr + 1.0) * 0.5
            arr = np.clip(arr, 0.0, 1.0)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            normal_image = Image.fromarray((arr * 255.0).astype(np.uint8))
        normal_image.save(out_path)


# scripts/dualprim/render_views.py imports math; keep it here too
import math


if __name__ == "__main__":
    main()
