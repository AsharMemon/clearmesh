"""Gate 0.5 — warm-start feasibility experiment.

The Tier B business case is "warm-start + 1000-iter refine ≈ teacher
quality in 1 min." This experiment empirically measures whether that
hypothesis holds, before we commit 3+ weeks to training a warm-start
predictor.

Design:
  1. Take a converged teacher primitives.json (from Gate 0 pass)
  2. Pick a trajectory snapshot from an earlier step (e.g. step 10000)
  3. Warm-start a fresh run from that snapshot
  4. Measure how many refinement iters are needed to match teacher
     quality (hole_metric within 5%)
  5. Repeat across multiple meshes and snapshot-steps

If the answer is "≤1000 iters," Tier B is feasible at current per-
iter cost and the neural warm-start predictor is worth training.
If the answer is "5000+ iters," Tier B wall time exceeds our budget
and we need kernel-level speedups first.

This script doesn't train the predictor — it just measures
feasibility. The predictor's targets are the trajectory snapshots
we save during teacher runs; this validates that starting from a
partially-refined state accelerates convergence as expected.

Usage:
    python scripts/dualprim/gate05_warmstart.py \\
        --teacher-dir /workspace/dualprim_round3/hole \\
        --out /workspace/gate05_warmstart_hole \\
        --refine-steps 200 500 1000 2000 \\
        --ref /workspace/test_box_hole.glb
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


_CANDIDATE_ROOTS = ["/workspace/clearmesh",
                    str(Path(__file__).resolve().parents[2])]
for _r in _CANDIDATE_ROOTS:
    if os.path.isdir(_r) and _r not in sys.path:
        sys.path.insert(0, _r)


def evaluate_mesh(ref: str, pred: str, out_prefix: str) -> dict:
    """Run hole_metric.py and parse its output into a dict."""
    env = os.environ.copy()
    env["PYOPENGL_PLATFORM"] = "egl"
    cmd = [
        sys.executable, "-u", "scripts/dualprim/hole_metric.py",
        "--ref", ref, "--pred", pred, "--resolution", "192",
    ]
    log_path = f"{out_prefix}_hole_metric.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    result = {"ok": proc.returncode == 0}
    if not result["ok"]:
        return result

    # Parse the printed output. Cheap but fragile — we control the format.
    text = open(log_path).read()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Mean mask IoU (all views):"):
            result["mask_iou"] = float(line.split()[-1])
        elif "sphere-only IoU:" in line:
            result["mask_iou_sphere"] = float(line.split()[-1])
        elif "hole-ring-only IoU:" in line:
            result["mask_iou_ring"] = float(line.split()[-1])
        elif line.startswith("Mean hole recall"):
            result["hole_recall"] = float(line.split("(")[0].split()[-1])
        elif "ring-only open %:" in line:
            # "ring-only open %:             33.3%  (n=8 hole-visible ring views)"
            pct_str = line.split()[-2] if "(" in line else line.split()[-1]
            # Strip "%" if present and "(n=..."
            for tok in line.split():
                if tok.endswith("%") and tok != "%":
                    try:
                        result["ring_open_pct"] = float(tok.rstrip("%"))
                        break
                    except ValueError:
                        pass
        elif "Through-hole open % (all):" in line:
            for tok in line.split():
                if tok.endswith("%") and tok != "%":
                    try:
                        result["through_hole_open_pct"] = float(tok.rstrip("%"))
                        break
                    except ValueError:
                        pass
    return result


def run_warmstart(
    *, ref: str, snapshot: str, out_dir: Path, refine_steps: int,
    k: int, resolution: int, rays: int, seed: int,
    nsq_init: str, fg_bias: float,
) -> dict:
    """Launch run_canary with --resume-primitives + num_iterations=refine_steps.

    Returns {ok, refine_steps, wall_time_s, metrics (dict from hole_metric)}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train.log"
    env = os.environ.copy()
    env["PYOPENGL_PLATFORM"] = "egl"
    cmd = [
        sys.executable, "-u", "scripts/dualprim/run_canary.py",
        "--input", ref,
        "--out", str(out_dir),
        "--mode", "mesh_rendered_views",
        "--k", str(k),
        "--iters", str(refine_steps),
        "--rays", str(rays),
        "--resolution", str(resolution),
        "--seed", str(seed),
        "--nsq-init", nsq_init,
        "--fg-bias", str(fg_bias),
        "--union-export",
        "--resume-primitives", snapshot,
    ]
    print(f"[gate05] warm-start refine-steps={refine_steps} snapshot={Path(snapshot).name}")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0
    result = {
        "ok": proc.returncode == 0 and (out_dir / "refit.glb").exists(),
        "refine_steps": refine_steps,
        "wall_time_s": wall,
        "snapshot": str(snapshot),
    }
    if result["ok"]:
        metrics = evaluate_mesh(ref, str(out_dir / "refit.glb"),
                                 str(out_dir / "eval"))
        result["metrics"] = metrics
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", required=True,
                    help="Dir containing trajectory/ snapshots + "
                         "primitives.json (the converged teacher).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--snapshot-steps", nargs="+", type=int,
                    default=[10000, 6000, 3000, 1000],
                    help="Which trajectory snapshots to use as warm-"
                         "start origins. Higher step = closer to "
                         "converged = fewer refine steps needed.")
    ap.add_argument("--refine-steps", nargs="+", type=int,
                    default=[200, 500, 1000, 2000, 5000],
                    help="Refinement iter budgets to measure.")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--rays", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nsq-init", default="coupled",
                    choices=["coupled", "independent"])
    ap.add_argument("--fg-bias", type=float, default=0.7)
    ap.add_argument("--repo-root", default="/workspace/clearmesh")
    args = ap.parse_args()

    teacher_dir = Path(args.teacher_dir)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    # Teacher quality — the target we're trying to recover
    print("=" * 70)
    print(f"[gate05] Teacher: {teacher_dir}")
    teacher_metrics = evaluate_mesh(
        args.ref, str(teacher_dir / "refit.glb"),
        str(out_root / "teacher_eval"),
    )
    print(f"[gate05] Teacher hole_metric: {json.dumps(teacher_metrics, indent=2)}")
    print("=" * 70)

    # Find available snapshots
    traj_dir = teacher_dir / "trajectory"
    if not traj_dir.exists():
        print(f"ERROR: no trajectory dir at {traj_dir} — teacher must have "
              f"been run with --trajectory-dir. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    available_snapshots = sorted(traj_dir.glob("step_*.json"))
    print(f"[gate05] {len(available_snapshots)} snapshots available: "
          f"{[p.name for p in available_snapshots]}")

    # Run cross-product of snapshots × refine_steps
    results = []
    for snap_step in args.snapshot_steps:
        snap_name = f"step_{snap_step:06d}.json"
        snap_path = traj_dir / snap_name
        if not snap_path.exists():
            print(f"[gate05] skipping snap={snap_name}: not available")
            continue
        for refine in args.refine_steps:
            out_dir = out_root / f"snap{snap_step}_refine{refine}"
            if (out_dir / "refit.glb").exists():
                print(f"[gate05] skipping {out_dir.name}: already done")
                # Re-evaluate metrics in case hole_metric changed
                metrics = evaluate_mesh(
                    args.ref, str(out_dir / "refit.glb"),
                    str(out_dir / "eval"),
                )
                r = {
                    "ok": True, "refine_steps": refine,
                    "snapshot": str(snap_path),
                    "snapshot_step": snap_step,
                    "metrics": metrics,
                    "wall_time_s": None,  # unknown (was a cache hit)
                }
            else:
                r = run_warmstart(
                    ref=args.ref, snapshot=str(snap_path), out_dir=out_dir,
                    refine_steps=refine, k=args.k, resolution=args.resolution,
                    rays=args.rays, seed=args.seed,
                    nsq_init=args.nsq_init, fg_bias=args.fg_bias,
                )
                r["snapshot_step"] = snap_step
            results.append(r)

            # Checkpoint results after each run
            with open(out_root / "gate05_results.json", "w") as f:
                json.dump({
                    "teacher": teacher_metrics,
                    "runs": results,
                }, f, indent=2)

    # Summary report
    print("\n" + "=" * 70)
    print("GATE 0.5 SUMMARY — warm-start refinement curve")
    print("=" * 70)
    print(f"Teacher IoU = {teacher_metrics.get('mask_iou', '—'):.3f}  "
          f"through-hole = {teacher_metrics.get('through_hole_open_pct', '—'):.1f}%")
    print()
    print(f"{'snap':>6} {'refine':>7} {'wall (s)':>10} {'IoU':>6} {'t-h open %':>12}")
    for r in results:
        if r.get("ok"):
            m = r["metrics"]
            wt = r.get("wall_time_s") or 0
            print(f"{r['snapshot_step']:>6} {r['refine_steps']:>7} {wt:>10.1f} "
                  f"{m.get('mask_iou', 0):>6.3f} "
                  f"{m.get('through_hole_open_pct', 0):>12.1f}")

    # Verdict: find the smallest refine_steps that recovers teacher quality
    # (hole_metric within 5% on both IoU and through-hole).
    teacher_iou = teacher_metrics.get("mask_iou", 1.0)
    teacher_pct = teacher_metrics.get("through_hole_open_pct", 0.0)
    thresh_iou = teacher_iou * 0.95
    thresh_pct = teacher_pct * 0.95 if teacher_pct > 0 else 0
    print()
    print(f"Recovery threshold: IoU ≥ {thresh_iou:.3f}, t-h open ≥ {thresh_pct:.1f}%")
    for snap_step in sorted(set(r.get("snapshot_step") for r in results)):
        snap_runs = [r for r in results if r.get("snapshot_step") == snap_step
                     and r.get("ok")]
        snap_runs.sort(key=lambda r: r["refine_steps"])
        recovered_at = None
        for r in snap_runs:
            m = r["metrics"]
            if (m.get("mask_iou", 0) >= thresh_iou
                    and m.get("through_hole_open_pct", 0) >= thresh_pct):
                recovered_at = r["refine_steps"]
                break
        if recovered_at:
            print(f"  snap={snap_step}: recovered at refine={recovered_at}")
        else:
            print(f"  snap={snap_step}: never recovered (ran up to "
                  f"{max(r['refine_steps'] for r in snap_runs) if snap_runs else '?'})")


if __name__ == "__main__":
    main()
