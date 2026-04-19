"""Build a diverse mesh corpus from Objaverse-LVIS for DualPrim teacher runs.

Why Objaverse-LVIS:
  - ~46k meshes (curated subset of full 800k Objaverse)
  - tagged with 1,156 LVIS categories, letting us filter for
    "primitive-decomposable" shapes (furniture, tools, containers,
    vehicles) and skip shapes that don't fit the model (rugs,
    plants, people)
  - widely-used, permissively-licensed, already in the TRELLIS.2
    training distribution — so the neural warm-start head trained on
    Objaverse will generalize to TRELLIS.2 outputs at inference

Output: a manifest JSON listing each downloaded mesh path + its LVIS
category tag. autonomous_runner consumes this via --mesh-list.

Usage:
    python scripts/dualprim/objaverse_sampler.py \\
        --num 300 \\
        --out-dir /workspace/objaverse_cache \\
        --categories furniture_misc tool container vehicle \\
        --max-faces 50000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# LVIS category → semantic group mapping
# ---------------------------------------------------------------------
# Not every LVIS category makes sense for primitive decomposition. A
# "rug" or "blanket" has no useful primitive structure. A "chair" or
# "camera" does. We group the ~1156 LVIS classes into coarse semantic
# buckets that reflect "decomposability" rather than real-world
# taxonomy. Users pick buckets via --categories.
#
# Caveat: this list is hand-curated from memory / intuition, not the
# full LVIS taxonomy. Missing a category just means "not in any
# bucket" — it gets skipped. Users can also pass --raw-lvis-names to
# bypass bucketing.

SEMANTIC_GROUPS: dict[str, list[str]] = {
    "furniture": [
        "chair", "armchair", "bench", "stool", "table", "desk",
        "cabinet", "dresser", "wardrobe", "bookcase", "shelf",
        "sofa", "couch", "loveseat", "bed", "crib", "nightstand",
        "lamp", "chandelier", "coffee_table", "side_table",
        "ottoman", "stool_(furniture)", "bar_stool",
    ],
    "container": [
        "bottle", "wine_bottle", "cup", "mug", "bowl", "plate",
        "dish", "pot", "pan", "bucket", "basket", "box", "crate",
        "jar", "can_(container)", "barrel", "kettle", "pitcher",
        "teapot", "vase", "tray", "cooler", "ice_bucket",
        "water_bottle", "flowerpot", "bin", "trash_can",
    ],
    "tool": [
        "hammer", "screwdriver", "wrench", "pliers", "saw",
        "drill", "chisel", "axe", "knife", "scissors", "shears",
        "ruler", "level", "tape_measure", "clamp", "crowbar",
    ],
    "vehicle": [
        "car_(automobile)", "truck", "van", "bus", "motorcycle",
        "bicycle", "tricycle", "scooter", "boat", "canoe", "kayak",
        "airplane", "helicopter", "train", "tractor", "forklift",
        "wagon", "cart", "wheelbarrow",
    ],
    "utensil": [
        "fork", "spoon", "ladle", "spatula", "whisk", "tongs",
        "can_opener", "bottle_opener", "corkscrew", "rolling_pin",
        "cutting_board",
    ],
    "appliance": [
        "blender", "toaster", "microwave", "oven", "refrigerator",
        "dishwasher", "washing_machine", "dryer", "coffee_maker",
        "rice_cooker", "crock_pot", "waffle_iron", "food_processor",
        "stove", "range_(kitchen)",
    ],
    "electronics": [
        "camera", "laptop", "computer", "keyboard_(electronic)",
        "mouse_(computer)", "phone", "telephone", "cell_phone",
        "speaker_(audio_equipment)", "headphones", "television",
        "monitor_(computer)", "radio", "tablet", "clock_radio",
    ],
    "sport": [
        "skateboard", "surfboard", "ski", "snowboard", "baseball_bat",
        "tennis_racket", "golf_club", "dumbbell", "barbell",
        "helmet", "bowling_ball", "bicycle_helmet",
    ],
}


def resolve_categories(group_names: list[str]) -> list[str]:
    """Expand semantic group names to their constituent LVIS class names."""
    out = []
    for g in group_names:
        if g in SEMANTIC_GROUPS:
            out.extend(SEMANTIC_GROUPS[g])
        else:
            # Treat unknown names as raw LVIS class names (pass-through)
            out.append(g)
    return list(set(out))


# ---------------------------------------------------------------------
# Mesh validation — reject the garbage before uploading to teacher runs
# ---------------------------------------------------------------------

def validate_mesh(path: str, max_faces: int, min_faces: int = 100) -> tuple[bool, str]:
    """Basic sanity checks. Returns (ok, reason).

    Gates out:
      - completely empty meshes
      - absurd face counts (the teacher optimizer won't converge)
      - degenerate extents (all points colinear, etc.)
      - NaN/inf vertices
    """
    try:
        import trimesh
        import numpy as np
        m = trimesh.load(path, force="mesh")
    except Exception as e:
        return False, f"load-failed: {e}"

    if len(m.faces) == 0 or len(m.vertices) == 0:
        return False, "empty-mesh"
    if len(m.faces) < min_faces:
        return False, f"too-few-faces ({len(m.faces)} < {min_faces})"
    if len(m.faces) > max_faces:
        return False, f"too-many-faces ({len(m.faces)} > {max_faces})"

    v = np.asarray(m.vertices, dtype=np.float64)
    if not np.isfinite(v).all():
        return False, "nan-verts"
    ext = v.max(axis=0) - v.min(axis=0)
    if (ext < 1e-6).any():
        return False, f"degenerate-extent {ext}"

    return True, "ok"


# ---------------------------------------------------------------------
# Main sampling flow
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=300,
                    help="Target count of valid meshes to collect. "
                         "Actual download count is larger (we over-"
                         "sample to account for validation failures).")
    ap.add_argument("--categories", nargs="+",
                    default=list(SEMANTIC_GROUPS.keys()),
                    help="Semantic groups to sample from (see "
                         "SEMANTIC_GROUPS in this file). Default = all.")
    ap.add_argument("--out-dir", required=True,
                    help="Where to cache downloaded GLBs + write manifest.")
    ap.add_argument("--max-faces", type=int, default=50_000,
                    help="Reject meshes with > this many faces. High face "
                         "counts blow up DualPrim's per-iter cost with no "
                         "quality benefit (the SQ implicit is the bottleneck, "
                         "not mesh resolution).")
    ap.add_argument("--min-faces", type=int, default=100,
                    help="Reject meshes below this face count (probably "
                         "broken / impostor).")
    ap.add_argument("--oversample-factor", type=float, default=1.8,
                    help="Download this × --num uids to account for "
                         "validation dropouts and misfiled categories.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--download-processes", type=int, default=8)
    args = ap.parse_args()

    try:
        import objaverse
    except ImportError:
        print("ERROR: `objaverse` package not installed. Run: pip install objaverse",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "meshes.json"

    lvis_names = resolve_categories(args.categories)
    print(f"[objaverse] resolved {len(args.categories)} groups → "
          f"{len(lvis_names)} LVIS classes")

    print(f"[objaverse] loading LVIS annotations...")
    lvis = objaverse.load_lvis_annotations()
    # LVIS annotations is {class_name: [uid1, uid2, ...]}

    # Collect UIDs matching our category filter
    candidates: list[tuple[str, str]] = []  # (uid, lvis_class)
    missing_classes = []
    for cls in lvis_names:
        uids = lvis.get(cls)
        if uids is None:
            missing_classes.append(cls)
            continue
        for uid in uids:
            candidates.append((uid, cls))

    if missing_classes:
        print(f"[objaverse] WARNING: {len(missing_classes)} unmatched LVIS "
              f"class names (typos or taxonomy mismatch): "
              f"{missing_classes[:10]}{'...' if len(missing_classes) > 10 else ''}")

    print(f"[objaverse] {len(candidates):,} candidate uids across "
          f"{len(set(c[1] for c in candidates))} matched classes")

    # Oversample by factor to cover validation dropouts
    target_download = int(args.num * args.oversample_factor)
    if target_download > len(candidates):
        print(f"[objaverse] WARNING: asked for {target_download} but only "
              f"{len(candidates)} candidates; using all")
        target_download = len(candidates)

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[:target_download]
    uids_to_download = [c[0] for c in selected]
    uid_to_class = {uid: cls for uid, cls in selected}

    print(f"[objaverse] downloading {len(uids_to_download)} objects "
          f"with {args.download_processes} processes...")
    paths = objaverse.load_objects(
        uids=uids_to_download,
        download_processes=args.download_processes,
    )
    # paths is {uid: local_path}
    print(f"[objaverse] downloaded {len(paths)} / {len(uids_to_download)}")

    # Validate and build manifest. We accept meshes in order until we
    # hit --num, so the manifest is the first args.num valid meshes
    # (skipping invalid ones). Deterministic given --seed.
    manifest = []
    validation_failures: dict[str, int] = {}
    for uid, src_path in paths.items():
        if len(manifest) >= args.num:
            break
        ok, reason = validate_mesh(src_path, args.max_faces, args.min_faces)
        if not ok:
            validation_failures[reason.split()[0]] = \
                validation_failures.get(reason.split()[0], 0) + 1
            continue
        manifest.append({
            "uid": uid,
            "lvis_class": uid_to_class[uid],
            "path": str(src_path),
        })

    print(f"[objaverse] {len(manifest)} valid meshes collected")
    if validation_failures:
        print(f"[objaverse] validation drops: {validation_failures}")

    with open(manifest_path, "w") as f:
        json.dump({
            "num": len(manifest),
            "categories": args.categories,
            "seed": args.seed,
            "meshes": manifest,
        }, f, indent=2)
    print(f"[objaverse] manifest -> {manifest_path}")

    # One-line summary of class distribution — helps spot single-class
    # collapse ("oh no, 250 of my 300 meshes are chairs")
    from collections import Counter
    class_counts = Counter(m["lvis_class"] for m in manifest)
    print(f"[objaverse] top 10 classes in manifest:")
    for cls, n in class_counts.most_common(10):
        print(f"    {cls:30s} {n:4d}")


if __name__ == "__main__":
    main()
