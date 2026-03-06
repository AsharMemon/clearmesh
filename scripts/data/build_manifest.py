#!/usr/bin/env python3
"""Build the final training manifest for Stage 2 + Easy3E.

Collects all generated data (coarse/fine pairs, conditioning renders,
O-Voxel files, SLAT latents) into a single JSON manifest that the
training scripts can load.

Usage:
    python build_manifest.py \
        --pairs_dir /workspace/data/training_pairs \
        --renders_dir /workspace/data/renders \
        --ovoxel_dir /workspace/data/ovoxel \
        --output /workspace/data/manifest_train.json \
        --val_split 0.05

    # Verify manifest (check all files exist):
    python build_manifest.py \
        --pairs_dir /workspace/data/training_pairs \
        --output /workspace/data/manifest_train.json \
        --verify_only
"""

import argparse
import json
import os
import random
from pathlib import Path

from tqdm import tqdm


def build_manifest(
    pairs_dir: str,
    renders_dir: str | None = None,
    ovoxel_dir: str | None = None,
    output_path: str = "/workspace/data/manifest_train.json",
    val_split: float = 0.05,
    verify_only: bool = False,
):
    """Build the training manifest.

    Scans the pairs directory and links in renders + O-Voxel data.
    Splits into train/val sets.

    Args:
        pairs_dir: Directory containing coarse/fine pairs.
        renders_dir: Directory containing conditioning renders.
        ovoxel_dir: Directory containing O-Voxel files.
        output_path: Output JSON path.
        val_split: Fraction of data for validation.
        verify_only: Only verify existing manifest, don't rebuild.
    """
    if verify_only:
        verify_manifest(output_path)
        return

    pairs_path = Path(pairs_dir)
    entries = []
    skipped = 0

    # Scan pairs directory
    uid_dirs = sorted([d for d in pairs_path.iterdir() if d.is_dir()])
    print(f"Scanning {len(uid_dirs)} pair directories...")

    for uid_dir in tqdm(uid_dirs, desc="Building manifest"):
        uid = uid_dir.name

        # Find coarse mesh
        coarse = uid_dir / "coarse.glb"
        if not coarse.exists():
            coarse = uid_dir / "coarse.obj"
        if not coarse.exists():
            skipped += 1
            continue

        # Find fine mesh (symlink to original)
        fine = None
        for ext in [".glb", ".obj", ".ply"]:
            candidate = uid_dir / f"fine{ext}"
            if candidate.exists():
                fine = candidate
                break
        if fine is None:
            skipped += 1
            continue

        entry = {
            "uid": uid,
            "coarse_mesh": str(coarse),
            "fine_mesh": str(fine),
        }

        # Link conditioning renders
        if renders_dir:
            render_dir = os.path.join(renders_dir, uid)
            if os.path.isdir(render_dir):
                renders = sorted(
                    [
                        os.path.join(render_dir, f)
                        for f in os.listdir(render_dir)
                        if f.endswith(".png") and "normal" not in f
                    ]
                )
                normals = sorted(
                    [
                        os.path.join(render_dir, f)
                        for f in os.listdir(render_dir)
                        if f.endswith(".png") and "normal" in f
                    ]
                )
                if renders:
                    entry["renders"] = renders
                if normals:
                    entry["normals"] = normals

        # Link O-Voxel file
        if ovoxel_dir:
            vxz = os.path.join(ovoxel_dir, f"{uid}.vxz")
            if os.path.exists(vxz):
                entry["ovoxel"] = vxz
            else:
                npy = os.path.join(ovoxel_dir, f"{uid}.npy")
                if os.path.exists(npy):
                    entry["ovoxel"] = npy

        # Link SLAT latents (if they were saved with pairs)
        slat = uid_dir / "slat.pt"
        if slat.exists():
            entry["slat"] = str(slat)
        dual_grid = uid_dir / "dual_grid.vxz"
        if dual_grid.exists():
            entry["dual_grid"] = str(dual_grid)

        # Link conditioning image (if saved during pair generation)
        cond_img = uid_dir / "conditioning.png"
        if cond_img.exists():
            entry["conditioning_image"] = str(cond_img)

        entries.append(entry)

    print(f"\nTotal entries: {len(entries)}")
    print(f"Skipped (incomplete): {skipped}")

    if not entries:
        print("ERROR: No valid entries found!")
        return

    # Split train/val
    random.seed(42)
    random.shuffle(entries)

    val_count = max(1, int(len(entries) * val_split))
    val_entries = entries[:val_count]
    train_entries = entries[val_count:]

    manifest = {
        "total": len(entries),
        "train_count": len(train_entries),
        "val_count": len(val_entries),
        "train": train_entries,
        "val": val_entries,
    }

    # Count what data types we have
    has_renders = sum(1 for e in entries if "renders" in e)
    has_normals = sum(1 for e in entries if "normals" in e)
    has_ovoxel = sum(1 for e in entries if "ovoxel" in e)
    has_slat = sum(1 for e in entries if "slat" in e)

    print(f"\nData coverage:")
    print(f"  With renders:    {has_renders}/{len(entries)}")
    print(f"  With normals:    {has_normals}/{len(entries)}")
    print(f"  With O-Voxel:    {has_ovoxel}/{len(entries)}")
    print(f"  With SLAT:       {has_slat}/{len(entries)}")
    print(f"\nSplit:")
    print(f"  Train: {len(train_entries)}")
    print(f"  Val:   {len(val_entries)}")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved: {output_path}")

    # Also save separate train/val lists for convenience
    train_path = output_path.replace(".json", "_train.json")
    val_path = output_path.replace(".json", "_val.json")
    with open(train_path, "w") as f:
        json.dump(train_entries, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_entries, f, indent=2)
    print(f"  Train list: {train_path}")
    print(f"  Val list:   {val_path}")


def verify_manifest(manifest_path: str):
    """Verify all files in a manifest exist."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_entries = manifest.get("train", []) + manifest.get("val", [])
    print(f"Verifying {len(all_entries)} entries...")

    missing = 0
    file_keys = ["coarse_mesh", "fine_mesh", "ovoxel", "slat", "dual_grid", "conditioning_image"]

    for entry in tqdm(all_entries, desc="Verifying"):
        for key in file_keys:
            if key in entry:
                path = entry[key]
                if not os.path.exists(path):
                    print(f"  MISSING: {entry['uid']}/{key}: {path}")
                    missing += 1

        # Check render lists
        for list_key in ["renders", "normals"]:
            if list_key in entry:
                for path in entry[list_key]:
                    if not os.path.exists(path):
                        print(f"  MISSING: {entry['uid']}/{list_key}: {path}")
                        missing += 1

    if missing == 0:
        print("All files verified OK!")
    else:
        print(f"\n{missing} files missing!")


def main():
    parser = argparse.ArgumentParser(description="Build training manifest")
    parser.add_argument(
        "--pairs_dir",
        type=str,
        required=True,
        help="Directory containing coarse/fine pairs",
    )
    parser.add_argument(
        "--renders_dir",
        type=str,
        default=None,
        help="Directory containing conditioning renders",
    )
    parser.add_argument(
        "--ovoxel_dir",
        type=str,
        default=None,
        help="Directory containing O-Voxel files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/data/manifest_train.json",
        help="Output manifest JSON path",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Fraction of data for validation (default: 5%%)",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing manifest",
    )
    args = parser.parse_args()

    build_manifest(
        args.pairs_dir,
        args.renders_dir,
        args.ovoxel_dir,
        args.output,
        args.val_split,
        args.verify_only,
    )


if __name__ == "__main__":
    main()
