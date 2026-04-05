#!/usr/bin/env python3
"""Create train/eval CSV manifests for voxelization training on SkullBreak.

Generates manifests in two modes:
1. Legacy mode (default): Maps defective_skull _surf.npy -> implant _vox.npz (2-column CSV)
2. Generated mode: Maps stage-1 DDIM completions (sample.npy, shift.npy, scale.npy) -> implant _vox.npz (4-column CSV)

Output: datasets/SkullBreak/voxelization/{train,eval}.csv

Usage:
    # Legacy manifests from _surf.npy + _vox.npz
    python create_skullbreak_manifests.py --dataset-root ../pcdiff/datasets/SkullBreak --mode legacy

    # Generated completion manifests (requires stage-1 outputs)
    python create_skullbreak_manifests.py --dataset-root ../pcdiff/datasets/SkullBreak --mode generated --completions-dir /path/to/completions
"""

import argparse
import csv
import sys
from pathlib import Path

CATEGORIES = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]


def read_split_ids(csv_path: Path) -> list[str]:
    """Read skull IDs from a split CSV (e.g., train.csv with entries like complete_skull/070_surf.npy)."""
    ids = []
    with csv_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Extract ID: "complete_skull/070_surf.npy" -> "070"
            stem = line.split("/")[-1].replace("_surf.npy", "")
            ids.append(stem)
    return ids


def create_legacy_manifests(root: Path, output_dir: Path):
    """Create 2-column manifests: pointcloud_npz, gt_psr_npz.
    
    For each (skull_id, category) pair, maps:
      defective_skull/{cat}/{id}_pc.npz -> implant/{cat}/{id}_vox.npz
    
    But we don't have _pc.npz files. Instead we'll use single-column legacy format
    where the base path resolves to {base}_pc.npz and {base}_vox.npz.
    
    Actually, looking at the SkullDataset more carefully:
    - For legacy format (1-col), it expects: basepath -> basepath_pc.npz and basepath_vox.npz
    - We need to create _pc.npz files from the _surf.npy surface point clouds
    
    Alternative: Use explicit 2-column format mapping _surf.npy paths to _vox.npz paths.
    But SkullDataset expects npz with 'points' key for legacy/explicit_npz format.
    
    Best approach: Create _pc.npz files from _surf.npy, then use 1-column legacy manifests.
    """
    train_ids = read_split_ids(root / "train.csv")
    test_ids = read_split_ids(root / "test.csv")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, skull_ids in [("train", train_ids), ("eval", test_ids)]:
        rows = []
        missing = []
        for skull_id in skull_ids:
            for category in CATEGORIES:
                # Check if _vox.npz exists
                vox_path = root / "implant" / category / f"{skull_id}_vox.npz"
                # For legacy format, we need _pc.npz alongside _vox.npz
                pc_path = root / "implant" / category / f"{skull_id}_pc.npz"

                if not vox_path.exists():
                    missing.append(str(vox_path))
                    continue

                if not pc_path.exists():
                    # Create _pc.npz from the defective skull _surf.npy
                    surf_path = root / "defective_skull" / category / f"{skull_id}_surf.npy"
                    if surf_path.exists():
                        _create_pc_npz(surf_path, pc_path)
                    else:
                        missing.append(f"surf:{surf_path}")
                        continue

                # Single-column legacy: base path (without _pc.npz or _vox.npz suffix)
                base = root / "implant" / category / skull_id
                rows.append([str(base)])

        manifest_path = output_dir / f"{split_name}.csv"
        with manifest_path.open("w", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        print(f"{split_name}: {len(rows)} entries written to {manifest_path}")
        if missing:
            print(f"  {len(missing)} entries skipped (missing files)")
            for m in missing[:5]:
                print(f"    {m}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")


def _create_pc_npz(surf_npy_path: Path, output_pc_npz: Path):
    """Convert a _surf.npy point cloud to _pc.npz format expected by SkullDataset."""
    import numpy as np

    points = np.load(str(surf_npy_path))
    # The voxelization pipeline expects 'points' key in the npz
    # Also normalize to the expected range if needed
    np.savez_compressed(str(output_pc_npz), points=points.astype(np.float32))


def create_generated_manifests(root: Path, completions_dir: Path, output_dir: Path):
    """Create 4-column manifests for generated completions:
    sample_points_path, shift_path, scale_path, gt_psr_path
    """
    train_ids = read_split_ids(root / "train.csv")
    test_ids = read_split_ids(root / "test.csv")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, skull_ids in [("train", train_ids), ("eval", test_ids)]:
        rows = []
        missing = []
        for skull_id in skull_ids:
            for category in CATEGORIES:
                # Check completion outputs
                comp_dir = completions_dir / category / f"{skull_id}_surf"
                sample_path = comp_dir / "sample.npy"
                shift_path = comp_dir / "shift.npy"
                scale_path = comp_dir / "scale.npy"
                vox_path = root / "implant" / category / f"{skull_id}_vox.npz"

                if not all(p.exists() for p in [sample_path, shift_path, scale_path]):
                    missing.append(f"completion:{comp_dir}")
                    continue
                if not vox_path.exists():
                    missing.append(f"vox:{vox_path}")
                    continue

                rows.append([str(sample_path), str(shift_path), str(scale_path), str(vox_path)])

        manifest_path = output_dir / f"{split_name}.csv"
        with manifest_path.open("w", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        print(f"{split_name}: {len(rows)} entries written to {manifest_path}")
        if missing:
            print(f"  {len(missing)} entries skipped (missing files)")
            for m in missing[:5]:
                print(f"    {m}")


def main():
    parser = argparse.ArgumentParser(description="Create voxelization manifests for SkullBreak")
    parser.add_argument("--dataset-root", type=str, default="../pcdiff/datasets/SkullBreak")
    parser.add_argument("--mode", choices=["legacy", "generated"], default="legacy")
    parser.add_argument("--completions-dir", type=str, help="Directory with stage-1 completion outputs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for manifests (default: {dataset-root}/voxelization)")
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir) if args.output_dir else root / "voxelization"

    if args.mode == "legacy":
        create_legacy_manifests(root, output_dir)
    elif args.mode == "generated":
        if not args.completions_dir:
            print("ERROR: --completions-dir required for generated mode")
            sys.exit(1)
        create_generated_manifests(root, Path(args.completions_dir).resolve(), output_dir)


if __name__ == "__main__":
    main()
