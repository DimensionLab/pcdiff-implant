#!/usr/bin/env python3
"""Precompute PSR indicator grids (_vox.npz) from GT implant .nrrd files.

For each implant .nrrd file, this script:
1. Loads the binary volume
2. Extracts surface points + normals via marching cubes
3. Normalizes to (0,1) space at resolution 512
4. Runs DPSR to produce the indicator grid
5. Saves as {implant_dir}/{category}/{id}_vox.npz with key 'psr'

Usage:
    python precompute_psr_grids.py --dataset-root ../pcdiff/datasets/SkullBreak
    python precompute_psr_grids.py --dataset-root ../pcdiff/datasets/SkullBreak --categories bilateral frontoorbital
"""

import argparse
import sys
from pathlib import Path

import nrrd
import numpy as np
import torch
from skimage import measure

# Add voxelization src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.dpsr import DPSR

RESOLUTION = 512
PSR_SIGMA = 0
CATEGORIES = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]
NUM_SURFACE_POINTS = 30720  # Match voxelization training config


def extract_surface_points_and_normals(volume: np.ndarray, num_points: int = NUM_SURFACE_POINTS):
    """Extract surface points and normals from a binary volume using marching cubes."""
    # Marching cubes to get mesh
    try:
        verts, faces, normals_mc, _ = measure.marching_cubes(volume, level=0.5)
    except ValueError:
        # Volume might be empty or all-filled
        return None, None

    if len(verts) == 0:
        return None, None

    # Compute per-vertex normals from the mesh if marching_cubes normals are unreliable
    # Use the marching_cubes normals directly (they're vertex normals)
    normals = normals_mc.copy()

    # Normalize normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normals = normals / norms

    # Subsample to desired number of points
    if len(verts) >= num_points:
        idx = np.random.choice(len(verts), num_points, replace=False)
    else:
        idx = np.random.choice(len(verts), num_points, replace=True)

    points = verts[idx].astype(np.float32)
    normals = normals[idx].astype(np.float32)

    return points, normals


def compute_psr_grid(points: np.ndarray, normals: np.ndarray, resolution: int = RESOLUTION):
    """Compute DPSR indicator grid from points and normals."""
    # Normalize points to (0, 1) space
    # Points are in voxel coordinates from marching cubes (0 to volume_size)
    # We need to normalize to (0, 1)
    points_norm = points / resolution  # Assuming volume is ~512^3 or we rescale

    # Actually, the volume dimensions might not be exactly 512.
    # Normalize to (0, 1) based on actual bounding box
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extent = (pmax - pmin).max()
    if extent < 1e-6:
        return None

    # Center and scale to (0, 1) with some padding
    center = (pmin + pmax) / 2
    scale = extent * 1.2  # 20% padding like the training code
    points_norm = (points - center) / scale + 0.5

    # Ensure in (0, 1) range
    points_norm = np.clip(points_norm, 0.001, 0.999)

    # Run DPSR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dpsr = DPSR(res=(resolution, resolution, resolution), sig=PSR_SIGMA).to(device)

    points_t = torch.from_numpy(points_norm).float().unsqueeze(0).to(device)
    normals_t = torch.from_numpy(normals).float().unsqueeze(0).to(device)

    with torch.no_grad():
        psr_grid = dpsr(points_t, normals_t)

    return psr_grid.squeeze(0).cpu().numpy()


def process_implant(nrrd_path: Path, output_path: Path, resolution: int = RESOLUTION):
    """Process a single implant .nrrd file to produce a _vox.npz PSR grid."""
    if output_path.exists():
        print(f"  SKIP (exists): {output_path}")
        return True

    # Load NRRD volume
    volume, header = nrrd.read(str(nrrd_path))

    # Ensure binary
    volume = (volume > 0).astype(np.float32)

    if volume.sum() < 100:
        print(f"  WARN: Nearly empty volume: {nrrd_path} ({volume.sum():.0f} voxels)")
        return False

    # Extract surface
    points, normals = extract_surface_points_and_normals(volume, NUM_SURFACE_POINTS)
    if points is None:
        print(f"  WARN: Could not extract surface from {nrrd_path}")
        return False

    # Compute PSR grid
    psr = compute_psr_grid(points, normals, resolution)
    if psr is None:
        print(f"  WARN: PSR computation failed for {nrrd_path}")
        return False

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), psr=psr.astype(np.float32))
    print(f"  OK: {output_path} shape={psr.shape}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Precompute PSR grids from implant NRRD files")
    parser.add_argument("--dataset-root", type=str, default="../pcdiff/datasets/SkullBreak",
                        help="Path to SkullBreak dataset root")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES,
                        help="Defect categories to process")
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help="PSR grid resolution")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just list files, don't process")
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    implant_dir = root / "implant"

    if not implant_dir.exists():
        print(f"ERROR: Implant directory not found: {implant_dir}")
        sys.exit(1)

    total = 0
    success = 0
    skipped = 0

    for category in args.categories:
        cat_dir = implant_dir / category
        if not cat_dir.exists():
            print(f"WARNING: Category directory not found: {cat_dir}")
            continue

        nrrd_files = sorted(cat_dir.glob("*.nrrd"))
        print(f"\n[{category}] Found {len(nrrd_files)} .nrrd files")

        for nrrd_file in nrrd_files:
            stem = nrrd_file.stem  # e.g., "000"
            output_path = cat_dir / f"{stem}_vox.npz"
            total += 1

            if args.dry_run:
                exists = "EXISTS" if output_path.exists() else "MISSING"
                print(f"  {exists}: {nrrd_file.name} -> {output_path.name}")
                continue

            print(f"Processing: {category}/{nrrd_file.name}")
            if process_implant(nrrd_file, output_path, args.resolution):
                success += 1
            else:
                print(f"  FAILED: {nrrd_file}")

    print(f"\nDone: {success}/{total} processed successfully")


if __name__ == "__main__":
    main()
