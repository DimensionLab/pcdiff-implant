import argparse
import csv
import multiprocessing as mp
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import nrrd
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class VoxelizationConfig:
    root: Path
    csv_path: Path
    num_points: int
    num_nn: int
    multiprocessing: bool
    threads: int
    overwrite: bool


def parse_args() -> VoxelizationConfig:
    parser = argparse.ArgumentParser(
        description="Prepare voxelization point clouds for the SkullBreak dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/SkullBreak"),
        help="Root directory of the SkullBreak dataset",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("datasets/SkullBreak/skullbreak.csv"),
        help="CSV listing complete-skull volumes (see split_skullbreak.py)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=30_720,
        help="Number of total points per sample (defective + implant)",
    )
    parser.add_argument(
        "--num-nn",
        type=int,
        default=3_072,
        help="Number of implant points",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(mp.cpu_count() - 1, 1),
        help="Number of workers when multiprocessing is enabled",
    )
    parser.add_argument(
        "--multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multiprocessing",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute samples even if cached .npz exists",
    )

    args = parser.parse_args()
    return VoxelizationConfig(
        root=args.root.expanduser().resolve(),
        csv_path=args.csv.expanduser().resolve(),
        num_points=args.num_points,
        num_nn=args.num_nn,
        multiprocessing=args.multiprocessing,
        threads=args.threads,
        overwrite=args.overwrite,
    )


def array2voxel(voxel_array):
    """
    convert a to a fixed size array to voxel_grid_index array
    (voxel_size*voxel_size*voxel_size)->(N*3)

    :input voxel_array: array with shape(voxel_size*voxel_size*voxel_size),the grid_index in
    :return grid_index_array: get from o3d.voxel_grid.get_voxels()
    """
    x, y, z = np.where(voxel_array == 1)
    index_voxel = np.vstack((x, y, z))
    grid_index_array = index_voxel.T
    return grid_index_array


def load_csv(csv_path: Path) -> list[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open() as fp:
        reader = csv.reader(fp)
        return [row[0] for row in reader if row]


def voxelization_output_paths(root: Path, case_id: str, defect: str) -> tuple[Path, Path]:
    target_dir = root / "voxelization"
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{case_id[-3:]}_{defect}"
    return target_dir / f"{suffix}_vox.npz", target_dir / f"{suffix}_pc.npz"


def sample_points(path: Path, k: int) -> np.ndarray:
    points = np.load(path)
    if points.shape[0] < k:
        raise ValueError(f"Point cloud {path} has {points.shape[0]} points < required {k}")
    idx = np.random.default_rng().choice(points.shape[0], k, replace=False)
    return points[idx]


def process_entry(
    root: Path,
    sample_path: Path,
    cfg: VoxelizationConfig,
) -> None:
    defects = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]
    base = sample_path.stem  # complete skull filename without extension

    for defect in defects:
        defective_pc = root / "defective_skull" / defect / f"{base}_surf.npy"
        implant_pc = root / "implant" / defect / f"{base}_surf.npy"
        gt_vox_path = root / "defective_skull" / defect / f"{base}.nrrd"

        vox_out, points_out = voxelization_output_paths(root, base, defect)
        if (not cfg.overwrite) and vox_out.exists() and points_out.exists():
            continue

        defective_points = sample_points(defective_pc, cfg.num_points - cfg.num_nn)
        implant_points = sample_points(implant_pc, cfg.num_nn)
        combined = np.concatenate((defective_points, implant_points), axis=0) / 512.0

        gt_vox, _ = nrrd.read(gt_vox_path)
        vox = np.full((512, 512, 512), 0.5, dtype=np.float32)
        vox[gt_vox > 0] = -0.5

        np.savez_compressed(points_out, points=combined)
        np.savez_compressed(vox_out, psr=vox)


def iter_samples(csv_rows: Iterable[str]) -> list[Path]:
    return [Path(row).expanduser().resolve() for row in csv_rows]


def run(cfg: VoxelizationConfig) -> None:
    csv_rows = load_csv(cfg.csv_path)
    entries = iter_samples(csv_rows)

    process = partial(process_entry, cfg.root, cfg=cfg)
    total_tasks = len(entries) * 5  # 5 defects per entry

    if cfg.multiprocessing and cfg.threads > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=cfg.threads) as pool:
            for _ in tqdm(pool.imap_unordered(process, entries), total=len(entries)):
                pass
    else:
        for entry in tqdm(entries):
            process(entry)

    print(f"Done processing {total_tasks} voxelization samples")


def main() -> None:
    cfg = parse_args()
    t_start = time.time()
    run(cfg)
    t_end = time.time()
    print(f"Total processing time: {t_end - t_start:.2f}s")


if __name__ == "__main__":
    main()
