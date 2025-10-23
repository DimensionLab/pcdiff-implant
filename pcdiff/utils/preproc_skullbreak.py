import argparse
import multiprocessing as mp
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import mcubes
import nrrd
import numpy as np
import open3d as o3d
from tqdm import tqdm

DEFECTS = [
    "bilateral",
    "frontoorbital",
    "parietotemporal",
    "random_1",
    "random_2",
]


@dataclass(frozen=True)
class SkullBreakSample:
    case_id: str
    defect: str
    defective_path: Path
    implant_path: Path
    complete_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess SkullBreak volumes into surface point clouds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("pcdiff/datasets/SkullBreak"),
        help="Root directory of the SkullBreak dataset",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=400_000,
        help="Number of Poisson-disk points sampled per surface",
    )
    parser.add_argument(
        "--multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multiprocessing during preprocessing",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(mp.cpu_count() - 1, 16),
        help="Number of worker processes when multiprocessing is enabled (capped at 16 to avoid spawn overhead)",
    )
    parser.add_argument(
        "--keep-mesh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the intermediate OBJ meshes on disk",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute point clouds even if cached .npy files already exist",
    )
    return parser.parse_args()


def build_samples(root: Path) -> list[SkullBreakSample]:
    complete_dir = root / "complete_skull"
    defective_dir = root / "defective_skull"
    implant_dir = root / "implant"

    if not complete_dir.is_dir():
        raise FileNotFoundError(f"Complete skull directory not found: {complete_dir}")

    samples: list[SkullBreakSample] = []
    for complete_path in sorted(complete_dir.glob("*.nrrd")):
        case_id = complete_path.stem
        for idx, defect in enumerate(DEFECTS):
            defective_path = defective_dir / defect / f"{case_id}.nrrd"
            implant_path = implant_dir / defect / f"{case_id}.nrrd"

            if not defective_path.is_file():
                raise FileNotFoundError(f"Missing defective volume: {defective_path}")
            if not implant_path.is_file():
                raise FileNotFoundError(f"Missing implant volume: {implant_path}")

            samples.append(
                SkullBreakSample(
                    case_id=case_id,
                    defect=defect,
                    defective_path=defective_path,
                    implant_path=implant_path,
                    complete_path=complete_path if idx == 0 else None,
                )
            )

    return samples


def _surface_paths(nrrd_path: Path) -> tuple[Path, Path]:
    obj_path = nrrd_path.with_name(f"{nrrd_path.stem}_surf.obj")
    npy_path = nrrd_path.with_name(f"{nrrd_path.stem}_surf.npy")
    return obj_path, npy_path


def generate_point_cloud(
    volume_path: Path,
    *,
    target_points: int,
    keep_mesh: bool,
    overwrite: bool,
) -> str:
    """Generate point cloud from volume. Returns 'processed', 'skipped', or 'failed'."""
    obj_path, npy_path = _surface_paths(volume_path)
    npy_path_tmp = npy_path.with_suffix('.npy.tmp')

    # Check if valid output already exists
    if not overwrite and npy_path.exists():
        try:
            data = np.load(npy_path, mmap_mode='r')
            if data.shape[0] > 100:  # Basic validation: at least 100 points
                return "skipped"
        except Exception:
            pass  # File corrupted, will regenerate

    try:
        volume, _ = nrrd.read(str(volume_path))
        verts, faces = mcubes.marching_cubes(volume, 0)
        mcubes.export_obj(verts, faces, str(obj_path))

        mesh = o3d.io.read_triangle_mesh(str(obj_path))
        if mesh.is_empty():
            raise RuntimeError(f"Empty mesh generated from volume {volume_path}")

        pc = mesh.sample_points_poisson_disk(target_points)
        pc_np = np.asarray(pc.points, dtype=np.float32)
        
        # Atomic write
        np.save(npy_path_tmp, pc_np)
        npy_path_tmp.rename(npy_path)

        if not keep_mesh:
            try:
                obj_path.unlink()
            except FileNotFoundError:
                pass
        
        # Clean up any leftover temp files
        if npy_path_tmp.exists():
            npy_path_tmp.unlink()
            
        return "processed"
    except Exception as e:
        # Clean up temp files on failure
        if npy_path_tmp.exists():
            npy_path_tmp.unlink()
        return "failed"


def process_sample(sample: SkullBreakSample, *, target_points: int, keep_mesh: bool, overwrite: bool) -> dict:
    """Process a sample and return statistics dict."""
    results = {"processed": 0, "skipped": 0, "failed": 0}
    
    if sample.complete_path is not None:
        status = generate_point_cloud(
            sample.complete_path,
            target_points=target_points,
            keep_mesh=keep_mesh,
            overwrite=overwrite,
        )
        results[status] += 1

    status = generate_point_cloud(
        sample.defective_path,
        target_points=target_points,
        keep_mesh=keep_mesh,
        overwrite=overwrite,
    )
    results[status] += 1

    status = generate_point_cloud(
        sample.implant_path,
        target_points=target_points,
        keep_mesh=keep_mesh,
        overwrite=overwrite,
    )
    results[status] += 1
    
    return results


def run(samples: list[SkullBreakSample], args: argparse.Namespace) -> None:
    worker = partial(
        process_sample,
        target_points=args.target_points,
        keep_mesh=args.keep_mesh,
        overwrite=args.overwrite,
    )

    total_stats = {"processed": 0, "skipped": 0, "failed": 0}
    
    if args.multiprocessing and args.threads > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.threads) as pool:
            for result in tqdm(pool.imap_unordered(worker, samples), total=len(samples)):
                for key in total_stats:
                    total_stats[key] += result[key]
    else:
        for sample in tqdm(samples):
            result = worker(sample)
            for key in total_stats:
                total_stats[key] += result[key]
    
    print(f"\nProcessing summary:")
    print(f"  Processed: {total_stats['processed']}")
    print(f"  Skipped (already done): {total_stats['skipped']}")
    print(f"  Failed: {total_stats['failed']}")


def main() -> None:
    args = parse_args()
    args.root = args.root.expanduser().resolve()

    print(f"Preprocessing SkullBreak dataset located at: {args.root}")

    t_start = time.time()
    samples = build_samples(args.root)
    if not samples:
        print("No volumes found to preprocess.")
        return

    run(samples, args)
    t_end = time.time()
    print(f"Done. Processed {len(samples)} entries in {t_end - t_start:.2f}s")


if __name__ == "__main__":
    main()
