import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path


DEFECTS = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]


@dataclass
class SplitConfig:
    root: Path
    seed: int
    train_ratio: float


def parse_args() -> SplitConfig:
    parser = argparse.ArgumentParser(
        description="Create voxelization train/eval splits for SkullBreak",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("pcdiff/datasets/SkullBreak"),
        help="Root directory of the SkullBreak dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used for splitting",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of voxelization samples used for training",
    )

    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("train-ratio must be within (0, 1)")

    return SplitConfig(
        root=args.root.expanduser().resolve(),
        seed=args.seed,
        train_ratio=args.train_ratio,
    )


def collect_voxelization_samples(cfg: SplitConfig) -> list[Path]:
    csv_path = cfg.root / "train.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing train.csv at {csv_path}. Run pcdiff/utils/split_skullbreak.py first."
        )

    samples: list[Path] = []
    with csv_path.open() as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row:
                continue
            base = Path(row[0]).stem
            for defect in DEFECTS:
                samples.append(cfg.root / "voxelization" / f"{base}_{defect}")

    if not samples:
        raise RuntimeError("No voxelization samples found. Run voxelization/utils/preproc_skullbreak.py first.")

    return samples


def write_csv(path: Path, rows: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        for row in rows:
            writer.writerow([row.as_posix()])


def main() -> None:
    cfg = parse_args()
    samples = collect_voxelization_samples(cfg)

    rng = random.Random(cfg.seed)
    rng.shuffle(samples)

    split_idx = int(len(samples) * cfg.train_ratio)
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    write_csv(cfg.root / "voxelization" / "train.csv", train_samples)
    write_csv(cfg.root / "voxelization" / "eval.csv", eval_samples)

    print(
        f"Created voxelization split with {len(train_samples)} training and {len(eval_samples)} evaluation samples"
    )


if __name__ == "__main__":
    main()
