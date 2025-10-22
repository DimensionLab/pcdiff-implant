import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SplitConfig:
    root: Path
    seed: int
    train_ratio: float
    include_csv: bool


def parse_args() -> SplitConfig:
    parser = argparse.ArgumentParser(
        description="Create train/test CSV splits for the SkullBreak dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/SkullBreak"),
        help="Root directory of the SkullBreak dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for splitting",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of cases assigned to the training split",
    )
    parser.add_argument(
        "--include-master-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit skullbreak.csv containing all case paths",
    )

    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("train-ratio must be within (0, 1)")

    return SplitConfig(
        root=args.root.expanduser().resolve(),
        seed=args.seed,
        train_ratio=args.train_ratio,
        include_csv=args.include_master_csv,
    )


def collect_complete_cases(root: Path) -> list[Path]:
    complete_dir = root / "complete_skull"
    if not complete_dir.is_dir():
        raise FileNotFoundError(f"Complete skull directory not found: {complete_dir}")

    cases = sorted(complete_dir.glob("*.nrrd"))
    if not cases:
        raise RuntimeError(f"No .nrrd files found in {complete_dir}")

    return cases


def write_csv(path: Path, rows: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        for row in rows:
            writer.writerow([row.as_posix()])


def main() -> None:
    cfg = parse_args()
    cases = collect_complete_cases(cfg.root)

    rng = random.Random(cfg.seed)
    rng.shuffle(cases)

    split_idx = int(len(cases) * cfg.train_ratio)
    train_cases = cases[:split_idx]
    test_cases = cases[split_idx:]

    if cfg.include_csv:
        write_csv(cfg.root / "skullbreak.csv", cases)

    write_csv(cfg.root / "train.csv", train_cases)
    write_csv(cfg.root / "test.csv", test_cases)

    print(
        f"Created SkullBreak split with {len(train_cases)} training and {len(test_cases)} testing cases"
    )


if __name__ == "__main__":
    main()
