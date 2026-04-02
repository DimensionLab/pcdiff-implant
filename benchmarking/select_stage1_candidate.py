#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateResult:
    dataset: str
    run_id: str
    runtime_sec_mean: float | None
    dice_mean: float | None
    bdice_10mm_mean: float | None
    hd95_mm_mean: float | None
    stage1_summary_path: str
    stage2_summary_path: str
    non_regressing: bool
    regression_reasons: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose the fastest non-regressing stage-1 configuration from archived benchmark artifacts."
    )
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        help="Dataset-specific baseline mapping in the form Dataset=/path/to/benchmark_summary.json.",
    )
    parser.add_argument(
        "--runs-root",
        required=True,
        help="Root containing <dataset>/<run-id>/stage2/benchmark_summary.json and stage1/benchmark_summary.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the selection report as JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_baselines(entries: list[str]) -> dict[str, Path]:
    baselines: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --baseline value: {entry!r}")
        dataset, raw_path = entry.split("=", 1)
        baselines[dataset] = Path(raw_path)
    return baselines


def metric_mean(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if not isinstance(value, dict):
        return None
    mean_value = value.get("mean")
    return float(mean_value) if mean_value is not None else None


def summarize_candidate(dataset: str, run_dir: Path, baseline_summary: dict[str, Any]) -> CandidateResult | None:
    stage1_summary_path = run_dir / "stage1" / "benchmark_summary.json"
    stage2_summary_path = run_dir / "stage2" / "benchmark_summary.json"
    if not stage1_summary_path.exists() or not stage2_summary_path.exists():
        return None

    stage1_summary = load_json(stage1_summary_path)
    stage2_summary = load_json(stage2_summary_path)

    runtime_sec_mean = metric_mean(stage1_summary, "runtime_sec")
    dice_mean = metric_mean(stage2_summary, "dice")
    bdice_mean = metric_mean(stage2_summary, "bdice_10mm")
    hd95_mean = metric_mean(stage2_summary, "hd95_mm")

    regression_reasons: list[str] = []
    baseline_dice = metric_mean(baseline_summary, "dice")
    baseline_bdice = metric_mean(baseline_summary, "bdice_10mm")
    baseline_hd95 = metric_mean(baseline_summary, "hd95_mm")

    if baseline_dice is not None and dice_mean is not None and dice_mean < baseline_dice:
        regression_reasons.append(f"dice {dice_mean:.6f} < baseline {baseline_dice:.6f}")
    if baseline_bdice is not None and bdice_mean is not None and bdice_mean < baseline_bdice:
        regression_reasons.append(f"bdice_10mm {bdice_mean:.6f} < baseline {baseline_bdice:.6f}")
    if baseline_hd95 is not None and hd95_mean is not None and hd95_mean > baseline_hd95:
        regression_reasons.append(f"hd95_mm {hd95_mean:.6f} > baseline {baseline_hd95:.6f}")

    return CandidateResult(
        dataset=dataset,
        run_id=run_dir.name,
        runtime_sec_mean=runtime_sec_mean,
        dice_mean=dice_mean,
        bdice_10mm_mean=bdice_mean,
        hd95_mm_mean=hd95_mean,
        stage1_summary_path=str(stage1_summary_path),
        stage2_summary_path=str(stage2_summary_path),
        non_regressing=not regression_reasons,
        regression_reasons=regression_reasons,
    )


def choose_best(candidates: list[CandidateResult]) -> CandidateResult | None:
    valid = [candidate for candidate in candidates if candidate.non_regressing and candidate.runtime_sec_mean is not None]
    if not valid:
        return None
    return min(valid, key=lambda candidate: candidate.runtime_sec_mean)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    baseline_paths = parse_baselines(args.baseline)
    baseline_summaries = {dataset: load_json(path) for dataset, path in baseline_paths.items()}

    candidates: list[CandidateResult] = []
    for dataset_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        baseline_summary = baseline_summaries.get(dataset_dir.name)
        if baseline_summary is None:
            continue
        for run_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            candidate = summarize_candidate(dataset_dir.name, run_dir, baseline_summary)
            if candidate is not None:
                candidates.append(candidate)

    best = choose_best(candidates)
    payload = {
        "schema_version": 1,
        "baseline_summary_paths": {dataset: str(path) for dataset, path in baseline_paths.items()},
        "runs_root": str(runs_root),
        "candidate_count": len(candidates),
        "best_candidate": None if best is None else best.__dict__,
        "candidates": [candidate.__dict__ for candidate in candidates],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
