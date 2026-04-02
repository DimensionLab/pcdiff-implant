#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AblationRun:
    run_id: str
    dataset: str
    sampling_method: str
    sampling_steps: int
    ensemble_size: int
    eval_path: str
    stage1_command: list[str]


def parse_csv(values: str) -> list[str]:
    return [item.strip() for item in values.split(",") if item.strip()]


def parse_int_csv(values: str) -> list[int]:
    return [int(item) for item in parse_csv(values)]


def build_run_id(method: str, steps: int, ensemble_size: int) -> str:
    return f"{method}-steps{steps}-ens{ensemble_size}"


def build_command(
    args: argparse.Namespace,
    dataset: str,
    stage1_dir: Path,
    method: str,
    steps: int,
    ensemble_size: int,
) -> list[str]:
    return [
        args.python,
        "pcdiff/test_completion.py",
        "--path",
        str(Path(args.dataset_csv_root) / dataset / "test.csv"),
        "--dataset",
        dataset,
        "--model",
        args.model,
        "--eval_path",
        str(stage1_dir),
        "--sampling_method",
        method,
        "--sampling_steps",
        str(steps),
        "--num_ens",
        str(ensemble_size),
        "--gpu",
        str(args.gpu),
        "--workers",
        str(args.workers),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible stage-1 ablation matrix and shell script."
    )
    parser.add_argument(
        "--datasets",
        default="SkullBreak",
        help="Comma-separated dataset names to include.",
    )
    parser.add_argument(
        "--sampling-methods",
        default="ddpm,ddim",
        help="Comma-separated sampling methods to sweep.",
    )
    parser.add_argument(
        "--ddpm-steps",
        default="1000",
        help="Comma-separated DDPM step counts.",
    )
    parser.add_argument(
        "--ddim-steps",
        default="250,100,50,25",
        help="Comma-separated DDIM step counts.",
    )
    parser.add_argument(
        "--ensemble-sizes",
        default="1,3,5",
        help="Comma-separated ensemble sizes.",
    )
    parser.add_argument(
        "--dataset-csv-root",
        default="pcdiff/datasets",
        help="Root containing <dataset>/test.csv.",
    )
    parser.add_argument(
        "--model",
        default="pcdiff/checkpoints/model_best.pth",
        help="Stage-1 checkpoint path passed to test_completion.py.",
    )
    parser.add_argument(
        "--output-root",
        default="benchmarking/runs/stage1_ablation",
        help="Root directory where per-run outputs and manifests are placed.",
    )
    parser.add_argument("--python", default="python", help="Python executable to use in generated commands.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for generated commands.")
    parser.add_argument("--workers", type=int, default=24, help="Data-loader worker count for generated commands.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = parse_csv(args.datasets)
    sampling_methods = parse_csv(args.sampling_methods)
    ddpm_steps = parse_int_csv(args.ddpm_steps)
    ddim_steps = parse_int_csv(args.ddim_steps)
    ensemble_sizes = parse_int_csv(args.ensemble_sizes)

    runs: list[AblationRun] = []
    for dataset in datasets:
        for sampling_method in sampling_methods:
            steps_list = ddpm_steps if sampling_method == "ddpm" else ddim_steps
            for steps in steps_list:
                for ensemble_size in ensemble_sizes:
                    run_id = build_run_id(sampling_method, steps, ensemble_size)
                    run_dir = output_root / dataset / run_id
                    stage1_dir = run_dir / "stage1"
                    runs.append(
                        AblationRun(
                            run_id=run_id,
                            dataset=dataset,
                            sampling_method=sampling_method,
                            sampling_steps=steps,
                            ensemble_size=ensemble_size,
                            eval_path=str(stage1_dir),
                            stage1_command=build_command(
                                args,
                                dataset,
                                stage1_dir,
                                sampling_method,
                                steps,
                                ensemble_size,
                            ),
                        )
                    )

    manifest = {
        "schema_version": 1,
        "purpose": "stage-1 sampling speed and candidate selection ablation",
        "selection_rule": {
            "target": "fastest stage-1 configuration that does not regress mean dice, 10 mm bDSC, or HD95 versus baseline after stage-2 evaluation",
            "required_stage2_artifacts": [
                "benchmark_cases.csv",
                "benchmark_summary.json",
                "benchmark_stage_report.json",
            ],
        },
        "defaults": {
            "dataset_csv_root": args.dataset_csv_root,
            "model": args.model,
            "python": args.python,
            "gpu": args.gpu,
            "workers": args.workers,
        },
        "runs": [asdict(run) for run in runs],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(Path(__file__).resolve().parents[1]))}",
        "",
        "# Generated by benchmarking/stage1_ablation_matrix.py",
        f"# Manifest: {manifest_path}",
        "",
    ]
    for run in runs:
        script_lines.append(f"mkdir -p {shlex.quote(run.eval_path)}")
        script_lines.append(" ".join(shlex.quote(part) for part in run.stage1_command))
        script_lines.append("")

    script_path = output_root / "run_stage1_ablation.sh"
    script_path.write_text("\n".join(script_lines), encoding="utf-8")
    script_path.chmod(0o755)

    print(f"Wrote {manifest_path}")
    print(f"Wrote {script_path}")
    print(f"Planned {len(runs)} runs")


if __name__ == "__main__":
    main()
