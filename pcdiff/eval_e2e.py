#!/usr/bin/env python3
"""
Full E2E evaluation harness for PCDiff comparing DDIM-50 vs DDPM-1000.

This script:
  1. Runs full test-set inference with DDIM-50 (num_ens=5) via distributed inference
  2. Runs full test-set inference with DDPM-1000 (num_ens=5) via distributed inference
  3. Computes voxelization metrics (DSC, bDSC, HD95) for both
  4. Creates comparison artifacts (mean±std, per-sample diffs)
  5. Records final metrics against acceptance criteria

Usage:
    python pcdiff/eval_e2e.py \
        --pcdiff-checkpoint path/to/model.pth \
        --vox-checkpoint voxelization/checkpoints/model_best.pt \
        --output-dir pcdiff/eval/e2e_comparison \
        --gpus 0,1

The script uses multiprocessing for parallel metric computation across GPUs.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "pcdiff"))
sys.path.insert(0, str(ROOT_DIR / "voxelization"))

# Acceptance criteria from PRD
ACCEPTANCE_CRITERIA = {
    "minimum": {"dsc": 0.85, "bdsc": 0.87, "hd95": 2.60},  # Must meet
    "target": {"dsc": 0.87, "bdsc": 0.89, "hd95": 2.45},  # Paper baseline
}

DEFECT_TYPES = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]


@dataclass
class SampleMetrics:
    """Metrics for a single sample."""

    case_id: str
    defect: str
    dsc: float
    bdsc: float
    hd95: float
    inference_time: float = 0.0


@dataclass
class MethodResults:
    """Results for one inference method (DDIM or DDPM)."""

    method: str
    sampling_steps: int
    num_ens: int
    samples: List[SampleMetrics] = field(default_factory=list)
    total_inference_time: float = 0.0
    total_voxelization_time: float = 0.0

    @property
    def mean_dsc(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.mean([s.dsc for s in self.samples]))

    @property
    def std_dsc(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.std([s.dsc for s in self.samples]))

    @property
    def mean_bdsc(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.mean([s.bdsc for s in self.samples]))

    @property
    def std_bdsc(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.std([s.bdsc for s in self.samples]))

    @property
    def mean_hd95(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.mean([s.hd95 for s in self.samples]))

    @property
    def std_hd95(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.std([s.hd95 for s in self.samples]))


@dataclass
class SampleInfo:
    """Information about a test sample."""

    case_id: str
    defect: str
    defective_npy: Path
    defective_nrrd: Path
    implant_nrrd: Path
    syn_dir_name: str  # e.g., "bilateral086_surf"


def read_test_dataset(csv_path: Path) -> List[SampleInfo]:
    """Read test dataset and enumerate all defect types."""
    csv_path = csv_path.expanduser().resolve()
    samples: List[SampleInfo] = []

    # CSV paths may be relative to repo root, not to CSV location
    repo_root = ROOT_DIR

    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            complete_path = Path(row[0])
            if not complete_path.is_absolute():
                # Try from repo root first (CSV contains paths like "datasets/SkullBreak/...")
                complete_path = (repo_root / complete_path).resolve()

            if not complete_path.exists():
                # Fallback: try from CSV directory
                complete_path = (csv_path.parent / Path(row[0])).resolve()

            # Extract case ID: stem is "086_surf", we need "086"
            stem = complete_path.stem  # e.g., "086_surf"
            if stem.endswith("_surf"):
                case_id = stem[:-5]  # Remove "_surf" suffix
            else:
                case_id = stem

            # Root is the SkullBreak directory (parent of complete_skull)
            root = complete_path.parent.parent

            for defect in DEFECT_TYPES:
                defective_npy = root / "defective_skull" / defect / f"{case_id}_surf.npy"
                defective_nrrd = root / "defective_skull" / defect / f"{case_id}.nrrd"
                implant_nrrd = root / "implant" / defect / f"{case_id}.nrrd"
                # The syn_dir_name should match what test_completion_distributed produces
                syn_dir_name = f"{defect}{case_id}_surf"

                if not (defective_npy.exists() and defective_nrrd.exists() and implant_nrrd.exists()):
                    continue

                samples.append(
                    SampleInfo(
                        case_id=case_id,
                        defect=defect,
                        defective_npy=defective_npy,
                        defective_nrrd=defective_nrrd,
                        implant_nrrd=implant_nrrd,
                        syn_dir_name=syn_dir_name,
                    )
                )

    if not samples:
        raise RuntimeError(f"No valid samples found in {csv_path}")
    return samples


def run_distributed_inference(
    checkpoint: Path,
    output_dir: Path,
    csv_path: Path,
    sampling_method: str,
    sampling_steps: int,
    num_ens: int,
    num_gpus: int,
) -> Tuple[float, bool]:
    """Run distributed PCDiff inference using torchrun.

    Returns (elapsed_time, success).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29501",
        str(ROOT_DIR / "pcdiff" / "test_completion_distributed.py"),
        "--path",
        str(csv_path),
        "--dataset",
        "SkullBreak",
        "--model",
        str(checkpoint),
        "--eval_path",
        str(output_dir),
        "--sampling_method",
        sampling_method,
        "--sampling_steps",
        str(sampling_steps),
        "--num_ens",
        str(num_ens),
        "--verify",
    ]

    print(f"\n{'=' * 60}")
    print(f"Running {sampling_method.upper()}-{sampling_steps} inference...")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=False,  # Show output in real-time
            text=True,
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0

        if not success:
            print(f"Inference failed with return code {result.returncode}")

        return elapsed, success

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Inference failed with exception: {e}")
        return elapsed, False


def compute_metrics_for_sample(
    sample_info: SampleInfo,
    syn_dir: Path,
    num_ens: int,
    vox_runner: Any,
    num_nn: int = 3072,
) -> Optional[SampleMetrics]:
    """Compute metrics for a single sample.

    Returns None if sample directory doesn't exist or computation fails.
    """
    import diplib as dip
    import nrrd
    from scipy import ndimage

    from voxelization.eval_metrics import bdc, dc, hd95
    from voxelization.src.utils import filter_voxels_within_radius

    sample_dir = syn_dir / sample_info.syn_dir_name

    if not sample_dir.exists():
        print(f"  Warning: Sample dir not found: {sample_dir}")
        return None

    # Load inference outputs
    try:
        sample_npy = sample_dir / "sample.npy"
        if not sample_npy.exists():
            print(f"  Warning: sample.npy not found in {sample_dir}")
            return None

        ensemble_points = np.load(sample_npy)  # (num_ens, num_nn, 3)
        defective_points = np.load(sample_info.defective_npy).astype(np.float32)

    except Exception as e:
        print(f"  Warning: Failed to load sample {sample_info.syn_dir_name}: {e}")
        return None

    # Run voxelization for each ensemble member
    completes = np.zeros((512, 512, 512), dtype=np.float32)
    reference_inputs = None

    for idx in range(num_ens):
        implant_pc = ensemble_points[idx]
        combined_points = np.concatenate([defective_points, implant_pc], axis=0)
        combined_norm = combined_points / 512.0

        psr_grid_np, inputs_tensor = vox_runner.generate_psr(combined_norm)
        out = np.zeros((512, 512, 512), dtype=np.float32)
        out[psr_grid_np <= 0] = 1
        out = ndimage.binary_dilation(out)
        completes += out

        if reference_inputs is None:
            reference_inputs = inputs_tensor

    # Majority vote
    threshold = int(math.ceil(num_ens / 2.0))
    mean_complete = np.zeros_like(completes, dtype=np.float32)
    mean_complete[completes >= threshold] = 1

    # Load ground truth
    defective_vol, header = nrrd.read(str(sample_info.defective_nrrd))
    mean_implant = mean_complete - defective_vol
    mean_implant = np.clip(mean_implant, 0.0, 1.0)
    raw_implant = mean_implant.copy()

    # Post-processing
    reference_implant_points = reference_inputs[:, -num_nn:, :].detach().cpu().squeeze(0)
    mean_implant = filter_voxels_within_radius(reference_implant_points, mean_implant)
    if not np.any(mean_implant):
        mean_implant = raw_implant

    mean_implant = mean_implant.astype(bool)
    mean_implant = dip.Opening(mean_implant, dip.SE((3, 3, 3)))
    mean_implant = dip.Label(mean_implant, mode="largest")
    mean_implant = dip.MedianFilter(mean_implant, dip.Kernel(shape="rectangular", param=(3, 3, 3)))
    mean_implant.Convert("BIN")
    mean_implant = dip.Closing(mean_implant, dip.SE((3, 3, 3)))
    mean_implant = dip.FillHoles(mean_implant)
    mean_implant = dip.Label(mean_implant, mode="largest")
    mean_implant = np.asarray(mean_implant, dtype=np.float32)

    if not np.any(mean_implant):
        mean_implant = raw_implant.astype(np.float32)
    mean_implant = mean_implant.astype(bool)

    # Load GT and compute metrics
    gt_implant, _ = nrrd.read(str(sample_info.implant_nrrd))
    spacing = np.asarray(
        [
            header["space directions"][0, 0],
            header["space directions"][1, 1],
            header["space directions"][2, 2],
        ]
    )

    dice = float(dc(mean_implant, gt_implant))
    bdice = float(bdc(mean_implant, gt_implant, defective_vol, voxelspacing=spacing, distance=10))
    haus = float(hd95(mean_implant, gt_implant, voxelspacing=spacing))

    return SampleMetrics(
        case_id=sample_info.case_id,
        defect=sample_info.defect,
        dsc=dice,
        bdsc=bdice,
        hd95=haus,
    )


def worker_compute_metrics(
    rank: int,
    gpu_id: int,
    samples: Sequence[SampleInfo],
    syn_dir: Path,
    num_ens: int,
    vox_config_path: Path,
    vox_checkpoint: Path,
    result_store,
) -> None:
    """Worker function for parallel metric computation."""
    import torch

    from voxelization.src import config as vox_config
    from voxelization.src.model import Encode2Points
    from voxelization.src.utils import load_config, load_model_manual

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Load voxelization model
    default_config = ROOT_DIR / "voxelization" / "configs" / "default.yaml"
    cfg = load_config(str(vox_config_path), str(default_config))
    cfg["test"]["model_file"] = str(vox_checkpoint.expanduser().resolve())

    model = Encode2Points(cfg).to(device)
    state_dict = torch.load(cfg["test"]["model_file"], map_location="cpu")
    load_model_manual(state_dict["state_dict"], model)
    model.eval()

    generator = vox_config.get_generator(model, cfg, device=device)

    # Create wrapper class
    class VoxRunner:
        def __init__(self, gen, dev):
            self.generator = gen
            self.device = dev

        @torch.no_grad()
        def generate_psr(self, combined_points_norm):
            import torch

            inputs = torch.from_numpy(combined_points_norm).float().unsqueeze(0).to(self.device)
            vertices, faces, points, normals, psr_grid = self.generator.generate_mesh(inputs)
            psr_grid_np = psr_grid.detach().cpu().numpy()[0]
            return psr_grid_np, points.detach().cpu()

    vox_runner = VoxRunner(generator, device)

    # Process samples
    results = []
    for i, sample in enumerate(samples):
        print(f"  [Rank {rank}] Processing {i + 1}/{len(samples)}: {sample.syn_dir_name}")
        metrics = compute_metrics_for_sample(sample, syn_dir, num_ens, vox_runner)
        if metrics:
            results.append(metrics)

    result_store[rank] = results


def compute_all_metrics(
    samples: List[SampleInfo],
    syn_dir: Path,
    num_ens: int,
    vox_config: Path,
    vox_checkpoint: Path,
    gpu_ids: List[int],
) -> List[SampleMetrics]:
    """Compute metrics for all samples using parallel workers."""
    print(f"\nComputing metrics for {len(samples)} samples using {len(gpu_ids)} GPUs...")

    ctx = get_context("spawn")
    manager = ctx.Manager()
    result_store = manager.dict()

    world_size = min(len(gpu_ids), len(samples))

    processes = []
    for rank in range(world_size):
        gpu_id = gpu_ids[rank % len(gpu_ids)]
        chunk = samples[rank::world_size]
        p = ctx.Process(
            target=worker_compute_metrics,
            args=(rank, gpu_id, chunk, syn_dir, num_ens, vox_config, vox_checkpoint, result_store),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Aggregate results
    all_metrics = []
    for partial in result_store.values():
        all_metrics.extend(partial)
    all_metrics.sort(key=lambda x: (x.case_id, x.defect))

    return all_metrics


def create_comparison_report(
    ddim_results: MethodResults,
    ddpm_results: MethodResults,
    output_dir: Path,
    checkpoint: Path,
    csv_path: Path,
) -> Dict[str, Any]:
    """Create comparison artifacts and report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create per-sample comparison
    ddim_by_key = {(s.case_id, s.defect): s for s in ddim_results.samples}
    ddpm_by_key = {(s.case_id, s.defect): s for s in ddpm_results.samples}

    all_keys = set(ddim_by_key.keys()) | set(ddpm_by_key.keys())

    per_sample_diffs = []
    for key in sorted(all_keys):
        case_id, defect = key
        ddim_s = ddim_by_key.get(key)
        ddpm_s = ddpm_by_key.get(key)

        if ddim_s and ddpm_s:
            per_sample_diffs.append(
                {
                    "case_id": case_id,
                    "defect": defect,
                    "ddim_dsc": ddim_s.dsc,
                    "ddpm_dsc": ddpm_s.dsc,
                    "delta_dsc": ddpm_s.dsc - ddim_s.dsc,
                    "ddim_bdsc": ddim_s.bdsc,
                    "ddpm_bdsc": ddpm_s.bdsc,
                    "delta_bdsc": ddpm_s.bdsc - ddim_s.bdsc,
                    "ddim_hd95": ddim_s.hd95,
                    "ddpm_hd95": ddpm_s.hd95,
                    "delta_hd95": ddpm_s.hd95 - ddim_s.hd95,
                }
            )

    # Aggregate by defect type
    defect_stats = {}
    for defect in DEFECT_TYPES:
        ddim_defect = [s for s in ddim_results.samples if s.defect == defect]
        ddpm_defect = [s for s in ddpm_results.samples if s.defect == defect]

        if ddim_defect and ddpm_defect:
            defect_stats[defect] = {
                "ddim": {
                    "dsc": float(np.mean([s.dsc for s in ddim_defect])),
                    "bdsc": float(np.mean([s.bdsc for s in ddim_defect])),
                    "hd95": float(np.mean([s.hd95 for s in ddim_defect])),
                    "n": len(ddim_defect),
                },
                "ddpm": {
                    "dsc": float(np.mean([s.dsc for s in ddpm_defect])),
                    "bdsc": float(np.mean([s.bdsc for s in ddpm_defect])),
                    "hd95": float(np.mean([s.hd95 for s in ddpm_defect])),
                    "n": len(ddpm_defect),
                },
            }

    # Check acceptance criteria
    meets_minimum = {
        "ddim": {
            "dsc": ddim_results.mean_dsc >= ACCEPTANCE_CRITERIA["minimum"]["dsc"],
            "bdsc": ddim_results.mean_bdsc >= ACCEPTANCE_CRITERIA["minimum"]["bdsc"],
            "hd95": ddim_results.mean_hd95 <= ACCEPTANCE_CRITERIA["minimum"]["hd95"],
        },
        "ddpm": {
            "dsc": ddpm_results.mean_dsc >= ACCEPTANCE_CRITERIA["minimum"]["dsc"],
            "bdsc": ddpm_results.mean_bdsc >= ACCEPTANCE_CRITERIA["minimum"]["bdsc"],
            "hd95": ddpm_results.mean_hd95 <= ACCEPTANCE_CRITERIA["minimum"]["hd95"],
        },
    }

    meets_target = {
        "ddim": {
            "dsc": ddim_results.mean_dsc >= ACCEPTANCE_CRITERIA["target"]["dsc"],
            "bdsc": ddim_results.mean_bdsc >= ACCEPTANCE_CRITERIA["target"]["bdsc"],
            "hd95": ddim_results.mean_hd95 <= ACCEPTANCE_CRITERIA["target"]["hd95"],
        },
        "ddpm": {
            "dsc": ddpm_results.mean_dsc >= ACCEPTANCE_CRITERIA["target"]["dsc"],
            "bdsc": ddpm_results.mean_bdsc >= ACCEPTANCE_CRITERIA["target"]["bdsc"],
            "hd95": ddpm_results.mean_hd95 <= ACCEPTANCE_CRITERIA["target"]["hd95"],
        },
    }

    # Create summary dict
    summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint),
        "dataset_csv": str(csv_path),
        "acceptance_criteria": ACCEPTANCE_CRITERIA,
        "ddim_50": {
            "sampling_method": "ddim",
            "sampling_steps": ddim_results.sampling_steps,
            "num_ens": ddim_results.num_ens,
            "n_samples": len(ddim_results.samples),
            "metrics": {
                "dsc": {"mean": ddim_results.mean_dsc, "std": ddim_results.std_dsc},
                "bdsc": {"mean": ddim_results.mean_bdsc, "std": ddim_results.std_bdsc},
                "hd95": {"mean": ddim_results.mean_hd95, "std": ddim_results.std_hd95},
            },
            "inference_time_seconds": ddim_results.total_inference_time,
            "meets_minimum": meets_minimum["ddim"],
            "meets_target": meets_target["ddim"],
        },
        "ddpm_1000": {
            "sampling_method": "ddpm",
            "sampling_steps": ddpm_results.sampling_steps,
            "num_ens": ddpm_results.num_ens,
            "n_samples": len(ddpm_results.samples),
            "metrics": {
                "dsc": {"mean": ddpm_results.mean_dsc, "std": ddpm_results.std_dsc},
                "bdsc": {"mean": ddpm_results.mean_bdsc, "std": ddpm_results.std_bdsc},
                "hd95": {"mean": ddpm_results.mean_hd95, "std": ddpm_results.std_hd95},
            },
            "inference_time_seconds": ddpm_results.total_inference_time,
            "meets_minimum": meets_minimum["ddpm"],
            "meets_target": meets_target["ddpm"],
        },
        "defect_type_breakdown": defect_stats,
    }

    # Write JSON artifacts
    with (output_dir / "comparison_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (output_dir / "per_sample_comparison.json").open("w") as f:
        json.dump(per_sample_diffs, f, indent=2)

    # Write CSV for easy analysis
    with (output_dir / "per_sample_comparison.csv").open("w", newline="") as f:
        if per_sample_diffs:
            writer = csv.DictWriter(f, fieldnames=per_sample_diffs[0].keys())
            writer.writeheader()
            writer.writerows(per_sample_diffs)

    # Write Markdown report
    write_markdown_report(output_dir / "comparison_report.md", summary, per_sample_diffs)

    return summary


def write_markdown_report(
    path: Path,
    summary: Dict[str, Any],
    per_sample: List[Dict],
) -> None:
    """Write a human-readable markdown report."""
    with path.open("w") as f:
        f.write("# PCDiff E2E Evaluation: DDIM-50 vs DDPM-1000\n\n")
        f.write(f"**Generated:** {summary['timestamp']}\n\n")
        f.write(f"**Checkpoint:** `{summary['checkpoint']}`\n\n")
        f.write(f"**Dataset:** `{summary['dataset_csv']}`\n\n")

        # Summary table
        f.write("## Summary Metrics\n\n")
        f.write("| Metric | DDIM-50 | DDPM-1000 | Δ (DDPM - DDIM) | Minimum | Target |\n")
        f.write("|--------|---------|-----------|-----------------|---------|--------|\n")

        ddim = summary["ddim_50"]["metrics"]
        ddpm = summary["ddpm_1000"]["metrics"]
        criteria = summary["acceptance_criteria"]

        for metric in ["dsc", "bdsc"]:
            ddim_val = ddim[metric]["mean"]
            ddpm_val = ddpm[metric]["mean"]
            delta = ddpm_val - ddim_val
            min_val = criteria["minimum"][metric]
            tgt_val = criteria["target"][metric]
            f.write(
                f"| {metric.upper()} | {ddim_val:.4f}±{ddim[metric]['std']:.4f} | "
                f"{ddpm_val:.4f}±{ddpm[metric]['std']:.4f} | {delta:+.4f} | "
                f"≥{min_val:.2f} | ≥{tgt_val:.2f} |\n"
            )

        # HD95 (lower is better)
        ddim_hd = ddim["hd95"]["mean"]
        ddpm_hd = ddpm["hd95"]["mean"]
        delta_hd = ddpm_hd - ddim_hd
        f.write(
            f"| HD95 | {ddim_hd:.4f}±{ddim['hd95']['std']:.4f} | "
            f"{ddpm_hd:.4f}±{ddpm['hd95']['std']:.4f} | {delta_hd:+.4f} | "
            f"≤{criteria['minimum']['hd95']:.2f} | ≤{criteria['target']['hd95']:.2f} |\n"
        )

        # Acceptance status
        f.write("\n## Acceptance Criteria Status\n\n")
        f.write("| Method | DSC ≥ 0.85 | bDSC ≥ 0.87 | HD95 ≤ 2.60 | All Minimum |\n")
        f.write("|--------|------------|-------------|-------------|-------------|\n")

        for method, key in [("DDIM-50", "ddim_50"), ("DDPM-1000", "ddpm_1000")]:
            mins = summary[key]["meets_minimum"]
            all_min = all(mins.values())
            f.write(
                f"| {method} | {'✓' if mins['dsc'] else '✗'} | "
                f"{'✓' if mins['bdsc'] else '✗'} | "
                f"{'✓' if mins['hd95'] else '✗'} | "
                f"{'✓' if all_min else '✗'} |\n"
            )

        f.write("\n| Method | DSC ≥ 0.87 | bDSC ≥ 0.89 | HD95 ≤ 2.45 | All Target |\n")
        f.write("|--------|------------|-------------|-------------|------------|\n")

        for method, key in [("DDIM-50", "ddim_50"), ("DDPM-1000", "ddpm_1000")]:
            tgts = summary[key]["meets_target"]
            all_tgt = all(tgts.values())
            f.write(
                f"| {method} | {'✓' if tgts['dsc'] else '✗'} | "
                f"{'✓' if tgts['bdsc'] else '✗'} | "
                f"{'✓' if tgts['hd95'] else '✗'} | "
                f"{'✓' if all_tgt else '✗'} |\n"
            )

        # Inference time comparison
        f.write("\n## Inference Time\n\n")
        ddim_time = summary["ddim_50"]["inference_time_seconds"]
        ddpm_time = summary["ddpm_1000"]["inference_time_seconds"]
        speedup = ddpm_time / ddim_time if ddim_time > 0 else 0
        f.write(f"- **DDIM-50:** {ddim_time:.1f}s ({ddim_time / 60:.1f} min)\n")
        f.write(f"- **DDPM-1000:** {ddpm_time:.1f}s ({ddpm_time / 60:.1f} min)\n")
        f.write(f"- **Speedup:** {speedup:.1f}x faster with DDIM\n")

        # Defect type breakdown
        f.write("\n## Breakdown by Defect Type\n\n")
        defect_stats = summary.get("defect_type_breakdown", {})
        if defect_stats:
            f.write("### DSC by Defect Type\n\n")
            f.write("| Defect | DDIM-50 | DDPM-1000 | Δ |\n")
            f.write("|--------|---------|-----------|----|\n")
            for defect in DEFECT_TYPES:
                if defect in defect_stats:
                    d = defect_stats[defect]
                    delta = d["ddpm"]["dsc"] - d["ddim"]["dsc"]
                    f.write(f"| {defect} | {d['ddim']['dsc']:.4f} | {d['ddpm']['dsc']:.4f} | {delta:+.4f} |\n")

        # Sample of per-case diffs (top 10 worst DDPM vs DDIM)
        f.write("\n## Per-Sample Comparison (Top 10 largest |Δ DSC|)\n\n")
        f.write("| Case | Defect | DDIM DSC | DDPM DSC | Δ DSC |\n")
        f.write("|------|--------|----------|----------|-------|\n")

        sorted_diffs = sorted(per_sample, key=lambda x: abs(x["delta_dsc"]), reverse=True)[:10]
        for row in sorted_diffs:
            f.write(
                f"| {row['case_id']} | {row['defect']} | {row['ddim_dsc']:.4f} | "
                f"{row['ddpm_dsc']:.4f} | {row['delta_dsc']:+.4f} |\n"
            )

        f.write("\n---\n")
        f.write("*Full per-sample data saved to `per_sample_comparison.csv`*\n")


def main():
    parser = argparse.ArgumentParser(description="E2E evaluation harness: DDIM-50 vs DDPM-1000")
    parser.add_argument(
        "--pcdiff-checkpoint", type=Path, required=True, help="Path to trained PCDiff checkpoint (.pth)"
    )
    parser.add_argument(
        "--vox-checkpoint",
        type=Path,
        default=Path("voxelization/checkpoints/model_best.pt"),
        help="Path to trained voxelization checkpoint",
    )
    parser.add_argument(
        "--vox-config",
        type=Path,
        default=Path("voxelization/configs/gen_skullbreak.yaml"),
        help="Voxelization config file",
    )
    parser.add_argument(
        "--dataset-csv", type=Path, default=Path("datasets/SkullBreak/test.csv"), help="Test dataset CSV"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pcdiff/eval/e2e_comparison"), help="Output directory for results"
    )
    parser.add_argument("--num-ens", type=int, default=5, help="Ensemble size for inference")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU IDs")
    parser.add_argument(
        "--skip-inference", action="store_true", help="Skip inference, only compute metrics from existing outputs"
    )
    parser.add_argument("--ddim-only", action="store_true", help="Only run DDIM-50 (for quick testing)")
    parser.add_argument("--ddpm-only", action="store_true", help="Only run DDPM-1000")
    args = parser.parse_args()

    # Resolve paths
    checkpoint = args.pcdiff_checkpoint.expanduser().resolve()
    vox_checkpoint = args.vox_checkpoint.expanduser().resolve()
    vox_config = args.vox_config.expanduser().resolve()
    csv_path = args.dataset_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    num_gpus = len(gpu_ids)

    if not checkpoint.exists():
        raise FileNotFoundError(f"PCDiff checkpoint not found: {checkpoint}")
    if not vox_checkpoint.exists():
        raise FileNotFoundError(f"Voxelization checkpoint not found: {vox_checkpoint}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    print("=" * 70)
    print("PCDiff E2E Evaluation Harness: DDIM-50 vs DDPM-1000")
    print("=" * 70)
    print(f"PCDiff checkpoint: {checkpoint}")
    print(f"Voxelization checkpoint: {vox_checkpoint}")
    print(f"Dataset CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Ensemble size: {args.num_ens}")
    print(f"GPUs: {gpu_ids}")
    print("=" * 70)

    # Read test dataset
    samples = read_test_dataset(csv_path)
    print(f"\nFound {len(samples)} test samples")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results
    ddim_results = MethodResults(method="ddim", sampling_steps=50, num_ens=args.num_ens)
    ddpm_results = MethodResults(method="ddpm", sampling_steps=1000, num_ens=args.num_ens)

    ddim_syn_dir = output_dir / "ddim_50" / "syn"
    ddpm_syn_dir = output_dir / "ddpm_1000" / "syn"

    # Run DDIM-50 inference
    if not args.ddpm_only:
        if not args.skip_inference:
            ddim_time, ddim_success = run_distributed_inference(
                checkpoint=checkpoint,
                output_dir=output_dir / "ddim_50",
                csv_path=csv_path,
                sampling_method="ddim",
                sampling_steps=50,
                num_ens=args.num_ens,
                num_gpus=num_gpus,
            )
            ddim_results.total_inference_time = ddim_time

            if not ddim_success:
                print("ERROR: DDIM-50 inference failed!")
                sys.exit(1)
        else:
            print("\nSkipping DDIM-50 inference (using existing outputs)...")

        # Compute DDIM metrics
        print("\n" + "=" * 60)
        print("Computing DDIM-50 voxelization metrics...")
        print("=" * 60)

        start_time = time.time()
        ddim_metrics = compute_all_metrics(
            samples=samples,
            syn_dir=ddim_syn_dir,
            num_ens=args.num_ens,
            vox_config=vox_config,
            vox_checkpoint=vox_checkpoint,
            gpu_ids=gpu_ids,
        )
        ddim_results.samples = ddim_metrics
        ddim_results.total_voxelization_time = time.time() - start_time

        print(f"\nDDIM-50 Results ({len(ddim_metrics)} samples):")
        print(f"  DSC:  {ddim_results.mean_dsc:.4f} ± {ddim_results.std_dsc:.4f}")
        print(f"  bDSC: {ddim_results.mean_bdsc:.4f} ± {ddim_results.std_bdsc:.4f}")
        print(f"  HD95: {ddim_results.mean_hd95:.4f} ± {ddim_results.std_hd95:.4f}")

    # Run DDPM-1000 inference
    if not args.ddim_only:
        if not args.skip_inference:
            ddpm_time, ddpm_success = run_distributed_inference(
                checkpoint=checkpoint,
                output_dir=output_dir / "ddpm_1000",
                csv_path=csv_path,
                sampling_method="ddpm",
                sampling_steps=1000,
                num_ens=args.num_ens,
                num_gpus=num_gpus,
            )
            ddpm_results.total_inference_time = ddpm_time

            if not ddpm_success:
                print("ERROR: DDPM-1000 inference failed!")
                sys.exit(1)
        else:
            print("\nSkipping DDPM-1000 inference (using existing outputs)...")

        # Compute DDPM metrics
        print("\n" + "=" * 60)
        print("Computing DDPM-1000 voxelization metrics...")
        print("=" * 60)

        start_time = time.time()
        ddpm_metrics = compute_all_metrics(
            samples=samples,
            syn_dir=ddpm_syn_dir,
            num_ens=args.num_ens,
            vox_config=vox_config,
            vox_checkpoint=vox_checkpoint,
            gpu_ids=gpu_ids,
        )
        ddpm_results.samples = ddpm_metrics
        ddpm_results.total_voxelization_time = time.time() - start_time

        print(f"\nDDPM-1000 Results ({len(ddpm_metrics)} samples):")
        print(f"  DSC:  {ddpm_results.mean_dsc:.4f} ± {ddpm_results.std_dsc:.4f}")
        print(f"  bDSC: {ddpm_results.mean_bdsc:.4f} ± {ddpm_results.std_bdsc:.4f}")
        print(f"  HD95: {ddpm_results.mean_hd95:.4f} ± {ddpm_results.std_hd95:.4f}")

    # Create comparison report
    if ddim_results.samples and ddpm_results.samples:
        print("\n" + "=" * 60)
        print("Creating comparison report...")
        print("=" * 60)

        summary = create_comparison_report(
            ddim_results=ddim_results,
            ddpm_results=ddpm_results,
            output_dir=output_dir,
            checkpoint=checkpoint,
            csv_path=csv_path,
        )

        print(f"\nComparison artifacts saved to: {output_dir}")
        print("  - comparison_summary.json")
        print("  - comparison_report.md")
        print("  - per_sample_comparison.json")
        print("  - per_sample_comparison.csv")

        # Print final summary
        print("\n" + "=" * 70)
        print("FINAL COMPARISON: DDIM-50 vs DDPM-1000")
        print("=" * 70)
        print(f"{'Metric':<10} {'DDIM-50':>20} {'DDPM-1000':>20} {'Δ':>12}")
        print("-" * 70)
        print(
            f"{'DSC':<10} {ddim_results.mean_dsc:>15.4f}±{ddim_results.std_dsc:.4f} "
            f"{ddpm_results.mean_dsc:>15.4f}±{ddpm_results.std_dsc:.4f} "
            f"{ddpm_results.mean_dsc - ddim_results.mean_dsc:>+12.4f}"
        )
        print(
            f"{'bDSC':<10} {ddim_results.mean_bdsc:>15.4f}±{ddim_results.std_bdsc:.4f} "
            f"{ddpm_results.mean_bdsc:>15.4f}±{ddpm_results.std_bdsc:.4f} "
            f"{ddpm_results.mean_bdsc - ddim_results.mean_bdsc:>+12.4f}"
        )
        print(
            f"{'HD95':<10} {ddim_results.mean_hd95:>15.4f}±{ddim_results.std_hd95:.4f} "
            f"{ddpm_results.mean_hd95:>15.4f}±{ddpm_results.std_hd95:.4f} "
            f"{ddpm_results.mean_hd95 - ddim_results.mean_hd95:>+12.4f}"
        )
        print("=" * 70)

    elif ddim_results.samples:
        # Save DDIM-only results
        with (output_dir / "ddim_50_metrics.json").open("w") as f:
            json.dump(
                {
                    "method": "ddim",
                    "sampling_steps": 50,
                    "num_ens": args.num_ens,
                    "n_samples": len(ddim_results.samples),
                    "dsc": {"mean": ddim_results.mean_dsc, "std": ddim_results.std_dsc},
                    "bdsc": {"mean": ddim_results.mean_bdsc, "std": ddim_results.std_bdsc},
                    "hd95": {"mean": ddim_results.mean_hd95, "std": ddim_results.std_hd95},
                },
                f,
                indent=2,
            )

    elif ddpm_results.samples:
        # Save DDPM-only results
        with (output_dir / "ddpm_1000_metrics.json").open("w") as f:
            json.dump(
                {
                    "method": "ddpm",
                    "sampling_steps": 1000,
                    "num_ens": args.num_ens,
                    "n_samples": len(ddpm_results.samples),
                    "dsc": {"mean": ddpm_results.mean_dsc, "std": ddpm_results.std_dsc},
                    "bdsc": {"mean": ddpm_results.mean_bdsc, "std": ddpm_results.std_bdsc},
                    "hd95": {"mean": ddpm_results.mean_hd95, "std": ddpm_results.std_hd95},
                },
                f,
                indent=2,
            )

    print("\nE2E evaluation complete!")


if __name__ == "__main__":
    main()
