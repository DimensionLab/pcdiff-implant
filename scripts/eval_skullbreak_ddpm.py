#!/usr/bin/env python3
"""
Evaluate PCDiff + voxelization on the SkullBreak test set using DDPM sampling and
ensemble averaging, reproducing the metrics reported in Table 1 of the paper.

The script:
  1. Splits the SkullBreak test CSV across the requested GPUs.
  2. Runs PCDiff inference with DDPM (default 1000 steps) generating `num_ens`
     implants per case.
  3. Converts each ensemble sample to a voxel implant with the voxelization model,
     aggregates them into a mean implant (majority vote), and applies the same
     post-processing as `voxelization/generate.py`.
  4. Computes DSC, boundary DSC (10 mm), and HD95 for every case.
  5. Aggregates mean metrics over the full test set, compares them to the paper
     results (Table 1), and writes a Markdown report.
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Sequence

import diplib as dip
import numpy as np
import torch
import yaml
from scipy import ndimage

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "pcdiff"))
sys.path.insert(0, str(ROOT_DIR / "voxelization"))

import nrrd  # noqa: E402
from pcdiff.test_completion import Model as PCDiffModel, get_betas  # noqa: E402
from voxelization.eval_metrics import bdc, dc, hd95  # noqa: E402
from voxelization.src import config as vox_config  # noqa: E402
from voxelization.src.model import Encode2Points  # noqa: E402
from voxelization.src.utils import load_config, load_model_manual, filter_voxels_within_radius  # noqa: E402

PAPER_METRICS_ENSEMBLE = {
    "dice": 0.87,
    "bdice": 0.89,
    "hd95": 2.45,
}


@dataclass
class SampleInfo:
    case_id: str
    defect: str
    defective_npy: Path
    defective_nrrd: Path
    implant_nrrd: Path


class PCDiffRunner:
    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        num_points: int,
        num_nn: int,
        num_ens: int,
        sampling_method: str,
        sampling_steps: int,
    ) -> None:
        self.device = device
        self.num_points = num_points
        self.num_nn = num_nn
        self.num_ens = num_ens
        self.sampling_method = sampling_method
        self.sampling_steps = sampling_steps

        betas = get_betas("linear", 0.0001, 0.02, 1000)

        class Args:
            pass

        args = Args()
        args.nc = 3
        args.num_points = num_points
        args.num_nn = num_nn
        args.attention = True
        args.dropout = 0.1
        args.embed_dim = 64
        args.loss_type = "mse"
        args.model_mean_type = "eps"
        args.model_var_type = "fixedsmall"
        args.width_mult = 1.0
        args.vox_res_mult = 1.0
        args.sampling_method = sampling_method
        args.sampling_steps = sampling_steps

        self.model = PCDiffModel(
            args,
            betas,
            args.loss_type,
            args.model_mean_type,
            args.model_var_type,
            args.width_mult,
            args.vox_res_mult,
        ).to(device)
        self.model.eval()

        checkpoint = Path(checkpoint).expanduser().resolve()
        state_dict = torch.load(checkpoint, map_location="cpu")["model_state"]
        if next(iter(state_dict.keys())).startswith("model.module."):
            state_dict = {k.replace("model.module.", "model."): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def generate_samples(self, defective_points: np.ndarray) -> np.ndarray:
        sv_points = self.num_points - self.num_nn
        if defective_points.shape[0] < sv_points:
            raise ValueError(
                f"Defective cloud has {defective_points.shape[0]} points, "
                f"but {sv_points} were expected."
            )

        idx = np.random.choice(defective_points.shape[0], sv_points, replace=False)
        partial_points_raw = defective_points[idx]

        pc_min = partial_points_raw.min(axis=0)
        pc_max = partial_points_raw.max(axis=0)
        shift = (pc_min + pc_max) / 2.0
        scale = (pc_max - pc_min).max() / 2.0
        if scale <= 0:
            raise ValueError("Invalid scale derived from defective point cloud.")
        scale = scale / 3.0

        partial_points = (partial_points_raw - shift) / scale
        partial_points = partial_points.astype(np.float32)

        pc_input = torch.from_numpy(partial_points).unsqueeze(0)  # (1, N, 3)
        pc_input = pc_input.transpose(1, 2).to(self.device)  # (1, 3, N)
        pc_input = pc_input.repeat(self.num_ens, 1, 1)

        noise_shape = torch.Size([self.num_ens, 3, self.num_nn])
        samples = self.model.gen_samples(
            pc_input,
            noise_shape,
            self.device,
            clip_denoised=False,
            sampling_method=self.sampling_method,
            sampling_steps=self.sampling_steps,
        )
        samples = samples.detach().cpu().numpy()
        samples = samples.transpose(0, 2, 1)  # (num_ens, num_nn, 3)

        return samples, shift.astype(np.float32), float(scale)


class VoxelizationRunner:
    def __init__(self, config_path: Path, checkpoint: Path, device: torch.device) -> None:
        config_path = config_path.expanduser().resolve()
        default_config = ROOT_DIR / "voxelization" / "configs" / "default.yaml"
        cfg = load_config(str(config_path), str(default_config))
        cfg["test"]["model_file"] = str(Path(checkpoint).expanduser().resolve())

        self.cfg = cfg
        self.device = device

        self.model = Encode2Points(cfg).to(device)
        state_dict = torch.load(cfg["test"]["model_file"], map_location="cpu")
        load_model_manual(state_dict["state_dict"], self.model)
        self.model.eval()

        self.generator = vox_config.get_generator(self.model, cfg, device=device)

    @torch.no_grad()
    def generate_psr(self, combined_points_norm: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        inputs = torch.from_numpy(combined_points_norm).float().unsqueeze(0).to(self.device)
        vertices, faces, points, normals, psr_grid = self.generator.generate_mesh(inputs)
        psr_grid_np = psr_grid.detach().cpu().numpy()[0]
        return psr_grid_np, points.detach().cpu()


def read_dataset(csv_path: Path) -> List[SampleInfo]:
    csv_path = csv_path.expanduser().resolve()
    base_dir = csv_path.parent
    samples: List[SampleInfo] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            complete_path = Path(row[0])
            if not complete_path.is_absolute():
                complete_path = (base_dir / complete_path).resolve()
            case_id = complete_path.stem
            root = complete_path.parent.parent
            for defect in ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]:
                defective_npy = root / "defective_skull" / defect / f"{case_id}_surf.npy"
                defective_nrrd = root / "defective_skull" / defect / f"{case_id}.nrrd"
                implant_nrrd = root / "implant" / defect / f"{case_id}.nrrd"
                if not (defective_npy.exists() and defective_nrrd.exists() and implant_nrrd.exists()):
                    continue
                samples.append(
                    SampleInfo(
                        case_id=case_id,
                        defect=defect,
                        defective_npy=defective_npy,
                        defective_nrrd=defective_nrrd,
                        implant_nrrd=implant_nrrd,
                    )
                )
    if not samples:
        raise RuntimeError(f"No valid samples found in {csv_path}")
    return samples


def evaluate_sample(
    sample: SampleInfo,
    pcdiff_runner: PCDiffRunner,
    vox_runner: VoxelizationRunner,
    num_ens: int,
) -> Dict[str, float]:
    defective_points = np.load(sample.defective_npy).astype(np.float32)
    ensemble_points, shift, scale = pcdiff_runner.generate_samples(defective_points)

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

    threshold = int(math.ceil(num_ens / 2.0))
    mean_complete = np.zeros_like(completes, dtype=np.float32)
    mean_complete[completes >= threshold] = 1

    defective_vol, header = nrrd.read(str(sample.defective_nrrd))
    mean_implant = mean_complete - defective_vol
    mean_implant = np.clip(mean_implant, 0.0, 1.0)
    raw_implant = mean_implant.copy()
    reference_implant_points = (
        reference_inputs[:, -pcdiff_runner.num_nn :, :]
        .detach()
        .cpu()
        .squeeze(0)
    )
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

    gt_implant, _ = nrrd.read(str(sample.implant_nrrd))
    spacing = np.asarray([
        header["space directions"][0, 0],
        header["space directions"][1, 1],
        header["space directions"][2, 2],
    ])

    dice = float(dc(mean_implant, gt_implant))
    bdice = float(bdc(mean_implant, gt_implant, defective_vol, voxelspacing=spacing, distance=10))
    haus = float(hd95(mean_implant, gt_implant, voxelspacing=spacing))

    return {
        "case_id": sample.case_id,
        "defect": sample.defect,
        "dice": dice,
        "bdice": bdice,
        "hd95": haus,
    }


def worker(
    rank: int,
    gpu_id: int,
    samples: Sequence[SampleInfo],
    args: argparse.Namespace,
    result_store,
) -> None:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    pcdiff_runner = PCDiffRunner(
        checkpoint=args.pcdiff_model,
        device=device,
        num_points=args.num_points,
        num_nn=args.num_nn,
        num_ens=args.num_ens,
        sampling_method=args.sampling_method,
        sampling_steps=args.sampling_steps,
    )
    vox_runner = VoxelizationRunner(
        config_path=args.vox_config,
        checkpoint=args.vox_model,
        device=device,
    )

    results = []
    for sample in samples:
        metrics = evaluate_sample(sample, pcdiff_runner, vox_runner, args.num_ens)
        results.append(metrics)
    result_store[rank] = results


def aggregate_results(result_store) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for partial in result_store.values():
        results.extend(partial)
    results.sort(key=lambda x: (x["case_id"], x["defect"]))
    return results


def spawn_worker(
    rank: int,
    gpu_ids: Sequence[int],
    samples: Sequence[SampleInfo],
    args: argparse.Namespace,
    result_store,
    world_size: int,
) -> None:
    gpu_id = gpu_ids[rank % len(gpu_ids)]
    chunk = samples[rank::world_size]
    worker(rank, gpu_id, chunk, args, result_store)


def write_markdown(
    markdown_path: Path,
    results: List[Dict[str, float]],
    means: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    markdown_path = markdown_path.expanduser().resolve()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# SkullBreak Evaluation (DDPM, Ensemble)\n\n")
        handle.write(f"- PCDiff checkpoint: `{args.pcdiff_model}`\n")
        handle.write(f"- Voxelization checkpoint: `{args.vox_model}`\n")
        handle.write(f"- Dataset CSV: `{args.dataset_csv}`\n")
        handle.write(f"- Sampling: **{args.sampling_method}** with **{args.sampling_steps}** steps\n")
        handle.write(f"- Ensemble size: **{args.num_ens}**\n")
        handle.write(f"- GPUs used: `{args.gpus}`\n\n")

        handle.write("## Mean Metrics\n\n")
        handle.write("| Metric | Paper (Ours n=5) | This run | Δ |\n")
        handle.write("|--------|------------------|----------|----|\n")
        for key, paper_value in PAPER_METRICS_ENSEMBLE.items():
            ours = means[key]
            delta = ours - paper_value
            handle.write(f"| {key.upper()} | {paper_value:.4f} | {ours:.4f} | {delta:+.4f} |\n")

        handle.write("\n## Per-case Summary (first 10 rows)\n\n")
        handle.write("| Case | Defect | DSC | bDSC | HD95 |\n")
        handle.write("|------|--------|-----|------|------|\n")
        for entry in results[:10]:
            handle.write(
                f"| {entry['case_id']} | {entry['defect']} | "
                f"{entry['dice']:.4f} | {entry['bdice']:.4f} | {entry['hd95']:.4f} |\n"
            )

        handle.write("\n_Full metrics saved alongside this report._\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SkullBreak DDPM ensemble metrics.")
    parser.add_argument("--pcdiff-model", type=Path, required=True, help="Path to trained PCDiff checkpoint (.pth).")
    parser.add_argument("--vox-model", type=Path, required=True, help="Path to trained voxelization checkpoint.")
    parser.add_argument("--vox-config", type=Path, default=Path("voxelization/configs/gen_skullbreak.yaml"))
    parser.add_argument("--dataset-csv", type=Path, required=True, help="SkullBreak CSV (e.g., test split).")
    parser.add_argument("--num-ens", type=int, default=5)
    parser.add_argument("--sampling-method", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num-points", type=int, default=30720)
    parser.add_argument("--num-nn", type=int, default=3072)
    parser.add_argument("--output-dir", type=Path, default=Path("my_results/skullbreak_eval_ddpm"))
    parser.add_argument("--markdown", type=Path, default=Path("my_results/skullbreak_eval_ddpm/report.md"))
    parser.add_argument("--metrics-yaml", type=Path, default=Path("my_results/skullbreak_eval_ddpm/metrics.yaml"))
    args = parser.parse_args()

    samples = read_dataset(args.dataset_csv)
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    world_size = min(len(gpu_ids), len(samples))
    if world_size == 0:
        raise RuntimeError("No GPUs specified for evaluation.")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ctx = get_context("spawn")
    manager = ctx.Manager()
    result_store = manager.dict()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=spawn_worker,
            args=(rank, gpu_ids, samples, args, result_store, world_size),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    results = aggregate_results(result_store)
    if not results:
        raise RuntimeError("No evaluation results gathered.")

    means = {
        "dice": float(np.mean([entry["dice"] for entry in results])),
        "bdice": float(np.mean([entry["bdice"] for entry in results])),
        "hd95": float(np.mean([entry["hd95"] for entry in results])),
    }

    metrics_yaml_path = args.metrics_yaml.expanduser().resolve()
    metrics_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"means": means, "samples": results}, handle)

    write_markdown(args.markdown, results, means, args)
    print(f"Markdown report written to {args.markdown}")
    print(f"Mean metrics: {means}")


if __name__ == "__main__":
    main()
