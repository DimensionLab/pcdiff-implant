#!/usr/bin/env python3
"""
SkullBreak Evaluation Script

Runs the full PCDiff + voxelization pipeline on a handful of random SkullBreak
cases and reports Dice (DSC), boundary Dice (bDSC, 10 mm), and HD95 metrics
against the ground-truth implants.
"""

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import diplib as dip
import numpy as np
import torch
import yaml
import nrrd

# Ensure project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pcdiff"))
sys.path.insert(0, str(PROJECT_ROOT / "voxelization"))

from test_completion import Model as PCDiffModel, get_betas  # noqa: E402
from voxelization.src.model import Encode2Points  # noqa: E402
from voxelization.src.utils import (  # noqa: E402
    filter_voxels_within_radius,
    load_config,
    load_model_manual,
)
from voxelization.src import config as vox_config  # noqa: E402
from voxelization.eval_metrics import bdc, dc, hd95  # noqa: E402

DEFECT_TYPES = ['bilateral', 'frontoorbital', 'parietotemporal', 'random_1', 'random_2']


@dataclass
class SampleInfo:
    case_id: str
    defect: str
    complete_path: Path
    defective_npy: Path
    defective_nrrd: Path
    implant_npy: Path
    implant_nrrd: Path
    vox_pointcloud: Path
    vox_grid: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PCDiff pipeline on random SkullBreak samples."
    )
    parser.add_argument(
        "--pcdiff-model",
        required=True,
        help="Path to trained PCDiff checkpoint (.pth).",
    )
    parser.add_argument(
        "--vox-model",
        required=True,
        help="Path to trained voxelization checkpoint (.pth).",
    )
    parser.add_argument(
        "--vox-config",
        default="voxelization/configs/gen_skullbreak.yaml",
        help="Voxelization config file (defaults to SkullBreak generation config).",
    )
    parser.add_argument(
        "--dataset-csv",
        default="pcdiff/datasets/SkullBreak/test.csv",
        help="CSV listing SkullBreak cases (use test split by default).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of random samples to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=30720,
        help="Total number of points used in PCDiff (defective + implant).",
    )
    parser.add_argument(
        "--num-nn",
        type=int,
        default=3072,
        help="Number of implant points predicted by PCDiff.",
    )
    parser.add_argument(
        "--sampling-method",
        choices=["ddpm", "ddim"],
        default="ddim",
        help="Sampling method for diffusion inference.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps (ignored for DDPM).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to run inference on (e.g. cuda:0 or cpu).",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Optional path to write metrics as YAML.",
    )
    return parser.parse_args()


def read_dataset(csv_path: Path) -> List[SampleInfo]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

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
            for defect in DEFECT_TYPES:
                defective_dir = complete_path.parent.parent / "defective_skull" / defect
                implant_dir = complete_path.parent.parent / "implant" / defect
                vox_dir = complete_path.parent.parent / "voxelization"

                defective_npy = defective_dir / f"{case_id}_surf.npy"
                defective_nrrd = defective_dir / f"{case_id}.nrrd"
                implant_npy = implant_dir / f"{case_id}_surf.npy"
                implant_nrrd = implant_dir / f"{case_id}.nrrd"
                vox_pc = vox_dir / f"{case_id}_{defect}_pc.npz"
                vox_grid = vox_dir / f"{case_id}_{defect}_vox.npz"

                if not defective_npy.exists() or not defective_nrrd.exists():
                    continue
                if not implant_nrrd.exists():
                    continue
                if not vox_pc.exists() or not vox_grid.exists():
                    continue

                samples.append(
                    SampleInfo(
                        case_id=case_id,
                        defect=defect,
                        complete_path=complete_path,
                        defective_npy=defective_npy,
                        defective_nrrd=defective_nrrd,
                        implant_npy=implant_npy,
                        implant_nrrd=implant_nrrd,
                        vox_pointcloud=vox_pc,
                        vox_grid=vox_grid,
                    )
                )

    if not samples:
        raise RuntimeError(f"No valid samples discovered in {csv_path}")

    return samples


class PCDiffRunner:
    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        num_points: int,
        num_nn: int,
        sampling_method: str = "ddim",
        sampling_steps: int = 50,
    ):
        self.device = device
        self.num_points = num_points
        self.num_nn = num_nn
        self.sampling_method = sampling_method
        self.sampling_steps = sampling_steps
        self.num_ens = 1

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
        args.sampling_method = sampling_method
        args.sampling_steps = sampling_steps

        args.time_num = len(betas)

        self.model = PCDiffModel(
            args,
            betas,
            "mse",
            "eps",
            "fixedsmall",
            width_mult=1.0,
            vox_res_mult=1.0,
        )
        self.model.to(device)
        self.model.eval()

        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"PCDiff checkpoint not found: {checkpoint}")

        state_dict = torch.load(checkpoint, map_location="cpu")["model_state"]
        # Strip DDP prefixes if present
        if next(iter(state_dict.keys())).startswith("model.module."):
            state_dict = {
                k.replace("model.module.", "model."): v for k, v in state_dict.items()
            }
        self.model.load_state_dict(state_dict, strict=True)

    def generate_implant(
        self,
        defective_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns:
            implant_points (num_nn, 3) in original coordinates (0..512)
            implant_points_normalized (num_nn, 3) in normalized space (-?, ?)
            shift (3,) and scale (float) used during normalization.
        """
        if defective_points.shape[1] != 3:
            raise ValueError("Defective point cloud must have shape (N, 3)")

        sv_points = self.num_points - self.num_nn
        if defective_points.shape[0] < sv_points:
            raise ValueError(
                f"Defective cloud has {defective_points.shape[0]} points "
                f"but {sv_points} are required."
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
        with torch.no_grad():
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

        implant_normalized = samples[0]
        implant_points = implant_normalized * scale + shift
        return implant_points.astype(np.float32), implant_normalized.astype(np.float32), shift.astype(np.float32), float(scale)


class VoxelizationRunner:
    def __init__(self, config_path: Path, checkpoint: Path, device: torch.device):
        config_path = config_path.expanduser().resolve()
        default_config = PROJECT_ROOT / "voxelization" / "configs" / "default.yaml"
        cfg = load_config(str(config_path), str(default_config))
        cfg["test"]["model_file"] = str(Path(checkpoint).expanduser().resolve())

        self.cfg = cfg
        self.device = device

        self.model = Encode2Points(cfg).to(device)
        state_dict = torch.load(cfg["test"]["model_file"], map_location="cpu")
        load_model_manual(state_dict["state_dict"], self.model)
        self.model.eval()

        self.generator = vox_config.get_generator(self.model, cfg, device=device)

    def generate_psr(
        self, combined_points_norm: np.ndarray
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Args:
            combined_points_norm: (N, 3) point cloud in [0, 1] space
        Returns:
            psr_grid: numpy array (512, 512, 512)
            inputs_tensor: torch tensor (1, N, 3) used for filtering utilities
        """
        inputs = torch.from_numpy(combined_points_norm).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, _, _, psr_grid = self.generator.generate_mesh(inputs)

        psr_grid_np = psr_grid.detach().cpu().numpy()[0]
        return psr_grid_np, inputs


def prepare_predicted_implant_volume(
    psr_grid: np.ndarray,
    defective_volume: np.ndarray,
    implant_points_norm: np.ndarray,
) -> np.ndarray:
    """
    Builds a binary implant segmentation from the PSR grid and post-processes
    it using the voxelization utilities to match the paper pipeline.
    """
    complete_volume = np.zeros_like(psr_grid, dtype=np.uint8)
    complete_volume[psr_grid <= 0] = 1

    implant_volume = complete_volume.astype(np.int16) - defective_volume.astype(np.int16)
    implant_volume = (implant_volume > 0).astype(np.uint8)

    implant_tensor = torch.from_numpy(implant_points_norm).float()
    filtered = filter_voxels_within_radius(implant_tensor, implant_volume)
    filtered = filtered.astype(bool)

    # Morphological cleanup (mirror voxelization/generate.py)
    implant = dip.Opening(filtered, dip.SE((3, 3, 3)))
    implant = dip.Label(implant, mode="largest")
    implant = dip.MedianFilter(implant, dip.Kernel(shape="rectangular", param=(3, 3, 3)))
    implant.Convert("BIN")
    implant = dip.Closing(implant, dip.SE((3, 3, 3)))
    implant = dip.FillHoles(implant)
    implant = dip.Label(implant, mode="largest")
    implant = np.asarray(implant, dtype=np.float32)

    return implant


def evaluate_sample(
    sample: SampleInfo,
    pcdiff_runner: PCDiffRunner,
    vox_runner: VoxelizationRunner,
) -> Dict[str, float]:
    defective_points = np.load(sample.defective_npy).astype(np.float32)

    implant_points, implant_norm, shift, scale = pcdiff_runner.generate_implant(defective_points)

    combined_points = np.concatenate([defective_points, implant_points], axis=0)
    combined_norm = combined_points / 512.0

    psr_grid, _ = vox_runner.generate_psr(combined_norm)

    defective_vol, header = nrrd.read(str(sample.defective_nrrd))
    defective_vol = defective_vol.astype(np.uint8)
    gt_implant, _ = nrrd.read(str(sample.implant_nrrd))
    gt_implant = gt_implant.astype(np.uint8)

    pred_implant = prepare_predicted_implant_volume(
        psr_grid,
        defective_vol,
        implant_points / 512.0,
    )

    spacing = np.asarray([
        header["space directions"][0, 0],
        header["space directions"][1, 1],
        header["space directions"][2, 2],
    ])

    dice = dc(pred_implant, gt_implant)
    bdice = bdc(pred_implant, gt_implant, defective_vol, voxelspacing=spacing, distance=10)
    haus95 = float(hd95(pred_implant, gt_implant, voxelspacing=spacing))

    return {
        "dice": float(dice),
        "bdice": float(bdice),
        "hd95": haus95,
        "shift": shift.tolist(),
        "scale": scale,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    samples = read_dataset(Path(args.dataset_csv))
    chosen_samples = random.sample(samples, k=min(args.num_samples, len(samples)))

    print(f"Evaluating {len(chosen_samples)} SkullBreak samples on {device}...\n")

    pcdiff_runner = PCDiffRunner(
        checkpoint=Path(args.pcdiff_model),
        device=device,
        num_points=args.num_points,
        num_nn=args.num_nn,
        sampling_method=args.sampling_method,
        sampling_steps=args.sampling_steps,
    )

    vox_runner = VoxelizationRunner(
        config_path=Path(args.vox_config),
        checkpoint=Path(args.vox_model),
        device=device,
    )

    metrics: List[Tuple[SampleInfo, Dict[str, float]]] = []
    for sample in chosen_samples:
        print(f"Processing case {sample.case_id} ({sample.defect})...")
        try:
            result = evaluate_sample(sample, pcdiff_runner, vox_runner)
            metrics.append((sample, result))
            print(
                f"  DSC={result['dice']:.4f}, "
                f"bDSC={result['bdice']:.4f}, "
                f"HD95={result['hd95']:.4f}"
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  ⚠️  Failed: {exc}")

    if not metrics:
        raise RuntimeError("No samples evaluated successfully.")

    mean_dice = np.mean([m["dice"] for _, m in metrics])
    mean_bdice = np.mean([m["bdice"] for _, m in metrics])
    mean_hd95 = np.mean([m["hd95"] for _, m in metrics])

    print("\nSummary (SkullBreak):")
    print(f"  Mean DSC : {mean_dice:.4f}")
    print(f"  Mean bDSC: {mean_bdice:.4f}")
    print(f"  Mean HD95: {mean_hd95:.4f}")

    if args.output_report:
        report = {
            "mean": {
                "dice": float(mean_dice),
                "bdice": float(mean_bdice),
                "hd95": float(mean_hd95),
            },
            "samples": [
                {
                    "case_id": sample.case_id,
                    "defect": sample.defect,
                    "dice": result["dice"],
                    "bdice": result["bdice"],
                    "hd95": result["hd95"],
                }
                for sample, result in metrics
            ],
        }
        with open(args.output_report, "w", encoding="utf-8") as handle:
            yaml.safe_dump(report, handle)
        print(f"\nReport written to {args.output_report}")


if __name__ == "__main__":
    main()
