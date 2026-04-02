"""
Proxy evaluation module for PCDiff training.

Runs fast DDIM-50 inference on a fixed validation subset every 50 epochs
and computes DSC/bDSC/HD95 metrics using the voxelization model.
"""

import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import ndimage

# Add paths for imports
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "pcdiff"))
sys.path.insert(0, str(ROOT_DIR / "voxelization"))

# These imports may fail if the environment is not fully set up
try:
    import diplib as dip
    import nrrd

    from voxelization.eval_metrics import bdc, dc
    from voxelization.eval_metrics import hd95 as compute_hd95
    from voxelization.src import config as vox_config
    from voxelization.src.model import Encode2Points
    from voxelization.src.utils import filter_voxels_within_radius, load_config, load_model_manual

    VOXELIZATION_AVAILABLE = True
except ImportError as e:
    VOXELIZATION_AVAILABLE = False
    VOXELIZATION_IMPORT_ERROR = str(e)


@dataclass
class ProxySample:
    """Information for a single proxy validation sample."""

    case_id: str
    defective_npy: Path
    defective_nrrd: Path
    implant_nrrd: Path


@dataclass
class ProxyEvalResult:
    """Result of evaluating a single sample."""

    case_id: str
    dsc: float
    bdsc: float
    hd95: float
    error: Optional[str] = None


class VoxelizationRunner:
    """Wrapper for the voxelization model (converts point clouds to voxel volumes)."""

    def __init__(self, config_path: Path, checkpoint: Path, device: torch.device):
        if not VOXELIZATION_AVAILABLE:
            raise RuntimeError(f"Voxelization dependencies not available: {VOXELIZATION_IMPORT_ERROR}")

        config_path = Path(config_path).expanduser().resolve()
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
    def generate_psr(self, combined_points_norm: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Generate PSR grid from normalized combined point cloud."""
        inputs = torch.from_numpy(combined_points_norm).float().unsqueeze(0).to(self.device)
        vertices, faces, points, normals, psr_grid = self.generator.generate_mesh(inputs)
        psr_grid_np = psr_grid.detach().cpu().numpy()[0]
        return psr_grid_np, points.detach().cpu()


def load_proxy_subset(subset_path: Path, base_dir: Optional[Path] = None) -> List[ProxySample]:
    """Load the fixed proxy validation subset from JSON file."""
    subset_path = Path(subset_path).resolve()

    with open(subset_path, "r") as f:
        data = json.load(f)

    if base_dir is None:
        base_dir = ROOT_DIR
    else:
        base_dir = Path(base_dir).resolve()

    samples = []
    for item in data["samples"]:
        sample = ProxySample(
            case_id=item["case_id"],
            defective_npy=base_dir / item["defective_npy"],
            defective_nrrd=base_dir / item["defective_nrrd"],
            implant_nrrd=base_dir / item["implant_nrrd"],
        )
        samples.append(sample)

    return samples


def run_pcdiff_inference_on_sample(
    model,
    defective_points: np.ndarray,
    num_points: int,
    num_nn: int,
    num_ens: int,
    device: torch.device,
    sampling_method: str = "ddpm",
    sampling_steps: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run PCDiff inference on a single defective skull point cloud.

    Args:
        model: PCDiff model (in eval mode) - both training and inference models support DDIM/DDPM
        defective_points: (N, 3) defective skull point cloud
        num_points: Total number of points (skull + implant)
        num_nn: Number of implant points to generate
        num_ens: Number of ensemble samples
        device: CUDA device
        sampling_method: "ddim" or "ddpm"
        sampling_steps: Number of diffusion steps (DDIM: typically 50, DDPM: always 1000)

    Returns:
        ensemble_points: (num_ens, num_nn, 3) generated implant point clouds (denormalized)
        shift: (3,) translation applied to input
        scale: float scaling factor applied to input
    """
    sv_points = num_points - num_nn  # Number of skull points to use

    if defective_points.shape[0] < sv_points:
        raise ValueError(f"Defective cloud has {defective_points.shape[0]} points, but {sv_points} were expected.")

    # Randomly sample skull points (same as eval script)
    idx = np.random.choice(defective_points.shape[0], sv_points, replace=False)
    partial_points_raw = defective_points[idx]

    # Compute normalization (same as training data loader)
    pc_min = partial_points_raw.min(axis=0)
    pc_max = partial_points_raw.max(axis=0)
    shift = (pc_min + pc_max) / 2.0
    scale = (pc_max - pc_min).max() / 2.0
    if scale <= 0:
        raise ValueError("Invalid scale derived from defective point cloud.")
    scale = scale / 3.0

    partial_points = (partial_points_raw - shift) / scale
    partial_points = partial_points.astype(np.float32)

    # Prepare input tensor
    pc_input = torch.from_numpy(partial_points).unsqueeze(0)  # (1, N, 3)
    pc_input = pc_input.transpose(1, 2).to(device)  # (1, 3, N)
    pc_input = pc_input.repeat(num_ens, 1, 1)  # (num_ens, 3, N)

    # Generate samples using DDIM or DDPM
    noise_shape = torch.Size([num_ens, 3, num_nn])
    samples = model.gen_samples(
        pc_input,
        noise_shape,
        device,
        clip_denoised=False,
        sampling_method=sampling_method,
        sampling_steps=sampling_steps,
    )

    # Convert to numpy and denormalize
    samples = samples.detach().cpu().numpy()
    samples = samples.transpose(0, 2, 1)  # (num_ens, num_nn, 3)

    # Denormalize the implant points
    ensemble_points = samples * scale + shift

    return ensemble_points, shift.astype(np.float32), float(scale)


def compute_metrics_for_sample(
    sample: ProxySample,
    ensemble_points: np.ndarray,
    vox_runner: VoxelizationRunner,
    num_nn: int,
) -> ProxyEvalResult:
    """
    Compute DSC/bDSC/HD95 metrics for a single sample.

    Args:
        sample: Sample information (paths to nrrd files)
        ensemble_points: (num_ens, num_nn, 3) generated implant point clouds
        vox_runner: Voxelization model wrapper
        num_nn: Number of implant points

    Returns:
        ProxyEvalResult with computed metrics
    """
    try:
        # Load defective skull points (for combining with implant)
        defective_points = np.load(sample.defective_npy).astype(np.float32)

        num_ens = ensemble_points.shape[0]
        completes = np.zeros((512, 512, 512), dtype=np.float32)
        reference_inputs = None

        # Process each ensemble sample
        for idx in range(num_ens):
            implant_pc = ensemble_points[idx]
            combined_points = np.concatenate([defective_points, implant_pc], axis=0)
            combined_norm = combined_points / 512.0  # Normalize for voxelization

            psr_grid_np, inputs_tensor = vox_runner.generate_psr(combined_norm)

            out = np.zeros((512, 512, 512), dtype=np.float32)
            out[psr_grid_np <= 0] = 1
            out = ndimage.binary_dilation(out)
            completes += out

            if reference_inputs is None:
                reference_inputs = inputs_tensor

        # Majority vote for ensemble
        threshold = int(math.ceil(num_ens / 2.0))
        mean_complete = np.zeros_like(completes, dtype=np.float32)
        mean_complete[completes >= threshold] = 1

        # Extract implant (subtract defective skull)
        defective_vol, header = nrrd.read(str(sample.defective_nrrd))
        mean_implant = mean_complete - defective_vol
        mean_implant = np.clip(mean_implant, 0.0, 1.0)
        raw_implant = mean_implant.copy()

        # Post-processing: filter voxels within radius of predicted implant points
        reference_implant_points = reference_inputs[:, -num_nn:, :].detach().cpu().squeeze(0)
        mean_implant = filter_voxels_within_radius(reference_implant_points, mean_implant)
        if not np.any(mean_implant):
            mean_implant = raw_implant

        # Morphological post-processing
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

        # Load ground truth implant
        gt_implant, _ = nrrd.read(str(sample.implant_nrrd))

        # Get voxel spacing from header
        spacing = np.asarray(
            [
                header["space directions"][0, 0],
                header["space directions"][1, 1],
                header["space directions"][2, 2],
            ]
        )

        # Compute metrics
        dice = float(dc(mean_implant, gt_implant))
        bdice = float(bdc(mean_implant, gt_implant, defective_vol, voxelspacing=spacing, distance=10))
        hd95_val = float(compute_hd95(mean_implant, gt_implant, voxelspacing=spacing))

        return ProxyEvalResult(
            case_id=sample.case_id,
            dsc=dice,
            bdsc=bdice,
            hd95=hd95_val,
        )

    except Exception as e:
        return ProxyEvalResult(
            case_id=sample.case_id,
            dsc=0.0,
            bdsc=0.0,
            hd95=float("inf"),
            error=str(e),
        )


def run_proxy_evaluation(
    pcdiff_model,
    vox_config_path: Path,
    vox_checkpoint_path: Path,
    subset_path: Path,
    device: torch.device,
    num_points: int = 30720,
    num_nn: int = 3072,
    num_ens: int = 1,
    sampling_method: str = "ddim",
    sampling_steps: int = 50,
    base_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Run proxy evaluation on the fixed validation subset.

    Args:
        pcdiff_model: PCDiff model (will be set to eval mode)
        vox_config_path: Path to voxelization config YAML
        vox_checkpoint_path: Path to voxelization checkpoint
        subset_path: Path to proxy validation subset JSON
        device: CUDA device
        num_points: Total points (skull + implant), default 30720
        num_nn: Implant points to generate, default 3072
        num_ens: Ensemble size (1 for proxy, 5 for full eval)
        sampling_method: "ddim" for proxy, "ddpm" for full eval
        sampling_steps: 50 for DDIM proxy, 1000 for DDPM
        base_dir: Base directory for resolving relative paths
        logger: Optional logger

    Returns:
        Dictionary with mean metrics: {"dsc": float, "bdsc": float, "hd95": float}
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not VOXELIZATION_AVAILABLE:
        logger.warning(f"Voxelization not available, skipping proxy eval: {VOXELIZATION_IMPORT_ERROR}")
        return {"dsc": 0.0, "bdsc": 0.0, "hd95": float("inf"), "error": "voxelization_unavailable"}

    # Load samples
    try:
        samples = load_proxy_subset(subset_path, base_dir)
    except Exception as e:
        logger.error(f"Failed to load proxy subset: {e}")
        return {"dsc": 0.0, "bdsc": 0.0, "hd95": float("inf"), "error": f"subset_load_failed: {e}"}

    logger.info(f"Running proxy evaluation on {len(samples)} samples with {sampling_method.upper()}-{sampling_steps}")

    # Initialize voxelization runner
    try:
        vox_runner = VoxelizationRunner(vox_config_path, vox_checkpoint_path, device)
    except Exception as e:
        logger.error(f"Failed to initialize voxelization runner: {e}")
        return {"dsc": 0.0, "bdsc": 0.0, "hd95": float("inf"), "error": f"vox_init_failed: {e}"}

    # Set model to eval mode
    pcdiff_model.eval()

    results: List[ProxyEvalResult] = []

    for i, sample in enumerate(samples):
        try:
            # Check if files exist
            if not sample.defective_npy.exists():
                logger.warning(f"Missing defective_npy: {sample.defective_npy}")
                results.append(ProxyEvalResult(sample.case_id, 0.0, 0.0, float("inf"), "missing_defective_npy"))
                continue
            if not sample.implant_nrrd.exists():
                logger.warning(f"Missing implant_nrrd: {sample.implant_nrrd}")
                results.append(ProxyEvalResult(sample.case_id, 0.0, 0.0, float("inf"), "missing_implant_nrrd"))
                continue

            # Load defective skull points
            defective_points = np.load(sample.defective_npy).astype(np.float32)

            # Run PCDiff inference
            with torch.no_grad():
                ensemble_points, shift, scale = run_pcdiff_inference_on_sample(
                    pcdiff_model,
                    defective_points,
                    num_points,
                    num_nn,
                    num_ens,
                    device,
                    sampling_method,
                    sampling_steps,
                )

            # Compute metrics
            result = compute_metrics_for_sample(sample, ensemble_points, vox_runner, num_nn)
            results.append(result)

            if result.error:
                logger.warning(f"Sample {sample.case_id}: error={result.error}")
            else:
                logger.info(
                    f"Sample {sample.case_id}: DSC={result.dsc:.4f}, bDSC={result.bdsc:.4f}, HD95={result.hd95:.2f}"
                )

        except Exception as e:
            logger.error(f"Failed to evaluate sample {sample.case_id}: {e}")
            results.append(ProxyEvalResult(sample.case_id, 0.0, 0.0, float("inf"), str(e)))

    # Set model back to train mode
    pcdiff_model.train()

    # Compute mean metrics (excluding failed samples)
    valid_results = [r for r in results if r.error is None]

    if not valid_results:
        logger.warning("No valid proxy evaluation results!")
        return {"dsc": 0.0, "bdsc": 0.0, "hd95": float("inf"), "error": "all_samples_failed"}

    mean_dsc = np.mean([r.dsc for r in valid_results])
    mean_bdsc = np.mean([r.bdsc for r in valid_results])
    mean_hd95 = np.mean([r.hd95 for r in valid_results])

    logger.info(
        f"Proxy eval complete: DSC={mean_dsc:.4f}, bDSC={mean_bdsc:.4f}, HD95={mean_hd95:.2f} "
        f"({len(valid_results)}/{len(samples)} samples)"
    )

    return {
        "dsc": float(mean_dsc),
        "bdsc": float(mean_bdsc),
        "hd95": float(mean_hd95),
        "num_valid": len(valid_results),
        "num_total": len(samples),
    }


def save_proxy_metrics(
    metrics: Dict[str, float],
    epoch: int,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Save proxy evaluation metrics to JSON file."""
    if logger is None:
        logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / f"proxy_eval_epoch_{epoch:04d}.json"

    data = {
        "epoch": epoch,
        "metrics": metrics,
    }

    with open(metrics_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Proxy metrics saved to: {metrics_file}")
    return metrics_file
