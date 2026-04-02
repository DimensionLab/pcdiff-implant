"""
prepare_pcdiff.py — One-time data preparation and evaluation utilities for PCDiff autoresearch.

This file is NOT modified by the AI agent. It provides:
  1. SkullBreak dataset loading (train + validation split)
  2. Chamfer Distance computation (proxy metric for fast iteration)
  3. DDIM-based fast inference for evaluation
  4. Result logging and comparison utilities
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration (adjust paths for RunPod vs local)
# ---------------------------------------------------------------------------

# Auto-detect environment
if os.path.exists("/workspace/pcdiff-implant"):
    PROJECT_ROOT = Path("/workspace/pcdiff-implant")
    DATASET_ROOT = PROJECT_ROOT / "datasets" / "SkullBreak"
elif os.path.exists("/home/mike/pcdiff-implant"):
    PROJECT_ROOT = Path("/home/mike/pcdiff-implant")
    DATASET_ROOT = PROJECT_ROOT / "datasets" / "SkullBreak"
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_ROOT = PROJECT_ROOT / "datasets" / "SkullBreak"

# Point cloud parameters (must match train_pcdiff.py)
NUM_POINTS = 4096  # Total points (defective + implant) — reduced for fast experiments
NUM_NN = 1024  # Implant points
SV_POINTS = NUM_POINTS - NUM_NN  # 3072 skull points

# Evaluation parameters
EVAL_CASES = 10  # Number of validation cases for proxy eval
DDIM_STEPS = 50  # DDIM sampling steps for fast eval

DEFECT_TYPES = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _nrrd_to_npy_path(path: str) -> str:
    """Convert .nrrd path to the preprocessed _surf.npy path."""
    if path.endswith(".nrrd"):
        stem = path[:-5]  # Remove .nrrd
        npy_path = stem + "_surf.npy"
        if os.path.exists(npy_path):
            return npy_path
    # Also check if .npy version exists directly
    npy_direct = path.rsplit(".", 1)[0] + ".npy" if "." in os.path.basename(path) else path + ".npy"
    if os.path.exists(npy_direct):
        return npy_direct
    # Return original path (np.load will fail with a clear error if neither exists)
    return path


def load_csv_entries(csv_path: str) -> list:
    """Load SkullBreak CSV and return list of (defective_path, implant_path) tuples.

    CSV paths are relative to PROJECT_ROOT/pcdiff/ (e.g. 'datasets/SkullBreak/...').
    """
    pcdiff_dir = str(PROJECT_ROOT)
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            defective = row[0].strip()
            implant = row[1].strip()
            if not os.path.isabs(defective):
                defective = os.path.join(pcdiff_dir, defective)
                implant = os.path.join(pcdiff_dir, implant)
            defective = _nrrd_to_npy_path(defective)
            implant = _nrrd_to_npy_path(implant)
            entries.append((defective, implant))
    return entries


def load_point_cloud(path: str, num_points: int) -> np.ndarray:
    """Load and randomly subsample a point cloud to num_points."""
    pc = np.load(path)
    n = pc.shape[0]
    idx = np.random.randint(0, n, num_points)
    return pc[idx, :]


def normalize_point_cloud(pc: np.ndarray) -> tuple:
    """Normalize point cloud using shape_bbox mode. Returns (normalized_pc, shift, scale)."""
    pc_max = pc.max(axis=0)
    pc_min = pc.min(axis=0)
    shift = ((pc_min + pc_max) / 2).reshape(1, 3)
    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
    scale = scale / 3  # Match original scaling
    return (pc - shift) / scale, shift, scale


def get_train_entries() -> list:
    """Get training data entries."""
    train_csv = str(DATASET_ROOT / "train.csv")
    return load_csv_entries(train_csv)


def get_test_entries() -> list:
    """Get test data entries."""
    test_csv = str(DATASET_ROOT / "test.csv")
    return load_csv_entries(test_csv)


def get_eval_subset(n: int = EVAL_CASES) -> list:
    """Get a fixed subset of test entries for proxy evaluation."""
    entries = get_test_entries()
    # Use deterministic selection for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.choice(len(entries), size=min(n, len(entries)), replace=False)
    return [entries[i] for i in sorted(indices)]


def prepare_eval_batch(entries: list, device: torch.device) -> tuple:
    """
    Prepare a batch of evaluation data.
    Returns (partial_x, gt_implant_points, shifts, scales) where:
      - partial_x: [B, 3, SV_POINTS] defective skull points (normalized)
      - gt_implant_points: list of [N, 3] ground truth implant arrays (unnormalized)
      - shifts, scales: normalization parameters for denormalization
    """
    partial_list = []
    gt_list = []
    shifts = []
    scales = []

    for defective_path, implant_path in entries:
        # Load defective skull
        defective_pc = load_point_cloud(defective_path, SV_POINTS)
        # Load ground truth implant (full, unnormalized)
        gt_implant = np.load(implant_path)

        # Normalize defective skull
        norm_defective, shift, scale = normalize_point_cloud(defective_pc)
        partial_list.append(torch.from_numpy(norm_defective).float().T)  # [3, SV_POINTS]
        gt_list.append(gt_implant)
        shifts.append((shift, scale))
        scales.append(scale)

    partial_x = torch.stack(partial_list).to(device)  # [B, 3, SV_POINTS]
    return partial_x, gt_list, shifts, scales


# ---------------------------------------------------------------------------
# Chamfer Distance (proxy metric)
# ---------------------------------------------------------------------------


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Chamfer Distance between two point clouds.
    pred: [N, 3], gt: [M, 3]
    Returns: mean of forward + backward nearest neighbor distances.
    """
    # Forward: for each point in pred, find nearest in gt
    # Using einsum for efficiency without scipy
    diff_fwd = pred[:, None, :] - gt[None, :, :]  # [N, M, 3]
    dist_fwd = np.sum(diff_fwd**2, axis=-1)  # [N, M]
    min_fwd = np.min(dist_fwd, axis=1)  # [N]

    # Backward: for each point in gt, find nearest in pred
    min_bwd = np.min(dist_fwd, axis=0)  # [M]

    return float(np.mean(min_fwd) + np.mean(min_bwd))


def chamfer_distance_batch(preds: list, gts: list) -> dict:
    """
    Compute Chamfer Distance for a batch of predictions vs ground truths.
    Returns dict with mean, std, and per-case distances.
    """
    distances = []
    for pred, gt in zip(preds, gts):
        cd = chamfer_distance(pred, gt)
        distances.append(cd)

    return {
        "chamfer_mean": float(np.mean(distances)),
        "chamfer_std": float(np.std(distances)),
        "chamfer_per_case": distances,
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate_model_loss(model, dataloader, device: torch.device, num_batches: int = 10) -> dict:
    """
    Evaluate model using validation loss (MSE on noise prediction).

    This is the PRIMARY proxy metric for autoresearch. It's fast (~30s),
    directly measures model quality, and doesn't depend on sampling method.

    Note: DDIM-50 produces very low quality output and even DDPM-1000 yields
    bulky meshes with the current model. Training loss is a more reliable
    signal for comparing experimental changes.

    Args:
        model: The Model instance (from train_pcdiff.py)
        dataloader: Validation data loader
        device: CUDA device
        num_batches: Number of batches to evaluate (0 = all)

    Returns:
        dict with val_loss_mean, val_loss_std, eval_time_sec
    """
    model.eval()
    t0 = time.time()
    losses = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if num_batches > 0 and i >= num_batches:
                break
            pc_in = data["train_points"].transpose(1, 2)
            if torch.cuda.is_available():
                pc_in = pc_in.to(device, non_blocking=True)
            batch_losses = model.get_loss_iter(pc_in)
            losses.extend(batch_losses.detach().cpu().tolist())

    metrics = {
        "val_loss_mean": float(np.mean(losses)),
        "val_loss_std": float(np.std(losses)),
        "val_loss_median": float(np.median(losses)),
        "val_loss_count": len(losses),
        "eval_time_sec": time.time() - t0,
    }

    model.train()
    return metrics


def evaluate_model_noise_prediction(
    model, dataloader, device: torch.device, timesteps_to_check: list = None, num_batches: int = 5
) -> dict:
    """
    Evaluate noise prediction accuracy at specific timesteps.

    Measures how well the model predicts noise at different diffusion stages:
    - Early timesteps (t~50): fine detail recovery
    - Mid timesteps (t~500): structural prediction
    - Late timesteps (t~950): coarse shape from noise

    Args:
        model: The Model instance
        dataloader: Validation data loader
        device: CUDA device
        timesteps_to_check: specific timesteps to evaluate (default: [50, 250, 500, 750, 950])
        num_batches: batches per timestep

    Returns:
        dict with per-timestep MSE and overall summary
    """
    if timesteps_to_check is None:
        timesteps_to_check = [50, 250, 500, 750, 950]

    model.eval()
    t0 = time.time()
    results = {}

    with torch.no_grad():
        for t_val in timesteps_to_check:
            t_losses = []
            for i, data in enumerate(dataloader):
                if i >= num_batches:
                    break
                pc_in = data["train_points"].transpose(1, 2)
                if torch.cuda.is_available():
                    pc_in = pc_in.to(device, non_blocking=True)

                B = pc_in.shape[0]
                t = torch.full((B,), t_val, dtype=torch.int64, device=device)

                # Get noise prediction loss at this specific timestep
                noise = torch.randn(pc_in[:, :, SV_POINTS:].shape, device=device)
                x_noisy = torch.cat(
                    [
                        pc_in[:, :, :SV_POINTS],
                        model.diffusion.q_sample(x_start=pc_in[:, :, SV_POINTS:], t=t, noise=noise),
                    ],
                    dim=-1,
                )
                pred_noise = model._denoise(x_noisy, t)[:, :, SV_POINTS:]
                mse = ((noise - pred_noise) ** 2).mean(dim=[1, 2])
                t_losses.extend(mse.cpu().tolist())

            results[f"t{t_val}_mse"] = float(np.mean(t_losses))

    results["noise_pred_mean"] = float(np.mean([v for k, v in results.items() if k.endswith("_mse")]))
    results["eval_time_sec"] = time.time() - t0

    model.train()
    return results


def evaluate_model(model, device: torch.device, entries: list = None, ddim_steps: int = DDIM_STEPS) -> dict:
    """
    Run generation-based evaluation (Chamfer Distance).

    WARNING: Use sparingly! DDIM-50 produces low quality output. This should
    only be used for milestone checks with DDPM-1000, not per-experiment eval.
    For per-experiment eval, use evaluate_model_loss() instead.

    Args:
        model: The Model instance (from train_pcdiff.py)
        device: CUDA device
        entries: eval entries (default: get_eval_subset())
        ddim_steps: number of DDIM sampling steps

    Returns:
        dict with chamfer_mean, chamfer_std, eval_time_sec, per_case details
    """
    if entries is None:
        entries = get_eval_subset()

    model.eval()
    t0 = time.time()

    partial_x, gt_list, shifts_scales, _ = prepare_eval_batch(entries, device)
    B = partial_x.shape[0]

    with torch.no_grad():
        gen_shape = (B, 3, NUM_NN)
        generated = model.gen_samples(
            partial_x,
            gen_shape,
            device,
            clip_denoised=False,
            sampling_method="ddim",
            sampling_steps=ddim_steps,
        )
        gen_implant = generated[:, :, SV_POINTS:].detach().cpu().numpy()

    pred_list = []
    for i in range(B):
        shift, scale = shifts_scales[i]
        pred_pc = gen_implant[i].T * scale + shift
        pred_list.append(pred_pc)

    metrics = chamfer_distance_batch(pred_list, gt_list)
    metrics["eval_time_sec"] = time.time() - t0
    metrics["ddim_steps"] = ddim_steps
    metrics["num_cases"] = B

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def log_experiment(experiment_id: str, metrics: dict, config: dict, accepted: bool, diff: str = "") -> None:
    """Log experiment results to JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)

    result = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": metrics,
        "config_summary": config,
        "accepted": accepted,
        "diff": diff,
    }

    log_file = RESULTS_DIR / "experiments.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def get_best_metric() -> float:
    """Get the best Chamfer Distance from experiment history."""
    log_file = RESULTS_DIR / "experiments.jsonl"
    if not log_file.exists():
        return float("inf")

    best = float("inf")
    with open(log_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("accepted"):
                cd = r["metrics"].get("chamfer_mean", float("inf"))
                best = min(best, cd)
    return best


def load_experiment_history() -> list:
    """Load full experiment history."""
    log_file = RESULTS_DIR / "experiments.jsonl"
    if not log_file.exists():
        return []
    with open(log_file) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Main: verify data access
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset root: {DATASET_ROOT}")

    train_csv = DATASET_ROOT / "train.csv"
    test_csv = DATASET_ROOT / "test.csv"

    print(f"Train CSV exists: {train_csv.exists()}")
    print(f"Test CSV exists: {test_csv.exists()}")

    if train_csv.exists():
        train_entries = get_train_entries()
        print(f"Training entries: {len(train_entries)}")

        # Verify first entry is loadable
        d, i = train_entries[0]
        print(f"Sample defective: {d} (exists: {os.path.exists(d)})")
        print(f"Sample implant: {i} (exists: {os.path.exists(i)})")

    if test_csv.exists():
        test_entries = get_test_entries()
        print(f"Test entries: {len(test_entries)}")

        eval_entries = get_eval_subset()
        print(f"Eval subset: {len(eval_entries)} cases")

    print("\nData preparation verified successfully.")
