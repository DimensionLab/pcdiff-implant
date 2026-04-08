#!/usr/bin/env python3
"""
Multi-GPU distributed inference for PCDiff completion model.

This script shards the test set across multiple GPUs using DistributedDataParallel,
ensuring deterministic non-overlapping splits and concurrency-safe output directories.

Usage:
    torchrun --nproc_per_node=N pcdiff/test_completion_distributed.py \
        --path datasets/SkullBreak/test.csv \
        --dataset SkullBreak \
        --model path/to/checkpoint.pth \
        --eval_path path/to/output \
        --sampling_method ddim \
        --sampling_steps 50 \
        --num_ens 5

The script supports both DDIM (fast, ~50 steps) and DDPM (full, 1000 steps) sampling.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Add pcdiff directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.skullbreak_data import SkullBreakDataset
from datasets.skullfix_data import SkullFixDataset
from test_completion import Model, get_betas


def setup_logging(output_dir: str, rank: int) -> logging.Logger:
    """Setup logging for distributed inference."""
    logger = logging.getLogger(f"pcdiff_inference_rank{rank}")
    logger.setLevel(logging.INFO)

    # Console handler (rank 0 only shows INFO+, others show WARNING+)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    console_formatter = logging.Formatter(f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (all ranks log to their own file)
    if rank == 0:
        log_file = os.path.join(output_dir, "inference.log")
    else:
        log_file = os.path.join(output_dir, f"inference_rank{rank}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def get_distributed_info() -> Tuple[int, int, int]:
    """Get distributed training info from environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank


def shard_dataset_indices(total_samples: int, world_size: int, rank: int) -> List[int]:
    """
    Deterministically shard dataset indices across ranks.

    Ensures:
    - No overlaps: Each sample is assigned to exactly one rank
    - No gaps: All samples are covered
    - Deterministic: Same indices for same (world_size, rank) pair

    Args:
        total_samples: Total number of samples in the dataset
        world_size: Number of distributed ranks
        rank: Current rank

    Returns:
        List of indices assigned to this rank
    """
    # Calculate samples per rank
    samples_per_rank = total_samples // world_size
    remainder = total_samples % world_size

    # Distribute remainder samples to first 'remainder' ranks
    if rank < remainder:
        start_idx = rank * (samples_per_rank + 1)
        end_idx = start_idx + samples_per_rank + 1
    else:
        start_idx = rank * samples_per_rank + remainder
        end_idx = start_idx + samples_per_rank

    return list(range(start_idx, end_idx))


def create_output_directory(base_dir: str, sample_name: str, rank: int) -> str:
    """
    Create a concurrency-safe output directory for a sample.

    Uses atomic directory creation to avoid race conditions.
    """
    output_dir = os.path.join(base_dir, sample_name)

    # Use os.makedirs with exist_ok=True (atomic on most filesystems)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def save_sample_outputs(
    output_dir: str,
    input_points: np.ndarray,
    sample_points: np.ndarray,
    shift: np.ndarray,
    scale: np.ndarray,
    rank: int,
) -> None:
    """
    Save inference outputs for a single sample.

    Outputs:
        input.npy: Original defective skull points
        sample.npy: Generated implant point clouds (num_ens, 3072, 3)
        shift.npy: Normalization shift vector
        scale.npy: Normalization scale value
        metadata.json: Inference metadata (rank, timestamp)
    """
    np.save(os.path.join(output_dir, "input.npy"), input_points)
    np.save(os.path.join(output_dir, "sample.npy"), sample_points)
    np.save(os.path.join(output_dir, "shift.npy"), shift)
    np.save(os.path.join(output_dir, "scale.npy"), scale)

    # Save metadata for debugging/auditing
    metadata = {
        "processed_by_rank": rank,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def get_dataset(path: str, num_points: int, num_nn: int, dataset: str) -> torch.utils.data.Dataset:
    """Load the appropriate dataset based on dataset name."""
    if dataset == "SkullBreak":
        return SkullBreakDataset(path=path, num_points=num_points, num_nn=num_nn, norm_mode="shape_bbox", eval=True)
    else:
        return SkullFixDataset(path=path, num_points=num_points, num_nn=num_nn, norm_mode="shape_bbox", eval=True)


def run_distributed_inference(opt, logger: logging.Logger) -> dict:
    """
    Run distributed inference on the test set.

    Returns:
        dict with inference statistics (samples processed, timing, etc.)
    """
    world_size, rank, local_rank = get_distributed_info()
    is_distributed = world_size > 1

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Initialize process group for distributed inference
    if is_distributed:
        dist.init_process_group(
            backend=opt.dist_backend,
            init_method="env://",
        )
        logger.info(f"Initialized distributed inference: world_size={world_size}, rank={rank}, local_rank={local_rank}")

    # Load full dataset
    full_dataset = get_dataset(opt.path, opt.num_points, opt.num_nn, opt.dataset)
    total_samples = len(full_dataset)

    if rank == 0:
        logger.info(f"Total test samples: {total_samples}")

    # Shard dataset indices deterministically
    my_indices = shard_dataset_indices(total_samples, world_size, rank)
    logger.info(f"Rank {rank}: Processing {len(my_indices)} samples (indices {my_indices[0]} to {my_indices[-1]})")

    # Create subset for this rank
    my_dataset = Subset(full_dataset, my_indices)

    # Create dataloader (no need for DistributedSampler since we already sharded)
    dataloader = DataLoader(
        my_dataset,
        batch_size=1,  # Process one sample at a time (ensembling handles batching internally)
        shuffle=False,  # Maintain deterministic order
        num_workers=min(opt.workers, 4),  # Limit workers per rank
        drop_last=False,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Load model
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(
        opt,
        betas,
        opt.loss_type,
        opt.model_mean_type,
        opt.model_var_type,
        width_mult=opt.width_mult,
        vox_res_mult=opt.vox_res_mult,
    )
    model = model.to(device)
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(opt.model, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]

    # Handle various checkpoint formats (DDP, torch.compile, or both)
    first_key = list(state_dict.keys())[0]
    new_state_dict = {}

    for k, v in state_dict.items():
        new_key = k
        # Handle DDP + torch.compile (model.module._orig_mod.*)
        if "module._orig_mod." in new_key:
            new_key = new_key.replace("module._orig_mod.", "")
        # Handle DDP only (model.module.*)
        elif ".module." in new_key:
            new_key = new_key.replace(".module.", ".")
        # Handle torch.compile only (model._orig_mod.*)
        elif "._orig_mod." in new_key:
            new_key = new_key.replace("._orig_mod.", ".")
        new_state_dict[new_key] = v

    state_dict = new_state_dict

    if first_key != list(state_dict.keys())[0]:
        logger.info(f"Transformed checkpoint keys: '{first_key}' -> '{list(state_dict.keys())[0]}'")

    model.load_state_dict(state_dict)
    logger.info(f"Loaded model checkpoint from: {opt.model}")

    # Create output directory structure
    output_base = os.path.join(opt.eval_path, "syn")
    os.makedirs(output_base, exist_ok=True)

    # Run inference
    inference_stats = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "samples_assigned": len(my_indices),
        "samples_processed": 0,
        "samples_failed": 0,
        "total_time_seconds": 0,
        "per_sample_times": [],
        "failed_samples": [],
    }

    start_time = time.time()

    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Rank {rank} inference",
        disable=(rank != 0 and not opt.verbose),
    )

    with torch.no_grad():
        for batch_idx, data in progress_bar:
            sample_start = time.time()

            try:
                # Get sample data
                pc = data["train_points"].transpose(1, 2).to(device)
                name = data["name"][0]

                # Extract sample name from path
                # e.g., "datasets/SkullBreak/defective_skull/bilateral/086_surf.npy" -> "bilateral086_surf"
                path_parts = name.split("/")
                defect_type = path_parts[-2]
                skull_id = path_parts[-1].split(".")[0]
                sample_name = f"{defect_type}{skull_id}"

                # Create output directory (concurrency-safe)
                sample_output_dir = create_output_directory(output_base, sample_name, rank)

                # Skip if already processed (allows for resume)
                sample_file = os.path.join(sample_output_dir, "sample.npy")
                if os.path.exists(sample_file) and not opt.overwrite:
                    logger.debug(f"Skipping {sample_name} (already processed)")
                    inference_stats["samples_processed"] += 1
                    continue

                # Ensemble: repeat input for multiple samples
                ensemble_num = opt.num_ens
                pc_batch = pc.repeat(ensemble_num, 1, 1)
                noise_shape = torch.Size([ensemble_num, 3, opt.num_nn])

                # Generate samples
                sample = model.gen_samples(
                    pc_batch,
                    noise_shape,
                    device,
                    clip_denoised=False,
                    sampling_method=opt.sampling_method,
                    sampling_steps=opt.sampling_steps,
                )
                sample = sample.detach().cpu()

                # Convert to numpy (shape: [num_ens, 3, num_points] -> [num_ens, num_points, 3])
                sample_np = sample.numpy().transpose(0, 2, 1)

                # Save outputs
                save_sample_outputs(
                    sample_output_dir,
                    input_points=data["train_points"].numpy(),
                    sample_points=sample_np,
                    shift=data["shift"].numpy(),
                    scale=data["scale"].numpy(),
                    rank=rank,
                )

                sample_time = time.time() - sample_start
                inference_stats["per_sample_times"].append(sample_time)
                inference_stats["samples_processed"] += 1

                if rank == 0:
                    progress_bar.set_postfix(
                        {
                            "sample": sample_name,
                            "time": f"{sample_time:.1f}s",
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to process sample {batch_idx}: {e}")
                inference_stats["samples_failed"] += 1
                inference_stats["failed_samples"].append(
                    {
                        "batch_idx": batch_idx,
                        "error": str(e),
                    }
                )

    inference_stats["total_time_seconds"] = time.time() - start_time

    # Log summary
    logger.info(
        f"Rank {rank} completed: {inference_stats['samples_processed']}/{inference_stats['samples_assigned']} samples "
        f"in {inference_stats['total_time_seconds']:.1f}s"
    )

    if inference_stats["samples_failed"] > 0:
        logger.warning(f"Rank {rank} had {inference_stats['samples_failed']} failed samples")

    # Synchronize before gathering stats
    if is_distributed:
        dist.barrier()

    # Gather statistics from all ranks (rank 0 only)
    all_stats = [inference_stats]
    if is_distributed and rank == 0:
        all_stats = [None] * world_size
        dist.gather_object(inference_stats, all_stats, dst=0)
    elif is_distributed:
        dist.gather_object(inference_stats, dst=0)

    # Write summary (rank 0 only)
    if rank == 0:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model_path": opt.model,
            "dataset_path": opt.path,
            "sampling_method": opt.sampling_method,
            "sampling_steps": opt.sampling_steps,
            "num_ensemble": opt.num_ens,
            "world_size": world_size,
            "total_samples": total_samples,
            "rank_stats": all_stats,
        }

        # Calculate totals
        total_processed = sum(s["samples_processed"] for s in all_stats if s)
        total_failed = sum(s["samples_failed"] for s in all_stats if s)
        total_time = max(s["total_time_seconds"] for s in all_stats if s)

        summary["total_processed"] = total_processed
        summary["total_failed"] = total_failed
        summary["wall_clock_time_seconds"] = total_time
        summary["throughput_samples_per_second"] = total_processed / total_time if total_time > 0 else 0

        summary_file = os.path.join(opt.eval_path, "inference_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("=" * 60)
        logger.info("INFERENCE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total samples processed: {total_processed}/{total_samples}")
        logger.info(f"Total failed: {total_failed}")
        logger.info(f"Wall clock time: {total_time:.1f}s")
        logger.info(f"Throughput: {summary['throughput_samples_per_second']:.2f} samples/s")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"Output directory: {output_base}")

    # Cleanup
    if is_distributed:
        dist.destroy_process_group()

    return inference_stats


def verify_outputs(eval_path: str, expected_samples: int, logger: logging.Logger) -> bool:
    """
    Verify that all expected outputs exist.

    Returns:
        True if all outputs exist, False otherwise
    """
    output_base = os.path.join(eval_path, "syn")

    if not os.path.exists(output_base):
        logger.error(f"Output directory does not exist: {output_base}")
        return False

    sample_dirs = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]

    missing = []
    incomplete = []

    for sample_dir in sample_dirs:
        sample_path = os.path.join(output_base, sample_dir)
        required_files = ["input.npy", "sample.npy", "shift.npy", "scale.npy"]

        for required_file in required_files:
            if not os.path.exists(os.path.join(sample_path, required_file)):
                incomplete.append((sample_dir, required_file))

    if len(sample_dirs) < expected_samples:
        logger.warning(f"Only {len(sample_dirs)}/{expected_samples} samples have output directories")

    if incomplete:
        logger.warning(
            f"Incomplete samples: {incomplete[:10]}..." if len(incomplete) > 10 else f"Incomplete samples: {incomplete}"
        )

    logger.info(f"Verification: {len(sample_dirs)} sample directories found")

    return len(sample_dirs) >= expected_samples and len(incomplete) == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU distributed PCDiff inference")

    # Data paths
    parser.add_argument("--path", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--dataset", type=str, required=True, choices=["SkullBreak", "SkullFix"], help="Dataset name")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--eval_path", type=str, required=True, help="Output directory for results")

    # Sampling parameters
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="ddim",
        choices=["ddpm", "ddim", "dpm_solver"],
        help="Sampling method (ddim for fast, ddpm for full)",
    )
    parser.add_argument(
        "--sampling_steps", type=int, default=50, help="Number of sampling steps (50 for DDIM, 1000 for DDPM)"
    )
    parser.add_argument("--num_ens", type=int, default=5, help="Number of ensemble samples")

    # Model parameters (must match training)
    parser.add_argument("--num_points", type=int, default=30720, help="Total points (skull + implant)")
    parser.add_argument("--num_nn", type=int, default=3072, help="Number of implant points to generate")
    parser.add_argument("--nc", type=int, default=3, help="Point dimension (3 for x,y,z)")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--schedule_type", type=str, default="linear")
    parser.add_argument("--time_num", type=int, default=1000)
    parser.add_argument("--attention", type=eval, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--model_mean_type", type=str, default="eps")
    parser.add_argument("--model_var_type", type=str, default="fixedsmall")
    parser.add_argument("--vox_res_mult", type=float, default=1.0)
    parser.add_argument("--width_mult", type=float, default=1.0)

    # Distributed parameters
    parser.add_argument(
        "--dist-backend", type=str, default="nccl", help="Distributed backend (nccl for GPU, gloo for CPU)"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")

    # Runtime options
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--verbose", action="store_true", help="Show progress on all ranks")
    parser.add_argument("--verify", action="store_true", help="Verify outputs after inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    opt = parse_args()

    # Get distributed info early
    world_size, rank, local_rank = get_distributed_info()

    # Create output directory (rank 0 creates, others wait)
    if rank == 0:
        os.makedirs(opt.eval_path, exist_ok=True)

    # Brief delay to ensure directory exists before other ranks proceed
    if world_size > 1:
        time.sleep(0.5)

    # Setup logging
    logger = setup_logging(opt.eval_path, rank)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("PCDiff Multi-GPU Distributed Inference")
        logger.info("=" * 60)
        logger.info(f"Model: {opt.model}")
        logger.info(f"Dataset: {opt.path}")
        logger.info(f"Output: {opt.eval_path}")
        logger.info(f"Sampling: {opt.sampling_method} ({opt.sampling_steps} steps)")
        logger.info(f"Ensemble: {opt.num_ens} samples")
        logger.info(f"World size: {world_size}")
        logger.info("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # Run inference
    try:
        stats = run_distributed_inference(opt, logger)

        # Verify outputs (rank 0 only)
        if opt.verify and rank == 0:
            # Load dataset to get expected sample count
            full_dataset = get_dataset(opt.path, opt.num_points, opt.num_nn, opt.dataset)
            expected = len(full_dataset)

            if verify_outputs(opt.eval_path, expected, logger):
                logger.info("Verification PASSED: All expected outputs exist")
            else:
                logger.warning("Verification FAILED: Some outputs are missing or incomplete")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
