#!/usr/bin/env python3
"""
Quick DDPM-1000 evaluation on 2 samples for hyperparameter search.

Usage:
    python pcdiff/quick_eval_ddpm.py --checkpoint path/to/model.pth --num-samples 2

This evaluates a checkpoint with DDPM-1000 sampling on a small number of samples
to quickly assess model quality during hyperparameter search.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add paths for imports
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "pcdiff"))
sys.path.insert(0, str(ROOT_DIR / "voxelization"))

from proxy_eval import (
    VoxelizationRunner,
    load_proxy_subset,
    run_pcdiff_inference_on_sample,
    compute_metrics_for_sample,
    ProxySample,
    VOXELIZATION_AVAILABLE,
)
from test_completion import Model, get_betas


def load_model(checkpoint_path: Path, device: torch.device, args) -> Model:
    """Load PCDiff model from checkpoint."""
    betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
    model = Model(args, betas, args.loss_type, args.model_mean_type, args.model_var_type,
                  args.width_mult, args.vox_res_mult)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model_state" in state_dict:
        model_state = state_dict["model_state"]
    elif "model" in state_dict:
        model_state = state_dict["model"]
    elif "state_dict" in state_dict:
        model_state = state_dict["state_dict"]
    else:
        model_state = state_dict

    # Remove DDP/compile prefixes if present
    # The checkpoint may have keys like "model.module.sa_layers..." which should become "model.sa_layers..."
    # or "model._orig_mod.sa_layers..." which should become "model.sa_layers..."
    cleaned_state = {}
    for k, v in model_state.items():
        new_key = k
        # Remove various prefixes that can appear from DDP or torch.compile
        prefixes_to_remove = ["module.", "_orig_mod."]
        changed = True
        while changed:
            changed = False
            for prefix in prefixes_to_remove:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
                # Also handle ".module." and "._orig_mod." anywhere in the key
                if ".module." in new_key:
                    new_key = new_key.replace(".module.", ".")
                    changed = True
                if "._orig_mod." in new_key:
                    new_key = new_key.replace("._orig_mod.", ".")
                    changed = True
        cleaned_state[new_key] = v

    model.load_state_dict(cleaned_state, strict=True)
    model = model.to(device)
    model.eval()

    return model


def run_quick_eval(
    checkpoint_path: Path,
    num_samples: int,
    num_ens: int,
    device: torch.device,
    args,
    logger: logging.Logger,
) -> dict:
    """Run quick DDPM-1000 evaluation on a few samples."""

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device, args)

    # Load voxelization runner
    vox_config = ROOT_DIR / "voxelization" / "configs" / "gen_skullbreak.yaml"
    vox_checkpoint = ROOT_DIR / "voxelization" / "checkpoints" / "model_best.pt"

    if not VOXELIZATION_AVAILABLE:
        logger.error("Voxelization dependencies not available")
        return {"error": "voxelization_unavailable"}

    logger.info("Initializing voxelization runner...")
    vox_runner = VoxelizationRunner(vox_config, vox_checkpoint, device)

    # Load proxy subset and take first N samples
    subset_path = ROOT_DIR / "pcdiff" / "proxy_validation_subset.json"
    all_samples = load_proxy_subset(subset_path, ROOT_DIR)
    samples = all_samples[:num_samples]

    logger.info(f"Running DDPM-1000 evaluation on {len(samples)} samples with num_ens={num_ens}")

    results = []
    total_time = 0

    for i, sample in enumerate(samples):
        logger.info(f"Processing sample {i+1}/{len(samples)}: case {sample.case_id}")

        start_time = time.time()

        # Load defective skull points
        defective_points = np.load(sample.defective_npy).astype(np.float32)

        # Run PCDiff inference with DDPM-1000
        with torch.no_grad():
            ensemble_points, shift, scale = run_pcdiff_inference_on_sample(
                model,
                defective_points,
                args.num_points,
                args.num_nn,
                num_ens,
                device,
                sampling_method="ddpm",
                sampling_steps=1000,
            )

        # Compute metrics
        result = compute_metrics_for_sample(sample, ensemble_points, vox_runner, args.num_nn)

        elapsed = time.time() - start_time
        total_time += elapsed

        results.append({
            "case_id": sample.case_id,
            "dsc": result.dsc,
            "bdsc": result.bdsc,
            "hd95": result.hd95,
            "time_s": elapsed,
            "error": result.error,
        })

        logger.info(f"  DSC={result.dsc:.4f}, bDSC={result.bdsc:.4f}, HD95={result.hd95:.2f} ({elapsed:.1f}s)")

    # Compute aggregated metrics
    valid_results = [r for r in results if r["error"] is None]

    if valid_results:
        mean_dsc = np.mean([r["dsc"] for r in valid_results])
        mean_bdsc = np.mean([r["bdsc"] for r in valid_results])
        mean_hd95 = np.mean([r["hd95"] for r in valid_results])
    else:
        mean_dsc = mean_bdsc = 0.0
        mean_hd95 = float("inf")

    summary = {
        "checkpoint": str(checkpoint_path),
        "sampling_method": "ddpm",
        "sampling_steps": 1000,
        "num_ens": num_ens,
        "num_samples": len(samples),
        "num_valid": len(valid_results),
        "total_time_s": total_time,
        "mean_dsc": mean_dsc,
        "mean_bdsc": mean_bdsc,
        "mean_hd95": mean_hd95,
        "per_sample": results,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"\n=== DDPM-1000 Evaluation Summary ===")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Samples: {len(valid_results)}/{len(samples)} valid")
    logger.info(f"Mean DSC: {mean_dsc:.4f}")
    logger.info(f"Mean bDSC: {mean_bdsc:.4f}")
    logger.info(f"Mean HD95: {mean_hd95:.2f}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"=====================================\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Quick DDPM-1000 evaluation")

    # Required
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # Evaluation settings
    parser.add_argument("--num-samples", type=int, default=2, help="Number of samples to evaluate")
    parser.add_argument("--num-ens", type=int, default=1, help="Ensemble size")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")

    # Model settings (should match training)
    parser.add_argument("--sampling-method", type=str, default="ddpm", help="Sampling method")
    parser.add_argument("--sampling-steps", type=int, default=1000, help="Sampling steps")
    parser.add_argument("--num-points", type=int, default=30720, help="Total points")
    parser.add_argument("--num-nn", type=int, default=3072, help="Implant points")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels")
    parser.add_argument("--beta-start", type=float, default=0.0001)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--schedule-type", type=str, default="linear")
    parser.add_argument("--time-num", type=int, default=1000)
    parser.add_argument("--attention", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--loss-type", type=str, default="mse")
    parser.add_argument("--model-mean-type", type=str, default="eps")
    parser.add_argument("--model-var-type", type=str, default="fixedsmall")
    parser.add_argument("--vox-res-mult", type=float, default=1.0)
    parser.add_argument("--width-mult", type=float, default=1.0)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Run evaluation
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    summary = run_quick_eval(
        checkpoint_path,
        args.num_samples,
        args.num_ens,
        device,
        args,
        logger,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ROOT_DIR / "pcdiff" / "eval" / f"quick_ddpm1000_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    main()
