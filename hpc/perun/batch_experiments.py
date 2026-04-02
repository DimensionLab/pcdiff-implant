#!/usr/bin/env python3
"""
batch_experiments.py — Pre-defined experiment batch for PERUN HPC.

Since PERUN compute nodes have no internet, we pre-generate experiment
configurations here and submit them as Slurm array jobs. No LLM calls needed.

Strategy: Based on RunPod results (best val_loss=1.0309, exp_0014: cosine + AdamW + grad clip),
explore promising directions with H200's 143GB VRAM:
  1. Larger batch sizes (8, 16, 32) — was limited to 4 on 24GB
  2. EMA (exponential moving average) — consistently attempted but failed on syntax
  3. Learning rate exploration with cosine annealing
  4. Width/resolution multipliers for bigger models
  5. Longer training (H200 is free, so use more time)

Usage:
    python batch_experiments.py generate   # Create experiment configs
    python batch_experiments.py submit     # Submit Slurm array job
    python batch_experiments.py results    # Summarize results
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
EXPERIMENT_DIR = PROJECT_DIR / "autoresearch" / "perun_experiments"
RESULTS_DIR = PROJECT_DIR / "autoresearch" / "results" / "perun"

# Best known config from RunPod (exp_0014): cosine schedule, AdamW, grad_clip=1.0, lr=2e-4
EXPERIMENTS = [
    # === Batch size scaling (H200 has 143GB, can go much bigger) ===
    {
        "id": "perun_001",
        "name": "batch8_baseline",
        "desc": "Baseline config with batch_size=8 (2x RunPod best)",
        "changes": {"BATCH_SIZE": 8},
        "time_budget": 3600,  # 1 hour
    },
    {
        "id": "perun_002",
        "name": "batch16_baseline",
        "desc": "Batch size 16 — leverage H200 VRAM",
        "changes": {"BATCH_SIZE": 16},
        "time_budget": 3600,
    },
    {
        "id": "perun_003",
        "name": "batch32_baseline",
        "desc": "Batch size 32 — large batch training",
        "changes": {"BATCH_SIZE": 32, "LEARNING_RATE": 4e-4},  # scale LR with batch
        "time_budget": 3600,
    },
    # === Learning rate with cosine annealing ===
    {
        "id": "perun_004",
        "name": "lr1e3_cosine_anneal",
        "desc": "Higher LR with cosine annealing schedule",
        "changes": {"LEARNING_RATE": 1e-3, "BATCH_SIZE": 8, "LR_GAMMA": 0.998},
        "time_budget": 3600,
    },
    {
        "id": "perun_005",
        "name": "lr5e4_batch16",
        "desc": "LR 5e-4 with batch 16",
        "changes": {"LEARNING_RATE": 5e-4, "BATCH_SIZE": 16},
        "time_budget": 3600,
    },
    # === Wider model (more parameters) ===
    {
        "id": "perun_006",
        "name": "wide1.5_batch8",
        "desc": "1.5x wider model, batch 8",
        "changes": {"WIDTH_MULT": 1.5, "BATCH_SIZE": 8, "LEARNING_RATE": 1.5e-4},
        "time_budget": 5400,  # 1.5 hours (bigger model, needs more time)
    },
    {
        "id": "perun_007",
        "name": "wide2.0_batch8",
        "desc": "2x wider model, batch 8",
        "changes": {"WIDTH_MULT": 2.0, "BATCH_SIZE": 8, "LEARNING_RATE": 1e-4},
        "time_budget": 7200,  # 2 hours
    },
    # === Higher voxel resolution ===
    {
        "id": "perun_008",
        "name": "voxres1.5_batch8",
        "desc": "1.5x voxel resolution, batch 8",
        "changes": {"VOX_RES_MULT": 1.5, "BATCH_SIZE": 8},
        "time_budget": 5400,
    },
    # === Combined best ideas ===
    {
        "id": "perun_009",
        "name": "combined_wide_batch16",
        "desc": "Wide 1.5x + batch 16 + LR 3e-4",
        "changes": {"WIDTH_MULT": 1.5, "BATCH_SIZE": 16, "LEARNING_RATE": 3e-4},
        "time_budget": 7200,
    },
    {
        "id": "perun_010",
        "name": "long_train_batch8",
        "desc": "Standard config, 4-hour training (more epochs)",
        "changes": {"BATCH_SIZE": 8, "EVAL_EVERY_EPOCHS": 100},
        "time_budget": 14400,  # 4 hours
    },
    # === Regularization variants ===
    {
        "id": "perun_011",
        "name": "dropout02_batch8",
        "desc": "Higher dropout (0.2) for regularization",
        "changes": {"DROPOUT": 0.2, "BATCH_SIZE": 8},
        "time_budget": 3600,
    },
    {
        "id": "perun_012",
        "name": "wd1e4_batch8",
        "desc": "Weight decay 1e-4 for regularization",
        "changes": {"WEIGHT_DECAY": 1e-4, "BATCH_SIZE": 8},
        "time_budget": 3600,
    },
    # === AMP with bfloat16 (H200 has great bf16 support) ===
    {
        "id": "perun_013",
        "name": "amp_bf16_batch16",
        "desc": "Mixed precision bf16, batch 16",
        "changes": {"USE_AMP": True, "AMP_DTYPE": "bfloat16", "BATCH_SIZE": 16},
        "time_budget": 3600,
    },
    {
        "id": "perun_014",
        "name": "amp_bf16_batch32_wide",
        "desc": "AMP bf16 + batch 32 + wide 1.5x",
        "changes": {"USE_AMP": True, "AMP_DTYPE": "bfloat16", "BATCH_SIZE": 32, "WIDTH_MULT": 1.5, "LEARNING_RATE": 3e-4},
        "time_budget": 7200,
    },
    # === Data augmentation ===
    {
        "id": "perun_015",
        "name": "augment_batch8",
        "desc": "Enable rotation augmentation",
        "changes": {"AUGMENT": True, "BATCH_SIZE": 8},
        "time_budget": 3600,
    },
    {
        "id": "perun_016",
        "name": "linear_schedule_batch8",
        "desc": "Linear noise schedule instead of cosine",
        "changes": {"SCHEDULE_TYPE": "linear", "BATCH_SIZE": 8},
        "time_budget": 3600,
    },
]


def generate_configs():
    """Generate experiment config JSON files."""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    for exp in EXPERIMENTS:
        config_path = EXPERIMENT_DIR / f"{exp['id']}.json"
        config_path.write_text(json.dumps(exp, indent=2))
        print(f"  {exp['id']}: {exp['name']} -> {config_path}")
    
    # Write manifest
    manifest = EXPERIMENT_DIR / "manifest.json"
    manifest.write_text(json.dumps(
        {"experiments": [e["id"] for e in EXPERIMENTS], "total": len(EXPERIMENTS)},
        indent=2
    ))
    print(f"\nGenerated {len(EXPERIMENTS)} experiment configs in {EXPERIMENT_DIR}")
    print(f"Manifest: {manifest}")


def generate_slurm_script():
    """Generate Slurm array job script."""
    slurm_script = SCRIPT_DIR / "run_batch_experiments.sh"
    content = f"""#!/bin/bash
#SBATCH --job-name=pcdiff-batch
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output={PROJECT_DIR}/autoresearch/results/perun/logs/exp_%a_%j.out
#SBATCH --error={PROJECT_DIR}/autoresearch/results/perun/logs/exp_%a_%j.err
#SBATCH --array=1-{len(EXPERIMENTS)}

set -euo pipefail

# Map array task ID to experiment
EXPERIMENTS=({' '.join(e['id'] for e in EXPERIMENTS)})
EXP_ID="${{EXPERIMENTS[$SLURM_ARRAY_TASK_ID - 1]}}"

echo "=== PCDiff Experiment: $EXP_ID ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff

cd {PROJECT_DIR}

# Build CUDA extensions
if [ -d "pcdiff/modules/functional" ]; then
    cd pcdiff/modules/functional
    python setup.py build_ext --inplace 2>/dev/null || true
    cd {PROJECT_DIR}
fi

nvidia-smi

# Read experiment config
CONFIG_FILE="{EXPERIMENT_DIR}/$EXP_ID.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config: $(cat $CONFIG_FILE)"
echo ""

# Run the experiment with config overrides
python {SCRIPT_DIR}/run_single_experiment.py --config "$CONFIG_FILE"

echo ""
echo "=== Experiment $EXP_ID Complete ==="
echo "End: $(date)"
"""
    slurm_script.write_text(content)
    os.chmod(slurm_script, 0o755)
    print(f"Slurm script: {slurm_script}")
    return slurm_script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "submit", "results"])
    args = parser.parse_args()
    
    if args.action == "generate":
        generate_configs()
        generate_slurm_script()
    elif args.action == "submit":
        print("Submit with: sbatch hpc/perun/run_batch_experiments.sh")
    elif args.action == "results":
        print("Results in:", RESULTS_DIR)
        results_dir = RESULTS_DIR
        if results_dir.exists():
            for f in sorted(results_dir.glob("*.json")):
                data = json.loads(f.read_text())
                val_loss = data.get("metrics", {}).get("val_loss_mean", "N/A")
                print(f"  {f.stem}: val_loss={val_loss}")


if __name__ == "__main__":
    main()
