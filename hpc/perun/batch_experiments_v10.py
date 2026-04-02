#!/usr/bin/env python3
"""
batch_experiments_v10.py — 1-hour experiments building on V7_002 winner.

V7_002 best config (val_loss=0.7728):
  - WIDTH_MULT=1.5, EMBED_DIM=96 (wider model)
  - BATCH_SIZE=2 (more gradient updates)
  - SCHEDULE_TYPE=sigmoid
  - BETA1=0.9
  - LR_WARMUP_EPOCHS=0
  - lr=2e-4

All experiments capped at 1 hour as agreed.
"""

import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
EXPERIMENT_DIR = PROJECT_DIR / "autoresearch" / "perun_experiments"

# Winner base config
BASE = {
    "LR_WARMUP_EPOCHS": 0,
    "WIDTH_MULT": 1.5,
    "EMBED_DIM": 96,
    "BATCH_SIZE": 2,
    "SCHEDULE_TYPE": "sigmoid",
    "BETA1": 0.9,
}

TIME_BUDGET = 3600  # 1 hour — strictly enforced

EXPERIMENTS = [
    # === LR variants around the winner ===
    {
        "id": "perun_v10_001",
        "name": "lr1e4_1h",
        "desc": "Winner + lower LR=1e-4",
        "changes": {**BASE, "LEARNING_RATE": 1e-4},
        "time_budget": TIME_BUDGET,
    },
    {
        "id": "perun_v10_002",
        "name": "lr3e4_1h",
        "desc": "Winner + higher LR=3e-4",
        "changes": {**BASE, "LEARNING_RATE": 3e-4},
        "time_budget": TIME_BUDGET,
    },
    {
        "id": "perun_v10_003",
        "name": "lr5e4_1h",
        "desc": "Winner + LR=5e-4",
        "changes": {**BASE, "LEARNING_RATE": 5e-4},
        "time_budget": TIME_BUDGET,
    },
    # === Regularization ===
    {
        "id": "perun_v10_004",
        "name": "wd1e4_1h",
        "desc": "Winner + weight_decay=1e-4",
        "changes": {**BASE, "WEIGHT_DECAY": 1e-4},
        "time_budget": TIME_BUDGET,
    },
    {
        "id": "perun_v10_005",
        "name": "dropout02_1h",
        "desc": "Winner + dropout=0.2",
        "changes": {**BASE, "DROPOUT": 0.2},
        "time_budget": TIME_BUDGET,
    },
    {
        "id": "perun_v10_006",
        "name": "no_dropout_1h",
        "desc": "Winner + no dropout",
        "changes": {**BASE, "DROPOUT": 0.0},
        "time_budget": TIME_BUDGET,
    },
    # === Wider model ===
    {
        "id": "perun_v10_007",
        "name": "width2_embed128_1h",
        "desc": "Even wider: WIDTH_MULT=2.0, EMBED_DIM=128",
        "changes": {**BASE, "WIDTH_MULT": 2.0, "EMBED_DIM": 128},
        "time_budget": TIME_BUDGET,
    },
    # === Augmentation ===
    {
        "id": "perun_v10_008",
        "name": "augment_1h",
        "desc": "Winner + data augmentation",
        "changes": {**BASE, "AUGMENT": True},
        "time_budget": TIME_BUDGET,
    },
    # === BS=1 (even more gradient updates) ===
    {
        "id": "perun_v10_009",
        "name": "bs1_1h",
        "desc": "BS=1 for maximum gradient updates per epoch",
        "changes": {**BASE, "BATCH_SIZE": 1},
        "time_budget": TIME_BUDGET,
    },
    # === Mixed precision ===
    {
        "id": "perun_v10_010",
        "name": "amp_bf16_1h",
        "desc": "Winner + AMP bfloat16 for faster training",
        "changes": {**BASE, "USE_AMP": True, "AMP_DTYPE": "bfloat16"},
        "time_budget": TIME_BUDGET,
    },
    # === Cosine schedule (was 2nd best in earlier rounds) ===
    {
        "id": "perun_v10_011",
        "name": "cosine_sched_1h",
        "desc": "Winner base but cosine schedule instead of sigmoid",
        "changes": {**BASE, "SCHEDULE_TYPE": "cosine"},
        "time_budget": TIME_BUDGET,
    },
    # === Higher voxel resolution ===
    {
        "id": "perun_v10_012",
        "name": "voxres1.5_1h",
        "desc": "Winner + VOX_RES_MULT=1.5",
        "changes": {**BASE, "VOX_RES_MULT": 1.5},
        "time_budget": TIME_BUDGET,
    },
]


def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    for exp in EXPERIMENTS:
        config_path = EXPERIMENT_DIR / f"{exp['id']}.json"
        config_path.write_text(json.dumps(exp, indent=2))
        print(f"  {exp['id']}: {exp['name']} — {exp['desc']}")

    # Generate Slurm array script
    exp_ids = ' '.join(e['id'] for e in EXPERIMENTS)
    slurm_script = SCRIPT_DIR / "run_v10_experiments.sh"
    slurm_script.write_text(f"""#!/bin/bash
#SBATCH --job-name=pcdiff-v10
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output={PROJECT_DIR}/autoresearch/results/perun/logs/v10_exp_%a_%j.out
#SBATCH --error={PROJECT_DIR}/autoresearch/results/perun/logs/v10_exp_%a_%j.err
#SBATCH --array=1-{len(EXPERIMENTS)}

set -euo pipefail

EXPERIMENTS=({exp_ids})
EXP_ID="${{EXPERIMENTS[$SLURM_ARRAY_TASK_ID - 1]}}"

echo "=== PCDiff V10 Experiment: $EXP_ID ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

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

CONFIG_FILE="{EXPERIMENT_DIR}/$EXP_ID.json"
python {SCRIPT_DIR}/run_single_experiment.py --config "$CONFIG_FILE"

echo "=== Experiment $EXP_ID Complete ==="
echo "End: $(date)"
""")
    os.chmod(slurm_script, 0o755)
    
    print(f"\nGenerated {len(EXPERIMENTS)} experiments (all 1h budget)")
    print(f"Submit: sbatch {slurm_script}")


if __name__ == "__main__":
    main()
