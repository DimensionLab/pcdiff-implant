#!/bin/bash
#SBATCH --job-name=pcdiff-train
#SBATCH --partition=GPU
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Single-GPU PCDiff training on PERUN H200
#
# Usage:
#   sbatch hpc/perun/train_single_gpu.sh
#   sbatch --time=48:00:00 hpc/perun/train_single_gpu.sh  # override time

set -euo pipefail

# Activate automatic scratch (Lustre fast I/O)
source .activate_scratch

echo "=== PCDiff Single-GPU Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start: $(date)"
echo ""

# Environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff

# Wandb config (key stored in ~/.bashrc or passed via --export)
export WANDB_PROJECT="pcdiff-implant-perun"
export WANDB_RUN_GROUP="single-gpu"
export WANDB_TAGS="perun,h200,single-gpu"
export WANDB_NOTES="SLURM_JOB_ID=$SLURM_JOB_ID node=$SLURM_NODELIST"

# Working directory is now in scratch (automatic)
cd ~/pcdiff-implant || true

# Build CUDA extensions if not already built
if [ -d "pcdiff/modules/functional" ]; then
    cd pcdiff/modules/functional
    python setup.py build_ext --inplace 2>/dev/null || true
    cd ~/pcdiff-implant
fi

# Show GPU info
nvidia-smi

# Run training
python autoresearch/train_pcdiff.py \
    --time-budget 82800
# Results automatically synced to ~/results_job_$SLURM_JOB_ID/ by epilog

echo ""
echo "=== Done ==="
echo "End: $(date)"
