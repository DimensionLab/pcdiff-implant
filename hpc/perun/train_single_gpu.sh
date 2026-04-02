#!/bin/bash
#SBATCH --job-name=pcdiff-train
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/mamuke588/pcdiff/logs/train_%j.out
#SBATCH --error=/scratch/mamuke588/pcdiff/logs/train_%j.err
#
# Single-GPU PCDiff training on PERUN H200
#
# Usage:
#   sbatch hpc/perun/train_single_gpu.sh
#   sbatch --time=48:00:00 hpc/perun/train_single_gpu.sh  # override time

set -euo pipefail

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

# Scratch paths
SCRATCH="/scratch/mamuke588/pcdiff"
CHECKPOINT_DIR="$SCRATCH/checkpoints/$SLURM_JOB_ID"
mkdir -p "$CHECKPOINT_DIR"

cd ~/pcdiff-implant

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
    --time-budget 82800 \
    2>&1 | tee "$SCRATCH/logs/train_${SLURM_JOB_ID}_console.log"

# Copy results to persistent storage
echo ""
echo "=== Copying results ==="
cp -r autoresearch/results/* "$SCRATCH/results/" 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "End: $(date)"
