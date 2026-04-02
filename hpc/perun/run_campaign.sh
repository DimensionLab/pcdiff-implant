#!/bin/bash
#SBATCH --job-name=pcdiff-campaign
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/mamuke588/pcdiff/logs/campaign_%j.out
#SBATCH --error=/scratch/mamuke588/pcdiff/logs/campaign_%j.err
#
# Automated experiment campaign on PERUN.
# Runs the autoresearch experiment loop (LLM-guided hyperparameter search).
#
# Usage:
#   sbatch hpc/perun/run_campaign.sh
#   sbatch --array=1-4 hpc/perun/run_campaign.sh  # 4 parallel campaigns

set -euo pipefail

echo "=== PCDiff Experiment Campaign ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff

# Wandb config
export WANDB_PROJECT="pcdiff-implant-perun"
export WANDB_RUN_GROUP="campaign-${SLURM_JOB_ID}"
export WANDB_TAGS="perun,h200,campaign,autoresearch"

# Scratch paths
SCRATCH="/scratch/mamuke588/pcdiff"
mkdir -p "$SCRATCH/results/campaign_${SLURM_JOB_ID}"

cd ~/pcdiff-implant

# Build CUDA extensions
if [ -d "pcdiff/modules/functional" ]; then
    cd pcdiff/modules/functional
    python setup.py build_ext --inplace 2>/dev/null || true
    cd ~/pcdiff-implant
fi

nvidia-smi

# Run experiment campaign
# Each experiment gets 15 min budget, run up to 100 experiments
python autoresearch/run_experiments.py \
    --max-experiments 100 \
    2>&1 | tee "$SCRATCH/logs/campaign_${SLURM_JOB_ID}_console.log"

# Sync results
echo ""
echo "=== Syncing results ==="
cp -r autoresearch/results/* "$SCRATCH/results/campaign_${SLURM_JOB_ID}/" 2>/dev/null || true

echo ""
echo "=== Campaign Complete ==="
echo "End: $(date)"
echo "Results in: $SCRATCH/results/campaign_${SLURM_JOB_ID}/"
