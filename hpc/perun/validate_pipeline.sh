#!/bin/bash
#SBATCH --job-name=pcdiff-validate
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/hpc/perun/validate_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/hpc/perun/validate_%j.err

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff

export WANDB_PROJECT="pcdiff-implant-perun"
export WANDB_RUN_GROUP="validation"
export WANDB_TAGS="perun,h200,validation"

cd ~/pcdiff-implant
echo "=== Pipeline Validation on PERUN H200 ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

# Run the autoresearch baseline with a 5-min budget to validate everything works
echo ">>> Running baseline training (5 min budget)..."
python autoresearch/train_pcdiff.py --baseline --time-budget 300 2>&1 | tail -30

echo ""
echo "=== Validation Complete ==="
echo "End: $(date)"
