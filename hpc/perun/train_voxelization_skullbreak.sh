#!/bin/bash
#SBATCH --job-name=vox-train
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/vox_train_%j.out

set -euo pipefail

PROJECT_DIR="/mnt/data/home/mamuke588/pcdiff-implant"
cd "$PROJECT_DIR/voxelization"

# Activate conda env
source /mnt/data/home/mamuke588/miniconda3/etc/profile.d/conda.sh
conda activate pcdiff

echo "=== VOXELIZATION TRAINING ON SKULLBREAK ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Set manifest paths
export PCDIFF_SKULLBREAK_TRAIN_CSV="$PROJECT_DIR/pcdiff/datasets/SkullBreak/voxelization/train.csv"
export PCDIFF_SKULLBREAK_EVAL_CSV="$PROJECT_DIR/pcdiff/datasets/SkullBreak/voxelization/eval.csv"

# Train voxelization model
python train.py configs/train_skullbreak.yaml

echo "=== DONE ==="
echo "End: $(date)"
