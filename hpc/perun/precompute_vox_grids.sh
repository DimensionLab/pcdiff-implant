#!/bin/bash
#SBATCH --job-name=vox-precompute
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/vox_precompute_%j.out

set -euo pipefail

PROJECT_DIR="/mnt/data/home/mamuke588/pcdiff-implant"
cd "$PROJECT_DIR/voxelization"

# Activate conda env
source /mnt/data/home/mamuke588/miniconda3/etc/profile.d/conda.sh
conda activate pcdiff

# Install nrrd if needed
pip install pynrrd scikit-image 2>/dev/null || true

echo "=== PRECOMPUTE PSR GRIDS FROM IMPLANT NRRD FILES ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Step 1: Precompute _vox.npz from implant .nrrd files
python precompute_psr_grids.py \
    --dataset-root "$PROJECT_DIR/pcdiff/datasets/SkullBreak" \
    --resolution 512

# Step 2: Create legacy manifests (defective skull -> implant PSR)
python create_skullbreak_manifests.py \
    --dataset-root "$PROJECT_DIR/pcdiff/datasets/SkullBreak" \
    --mode legacy

echo "=== DONE ==="
echo "End: $(date)"
