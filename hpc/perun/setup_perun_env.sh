#!/bin/bash
# setup_perun_env.sh — One-time environment setup on PERUN login node
# Run this after first SSH login to PERUN.
#
# Usage: bash ~/pcdiff-implant/hpc/perun/setup_perun_env.sh

set -euo pipefail

echo "=== PERUN Environment Setup for PCDiff ==="

# 1. Install Miniconda if not present
if [ ! -d "$HOME/miniconda3" ]; then
    echo ">> Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    echo ">> Miniconda installed. Please 'source ~/.bashrc' then re-run this script."
    exit 0
fi

# Make sure conda is available
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# 2. Create pcdiff conda environment
if ! conda env list | grep -q "pcdiff"; then
    echo ">> Creating pcdiff conda environment..."
    conda create -n pcdiff python=3.11 -y
fi

echo ">> Activating pcdiff environment..."
conda activate pcdiff

# 3. Install PyTorch with CUDA 12.x support (H200 compatible)
echo ">> Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install project dependencies
echo ">> Installing project dependencies..."
if [ -f "$HOME/pcdiff-implant/requirements.txt" ]; then
    pip install -r "$HOME/pcdiff-implant/requirements.txt"
fi

# Additional dependencies for autoresearch
pip install wandb scipy trimesh scikit-learn open3d nrrd

# 5. Configure wandb (non-interactive)
echo ">> Configuring Weights & Biases..."
# The API key should be set via WANDB_API_KEY env var in job scripts
# Do NOT hardcode the key here
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY"
    echo ">> wandb configured."
else
    echo ">> WANDB_API_KEY not set. Set it in your job scripts or ~/.bashrc"
    echo "   export WANDB_API_KEY=<your-key>"
fi

# 6. Build CUDA extensions for PCDiff (if needed)
echo ">> Building CUDA extensions..."
cd "$HOME/pcdiff-implant"
if [ -d "pcdiff/modules/functional" ]; then
    cd pcdiff/modules/functional
    # Build point-cloud ops with CUDA
    python setup.py build_ext --inplace 2>/dev/null || \
        echo ">> Warning: CUDA extensions build should be done on a compute node with GPU access"
    cd "$HOME/pcdiff-implant"
fi

# 7. Create scratch directory structure
SCRATCH="/scratch/mamuke588"
if [ -d "/scratch" ]; then
    mkdir -p "$SCRATCH/pcdiff/{checkpoints,logs,results,data}"
    echo ">> Scratch directories created at $SCRATCH/pcdiff/"
fi

# 8. Verify setup
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || \
    echo ">> Note: CUDA not available on login node. Will work on compute nodes."
python -c "import wandb; print(f'wandb: {wandb.__version__}')" 2>/dev/null || \
    echo ">> wandb not installed properly"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Clone/sync your code to ~/pcdiff-implant on PERUN"
echo "  2. Copy dataset to $SCRATCH/pcdiff/data/ or ~/pcdiff-implant/datasets/"
echo "  3. Submit a job: sbatch hpc/perun/train_single_gpu.sh"
