# Quick Start Guide (uv + Python 3.10 + PyTorch 2.5)

Get up and running in 5 minutes! 🚀

## Prerequisites
- Python 3.10 (installed via `uv python install 3.10`)
- NVIDIA GPU with CUDA 12.4+ or 13.0+ (H100 recommended)
- uv package manager

## Install uv
```bash
# Option 1: Using pip
pip install uv

# Option 2: Standalone installer (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Automated Setup (Easiest!)

```bash
cd /home/michaltakac/pcdiff-implant

# Run the setup script (defaults to CUDA 12.4, compatible with CUDA 13)
./setup_uv.sh

# For different CUDA versions:
./setup_uv.sh cu124  # CUDA 12.4 (default, works with CUDA 13 drivers)
./setup_uv.sh cu121  # CUDA 12.1
./setup_uv.sh cpu    # CPU only
```

That's it! The script will:
- Create a virtual environment
- Install PyTorch with CUDA support
- Install all dependencies
- Install PyTorch3D and PyTorch Scatter
- Verify everything works

## Manual Setup

If you prefer to install manually:

```bash
# 1. Create and activate virtual environment
uv python install 3.10  # Download Python 3.10 if needed
uv venv --python 3.10
source .venv/bin/activate

# 2. Install PyTorch 2.5.0 with CUDA 12.4 (works with CUDA 13 drivers)
uv pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install "torchvision==0.20.0" --index-url https://download.pytorch.org/whl/cu124

# 3. Install project dependencies
uv pip install -e .

# 4. Install PyTorch3D (optional, for voxelization only)
# First try pre-built wheels (fast)
uv pip install --no-deps fvcore iopath
uv pip install pytorch3d

# If that fails, build from source (~5-10 min)
# Make sure CUDA_HOME matches PyTorch's CUDA version
export CUDA_HOME=/usr/local/cuda-12.4  # Match PyTorch's CUDA 12.4
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# 5. Install PyTorch Scatter (for voxelization, pre-built wheel)
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

## Verify Installation

```bash
# Activate the virtual environment first!
source .venv/bin/activate

# Quick test
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import pytorch3d, torch_scatter; print('All dependencies OK!')"
```

> **💡 Tip**: If you get `ModuleNotFoundError`, make sure you activated the virtual environment with `source .venv/bin/activate`

## Next Steps

### 1. Download Datasets
- **SkullBreak**: https://www.fit.vutbr.cz/~ikodym/skullbreak_training.zip
- **SkullFix**: https://files.icg.tugraz.at/f/2c5f458e781a42c6a916/?dl=1

Organize as shown in [README.md](./README.md#data).

### 2. Preprocess Data
```bash
# Make sure virtual environment is activated!
source .venv/bin/activate

python3 pcdiff/utils/preproc_skullbreak.py
python3 pcdiff/utils/preproc_skullfix.py
```

### 3. Create Train/Test Split
```bash
source .venv/bin/activate

python3 pcdiff/utils/split_skullbreak.py
python3 pcdiff/utils/split_skullfix.py
```

### 4. Train Point Cloud Diffusion Model
```bash
source .venv/bin/activate

python3 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak
```

### 5. Train Voxelization Network
```bash
source .venv/bin/activate

python3 voxelization/train.py voxelization/configs/train_skullbreak.yaml
```

### 6. Generate Implants
```bash
source .venv/bin/activate

# Point cloud generation
python3 pcdiff/test_completion.py \
    --path pcdiff/datasets/SkullBreak/test.csv \
    --dataset SkullBreak \
    --model MODELPATH \
    --eval_path pcdiff/datasets/SkullBreak/results

# Voxelization
python3 voxelization/generate.py voxelization/configs/gen_skullbreak.yaml
```

## Troubleshooting

### ModuleNotFoundError (mcubes, torch, etc.)

**Problem**: `ModuleNotFoundError: No module named 'mcubes'` or similar errors

**Solution**: You forgot to activate the virtual environment!
```bash
# Always activate before running any Python scripts
source .venv/bin/activate

# Verify it's activated (you should see (.venv) in your prompt)
which python3  # Should show: /home/YOUR_USER/pcdiff-implant/.venv/bin/python3

# Now run your script
python3 pcdiff/utils/preproc_skullbreak.py
```

### CUDA not available
```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu124
```

### PyTorch3D installation fails (CUDA version mismatch)

**Problem**: PyTorch3D build fails with "CUDA version (13.0) mismatches the version that was used to compile PyTorch (12.4)"

**Solution 1 - Install matching CUDA toolkit** (Recommended):
```bash
# Install CUDA 12.4 toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent --override

# Set environment variables
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Re-run setup
./setup_uv.sh
```

**Solution 2 - Skip PyTorch3D** (if only using point cloud diffusion):
```bash
# PyTorch3D is only needed for voxelization
# The main point cloud diffusion model works without it
# Just proceed with the other dependencies
```

**Solution 3 - Manual build**:
```bash
# Clone and build from source with proper CUDA paths
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
export CUDA_HOME=/usr/local/cuda-12.4  # Match PyTorch's CUDA version
uv pip install -e .
```

### diplib fails
```bash
# Install system dependencies first
sudo apt-get update
sudo apt-get install libdip-dev
```

## Command Cheatsheet

```bash
# Activate environment
source .venv/bin/activate

# Update dependencies
uv pip install --upgrade -e .

# Check what's installed
uv pip list

# Deactivate environment
deactivate

# Remove environment (start fresh)
rm -rf .venv
```

## Resources

- 📖 [SETUP.md](./SETUP.md) - Comprehensive setup guide
- 🔄 [MIGRATION.md](./MIGRATION.md) - Migrating from conda
- 🧠 [pcdiff/README.md](./pcdiff/README.md) - Point cloud diffusion details
- 🔲 [voxelization/README.md](./voxelization/README.md) - Voxelization details
- 📄 [Paper](https://arxiv.org/abs/2303.08061) - Original MICCAI 2023 paper

## Common Workflows

### Development
```bash
# Activate environment
source .venv/bin/activate

# Run experiments
python3 pcdiff/train_completion.py --path datasets/SkullBreak/train.csv --dataset SkullBreak

# View tensorboard logs
tensorboard --logdir=./logs
```

### Production
```bash
# Generate implants with ensembling
python3 pcdiff/test_completion.py \
    --path datasets/SkullFix/test.csv \
    --dataset SkullFix \
    --num_ens 5 \
    --model MODELPATH \
    --eval_path datasets/SkullFix/results
```

## Performance Tips

- **GPU Memory**: Reduce batch size if you run out of VRAM
- **Speed**: Use `--num_workers` to parallelize data loading
- **Quality**: Use ensembling (`--num_ens 5`) for best results
- **Debugging**: Use `ipdb` for interactive debugging

---

**Questions?** Check the full documentation in [SETUP.md](./SETUP.md) or open an issue on GitHub.

