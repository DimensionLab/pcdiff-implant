# Quick Start Guide (uv + Python 3.14 + CUDA 13)

Get up and running in 5 minutes! 🚀

## Prerequisites
- Python 3.14 installed (`python3 --version`)
- NVIDIA GPU with CUDA 13.0.2+ (H100 recommended)
- uv package manager

## Install uv
```bash
# Option 1: Using pip (requires Python 3.14)
pip install uv

# Option 2: Standalone installer (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Automated Setup (Easiest!)

```bash
cd /home/michaltakac/pcdiff-implant

# Run the setup script (defaults to CUDA 13.0)
./setup_uv.sh

# For different CUDA versions:
./setup_uv.sh cu130  # CUDA 13.0 (default)
./setup_uv.sh cu124  # CUDA 12.4
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
uv venv --python python3.14
source .venv/bin/activate

# 2. Install PyTorch with CUDA 13 (H100)
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu130

# 3. Install project dependencies
uv pip install -e .

# 4. Install PyTorch3D (for voxelization)
uv pip install pytorch3d

# 5. Install PyTorch Scatter (for voxelization)
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu130.html
```

## Verify Installation

```bash
# Quick test
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import pytorch3d, torch_scatter; print('All dependencies OK!')"
```

## Next Steps

### 1. Download Datasets
- **SkullBreak**: https://www.fit.vutbr.cz/~ikodym/skullbreak_training.zip
- **SkullFix**: https://files.icg.tugraz.at/f/2c5f458e781a42c6a916/?dl=1

Organize as shown in [README.md](./README.md#data).

### 2. Preprocess Data
```bash
python3 pcdiff/utils/preproc_skullbreak.py
python3 pcdiff/utils/preproc_skullfix.py
```

### 3. Create Train/Test Split
```bash
python3 pcdiff/utils/split_skullbreak.py
python3 pcdiff/utils/split_skullfix.py
```

### 4. Train Point Cloud Diffusion Model
```bash
python3 pcdiff/train_completion.py \
    --path datasets/SkullBreak/train.csv \
    --dataset SkullBreak
```

### 5. Train Voxelization Network
```bash
python3 voxelization/train.py voxelization/configs/train_skullbreak.yaml
```

### 6. Generate Implants
```bash
# Point cloud generation
python3 pcdiff/test_completion.py \
    --path datasets/SkullBreak/test.csv \
    --dataset SkullBreak \
    --model MODELPATH \
    --eval_path datasets/SkullBreak/results

# Voxelization
python3 voxelization/generate.py voxelization/configs/gen_skullbreak.yaml
```

## Troubleshooting

### CUDA not available
```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu124
```

### PyTorch3D installation fails
```bash
# Try building from source
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && uv pip install -e .
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

