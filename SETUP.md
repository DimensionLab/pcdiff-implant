# Modern Setup Guide (Python 3.10 + uv)

This guide explains how to set up the project using Python 3.10 and `uv` for dependency management.

## Prerequisites

1. **Python 3.10** (installed via `uv python install 3.10`)
2. **uv** package manager ([installation guide](https://github.com/astral-sh/uv))
3. **CUDA-capable GPU** (H100 recommended, tested with CUDA 13.0.2)

**Note**: We use PyTorch 2.5.0 + CUDA 12.4 binaries which are forward-compatible with CUDA 13 drivers and have pre-built torch-scatter wheels.

## Installation Options

### Option 1: Unified Environment (Recommended)

Install all dependencies in a single environment using the root `pyproject.toml`:

```bash
# Create virtual environment and install dependencies
uv python install 3.10  # Install Python 3.10 if needed
uv venv --python 3.10
source .venv/bin/activate  # On Linux/Mac
# or: .venv\Scripts\activate  # On Windows

# Install PyTorch 2.5.0 with CUDA 12.4 (compatible with CUDA 13 drivers)
uv pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install "torchvision==0.20.0" --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
uv pip install -e .

# Install PyTorch3D (required for voxelization network, builds from source)
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install PyTorch Scatter (required for voxelization network, pre-built wheel)
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

### Option 2: Separate Environments

If you prefer to keep the environments separate (e.g., for dependency isolation):

#### For Point Cloud Diffusion Model:
```bash
cd pcdiff
uv python install 3.10  # Install Python 3.10 if needed
uv venv --python 3.10
source .venv/bin/activate

# Install PyTorch with CUDA 12.4
uv pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install "torchvision==0.20.0" --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
uv pip install -e .
```

#### For Voxelization Network:
```bash
cd voxelization
uv python install 3.10  # Install Python 3.10 if needed
uv venv --python 3.10
source .venv/bin/activate

# Install PyTorch with CUDA 12.4
uv pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install "torchvision==0.20.0" --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
uv pip install -e .

# Install PyTorch3D and PyTorch Scatter
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

## Installing PyTorch3D & PyTorch Scatter

### Prerequisites
```bash
# Python 3.10 is installed automatically by uv
# Verify CUDA is accessible
nvcc --version  # should show 13.0
```

### Installation
```bash
# PyTorch3D (~5-10 minutes, must build from source)
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# PyTorch Scatter (pre-built wheel, instant)
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

## Verifying Installation

### Test Point Cloud Diffusion Model
```python
python3 -c "import torch; import trimesh; import open3d; print('PCD environment OK')"
```

### Test Voxelization Network
```python
python3 -c "import torch; import pytorch3d; import torch_scatter; print('Voxelization environment OK')"
```

## CUDA Compatibility Notes

The original environments used CUDA 10.1 and 11.3. Modern PyTorch (2.x) supports CUDA 11.8, 12.1, and 12.4:

- **For CUDA 13.0**: `--index-url https://download.pytorch.org/whl/cu130`
- **For CUDA 12.4**: `--index-url https://download.pytorch.org/whl/cu124`
- **CPU only**: `--index-url https://download.pytorch.org/whl/cpu`

Check your CUDA version:
```bash
nvcc --version  # or nvidia-smi
```

## Troubleshooting

### PyTorch3D or PyTorch Scatter build errors

**Missing Python.h:**
```bash
sudo apt-get install python3.12-dev
```

**Missing CUDA or build tools:**
```bash
# On Ubuntu/Debian:
sudo apt-get install build-essential libgl1-mesa-dev libglib2.0-0

# Check CUDA is properly installed
nvcc --version
```

### diplib installation fails
```bash
# diplib might need system dependencies
# On Ubuntu/Debian:
sudo apt-get install libdip-dev

# If still failing, you may need to build from source or use conda
conda install diplib -c diplib
```

### Version conflicts
```bash
# Start fresh
rm -rf .venv
uv python install 3.10  # Install Python 3.10 if needed
uv venv --python 3.10
source .venv/bin/activate
# Then reinstall following steps above
```

## Migration Notes from Conda Environments

**Key changes from original setup:**
- Python 3.6/3.8 → Python 3.10
- PyTorch 1.7.1/1.12.0 → PyTorch 2.x
- Conda/Mamba → uv
- CUDA 10.1/11.3 → CUDA 12.1 (or your system version)
- Unified environment possible (original used two separate environments)

**Compatibility considerations:**
- All major dependencies support Python 3.10
- PyTorch 2.x is largely backward compatible with 1.x models
- Some old model checkpoints might need conversion (test thoroughly)
- Performance should be equal or better with modern PyTorch

## Next Steps

After installation, proceed with:
1. [Data preprocessing and training the Point Cloud Diffusion Model](./pcdiff/README.md)
2. [Training the Voxelization Network](./voxelization/README.md)

