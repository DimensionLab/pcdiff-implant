# Modern Setup Guide (Python 3.14 + uv)

This guide explains how to set up the project using Python 3.14 and `uv` for dependency management.

## Prerequisites

1. **Python 3.14** installed on your system (accessible via `python3`)
2. **uv** package manager ([installation guide](https://github.com/astral-sh/uv))
3. **CUDA-capable GPU** (NVIDIA A100 recommended, as per original paper)
4. **CUDA Toolkit 13.0+** installed on your system (tested with CUDA 13.0.2)

## Installation Options

### Option 1: Unified Environment (Recommended)

Install all dependencies in a single environment using the root `pyproject.toml`:

```bash
# Create virtual environment and install dependencies
uv venv --python python3.14
source .venv/bin/activate  # On Linux/Mac
# or: .venv\Scripts\activate  # On Windows

# Install PyTorch with CUDA support (adjust for your CUDA version)
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu130

# Install all other dependencies
uv pip install -e .

# Install PyTorch3D (required for voxelization network)
# Note: This can be tricky - see "Installing PyTorch3D" section below
uv pip install pytorch3d

# Install PyTorch Scatter (required for voxelization network)
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
```

### Option 2: Separate Environments

If you prefer to keep the environments separate (e.g., for dependency isolation):

#### For Point Cloud Diffusion Model:
```bash
cd pcdiff
uv venv --python python3.14
source .venv/bin/activate

# Install PyTorch with CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
uv pip install -e .
```

#### For Voxelization Network:
```bash
cd voxelization
uv venv --python python3.14
source .venv/bin/activate

# Install PyTorch with CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
uv pip install -e .

# Install PyTorch3D and PyTorch Scatter (see below)
uv pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0%2Bcu130.html
```

## Installing PyTorch3D

PyTorch3D can be challenging to install. Here are the recommended methods:

### Installing from source (recommended for Python 3.14)
```bash
# Install build dependencies
uv pip install fvcore iopath

# Clone and build
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
uv pip install -e .
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

### diplib installation fails
```bash
# diplib might need system dependencies
# On Ubuntu/Debian:
sudo apt-get install libdip-dev

# If still failing, you may need to build from source or use conda
conda install diplib -c diplib
```

### PyTorch3D build errors
Make sure you have:
```bash
# On Ubuntu/Debian:
sudo apt-get install libgl1-mesa-dev libglib2.0-0

# Check CUDA is properly installed
nvcc --version
```

### Version conflicts
```bash
# Start fresh
rm -rf .venv
uv venv --python python3.14
source .venv/bin/activate
# Then reinstall following steps above
```

## Migration Notes from Conda Environments

**Key changes from original setup:**
- Python 3.6/3.8 → Python 3.14
- PyTorch 1.7.1/1.12.0 → PyTorch 2.x
- Conda/Mamba → uv
- CUDA 10.1/11.3 → CUDA 12.1 (or your system version)
- Unified environment possible (original used two separate environments)

**Compatibility considerations:**
- All major dependencies support Python 3.14
- PyTorch 2.x is largely backward compatible with 1.x models
- Some old model checkpoints might need conversion (test thoroughly)
- Performance should be equal or better with modern PyTorch

## Next Steps

After installation, proceed with:
1. [Data preprocessing and training the Point Cloud Diffusion Model](./pcdiff/README.md)
2. [Training the Voxelization Network](./voxelization/README.md)

