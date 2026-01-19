# Server Setup Documentation

This document describes the server environment for running PCDiff-Implant.

## Server Overview

| Component | Details |
|-----------|---------|
| **Platform** | RunPod.io Pod |
| **OS** | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| **Kernel** | Linux 6.8.0-64-generic x86_64 |
| **Architecture** | x86_64 (GNU/Linux) |

## Hardware

### CPU
| Spec | Value |
|------|-------|
| **Model** | AMD EPYC 7763 64-Core Processor |
| **vCPUs** | 252 |

### Memory
| Spec | Value |
|------|-------|
| **Total RAM** | 944 GB |
| **Available** | ~875 GB |
| **Swap** | None |

### Storage
| Mount | Size | Available | Use |
|-------|------|-----------|-----|
| `/` (overlay) | 20 GB | 20 GB | 3% |
| `/workspace` (network storage ID wzcc20z2mg called "crainial-implant") | 629 TB (50 GB ours) | 215 TB (irrelevant, we need to care about our own limited storage) | 66% (irrelevant, we need to care about our own limited storage) |

### GPU
| Spec | Value |
|------|-------|
| **Model** | NVIDIA A100 80GB PCIe |
| **VRAM** | 80 GB HBM2e |
| **Bus** | PCIe (0000:00:08.0) |
| **UUID** | GPU-997a34b9-b8e9-609f-fdc5-1c6771231b2d |
| **DMA Size** | 47 bits |

## NVIDIA Software Stack

### Driver
| Component | Version |
|-----------|---------|
| **NVIDIA Driver** | 570.172.08 |
| **Driver Type** | NVIDIA UNIX Open Kernel Module |
| **GPU Firmware** | 570.172.08 |

### CUDA
| Component | Version |
|-----------|---------|
| **CUDA Toolkit** | 12.4 |
| **nvcc** | V12.4.131 (release 12.4) |
| **CUDA Path** | `/usr/local/cuda-12.4` (symlinked to `/usr/local/cuda`) |

## Python Environment

### System Python
| Component | Version |
|-----------|---------|
| **Python** | 3.11.10 |
| **Path** | `/usr/bin/python3` |

### PyTorch Stack
| Package | Version |
|---------|---------|
| **PyTorch** | 2.4.1+cu124 |
| **TorchVision** | 0.19.1+cu124 |
| **TorchAudio** | 2.4.1+cu124 |
| **CUDA (PyTorch)** | 12.4 |
| **cuDNN** | 9.1.0 (90100) |

### Key Dependencies
| Package | Version |
|---------|---------|
| **NumPy** | 1.26.3 |
| **nvidia-cuda-runtime-cu12** | 12.4.99 |
| **nvidia-cuda-nvrtc-cu12** | 12.4.99 |
| **nvidia-cuda-cupti-cu12** | 12.4.99 |

## Environment Setup for PCDiff-Implant

The project requires Python 3.10 for compatibility. Create a virtual environment:

```bash
# Create environment with uv (Python 3.10 required)
uv venv --python 3.10 && source .venv/bin/activate

# Install PyTorch with CUDA 12.4
uv pip install "torch==2.5.0" "torchvision==0.20.0" --index-url https://download.pytorch.org/whl/cu124

# Install project and special dependencies
uv pip install -e .
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

## Training Commands

Once the environment is set up and CUDA is accessible:

```bash
# Activate environment
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 compute capability

# Single GPU training (SkullBreak)
python pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 8

# Multi-GPU training (8x GPU)
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 --lr 1.6e-3
```

## Notes

- **GPU Access**: The A100 80GB GPU is fully accessible for PyTorch CUDA operations. Training and inference work correctly.
- **NVML**: `nvidia-smi` may fail with "Failed to initialize NVML: Unknown Error" in containerized environments, but this does not affect PyTorch CUDA functionality.
- **Network Storage**: The `/workspace` mount is network-attached storage (MFS) with high capacity suitable for datasets and model checkpoints.
- **Container Environment**: This is a containerized RunPod environment; some host-level tools may be unavailable.

## Verification Commands

```bash
# Check GPU availability in PyTorch
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check CUDA compiler
nvcc --version

# Check driver version
cat /proc/driver/nvidia/version

# Check GPU info
cat /proc/driver/nvidia/gpus/*/information
```
