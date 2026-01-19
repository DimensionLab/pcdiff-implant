# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PCDiff-Implant generates patient-specific cranial implants using a two-stage pipeline:
1. **PCDiff** - Point cloud diffusion model that generates implant point clouds from defective skull inputs
2. **Voxelization** - Neural surface reconstruction that converts point clouds to watertight meshes

Based on MICCAI 2023 paper "Point Cloud Diffusion Models for Automatic Implant Generation" - [link to paper stored locally](paper/pcdiff_paper.pdf).

## Common Commands

### Setup
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

### Data Preparation
```bash
python pcdiff/utils/preproc_skullbreak.py   # Preprocess (uses multiprocessing)
python pcdiff/utils/split_skullbreak.py     # Create train/test splits
```

### Training

**PCDiff (single GPU):**
```bash
python pcdiff/train_completion.py --path pcdiff/datasets/SkullBreak/train.csv --dataset SkullBreak --bs 8
```

**PCDiff (multi-GPU with torchrun):**
```bash
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv --dataset SkullBreak \
    --bs 64 --lr 1.6e-3  # Scale batch size (N×8) and LR (N×2e-4) with GPU count
```

**Voxelization:**
```bash
cd voxelization && python train.py configs/train_skullbreak.yaml
# With wandb: python train.py configs/train_skullbreak.yaml --wandb-project my-project
```

### Inference

**PCDiff (DDIM for fast inference):**
```bash
python pcdiff/test_completion.py \
    --path pcdiff/datasets/SkullBreak/test.csv --dataset SkullBreak \
    --model pcdiff/output/train_completion/LATEST/model_best.pth \
    --sampling_method ddim --sampling_steps 50 --num_ens 5
```

**Voxelization:**
```bash
cd voxelization && python generate.py configs/gen_skullbreak.yaml
```

**Full pipeline (single skull):**
```bash
python run_single_inference.py \
    --input path/to/defective_skull.npy \
    --pcdiff_model path/to/pcdiff_model.pth \
    --vox_model path/to/vox_model.pt \
    --output_dir results
```

### Web Viewer
```bash
cd web_viewer && ./start_dev.sh   # Dev mode at localhost:5173
# Convert results: python pcdiff/utils/convert_to_web.py inference_results --batch
```

## Architecture

```
Input (Defective Skull NPY, 512³)
    ↓
PCDiff Model (PVCNN diffusion, 1000 DDPM or 50 DDIM steps)
    ↓
Implant Point Cloud (3072 points)
    ↓
Voxelization Model (Encode2Points + DPSR)
    ↓
Watertight Mesh (PLY/STL/NRRD export)
```

### Key Directories
- `pcdiff/` - Diffusion model: `train_completion.py`, `test_completion.py`, model in `model/`, custom CUDA ops in `modules/`
- `voxelization/` - Surface reconstruction: `train.py`, `generate.py`, network in `src/network/`
- `pcdiff/datasets/` - Data loading and CSV splits for SkullBreak/SkullFix
- `voxelization/configs/` - YAML configs for training and generation

### Model Components
- **PCDiff**: PVCNN2Base with point-voxel convolutions, PointNet encoder, ball query KNN
- **Voxelization**: Local pooling PointNet encoder → 3D UNet → Simple local decoder → Differentiable Poisson Surface Reconstruction (DPSR)

## Distributed Training Notes

- PCDiff uses `torchrun` with NCCL backend for DDP
- Scale batch size and learning rate linearly with GPU count
- Model uses GroupNorm (handles small per-GPU batches well)
- Debug NCCL issues: `export NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL`
- Run in tmux/screen for persistent sessions

## Output Formats
- NPY: Raw arrays for processing
- PLY: Point clouds and meshes for visualization
- STL: Watertight meshes for 3D printing
- NRRD: Medical imaging format
