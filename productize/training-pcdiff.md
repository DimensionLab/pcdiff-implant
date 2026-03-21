# PCDiff Model Training

This document describes how to train the PCDiff point cloud diffusion model for cranial implant generation.

## Overview

PCDiff uses a Point-Voxel CNN (PVCNN) backbone with a Gaussian diffusion process to generate implant point clouds conditioned on defective skull inputs.

## Training Command

```bash
cd /workspace/pcdiff-implant
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="8.0"

python pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 8 \
    --niter 15000 \
    --checkpoint_dir pcdiff/checkpoints
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--bs` | 8 | Batch size |
| `--niter` | 15000 | Number of training epochs |
| `--lr` | 2e-4 | Learning rate |
| `--time_num` | 1000 | Diffusion timesteps (T) |
| `--beta_start` | 0.0001 | Noise schedule start |
| `--beta_end` | 0.02 | Noise schedule end |
| `--num_points` | 30720 | Total points (skull + implant) |
| `--num_nn` | 3072 | Implant points to generate |
| `--embed_dim` | 64 | Time embedding dimension |
| `--attention` | True | Use attention in PVCNN |
| `--dropout` | 0.1 | Dropout rate |

## Checkpointing

Checkpoints are saved to `pcdiff/checkpoints/`:
- `model_best.pth` - Best model (lowest loss)
- `model_latest.pth` - Most recent checkpoint
- `model_epoch_N.pth` - Periodic checkpoints (every 5 epochs, keeps last 3)

## Dataset

**SkullBreak** (primary dataset):
- Train: 427 samples
- Test: 28 samples
- Input: Defective skull point cloud (27648 points)
- Output: Implant point cloud (3072 points)

## Model Architecture

```
Input: [B, 3, 30720] (defective skull + noise for implant region)
       ↓
PVCNN2Base (Point-Voxel CNN)
  - 4 Set Abstraction blocks
  - 4 Feature Propagation blocks
  - Time embedding via sinusoidal encoding
       ↓
Output: [B, 3, 30720] (predicted noise)
```

## Training Loss

MSE loss between predicted noise and actual noise added during diffusion forward process.

## Monitoring

Training logs to:
- Console (every 10 iterations)
- Weights & Biases (if `--no-wandb` not set)

Key metrics to watch:
- `train/loss`: Should decrease over time
- `diag/mse_bt`: Reconstruction MSE at each timestep

## Resume Training

```bash
python pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --model pcdiff/checkpoints/model_latest.pth
```

## Multi-GPU Training

```bash
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 --lr 1.6e-3
```
