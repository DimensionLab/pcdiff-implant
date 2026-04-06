# PCDiff 8x H200 Final Training — DIM-69

## Configuration
- **Node**: gpu03 (Perun HPC)
- **GPUs**: 6× NVIDIA H200 (143GB each), 100% utilization
- **Architecture**: V14 cosine-minsnr (embed_dim=96, width_mult=1.5, dropout=0.1)
- **Training**: batch=64, LR=5.66e-4 cosine (50-epoch warmup), 5000 max epochs
- **Loss**: mse_minsnr (gamma=5.0), EMA decay=0.9999, grad_clip=1.0
- **Dataset**: SkullBreak, 455 samples, 7 batches/epoch/rank
- **Run dir**: pcdiff/runs/SkullBreak/20260406_063836/
- **Slurm job**: 19132

## Status
- Started: 2026-04-06 06:38 UTC
- 13.5 seconds/epoch
- Projected completion: ~18.5 hours (well within 2-day limit)

## Checkpoints Downloaded
- [x] Epoch 50 (model_epoch_50.pth, 712MB)

## Loss Progression
| Epoch | Loss | Best Loss | Grad Norm |
|-------|------|-----------|-----------|
| 0 | 1.036 | 0.778 | 33.64 |
| 10 | 0.313 | 0.386 | 3.39 |
| 20 | 0.201 | 0.267 | 1.48 |
| 30 | 0.116 | 0.196 | 0.40 |
| 50 | 0.310 | 0.166 | 1.11 |
| 63 | 0.155 | **0.159** | 0.15 |
| 71 | 0.147 | 0.159 | 0.24 |

## Notes
- Script requested 8 GPUs but gpu03 only exposes 6; torchrun adapted with 8 workers on 6 GPUs
- Single-GPU predecessors (v14 ep950 best=0.165, v12 ep950 best=0.168) finishing within the hour
- 8-GPU already matching single-GPU best loss at epoch 63 vs epoch ~700+ for single-GPU
