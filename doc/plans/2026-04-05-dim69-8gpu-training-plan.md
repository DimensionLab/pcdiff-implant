# DIM-69 Plan: 2-Day PCDiff Training on 8x H200

**Date:** 2026-04-05
**Author:** CTO (ba6d3030)
**Status:** Draft — pending V14/V12 completion

## Prerequisites (must complete before launch)

1. **V14 cosine-minsnr training completes** (job 18923, ~13h remaining) — best train loss 0.1648
2. **V12 training completes** (job 18922, ~13h remaining) — best train loss 0.1678
3. **DIM-7 vox-retrain-v3 evaluation** — need boundary metrics before committing to architecture
4. **Winner selection**: Compare V14 vs V12 final checkpoints on full SkullBreak eval

## Architecture Decision

Based on experiments so far, **V14 (cosine schedule + min-SNR-γ weighting)** is the leading candidate:
- Best training loss: 0.1648 (vs V12's 0.1678)
- Cosine schedule showed 17% val loss improvement over linear in V12 experiments
- Min-SNR-γ (γ=5.0) provides lowest variance and fastest convergence

Key config from V14:
- `SCHEDULE_TYPE = "cosine"`, `LOSS_TYPE = "mse_minsnr"`, `MIN_SNR_GAMMA = 5.0`
- `EMBED_DIM = 96`, `WIDTH_MULT = 1.5`, `DROPOUT = 0.1`
- `OPTIMIZER_TYPE = "adamw"`, `LR = 2e-4`, `BETA1 = 0.9`
- `EMA_DECAY = 0.9999`, `GRAD_CLIP = 1.0`
- `BATCH_SIZE = 2` (single GPU)

## 8x H200 Training Plan

### Infrastructure
- Perun HPC GPU partition: 8x NVIDIA H200 per node (143 GB each)
- 128 CPUs, ~2.2 TB RAM per node
- No time limit on GPU partition
- Slurm: `--account=perun2501174 --qos=perun2501174`

### Scaling Strategy
- Use `torchrun --nproc_per_node=8` with PyTorch DDP (already supported in `train_completion.py`)
- Global batch size: 64 (8 per GPU × 8 GPUs)
- Learning rate: **sqrt scaling recommended** → `LR = 2e-4 × √8 ≈ 5.66e-4`
  - Linear scaling (1.6e-3) risks divergence with cosine schedule + min-SNR
  - Conservative sqrt scaling is safer for our architecture (GroupNorm, small dataset)
- Warmup: Add 50-epoch LR warmup from 1e-5 → 5.66e-4
- Data workers: 16 per GPU (128 CPUs available)

### Checkpoint Strategy (per issue requirements)
- `model_best.pth` — overwritten whenever val loss improves
- `model_latest.pth` — overwritten every epoch
- Download locally every 100 epochs (rsync cron or heartbeat script)
- Full epoch logs committed + pushed on every heartbeat

### Estimated Training Time
- Single GPU: ~2.4 min/epoch (observed from vox-retrain-v3 timing)
- 8x GPU: ~18 sec/epoch (theoretical 8x speedup)
- 2 days = 2880 min → ~9600 epochs possible
- Target: 2000-5000 epochs (well past convergence based on V14 data)

## Slurm Script

```bash
#!/bin/bash
#SBATCH --job-name=pcdiff-8gpu-final
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

set -euo pipefail
source .activate_scratch

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff
export CUDA_HOME=$CONDA_PREFIX

# NCCL configuration for single-node multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

cd /mnt/data/home/mamuke588/pcdiff-implant

torchrun --nproc_per_node=8 pcdiff/train_completion.py \
  --path pcdiff/datasets/SkullBreak/train.csv \
  --dataset SkullBreak \
  --bs 64 \
  --lr 5.66e-4 \
  --niter 5000 \
  --schedule cosine \
  --loss-type mse_minsnr \
  --workers 16 \
  --save-dir $SCRATCH/checkpoints/pcdiff-8gpu-final
```

## Open Questions

1. Should we use the modified `train_pcdiff_modified.py` (autoresearch version with EMA, grad clip, width_mult) or the standard `train_completion.py`? The modified version has more features but hasn't been tested with DDP.
2. Do we want TensorBoard logging alongside file logs? Need to add `--tensorboard` flag if so.
3. Should we also run a parallel 8-GPU voxelization training?

## Next Steps

- [ ] Wait for V14/V12 completion (~13h)
- [ ] Run final eval comparison on both checkpoints
- [ ] Verify `train_completion.py` supports all V14 config flags (cosine schedule, min-SNR, etc.)
- [ ] If not, port those features from `train_pcdiff_modified.py` into `train_completion.py`
- [ ] Adapt winning config into DDP-compatible Slurm script
- [ ] Test 1-epoch DDP run before committing 2-day allocation
- [ ] Submit production training job
