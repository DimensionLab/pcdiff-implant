# PCDiff Autoresearch Program

You are an AI research agent optimizing a Point Cloud Diffusion model (PCDiff) for cranial implant generation. The model generates 3D implant point clouds from defective skull scans.

## Goal

**Minimize Chamfer Distance** between generated and ground-truth implant point clouds on the SkullBreak validation set. Lower is better.

## What You Can Modify

You may ONLY modify `train_pcdiff.py`. The file contains clearly marked hyperparameter constants at the top and the full model architecture.

### Fair game for modification:
- All `UPPERCASE` hyperparameter constants (learning rate, batch size, schedules, etc.)
- Noise schedule (`get_betas` function and `SCHEDULE_TYPE`)
- Model architecture: attention patterns, block configs, width/depth multipliers
- Optimizer choice and configuration
- LR schedule strategy
- Loss function
- Data augmentation
- Training loop logic (e.g., EMA, curriculum, etc.)

### Do NOT modify:
- `prepare_pcdiff.py` (data loading and evaluation utilities)
- The evaluation protocol (Chamfer Distance on fixed 10-case subset via DDIM-50)
- Point cloud dimensions: NUM_POINTS=30720, NUM_NN=3072, SV_POINTS=27648
- The `SkullBreakDatasetSimple` class interface

## Research Directions (Prioritized)

### High Priority (likely high impact)
1. **Cosine noise schedule**: Change `SCHEDULE_TYPE = "cosine"`. The cosine schedule from "Improved DDPM" is well-known to improve diffusion quality, especially for structured data.
2. **EMA (Exponential Moving Average)**: Add EMA of model weights for evaluation. Use decay=0.9999. This is a nearly universal improvement for diffusion models.
3. **Gradient clipping**: Set `GRAD_CLIP = 1.0`. Stabilizes training, especially with larger models.
4. **Learning rate**: Try cosine annealing (`CosineAnnealingLR`) instead of the current warmup + exponential decay.

### Medium Priority
5. **Width multiplier**: Try `WIDTH_MULT = 1.5` or `2.0`. Larger model capacity may help.
6. **v-prediction**: Change `MODEL_MEAN_TYPE` from `"eps"` to `"v"` (velocity prediction). Requires modifying the loss and prediction code. Better signal-to-noise ratio.
7. **Data augmentation**: Enable `AUGMENT = True`. Also try adding point jitter (small Gaussian noise to coordinates).
8. **Optimizer**: Try AdamW with weight_decay=0.01, or experiment with beta1=0.9 instead of 0.5.

### Lower Priority (explore after fundamentals)
9. **Attention pattern**: Currently alternating layers have attention. Try attention on every layer, or only on the deepest layers.
10. **Fewer timesteps**: Reduce `NUM_TIMESTEPS` from 1000 to 500 or 200 with adjusted schedule.
11. **Mixed precision**: Enable `USE_AMP = True` with `AMP_DTYPE = "bfloat16"` for faster training.
12. **Architecture variants**: Modify sa_blocks/fp_blocks in PVCNN2 class — change channel widths, number of blocks, voxel resolutions.

## Rules

1. Make ONE change at a time (or a small coherent group of related changes).
2. The training script must remain runnable: `python train_pcdiff.py --time-budget 900`
3. After training, the script automatically evaluates and prints Chamfer Distance.
4. If the metric improves, the change is ACCEPTED. If it worsens or the script crashes, REJECTED.
5. Always keep the file self-contained — do not add new imports or external dependencies.
6. Preserve the `Model.gen_samples()` interface — it's used by the evaluation code.

## Current Baseline

The current configuration matches the paper's defaults:
- Linear beta schedule, 1000 timesteps
- Adam optimizer, lr=2e-4, beta1=0.5
- PVCNN2 with embed_dim=64, attention=True, dropout=0.1
- No augmentation, no gradient clipping, no EMA
