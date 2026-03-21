# SkullBreak Evaluation (DDPM, Ensemble)

- PCDiff checkpoint: `pcdiff/output/train_completion/2025-10-24-06-39-48/epoch_14999.pth`
- Voxelization checkpoint: `voxelization/out/skullbreak/model_best.pt`
- Dataset CSV: `pcdiff/datasets/SkullBreak/test.csv`
- Sampling: **ddpm** with **50** steps
- Ensemble size: **1**
- GPUs used: `0,1,2,3,4,5,6,7`

## Mean Metrics

| Metric | Paper (Ours n=5) | This run | Δ |
|--------|------------------|----------|----|
| DICE | 0.8700 | 0.0003 | -0.8697 |
| BDICE | 0.8900 | 0.0003 | -0.8897 |
| HD95 | 2.4500 | 193.2715 | +190.8215 |

## Per-case Summary (first 10 rows)

| Case | Defect | DSC | bDSC | HD95 |
|------|--------|-----|------|------|
| 003 | bilateral | 0.0000 | 0.0000 | 196.1262 |
| 003 | frontoorbital | 0.0000 | 0.0000 | 102.7003 |
| 003 | parietotemporal | 0.0000 | 0.0000 | 164.6589 |
| 003 | random_1 | 0.0000 | 0.0000 | 217.3841 |
| 003 | random_2 | 0.0000 | 0.0000 | 186.7055 |
| 004 | bilateral | 0.0000 | 0.0000 | 166.6080 |
| 004 | frontoorbital | 0.0000 | 0.0000 | 178.6590 |
| 004 | parietotemporal | 0.0000 | 0.0000 | 226.4219 |
| 004 | random_1 | 0.0000 | 0.0000 | 197.7024 |
| 004 | random_2 | 0.0000 | 0.0000 | 161.9180 |

_Full metrics saved alongside this report._
