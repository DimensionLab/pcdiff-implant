# PCDiff 8x H200 Final Training — DIM-69

## Status: RUNNING
- Latest epoch: 101/5000
- Latest loss: 0.2454
- Best loss: 0.154436 (epoch 98)
- Rate: ~13.5s/epoch, ETA: ~18h total

## Checkpoints Downloaded
- [x] Epoch 50 (712MB)
- [x] Epoch 100 (712MB)

## Loss Progression
| Epoch | Loss | Grad Norm | Best Loss |
|-------|------|-----------|-----------|
| 0 | 1.0364 | 33.6411 | 0.7777 |
| 10 | 0.3125 | 3.3932 | 0.3857 |
| 20 | 0.2005 | 1.4825 | 0.2725 |
| 30 | 0.1158 | 0.3962 | 0.1958 |
| 40 | 0.2031 | 0.2315 | 0.1958 |
| 50 | 0.3095 | 1.1072 | 0.1733 |
| 60 | 0.2175 | 0.2689 | 0.1662 |
| 70 | 0.1542 | 0.3053 | 0.1586 |
| 80 | 0.2679 | 0.2388 | 0.1586 |
| 90 | 0.2245 | 0.2562 | 0.1569 |
| 100 | 0.2735 | 0.2111 | 0.1544 |
| 101 | 0.2454 | 0.1428 | 0.1544 |

## Gating Decisions
- 2026-04-06 06:50:10,202 :   Decision: continue
- 2026-04-06 07:01:10,194 :   Decision: continue

## Configuration
- Node: gpu03 (6× H200, 100% util)
- Architecture: V14 cosine-minsnr
- Batch: 64, LR: 5.66e-4 cosine, warmup 50 epochs
- Loss: mse_minsnr (gamma=5.0), EMA=0.9999, grad_clip=1.0
- Run dir: pcdiff/runs/SkullBreak/20260406_063836/
- Slurm job: 19132
