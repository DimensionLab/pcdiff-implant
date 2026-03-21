# End-to-End Inference Pipeline

This document describes how to run the complete PCDiff + Voxelization pipeline for cranial implant generation and evaluation.

## Pipeline Overview

```
Defective Skull (512^3 volume or point cloud)
    ↓
PCDiff Model (Point Cloud Diffusion)
    ↓
Implant Point Cloud (3072 points)
    ↓
Voxelization Model (Neural Surface Reconstruction)
    ↓
Watertight Implant Mesh + Volume
    ↓
Evaluation Metrics (DSC, bDSC, HD95)
```

## Model Checkpoints

| Model | Checkpoint Location |
|-------|---------------------|
| PCDiff | `pcdiff/checkpoints/model_best.pth` |
| Voxelization | `voxelization/checkpoints/model_best.pt` |

## Step 1: PCDiff Inference

Generate implant point clouds from the test set:

```bash
cd /workspace/pcdiff-implant
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="8.0"

python pcdiff/test_completion.py \
    --path pcdiff/datasets/SkullBreak/test.csv \
    --dataset SkullBreak \
    --model pcdiff/checkpoints/model_best.pth \
    --eval_path pcdiff/output/inference_results \
    --sampling_method ddim \
    --sampling_steps 50 \
    --num_ens 5 \
    --gpu 0
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--sampling_method` | `ddpm` (slow, 1000 steps) or `ddim` (fast, configurable) |
| `--sampling_steps` | Number of diffusion steps (50 for DDIM is good) |
| `--num_ens` | Ensemble samples per skull (5 recommended) |
| `--eval_path` | Output directory for results |

### Output Structure

```
pcdiff/output/inference_results/syn/
├── <sample_name>/
│   ├── input.npy      # Defective skull point cloud
│   ├── sample.npy     # Generated implant point clouds [num_ens, N, 3]
│   ├── shift.npy      # Normalization shift
│   └── scale.npy      # Normalization scale
```

## Step 2: Voxelization

Convert point clouds to volumetric meshes and evaluate:

```bash
cd /workspace/pcdiff-implant/voxelization

python generate.py configs/gen_skullbreak.yaml
```

### Configuration (`configs/gen_skullbreak.yaml`)

Key settings to verify/update:

```yaml
data:
  path: ../pcdiff/output/inference_results/syn  # Path to PCDiff output
  dset: SkullBreak

test:
  model_file: checkpoints/model_best.pt  # Voxelization model

generation:
  num_ensemble: 5  # Must match PCDiff --num_ens
  compute_eval_metrics: true
  save_ensemble_implants: false
```

### Output

For each sample:
- `mean_impl.nrrd` - Final averaged implant volume
- `eval_metrics.yaml` - DSC, bDSC, HD95 metrics

## Step 3: Aggregate Evaluation Results

After voxelization completes, aggregate the evaluation metrics:

```bash
cd /workspace/pcdiff-implant

python -c "
import os
import yaml
import numpy as np
from pathlib import Path

results_dir = Path('pcdiff/output/inference_results/syn')
metrics = {'dice': [], 'bdice': [], 'haussdorf95': []}

for sample_dir in results_dir.iterdir():
    metrics_file = sample_dir / 'eval_metrics.yaml'
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = yaml.safe_load(f)
            for k in metrics:
                if k in m:
                    metrics[k].append(m[k])

print('Evaluation Results (SkullBreak Test Set)')
print('=' * 50)
print(f'Samples: {len(metrics[\"dice\"])}')
print(f'DSC:   {np.mean(metrics[\"dice\"]):.4f} ± {np.std(metrics[\"dice\"]):.4f}')
print(f'bDSC:  {np.mean(metrics[\"bdice\"]):.4f} ± {np.std(metrics[\"bdice\"]):.4f}')
print(f'HD95:  {np.mean(metrics[\"haussdorf95\"]):.2f} ± {np.std(metrics[\"haussdorf95\"]):.2f} mm')
"
```

## Single Skull Inference

For a single defective skull input:

```bash
python run_single_inference.py \
    --input path/to/defective_skull.npy \
    --pcdiff_model pcdiff/checkpoints/model_best.pth \
    --vox_model voxelization/checkpoints/model_best.pt \
    --output_dir results/my_skull \
    --num_ens 5 \
    --gpu 0
```

### Outputs

- `skull_complete.ply/stl` - Complete skull with implant
- `implant_only.ply/stl` - Generated implant mesh
- `implant_volume.nrrd` - Implant as 3D volume
- Various point cloud exports

## Acceptance Criteria

Target metrics from acceptance-criteria.md:

| Metric | Minimum | Target |
|--------|---------|--------|
| DSC | ≥ 0.85 | ≥ 0.87 |
| bDSC | ≥ 0.87 | ≥ 0.89 |
| HD95 | ≤ 2.60 mm | ≤ 2.45 mm |

## Troubleshooting

### CUDA Out of Memory

Reduce ensemble size or use DDIM with fewer steps:
```bash
--num_ens 1 --sampling_method ddim --sampling_steps 25
```

### Path Issues

Ensure paths in `gen_skullbreak.yaml` point to correct locations. The `data.path` should be the directory containing PCDiff inference outputs.

### Missing Ground Truth

Evaluation metrics require the SkullBreak dataset with ground truth implants in:
```
datasets/SkullBreak/implant/<sample_name>/<xxx>.nrrd
```
