# Baseline Reproduction Report — 2026-03-24

## Summary

Baseline reproduction for the pcdiff-implant SkullBreak pipeline has been **partially verified** on CPU. Full GPU inference remains blocked by the absence of a CUDA-capable GPU on the development server.

## What Was Verified

### Stage 1: Point Cloud Diffusion (pcdiff)

| Check | Status | Details |
|-------|--------|---------|
| Checkpoint loads | **PASS** | `model_best.pth` — epoch 7719, loss 0.139282, 27.6M params |
| Data loader works | **PASS** | SkullBreakDataset reads test.csv (23 cases × 5 defects = 115 samples) |
| File paths resolve | **PASS** | CSV paths fixed to relative, data loader resolves against CSV dir |
| Sample shape correct | **PASS** | (950, 3) — 1000 points minus 50 nearest neighbors |
| DDP checkpoint compat | **PASS** | No `module.` prefix; `model.` prefix stripped correctly |

**Checkpoint hashes:**
- `model_best.pth`: sha256=26e204df4eb5527783738eeff4b59c5f061aa89a
- `model_latest.pth`: sha256=f6d348cf167bfa8f3951d4de6af1bf0fad9d5db8
- `pcdiff_model_best.pth`: sha256=a8ca1a750efc0d149c13e21b6de982717a057097

### Stage 2: Voxelization

| Check | Status | Details |
|-------|--------|---------|
| Checkpoint loads | **PASS** | `model_best.pt` — 129 state dict keys |
| Config files valid | **PASS** | 5 YAML configs with env-var interpolation |
| pytorch3d dependency | **BLOCKED** | Not installed (requires CUDA for build) |

**Checkpoint hash:**
- `model_best.pt`: sha256=ea4e992cf2b89750353d5129a54be7c75bc17092

### Environment

| Component | Status |
|-----------|--------|
| Python 3.10.20 | OK |
| torch 2.5.0+cpu | OK (CPU only) |
| torchvision | MISSING |
| numpy, nrrd, mcubes, trimesh | OK |
| open3d, torch_scatter | OK |
| pytorch3d | MISSING (needs CUDA) |
| CUDA / nvidia-smi | NOT AVAILABLE |

## Gaps Between Claimed and Reproducible Performance

### Critical

1. **No GPU available on dev server** — Cannot run actual inference (DDPM/DDIM sampling). Stage 1 inference requires CUDA for reasonable runtime (~1.5 min per case with DDIM-50 on GPU vs hours on CPU).

2. **CSV paths were broken** — Original CSVs had absolute macOS developer paths (`/Users/michaltakac/...`) and referenced `.nrrd` files instead of `_surf.npy`. Fixed in this session.

3. **pytorch3d not installable without CUDA** — Stage 2 voxelization requires pytorch3d which needs a CUDA dev toolkit to compile.

### Medium

4. **torchvision missing** — Not installed in current venv. May be needed for some transforms.

5. **No published baseline metrics** — The repo does not include reference Dice/bDice/HD95 scores from the original MICCAI 2023 paper runs. Without ground truth numbers, we cannot confirm reproduction matches claimed performance.

## Reproduction Steps (Once GPU Available)

```bash
cd /home/mike/pcdiff-implant
source .venv/bin/activate

# Stage 1: Generate point cloud completions (DDIM-50, ensemble=1)
python pcdiff/test_completion.py \
  --path pcdiff/datasets/SkullBreak/test.csv \
  --dataset SkullBreak \
  --model pcdiff/checkpoints/model_best.pth \
  --eval_path results/skullbreak_baseline \
  --num_points 1000 --num_nn 50 \
  --sampling_method ddim --sampling_steps 50 \
  --num_ens 1 --gpu 0

# Stage 2: Voxelize point clouds to mesh
python voxelization/generate.py \
  --config voxelization/configs/gen_skullbreak.yaml

# Metrics will be in:
# results/skullbreak_baseline/benchmark_cases.csv
# results/skullbreak_baseline/benchmark_summary.json
```

## Changes Made

1. **Fixed CSV paths** — Rewrote `train.csv`, `test.csv`, `skullbreak.csv` from absolute macOS paths to relative paths with `_surf.npy` suffix
2. **Updated data loader** — `skullbreak_data.py` now resolves relative CSV paths against the CSV file's directory
