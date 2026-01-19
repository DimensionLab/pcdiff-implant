# Project Build - Activity Log

## Current Status
**Last Updated:** 2026-01-19
**Tasks Completed:** PRD drafted; plan harness created; prior-run forensics captured; baseline assets verified
**Current Task:** Task 2 - Standardize experiment artifact layout + reproducibility metadata 

---

## Session Log

### 2025-01-15 10:00:00 - Initial setup

This is a first note in this activity sheet. We will track all of the activities done during the productization of the point cloud diffusion + voxelization models for automatic cranial implant generation (this project). Our goal is to reproduce the evaluation metrics - DSC close to 0.87, bDSC close to 0.89, and HD95 close to 2.45.

We need to make sure we can properly evaluate the evaluation metrics and "visual" (spatial) shape and positioning of the generated implant within the skull defect. We only care about SKullBreak dataset.

### 2026-01-19 00:00:00 - Training PRD + prior-run forensics (no eval run yet)

Summary:
- Read `paper/pcdiff_paper.pdf` and extracted the **paper-true** training/eval baseline (SkullBreak: 30,720 points, \(T=1000\), \(\beta\) 1e-4→0.02 linear, Adam lr 2e-4, batch 8, 15k epochs; time-embed dim 64; attention placement per paper table).
- Audited prior **8×H100** training snapshot under `productize/previous-training/run-20251023_193521-8jukhu0i/`:
  - args show `--bs 64 --lr 1.6e-3 --niter 15000` (linear LR scaling from the paper’s lr 2e-4 @ bs=8)
  - `files/output.log` contains epochs 0→1999; from ~500 onward the loss shows a “spiky plateau” band (high variance, mean ~0.187 for epochs ≥500; spikes up to ~0.78).
- Updated `productize/PRD.md` and created `productize/plan.md` harness describing:
  - a hard **700-epoch gating loop** with continue/stop decisions at **50/100/200/500/700**
  - **proxy eval every 50 epochs** and a required **full eval** that compares **DDIM-50** vs **DDPM-1000**
  - requirement that training + inference use **all available GPUs**

Files changed:
- `productize/PRD.md`
- `productize/plan.md`
- `productize/activity.md` (this entry reformats the log)

Commits:
- None in this step (no evaluation run executed yet; commits will be created after each eval run as required).

### 2026-01-19 00:00:00 - Ensure W&B is in training env + log best checkpoint as W&B artifact (no eval run yet)

Summary:
- Made `wandb` a **default dependency** in `pyproject.toml` so `uv pip install -e .` includes it for the PCDiff training venv.
- Updated `pcdiff/train_completion.py` so rank‑0 uploads the **best checkpoint** (`model_best.pth`) to Weights & Biases as an **Artifact** (type `model`, aliases `best` and `epoch_<N>`), enabling manual download later.
- Added a friendlier log message reminding to run `wandb login` / set `WANDB_API_KEY` (you’ll do login manually).

Files changed:
- `pyproject.toml`
- `pcdiff/train_completion.py`
- `productize/activity.md` (this entry)

Commits:
- None in this step (no evaluation run executed yet).

### 2026-01-19 10:00:00 - Baseline assets verification (Task 1)

Summary:
- Verified all baseline assets required for the training harness:
  1. **Paper PDF**: `paper/pcdiff_paper.pdf` present (44MB)
  2. **SkullBreak CSVs**: Located at `datasets/SkullBreak/` (not `pcdiff/datasets/`)
     - `train.csv`: 86 samples (skulls 000-085)
     - `test.csv`: 28 samples (skulls 086-113)
  3. **Prior 8xH100 run**: `productize/previous-training/run-20251023_193521-8jukhu0i/`
     - Args: `--bs 64 --lr 1.6e-3 --wandb-project pcdiff`
     - Git commit: `33a4296d2654505d60558e23ad4d62a04248c343`
     - GPUs: 8x NVIDIA H100 80GB HBM3
     - Logged epochs 0-1999; "spiky plateau" confirmed (loss range ~0.03-0.47 after epoch 500)
  4. **Git branch**: Confirmed `feat/runpod-improvements`

Note: The CSV path in `plan.md` said `pcdiff/datasets/SkullBreak/` but actual location is `datasets/SkullBreak/`. The training script uses relative paths from CSVs.

Files verified:
- `paper/pcdiff_paper.pdf`
- `datasets/SkullBreak/train.csv`
- `datasets/SkullBreak/test.csv`
- `productize/previous-training/run-20251023_193521-8jukhu0i/files/output.log`
- `productize/previous-training/run-20251023_193521-8jukhu0i/files/wandb-metadata.json`

Task 1 marked as `passes: true` in `productize/plan.md`.