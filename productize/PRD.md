# DimensionLab CrAInial AI — PCDiff Training PRD (SkullBreak)

This document is the **token-efficient PRD** for getting PCDiff training to **meet or beat** the SkullBreak acceptance criteria, with a **fast “gating” loop** that detects the prior “spiky plateau” early and stops quickly.

Non‑negotiables (must-haves) are embedded throughout, but the source of truth is:
- `paper/pcdiff_paper.pdf` (local) — **must be re-read if unsure**
- `productize/acceptance-criteria.md`
- `productize/inference-e2e.md`
- prior 8×H100 run snapshot under `productize/previous-training/`

## Problem / Goal

- **Goal**: Train a PCDiff model for **SkullBreak only** that achieves:
  - **DSC** ≥ 0.85 (target ≥ 0.87)
  - **bDSC** ≥ 0.87 (target ≥ 0.89)
  - **HD95** ≤ 2.60 mm (target ≤ 2.45 mm)
- **Secondary goal**: Enable a repeatable, multi‑GPU training + evaluation loop that can reliably detect and avoid the **“spiky plateau”** failure mode seen in previous training.

## Scope

- **In scope**:
  - Multi‑GPU training for PCDiff (use *all available GPUs*)
  - Automated early-stop + divergence stop + plateau detection with decision checkpoints at **50 / 100 / 200 / 500 / 700**
  - Evaluation metrics computed **every 50 training epochs** (defined below as “proxy eval” vs “full eval”)
  - End‑to‑end evaluation that runs **both**:
    - **DDIM** (50 steps)
    - **DDPM** (1000 steps)
- **Out of scope (for this PRD)**:
  - SkullFix
  - UI / product app work
  - Dataset creation beyond SkullBreak’s existing CSVs

## Baselines (must align to paper first)

### Paper baseline (PCDiff paper, SkullBreak)
From `paper/pcdiff_paper.pdf`:
- **Points**: total 30,720 = skull 27,648 + implant 3,072
- **Diffusion**: \(T=1000\), linear \(\beta\) schedule \(1e^{-4}\to 0.02\)
- **Optimizer**: Adam, **lr=2e‑4**, **batch=8**, **epochs=15,000**
- **Time embedding**: sinusoidal dim 64 (MLP 64→64→64), concatenated before every SA/FP block
- **Backbone**: PointNet++ style with Point‑Voxel Convs (PVCNN)
- **Attention placement (paper)**: only in specific blocks (see architecture table in paper supplement)

### Observed 8×H100 prior run (what happened)
From `productize/previous-training/run-20251023_193521-8jukhu0i/`:
- **Command args**: `--bs 64 --lr 1.6e-3 --niter 15000` on **8×H100**
  - Note: `1.6e-3` is **linear LR scaling** from paper \(2e^{-4}\times(64/8)\)
- **Loss behavior (logged 0→1999 epochs)**:
  - After ~500 epochs: **mean ~0.187**, with **spikes** (min ~0.03, max ~0.78)
  - This matches the “spiky plateau” description and justifies a **hard gating loop**.

## PRD: Training System Requirements

- **Distributed training**:
  - Training MUST use **all available GPUs** via `torchrun --nproc_per_node=<N>`
  - Deterministic setup: fixed seed, log rank‑0 only for stdout, but aggregate metrics across ranks
- **Logging/artifacts**:
  - Persist: `train.log`, `config.yaml/json`, git commit hash, dataset hash (train/test CSV)
  - Log per step: loss, grad-norm, lr, EMA loss (if used), data timing
  - Log per eval: DSC/bDSC/HD95 + inference settings (sampler, steps, ensemble size)
- **Safety stops**:
  - Stop immediately on NaN/Inf, exploding grad norm, or catastrophic loss divergence
  - Always checkpoint at each decision epoch (50/100/200/500/700)

## PRD: Training Protocol (the “700‑epoch gating loop”)

### Definitions
- “**Epoch**” here is the first counter in existing logs: `[epoch/15000][batch/7]`.
- “**Proxy eval**” = fast, repeatable metric eval used every **50 epochs** to guide early stopping.
- “**Full eval**” = end‑to‑end evaluation on the full SkullBreak test set, run after training stops (see below).

### Phase 0 — paper parity check (mandatory, short)
Run a small sanity training job (e.g., 5–10 epochs) to validate:
- data loads correctly, normalization matches paper ([−3,3] range)
- DDP correctness (no rank drift), deterministic seed behavior
- loss decreases from initial value and doesn’t NaN

### Phase 1 — gating run (max 700 epochs, mandatory)

**Hard cap:** 700 epochs. We decide to continue/stop at **50 / 100 / 200 / 500 / 700**.

#### What runs every epoch (training)
- Forward/backward + optimizer step (DDP)
- Log: loss (per batch + epoch average), lr, grad-norm, data time, step time

#### What runs every 50 epochs (proxy eval; mandatory)
Proxy eval is designed to be **fast** but still predictive.
- **PCDiff inference**:
  - sampler: **DDIM**
  - steps: **50** (match non‑negotiable inference setting early)
  - ensemble: **n=1** for proxy eval (speed); optionally n=3 if budget allows
  - evaluate on a **fixed validation subset** (e.g. 10–20 SkullBreak test volumes) to keep runtime bounded
- **Voxelization + metrics**:
  - run voxelization on the proxy outputs
  - compute **DSC / bDSC / HD95**
- Log the proxy metrics (and the exact subset list) as artifacts.

#### Decision checkpoints (50/100/200/500/700)
At each checkpoint we compute:
- **Loss summary**: running median + 10/90 percentiles over the last 50 epochs
- **Proxy metrics trend**: best‑so‑far + slope over the last 2 proxy eval points

Decision policy (token-efficient, but strict):
- **Stop (divergence)** if any of:
  - NaN/Inf loss, NaN/Inf gradients
  - loss median increases for 2 consecutive checkpoints *and* 90‑percentile spikes worsen
  - proxy DSC collapses (e.g. drops by >0.05 vs best‑so‑far)
- **Stop (plateau)** if both:
  - proxy metrics improve by < **ΔDSC 0.005** and < **ΔbDSC 0.005** over the last 100 epochs, and
  - loss median is in a high-variance band (“spiky plateau”: wide 10/90 spread) without downward trend
- **Continue** otherwise, until the next checkpoint (but never beyond 700).

#### “Escape hatch” interventions (allowed, but bounded)
If the run is trending toward plateau at 100 or 200 (not yet a hard stop), apply at most **one** intervention and reassess at the next checkpoint:
- reduce LR by ×0.5 (or switch from linear LR scaling to sqrt scaling)
- enable grad clipping (e.g. clip-norm 1.0)
- enable EMA of weights for sampling (EMA often stabilizes diffusion sampling quality)

If plateau persists after one intervention → stop at 500 or earlier.

## PRD: Architecture Plan (“paper parity” first, then improvements)

### A. Paper parity checklist (must satisfy)
From the paper supplement architecture table:
- time embedding dim = **64**, concatenated before each SA/FP block
- SA/FP PVConv block counts and voxel resolutions match the paper table
- attention placement matches the table (don’t enable “attention everywhere” unless explicitly experimenting)
- points: **30,720 total**, implant points **3,072**
- diffusion: \(T=1000\), \(\beta\) linear 1e‑4→0.02

### B. Controlled architecture improvements (only after A)
Goal: improve metrics (especially HD95) without destabilizing training.
- **Capacity**: modest width increase (e.g. `width_mult` 1.25–1.5)
- **Attention**: keep paper placement, but allow an experiment that adds attention to one extra block at a time
- **Regularization**: keep dropout ~0.1 initially; only tune once parity is confirmed

## PRD: Hyperparameter Strategy (avoid “spiky plateau”)

### Baseline (paper-true)
- Adam, lr **2e‑4**, batch **8**, epochs 15k (paper reference)

### Multi‑GPU scaling policy (explicit choice per run)
The prior 8×H100 run used **linear LR scaling**: \(2e^{-4} \times (64/8) = 1.6e^{-3}\).
We will explicitly test scaling strategies inside the **700‑epoch gate**:
- **S1 (paper LR)**: keep lr=2e‑4, increase global batch; rely on gradient noise reduction
- **S2 (sqrt scaling)**: lr = \(2e^{-4} \times \sqrt{(B/8)}\)
- **S3 (linear scaling + warmup)**: lr = \(2e^{-4} \times (B/8)\) + warmup (if supported)

Success criterion for scaling choice: best proxy DSC/bDSC + stable loss trend (reduced spikes).

## PRD: Multi‑GPU Inference + Evaluation (E2E)

### Full eval trigger
Run full eval when training stops (50/100/200/500/700), and **commit** results after the eval run.

### Full eval requirements (non‑negotiable)
For the same trained checkpoint, run **two** complete pipelines:

1) **DDIM-50**
- PCDiff inference: `--sampling_method ddim --sampling_steps 50 --num_ens 5`
- Voxelization: compute metrics

2) **DDPM-1000**
- PCDiff inference: `--sampling_method ddpm` (1000 steps implied/forced)
- Voxelization: compute metrics

### Multi‑GPU requirement
Inference MUST use **all available GPUs**:
- shard the test set across ranks (each rank writes a disjoint subset under the same `eval_path`)
- ensure deterministic output naming and safe concurrency (per-sample directory locks or rank-prefixed staging)

### Comparison output (required)
Produce a single comparison artifact containing:
- mean±std of DSC/bDSC/HD95 for DDIM-50 and DDPM-1000
- per-sample metrics diffs (DDPM vs DDIM) to detect regressions

## PRD: Experiment Matrix (minimal set)

Within the gating budget, run:
- **E0**: strict paper parity (paper LR, paper arch placement)
- **E1**: LR scaling S2 (sqrt scaling)
- **E2**: LR scaling S3 (linear + warmup) or LR×0.5 intervention

Pick the best checkpoint by **proxy metrics**, then validate with **full eval**.

## PRD: Operational Process (must follow)

- **Activity log**: update `productize/activity.md` after each change (summary is enough).
- **Branch discipline**: stay on `feat/runpod-improvements`.
- **Commits**:
  - After each **evaluation run** (proxy or full), commit with a message describing:
    - what changed
    - which checkpoint was evaluated
    - DDIM-50 vs DDPM-1000 metrics (if full eval)
    - pointers to logs/artifacts paths
