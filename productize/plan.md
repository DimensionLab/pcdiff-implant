# Project Plan

## Overview
Build a **repeatable, multi‑GPU** training + inference + evaluation harness for **PCDiff on SkullBreak**, with a hard **700‑epoch gating loop** (decision points at 50/100/200/500/700), **proxy eval every 50 epochs**, and **full E2E eval** comparing **DDIM‑50** vs **DDPM‑1000**.

**Reference:** `PRD.md`

---

## Task List

```json
[
  {
    "category": "setup",
    "description": "Confirm baseline assets and repo state (paper, dataset splits, prior-run logs, branch)",
    "steps": [
      "Verify `paper/pcdiff_paper.pdf` is present and readable",
      "Verify SkullBreak CSVs exist and are consistent: `datasets/SkullBreak/train.csv` and `datasets/SkullBreak/test.csv`",
      "Verify prior 8xH100 run snapshot exists under `productize/previous-training/` and extract key args/loss stats",
      "Verify git branch is `feat/runpod-improvements` and working tree is clean or changes are intentional"
    ],
    "passes": true
  },
  {
    "category": "setup",
    "description": "Standardize experiment artifact layout + reproducibility metadata",
    "steps": [
      "Define a single run directory schema for training and eval outputs (configs, logs, checkpoints, metrics)",
      "Ensure every run records: git commit hash, CLI args, dataset hash (CSV), GPU count, and seed",
      "Ensure rank-0 writes canonical logs while all ranks can write sharded inference outputs safely",
      "Document artifact locations in `productize/activity.md` after first run is created"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement/verify true multi-GPU training invocation (DDP) uses all available GPUs",
    "steps": [
      "Add or update a single entrypoint script/command for PCDiff training using `torchrun --nproc_per_node=<N>`",
      "Verify DDP backend is NCCL and ranks initialize correctly",
      "Verify per-rank batch size and effective global batch are logged",
      "Run a short sanity training (5–10 epochs) to confirm stable loss and no NaNs"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Implement the 700-epoch gating loop with decision checkpoints (50/100/200/500/700)",
    "steps": [
      "Add configuration for decision epochs and hard max (700)",
      "Implement stop-on-divergence rules (NaN/Inf, exploding gradients, catastrophic loss behavior)",
      "Implement plateau detection rules (high-variance loss band + no proxy metric improvement)",
      "Ensure checkpoints are saved at each decision epoch and are uniquely identifiable",
      "Verify that wandb is installed and logs and best model are stored there"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Proxy evaluation every 50 epochs (fast metrics loop)",
    "steps": [
      "Define a fixed validation subset of SkullBreak test cases (10–20) and persist the list as an artifact",
      "Run PCDiff inference for proxy eval using DDIM-50 (and n=1 ensemble by default)",
      "Run voxelization + metrics (DSC/bDSC/HD95) on the proxy outputs",
      "Log proxy metrics + settings as structured artifacts and append summary to `productize/activity.md`"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Multi-GPU inference sharding for test-time sampling (required for DDIM-50 and DDPM-1000)",
    "steps": [
      "Shard the SkullBreak test set across ranks deterministically (no overlaps, no gaps)",
      "Ensure output directories are concurrency-safe (one sample dir per case, no clobbering)",
      "Verify GPU utilization across all devices during inference",
      "Verify a full test-set inference completes without missing outputs"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Full E2E evaluation harness (DDIM-50 vs DDPM-1000) with comparable outputs",
    "steps": [
      "Run full test-set inference with DDIM-50 (num_ens=5) and compute voxelization metrics",
      "Run full test-set inference with DDPM-1000 (num_ens=5) and compute voxelization metrics",
      "Aggregate and store comparison artifact: mean±std and per-sample diffs (DDPM vs DDIM)",
      "Record final metrics against acceptance criteria thresholds in `productize/activity.md`"
    ],
    "passes": true
  },
  {
    "category": "feature",
    "description": "Run the minimal experiment matrix inside the gating budget (E0/E1/E2) and select best checkpoint",
    "steps": [
      "E0: strict paper parity run (paper LR and paper-aligned architecture settings) through gating",
      "E1: run sqrt LR scaling strategy through gating",
      "E2: run linear+warmup (if supported) or a single bounded intervention (e.g. LR×0.5 / grad clip / EMA) through gating",
      "Select best checkpoint based on proxy metrics, then validate with full E2E eval"
    ],
    "passes": true
  },
  {
    "category": "testing",
    "description": "Acceptance criteria verification and regression guardrails",
    "steps": [
      "Verify final DDIM-50 and DDPM-1000 metrics meet or exceed minimum thresholds (DSC≥0.85, bDSC≥0.87, HD95≤2.60)",
      "If thresholds are met, verify whether targets are hit (DSC≥0.87, bDSC≥0.89, HD95≤2.45)",
      "Store a frozen evaluation report artifact tied to the winning checkpoint and commit hash",
      "Add a lightweight reproducibility checklist to `productize/activity.md` for re-runs"
    ],
    "passes": true
  },
  {
    "category": "process",
    "description": "Commit discipline for evaluation runs (required operational rule)",
    "steps": [
      "After each proxy eval or full eval run, create a git commit on `feat/runpod-improvements`",
      "Commit message must include: what changed, checkpoint evaluated, and key metrics (DDIM-50 vs DDPM-1000 if full eval)",
      "Commit body must include pointers to run directories/logs/metrics artifacts",
      "Ensure `productize/activity.md` is updated in the same commit with a dated summary entry"
    ],
    "passes": true
  },
  {
    "category": "custom",
    "description": "You now have 8x H100 GPU. Improve pcdiff model towards baseline evaluation metrics using DDPM-1000, using uv's .venv (`source .venv/bin/activate`) and wandb for storing run logs (already logged in)",
    "steps": [
      "Start with currently trained pcdiff model to run full DDPM-1000 eval",
      "Do a hyperparameter search to find the best setup for improving model towards evaluation metrics, running max 25-50 epoch training runs, doing full eval using DDPM-1000, 1 ensemble, on max 2 samples",
      "If thresholds are met, verify whether targets are hit (DSC≥0.87, bDSC≥0.89, HD95≤2.45)",
      "If after few hyperparameter searches the targets are not met, start improving model architecture based on the latest research into the area (re-read the paper @paper/pcdiff_paper.pdf, check latest research on the web)",
      "Again, if thresholds are met, verify whether targets are hit (DSC≥0.87, bDSC≥0.89, HD95≤2.45)",
      "Go back to hyperparameter search if targets are not being hit and start over",
      "Once evaluation metrics compared to baseline are satisfactory, store a frozen evaluation report artifact tied to the winning checkpoint and commit hash",
      "Commit body must include pointers to run directories/logs/metrics artifacts",
      "Ensure `productize/activity.md` is updated in the same commit with a dated summary entry"
    ],
    "passes": true
  },
  {
    "category": "custom",
    "description": "Run a full training run for pcdiff model on 8x H100 GPU, evaluation metrics using DDPM-1000, using uv's .venv (`source .venv/bin/activate`) and wandb for storing run logs (already logged in). We have only limited budget of 1000€, current server costs 19.12€/hour!",
    "steps": [
      "Only continue if you re-read the paper @paper/pcdiff_paper.pdf",
      "Set up correct hyperparameters for multi-GPU training of pcdiff and inference runs",
      "Evaluate agains target baselines (DSC≥0.87, bDSC≥0.89, HD95≤2.45)",
      "Store an evaluation report, winning checkpoint and commit hash",
      "Commit body must include pointers to run directories/logs/metrics artifacts",
      "Ensure `productize/activity.md` is updated in the same commit with a dated summary entry"
    ],
    "passes": false
  }
]
```

---

## Agent Instructions
- Read `activity.md` first to understand current state
- Find next task with `"passes": false`
- Complete all steps for that task
- Verify in browser
- Update task to `"passes": true`
- Log completion in `activity.md`
- Repeat until all tasks pass

Important: Only modify the passes field. Do not remove or rewrite tasks.

### Completion Criteria

All tasks marked with `"passes": true`

### Updating activity.md

This file logs what the agent accomplishes during each iteration:

```markdown
# Project Build - Activity Log

## Current Status
**Last Updated:** 
**Tasks Completed:** 
**Current Task:** 

---

## Session Log

<!-- Agent will append dated entries here -->
```

