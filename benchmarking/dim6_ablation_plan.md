# DIM-6: Stage-1 Sampling Ablation Plan

Date: 2026-04-10
Author: CTO (ba6d3030)
Status: IN PROGRESS — awaiting corrected stage-2 eval results from PERUN

## Context

DIM-5 proposed architecture upgrades. The near-term track is inference speedup
via DDIM step reduction, tested against the locked 6GPU PCDiff model (cosine
schedule, embed_dim=96, width_mult=1.5, mse_minsnr loss).

### Locked Baseline (published PCDiff n=5)
- DSC: 0.87
- bDSC (10mm): 0.89
- HD95: 2.45mm

### Acceptance Thresholds
- DSC >= 0.85 (minimum), target >= 0.87
- bDSC >= 0.87 (minimum), target >= 0.89
- HD95 <= 2.60mm (maximum), target <= 2.45mm

### Models
- PCDiff 6GPU: `pcdiff/runs/SkullBreak/20260407_181433/checkpoints/model_best.pth`
- Voxelization retrained: `voxelization/out/skullbreak_6gpu/model_best.pt`
  (trained on DDIM-50 6GPU outputs, best val psr_l2=0.4298 at epoch 100)

### Critical Blocker
ALL prior stage-2 evaluations produced "no valid results" because they used
the OLD voxelization model (trained on embed_dim=64 outputs). The corrected
eval (PERUN job 20004, dependency on retrain job 19732) should resolve this.

---

## Experiment Matrix

### Phase 1: DDIM Step Sweep (Primary — SPEED vs QUALITY tradeoff)

All configs use the 6GPU PCDiff model, cosine schedule, ens=1.

| Config | DDIM Steps | Ensemble | Per-Sample Time (H200) | Stage-1 Data |
|--------|-----------|----------|----------------------|--------------|
| ddim-10 | 10 | 1 | ~8.9s | EXISTS (sweep 19586) |
| ddim-15 | 15 | 1 | ~13.4s | EXISTS (sweep 19586) |
| ddim-20 | 20 | 1 | ~17.8s | EXISTS (sweep 19586) |
| ddim-25 | 25 | 1 | ~22.3s | EXISTS (sweep 19586) |
| ddim-35 | 35 | 1 | ~31.3s | EXISTS (sweep 19586) |
| ddim-50 | 50 | 1 | ~44.5s | EXISTS (sweep 19586) |
| ddim-100 | 100 | 1 | ~88.9s | EXISTS (sweep 19413) |
| ddim-200 | 200 | 1 | ~177.8s | EXISTS (sweep 19413) |
| ddim-250 | 250 | 1 | ~222s | EXISTS (sweep DIM-79) |
| ddpm-1000 | 1000 | 1 | ~890s (est) | NEEDS GENERATION |

Note: DDIM-5 eliminated (100% NaN outputs). DDPM-1000 is the quality ceiling
but ~20x slower than DDIM-50 — included as upper bound reference only.

### Phase 2: Ensemble Sweep (on top 2-3 fastest non-regressing configs)

| Config | DDIM Steps | Ensemble | Est. Time | Stage-1 Data |
|--------|-----------|----------|----------|--------------|
| ddim-50-ens3 | 50 | 3 | ~133s | EXISTS (DIM-79) |
| ddim-50-ens5 | 50 | 5 | ~222s | NEEDS GENERATION |
| ddim-25-ens3 | 25 | 3 | ~67s | NEEDS GENERATION |
| ddim-25-ens5 | 25 | 5 | ~112s | NEEDS GENERATION |

Ensemble configs use majority voting (legacy strategy) in voxelization.

### Phase 3: Candidate Reranking vs Majority Voting

For ensemble configs (ens >= 2), compare:
1. **Legacy (majority voting)**: `completes >= ceil(num_ensemble/2)` — current default
2. **Per-case best selection**: For each of the N ensemble members, compute
   individual DSC against GT. Pick the single best member per case.
3. **Confidence-weighted**: Weight each ensemble member's output by the
   voxelization model's confidence (inverse of psr_l2 magnitude near surface).

This phase reuses the SAME stage-1 outputs from Phase 2 — only the stage-2
aggregation strategy changes. Implemented as a post-processing comparison,
not a separate generation run.

---

## Execution Plan

### Step 1: Verify Corrected Stage-2 Results (CRITICAL PATH)

Once VPN/PERUN is accessible:
1. Check if job 19732 (vox retrain) completed
2. Check if job 20004 (corrected s2 eval) completed
3. Retrieve `benchmark_summary.json` for each tested config
4. Record DSC, bDSC, HD95 for DDIM-25, DDIM-50, DDIM-100, DDIM-200

### Step 2: Run Missing Stage-1 Generations

Submit to PERUN (H200):
- DDPM-1000-ens1 (quality ceiling reference)
- DDIM-50-ens5 (if DDIM-50-ens3 shows improvement)
- DDIM-25-ens3 and DDIM-25-ens5 (if DDIM-25 solo is non-regressing)

### Step 3: Run Stage-2 Evaluation for All Phase 1 + Phase 2 Configs

Using the retrained 6GPU voxelization model:
```bash
VOX_MODEL="voxelization/out/skullbreak_6gpu/model_best.pt"
# For each config: run voxelization/generate.py with correct num_ensemble
```

### Step 4: Reranking Comparison (Phase 3)

For each ensemble config:
1. Generate individual ensemble member meshes (save_ensemble_implants: True)
2. Compute per-member DSC against GT
3. Compare: majority vote DSC vs best-member DSC vs confidence-weighted DSC
4. Report delta

### Step 5: Selection Decision

Using `benchmarking/select_stage1_candidate.py`:
1. Input all benchmark_summary.json files
2. Apply decision rule: fastest config where DSC >= baseline, bDSC >= baseline, HD95 <= baseline
3. Output selection_report.json
4. Archive all benchmark artifacts

---

## Current Blockers

1. **VPN down** — cannot access PERUN to check job status or retrieve results
2. **Retrained vox model** — need confirmation that job 19732 finished and
   model is usable (best at epoch 100, val_psr_l2=0.4298)
3. **No valid stage-2 metrics** — everything depends on corrected eval results

## Decision Rule (from workflow spec)

Select the FASTEST stage-1 configuration whose downstream stage-2 metrics do
NOT regress against locked baseline:
- mean dice >= baseline
- mean bdice_10mm >= baseline
- mean hd95_mm <= baseline
