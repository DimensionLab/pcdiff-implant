# Inference Speedup Research Report — CrAInial Pipeline

**Date:** 2026-04-03
**Author:** CTO (ba6d3030)
**Issue:** DIM-64

---

## Executive Summary

This report surveys state-of-the-art techniques for speeding up inference in our two-stage cranial implant pipeline (PCDiff point cloud diffusion + voxelization surface reconstruction). Key finding: **PCDiff is NOT the accuracy SOTA** for cranial implants (DSC ~0.86 vs 0.94+ from Wodzinski et al. 2024), but its probabilistic multi-proposal generation is a genuine differentiator. Multiple techniques can yield **50-1000x total inference speedup** with minimal accuracy loss.

---

## Part 1: Competitive Landscape — Is PCDiff State-of-the-Art?

### PCDiff Performance (Our Baseline)
- SkullBreak: DSC ~0.86, bDSC ~0.88
- SkullFix: DSC ~0.90, bDSC ~0.92-0.93, HD95 ~1.69-1.73

### Competing Methods (Ranked by SkullBreak DSC)

| Method | Year | DSC (SkullBreak) | Notes |
|--------|------|-------------------|-------|
| Wodzinski et al. (LDM aug + CNN) | 2024 | **>0.94** | Self-claims "most accurate and robust" |
| Deep Learnable Symmetry | 2024 | Competitive w/ above | Exploits bilateral skull symmetry |
| 2D U-Net baseline | 2021 | ~0.87 | Simple but effective |
| **PCDiff (ours)** | 2023 | **~0.86** | Point cloud diffusion |
| CraNeXt | 2024 | ~0.80 | Skull categorization |
| 3D Sparse U-Net | 2021 | ~0.71 | Sparse approach |

### Direct Competitors to PCDiff
1. **"Rectified Flow for Efficient Automatic Implant Generation"** (Zhou et al., Feb 2025) — directly targets PCDiff's slow multi-step sampling, uses rectified flow for faster inference
2. **Occupancy Networks for Cranioplasty** (Mazzocchetti et al., 2024, IEEE Access) — claims "fast and resource-efficient" vs PCDiff
3. **IDNet** (Ji et al., 2025) — diffusion model variant for cranio-maxillofacial defects

### Assessment
PCDiff's value is **methodological novelty** (first point cloud diffusion for implants) and **probabilistic generation** (multiple proposals), NOT raw accuracy. Our marketing should say "competitive diffusion-based approach" not "state-of-the-art."

---

## Part 2: Stage-1 (PCDiff) Speedup Techniques

### 2.1 DPM-Solver++ — Drop-in Fast Sampler (NO RETRAINING)

- **Paper:** Lu et al. 2022, arXiv:2211.01095
- **What:** High-order ODE solver for diffusion reverse process. 15-20 steps instead of 1000.
- **Speedup:** ~50-66x
- **Risk:** Very low — near-identical quality to 1000-step DDPM in image/3D domains
- **Effort:** Zero retraining, just swap the sampler
- **Priority:** ★★★★★ IMMEDIATE WIN

### 2.2 Consistency Models / Distillation

- **Paper:** Song et al. 2023, arXiv:2303.01469
- **3D-specific:** ConTiCoM-3D (Eilermann et al. 2025, arXiv:2509.01492) — consistency models for 3D point clouds
- **What:** Learn to map any ODE trajectory point to clean data in 1-2 steps
- **Speedup:** 100-500x (1-2 steps)
- **Risk:** Medium — 1 step may lose boundary detail; 2-4 steps likely sufficient for clinical quality
- **Effort:** Distillation training from existing DDPM
- **Priority:** ★★★★☆ SHORT-TERM

### 2.3 Progressive Distillation

- **Paper:** Salimans & Ho 2022, arXiv:2202.00512
- **What:** Iteratively halve the steps: 1000→500→...→4 steps
- **Speedup:** 125-250x (4-8 steps)
- **Risk:** Low-medium — controlled quality/speed trade-off at each round
- **Effort:** Multiple distillation rounds, moderate compute
- **Priority:** ★★★☆☆ ALTERNATIVE to consistency models

### 2.4 Flow Matching (Architecture Change)

- **Paper:** Lipman et al. 2022, arXiv:2210.02747
- **3D-specific:** "Not-So-Optimal Transport Flows for 3D Point Cloud Generation" 2025, arXiv:2502.12456
- **What:** Replace DDPM with flow matching — straighter ODE trajectories = fewer steps
- **Speedup:** 20-100x (10-50 steps)
- **Risk:** Low — same architecture, different training objective
- **Effort:** Full retrain
- **Priority:** ★★★★☆ for next major model version

### 2.5 Sparse Point-Voxel Diffusion (Architecture Speedup)

- **Paper:** SPVD, Romanelis et al. 2024, arXiv:2408.06145
- **What:** Replace dense 3D convolutions with sparse convolutions (skip empty space)
- **Speedup:** 3-10x per step + scales to higher point counts
- **Risk:** None (orthogonal to sampling speedups)
- **Effort:** Architecture change + retrain
- **Priority:** ★★★☆☆ LONG-TERM, combines multiplicatively with above

---

## Part 3: Stage-2 (Voxelization) Speedup Techniques

### 3.1 NKSR + TorchSparse++ (Sparse Convolutions)

- **Paper:** Huang et al. (NVIDIA) CVPR 2023, arXiv:2305.19590
- **Engine:** TorchSparse++, arXiv:2311.12862
- **What:** Sparse voxel convolutions for neural kernel surface reconstruction
- **Speedup:** 10-50x vs dense voxel approaches
- **Applicability:** Direct — cranial implants are sparse 3D structures
- **Priority:** ★★★★☆ HIGH

### 3.2 Hash-Grid Neural Representations (Instant-NGP style)

- **Paper:** Neuralangelo, Li et al. (NVIDIA) CVPR 2023, arXiv:2306.03092
- **Foundation:** Instant-NGP, Müller et al. 2022, arXiv:2201.05989
- **What:** Multi-resolution hash tables replace dense feature grids
- **Speedup:** 20-100x vs MLP-based implicit functions
- **Applicability:** Adaptable — replace dense SDF grid with hash encoding
- **Priority:** ★★★☆☆ MEDIUM (requires architecture adaptation)

### 3.3 Convolutional Occupancy Networks

- **Paper:** Peng et al. (MPI) ECCV 2020, arXiv:2003.04618
- **What:** Feature plane decomposition reduces 3D to three 2D conv passes
- **Speedup:** 5-15x vs vanilla occupancy networks
- **Priority:** ★★★☆☆ MEDIUM

### 3.4 Mixed Precision + TensorRT Deployment

- **Paper:** Mixed Precision PointPillars, arXiv:2601.12638
- **What:** FP16/INT8 quantization + TensorRT graph optimization
- **Speedup:** 2-4x on top of any approach
- **Applicability:** Universal — applies to all neural components
- **Priority:** ★★★★★ ALWAYS DO THIS for production deployment

### 3.5 Model Pruning + Knowledge Distillation

- **Paper:** 3D PC Network Pruning, arXiv:2408.14601
- **What:** Remove 50-80% of weights with <1% accuracy drop
- **Speedup:** 2-5x (pruning) or 4-10x (pruning + quantization)
- **Priority:** ★★★☆☆ DEPLOYMENT OPTIMIZATION

---

## Part 4: Recommended Strategy

### Phase 1 — Immediate (No Retraining Required)
1. **Apply DPM-Solver++ to PCDiff** — swap sampler, ~50x speedup on stage-1
2. **Apply FP16 mixed precision** to both stages — ~2x additional speedup
3. **Benchmark current inference time** end-to-end to establish baseline

**Expected result:** ~100x total speedup on stage-1, ~2x on stage-2

### Phase 2 — Short-Term (Distillation, 1-2 weeks)
1. **Consistency distillation** of PCDiff to 2-4 step model
2. **TensorRT export** for both stages
3. Validate clinical metrics (DSC, bDSC, HD95) are preserved

**Expected result:** ~250-500x stage-1 speedup, ~4x stage-2 speedup

### Phase 3 — Next Model Version (1-2 months)
1. **Retrain with flow matching** instead of DDPM
2. **Adopt SPVD sparse architecture** for diffusion backbone
3. **Replace voxelization with NKSR** or hash-grid approach
4. **Apply consistency distillation** on top of flow matching model

**Expected result:** ~500-1000x stage-1, ~50x stage-2

### Phase 4 — Accuracy Improvement (Parallel Track)
1. **Study Wodzinski et al. 2024** approach (LDM augmentation + CNN) — they reach DSC >0.94
2. **Investigate deep learnable symmetry** exploitation
3. **Consider hybrid approach:** diffusion for multi-proposal generation + symmetry-aware refinement

---

## Key Papers to Acquire

| Paper | ArXiv | Relevance |
|-------|-------|-----------|
| DPM-Solver++ | 2211.01095 | Immediate sampler speedup |
| Consistency Models | 2303.01469 | Few-step distillation |
| ConTiCoM-3D | 2509.01492 | Consistency models for 3D PC |
| Progressive Distillation | 2202.00512 | Controlled step reduction |
| Flow Matching | 2210.02747 | Better training objective |
| SPVD | 2408.06145 | Sparse point-voxel diffusion |
| NKSR | 2305.19590 | Fast surface reconstruction |
| Neuralangelo | 2306.03092 | Hash-grid neural surfaces |
| Rectified Flow for Implants | (Zhou et al. 2025) | Direct competitor |
| Wodzinski et al. 2024 | (CBM journal) | Accuracy SOTA reference |

---

## Conclusion

The single highest-leverage action is **applying DPM-Solver++ to our existing PCDiff model** — zero retraining, ~50x inference speedup. For RunPod serverless (24-96GB GPUs), this could reduce stage-1 inference from ~60s to ~1-2s.

The full optimization pipeline (Phases 1-3) can plausibly achieve **sub-second total inference** for the complete skull-scan-to-implant pipeline, enabling real-time clinical workflows.

PCDiff's accuracy gap vs. Wodzinski et al. (0.86 vs 0.94+ DSC) should be addressed in parallel — our differentiator (multi-proposal probabilistic generation) is real, but the accuracy gap is substantial.
