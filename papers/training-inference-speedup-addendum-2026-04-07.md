# Training & Inference Speedup Research Addendum

**Date:** 2026-04-07
**Author:** CTO (ba6d3030)
**Issue:** DIM-64
**Updates:** Complements inference-speedup-research-2026-04-03.md with training speedup and additional inference/voxelization techniques.

---

## Part A: Training Speedup Techniques

### A.1 FastDiT-3D: Extreme Masking (~15x Training Speedup)

- **Paper:** Mo et al. 2024, arXiv:2312.07231 (ECCV 2024)
- **Key Idea:** Apply ~99% voxel masking during denoising in a masked diffusion transformer. Voxel-aware masking adaptively focuses on foreground. Combined with Mixture-of-Experts for multi-category generation.
- **Claimed Speedup:** Uses only 6.5% of original training cost (~15x speedup)
- **Accuracy:** IMPROVES over baseline (1-NN Accuracy and Coverage at 128³ resolution)
- **PDF:** `pdfs/fastdit3d-extreme-masking-2312.07231.pdf`
- **Priority:** ★★★★★ Highest — if we adopt DiT-style backbone, this is transformative

### A.2 PointDif: Point Cloud Pre-training with Diffusion

- **Paper:** Zheng et al. 2024, arXiv:2311.14960 (CVPR 2024)
- **Key Idea:** Pre-training via conditional point-to-point generation with recurrent uniform sampling for balanced noise-level supervision.
- **Benefit:** Learned representations reduce downstream training time. +2.4% classification on ScanObjectNN.
- **PDF:** `pdfs/pointdif-pre-training-2311.14960.pdf`
- **Priority:** ★★★☆☆ MEDIUM — useful if we need to train on new skull defect types

### A.3 PDR: Point Diffusion-Refinement Paradigm

- **Paper:** Lyu et al. 2022, arXiv:2112.03530 (ICLR 2022)
- **Key Idea:** Two-stage: coarse DDPM generation + refinement network (RFNet). RFNet provides 50x acceleration of iterative DDPM process.
- **Accuracy:** OUTPERFORMS SOTA on benchmarks at time of publication
- **Applicability:** Directly relevant to cranial implant pipeline — coarse implant shape via diffusion, detail via refinement
- **PDF:** `pdfs/pdr-point-diffusion-refinement-2112.03530.pdf`
- **Priority:** ★★★★☆ HIGH — refinement approach could replace multi-step sampling

---

## Part B: Additional Inference Speedup Techniques

### B.1 LION: Latent Point Diffusion Models

- **Paper:** Zeng et al. 2022, arXiv:2210.06978 (NeurIPS 2022)
- **Key Idea:** VAE with hierarchical latent space (global shape latent + point-structured latent). Diffusion operates in compressed latent space.
- **Benefit:** Inherently faster — diffusion in lower-dimensional latent space. Also enables interpolation/editing.
- **PDF:** `pdfs/lion-latent-point-diffusion-2210.06978.pdf`
- **Priority:** ★★★★☆ HIGH — latent space diffusion is the dominant paradigm in 2D (Stable Diffusion) and should be adopted for 3D

### B.2 GaussianAnything: Point Cloud Flow Matching

- **Paper:** Lan et al. 2025, arXiv:2411.08033 (ICLR 2025)
- **Key Idea:** Replaces diffusion with flow matching in point cloud-structured latent space. Geometry-texture disentanglement.
- **Benefit:** Flow matching typically needs fewer sampling steps. Interactive editing capability.
- **PDF:** `pdfs/gaussian-anything-2411.08033.pdf`
- **Priority:** ★★★★☆ HIGH — aligns with our Phase 3 flow matching strategy

### B.3 FastPoint: FPS Acceleration (2.55x)

- **Paper:** Lee et al. 2025, arXiv:2507.23480 (ICCV 2025)
- **Key Idea:** Predicts distance curve during farthest point sampling to identify subsequent sample points without exhaustive pairwise computation.
- **Benefit:** 2.55x end-to-end speedup on RTX 3090. Zero accuracy loss.
- **Priority:** ★★★☆☆ MEDIUM — applicable to point cloud preprocessing in our pipeline

### B.4 MLPCM: Multi-Scale Latent Point Cloud Model

- **Paper:** arXiv:2412.19413
- **Key Idea:** Multi-scale latent representations for point cloud generation.
- **PDF:** `pdfs/mlpcm-multi-scale-latent-point-2412.19413.pdf` (already existed)

---

## Part C: Efficient Voxelization / Sparse Convolution

### C.1 TorchSparse: Efficient Point Cloud Inference Engine

- **Paper:** Tang et al. 2022, arXiv:2204.10319 (MLSys 2022)
- **Key Idea:** Adaptive matrix multiplication grouping + vectorized quantized fused locality-aware memory access.
- **Speedup:** 1.6x over MinkowskiEngine, 1.5x over SpConv
- **PDF:** `pdfs/torchsparse-efficient-inference-2204.10319.pdf`
- **Priority:** ★★★★☆ HIGH — foundational for any sparse voxel work

### C.2 Minuet: Accelerating 3D Sparse Convolutions on GPUs

- **Paper:** Yang et al. 2024, arXiv:2401.06145
- **Key Idea:** Replaces hash tables with segmented sorting double-traversed binary search for map step. Lightweight autotuning.
- **Speedup:** 1.74x average end-to-end (up to 2.22x). 15.8x average for map step alone.
- **PDF:** `pdfs/minuet-sparse-conv-gpu-2401.06145.pdf`
- **Priority:** ★★★★☆ HIGH — next-gen sparse convolution engine

### C.3 Spira: Accelerating Sparse Convolutions in Voxel-Based Networks

- **Paper:** Adamopoulos et al. 2025, arXiv:2511.20834
- **Key Idea:** Voxel-property-aware sparse convolution. One-shot search (zero preprocessing). Dual-dataflow execution. Network-wide parallelization.
- **Speedup:** 1.71x average end-to-end (up to 2.31x)
- **PDF:** `pdfs/spira-sparse-conv-voxel-2511.20834.pdf`
- **Priority:** ★★★★☆ HIGH — latest state-of-the-art in sparse convolution

---

## Part D: Cranial Implant Specific — Symmetry Enforcement

### D.1 Deep Learnable Symmetry Enforcement for Skull Reconstruction

- **Paper:** Wodzinski et al. 2024, arXiv:2411.17342
- **Key Idea:** Learnable symmetry enforcement as training objective. Exploits bilateral skull symmetry.
- **Speedup:** <500 GPU hours vs >100,000 GPU hours for best methods (200x reduction)
- **Accuracy:** IMPROVES significantly — DSC 0.94 vs 0.84 baseline
- **PDF:** `pdfs/wodzinski-symmetry-enforcement-2411.17342.pdf`
- **Priority:** ★★★★★ CRITICAL — combines speedup AND accuracy improvement. Should be our #1 accuracy initiative.

---

## Part E: Updated Strategy — Integrated Training + Inference Plan

### Revised Phase Recommendations

**Phase 1 (Immediate — No Retraining):** Unchanged from original report.
- DPM-Solver++ swap, FP16 mixed precision
- Expected: ~100x stage-1 speedup

**Phase 2 (Short-Term — 2-3 weeks):**
1. Add symmetry enforcement loss to PCDiff training (from D.1) — simultaneous accuracy + training speed improvement
2. Consistency distillation of PCDiff to 2-4 steps (ConTiCoM-3D approach)
3. Adopt Minuet or Spira for sparse convolution backend in voxelization
4. TensorRT export for production

**Phase 3 (Next Model Version — 1-2 months):**
1. Retrain with flow matching (GaussianAnything-style architecture)
2. Adopt LION-style latent space diffusion for compressed representation
3. Consider FastDiT-3D masking for training acceleration
4. Integrate PDR refinement network for post-diffusion detail enhancement
5. Apply FDS training-free refinement on top of flow matching

**Phase 4 (Accuracy + Speed Fusion):**
1. Full integration of symmetry enforcement with diffusion pipeline
2. Hybrid: diffusion for multi-proposal → symmetry-aware refinement → sparse voxelization
3. Target: DSC >0.94 with <1 second total inference

### Expected Cumulative Impact

| Phase | Training Speedup | Inference Speedup | Accuracy Impact |
|-------|-----------------|-------------------|-----------------|
| Phase 1 | None | ~100x | Preserved |
| Phase 2 | 2-3x (symmetry) | ~300x total | Improved (+5-8% DSC) |
| Phase 3 | ~15x (FastDiT masking) | ~1000x total | Preserved or improved |
| Phase 4 | Further gains | Sub-second | DSC >0.94 |

### Newly Downloaded Papers (9 additional)

19. fastdit3d-extreme-masking-2312.07231.pdf (training speedup)
20. pointdif-pre-training-2311.14960.pdf (pre-training paradigm)
21. pdr-point-diffusion-refinement-2112.03530.pdf (refinement approach)
22. lion-latent-point-diffusion-2210.06978.pdf (latent space diffusion)
23. gaussian-anything-2411.08033.pdf (flow matching for point clouds)
24. torchsparse-efficient-inference-2204.10319.pdf (sparse conv engine)
25. minuet-sparse-conv-gpu-2401.06145.pdf (GPU sparse conv acceleration)
26. spira-sparse-conv-voxel-2511.20834.pdf (latest sparse conv)
27. wodzinski-symmetry-enforcement-2411.17342.pdf (symmetry + accuracy + speed)

---

## Conclusion

This addendum adds **training speedup techniques** (FastDiT-3D: 15x training speedup) and **critical accuracy findings** (symmetry enforcement: 200x GPU hour reduction + DSC 0.94). The most impactful new insight is that **symmetry enforcement** simultaneously addresses our accuracy gap (0.86→0.94 DSC) AND reduces training compute by 200x — this should be our highest priority alongside DPM-Solver++ inference speedup.

For the Perun HPC 8x H200 training run planned in DIM-69, I recommend:
1. Integrate symmetry enforcement loss BEFORE the training run
2. Use DPM-Solver++ validation during training (faster epoch evaluation)
3. Benchmark against FastDiT-3D masking if using DiT backbone
