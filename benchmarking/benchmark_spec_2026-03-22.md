# Benchmark Specification

Date: 2026-03-22

## Scope

This document defines the benchmark that the current `pcdiff-implant` codebase actually implements. It is intended to be the source of truth for reproduction work and for auditing README-level claims. Our reproduction effort focuses on **SkullBreak** only; SkullFix support exists in the upstream code but is out of scope for our benchmarking.

## Benchmark Stages

### Stage 1: Point-Cloud Completion

- Entry point: `pcdiff/test_completion.py`
- Input split:
  - SkullBreak: `pcdiff/datasets/SkullBreak/test.csv`
- Output root: `--eval_path`
- Canonical emitted artifacts:
  - `benchmark_cases.csv`
  - `benchmark_summary.json`
  - `benchmark_stage_report.json`

### Stage 2: Voxelization and Metric Computation

- Entry point: `voxelization/generate.py`
- Input root:
  - SkullBreak: point-cloud outputs under `results/syn`
- Output artifacts per case:
  - `mean_impl.nrrd`
  - `eval_metrics.yaml`
- Canonical emitted aggregate artifacts:
  - `benchmark_cases.csv`
  - `benchmark_summary.json`
  - `benchmark_stage_report.json`

## Dataset Semantics Implemented by the Repo

### SkullBreak

- Split unit: complete skull case, then expanded to five defects during dataset loading.
- Defect families:
  - `bilateral`
  - `frontoorbital`
  - `parietotemporal`
  - `random_1`
  - `random_2`
- Point-cloud preprocessing:
  - marching cubes over `.nrrd` volumes
  - Poisson-disk sampling to `400000` surface points
- Split generator behavior:
  - `pcdiff/utils/split_skullbreak.py` shuffles complete skulls with seed `42`
  - default train/test ratio is `0.8`
  - it can also emit `skullbreak.csv`
- Dataset loader behavior:
  - `pcdiff/datasets/skullbreak_data.py` expands each complete-skull CSV row into five defect-specific examples

### SkullFix (out of scope)

SkullFix support exists in the upstream codebase but is not part of our reproduction effort. See the upstream README for details.

## Metric Definitions Implemented by the Repo

Metrics are computed in `voxelization/generate.py` using helpers from `voxelization/eval_metrics.py`.

- DSC:
  - Dice coefficient between predicted implant volume and ground-truth implant volume
- 10 mm bDSC:
  - Dice coefficient after masking both predicted and ground-truth implants to voxels within `10` millimeters of the defective skull
  - the distance transform uses the volume `voxelspacing`
- HD95:
  - symmetric 95th percentile Hausdorff distance between predicted and ground-truth implant surfaces
  - measured in millimeters via `voxelspacing`
- Runtime:
  - per-case wall-clock time measured around stage-2 generation
- GPU peak memory:
  - `torch.cuda.max_memory_allocated` when CUDA is active

These implementations match the external metric reference linked by the README: `https://github.com/OldaKodym/evaluation_metrics`.

## Important Reproducibility Mismatches

- Missing datasets:
  - SkullBreak volumes are absent; only CSV metadata is present locally
- Invalid checked-in split paths:
  - sampled SkullBreak CSV entries point to another machine's absolute macOS paths
- Ambiguous path conventions:
  - point-cloud docs use `pcdiff/datasets/...`
  - voxelization configs use `datasets/...`
- Stage-2 checkpoint/config mismatch:
  - generation configs expect checkpoint paths that are not the checked-in checkpoint location

## Claim Discipline

Do not treat README tables as reproduced from this checkout unless the following are archived together:

- stage 1 `benchmark_stage_report.json`
- stage 2 `benchmark_stage_report.json`
- the exact checkpoints used for both stages
- the exact configs or CLI commands
- local split files generated in the workspace
- the commit SHA recorded by each stage report

If any of those are missing, the result should be treated as exploratory rather than canonical.
