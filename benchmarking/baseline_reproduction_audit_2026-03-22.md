# Baseline Reproduction Audit

Date: 2026-03-22

## Scope

Audit the current `pcdiff-implant` workspace for reproducibility of the checked-in baseline on SkullBreak and SkullFix, with emphasis on whether the repository alone is sufficient to rerun the published pipeline.

## Result

The baseline is not currently reproducible from this workspace as checked out.

The primary blockers are external artifact availability rather than a single code crash:

- The Python ML runtime is not installed in the active environment.
- The expected dataset volumes are not present in the workspace.
- The checked-in SkullBreak split CSVs point to absolute paths from another machine (`/Users/michaltakac/...`) and all sampled entries are missing locally.
- The voxelization generation configs point at result and checkpoint paths that do not exist in this checkout.

## Evidence

### Environment

- Host Python is `3.12.3`.
- `uv` is available at `/home/mike/.local/bin/uv`.
- `nvidia-smi` was not available in `PATH` during the audit.
- Required runtime imports were missing from the default interpreter:
  - `torch`
  - `torchvision`
  - `numpy`
  - `nrrd`
  - `mcubes`
  - `trimesh`
- Voxelization-only imports were also unavailable:
  - `pytorch3d`
  - `torch_scatter`
  - `open3d`

### Dataset State

- `pcdiff/datasets/SkullBreak/` exists only with CSV metadata.
- `pcdiff/datasets/SkullFix/` is absent in this workspace.
- `pcdiff/datasets/SkullBreak/train.csv` has `91` rows.
- `pcdiff/datasets/SkullBreak/test.csv` has `23` rows.
- Sampled SkullBreak CSV entries reference absolute macOS paths under `/Users/michaltakac/projects/dimensionlab/pcdiff-implant/...`.
- Sampled CSV targets do not exist locally, so the advertised evaluation commands cannot start from the current workspace.

### Checkpoint Inventory

The workspace does include candidate checkpoints, which is useful for later reproduction once the environment and data are restored:

- `pcdiff/checkpoints/model_best.pth`
  - size: `332166569` bytes
  - sha256: `26e204df4eb5527783738eeff4b59c5f061aa89a9256a0b64b08a3577d628cf9`
- `pcdiff/checkpoints/model_latest.pth`
  - size: `332168949` bytes
  - sha256: `f6d348cf167bfa8f3951d4de6af1bf0fad9d5db863db7f9ca92f81158a1bfede`
- `pcdiff/checkpoints/pcdiff_model_best.pth`
  - size: `332166569` bytes
  - sha256: `a8ca1a750efc0d149c13e21b6de982717a057097d8b4f8edbf70940005e7543e`
- `voxelization/checkpoints/model_best.pt`
  - size: `4392447` bytes
  - sha256: `ea4e992cf2b89750353d5129a54be7c75bc17092153094c3670ebf7bfc663b38`

## Path Integrity Findings

The repo mixes several path conventions:

- README and quickstart commands use `pcdiff/datasets/...`.
- `voxelization/configs/gen_skullbreak.yaml` uses `datasets/SkullBreak/results/syn`.
- `voxelization/configs/gen_skullfix.yaml` uses `datasets/SkullFix/results/syn`.
- Both generation configs point to `voxelization/out/.../model_best.pt`, while the checked-in checkpoint is under `voxelization/checkpoints/model_best.pt`.

This means that even after environment setup, the canonical baseline invocation is underspecified. A user cannot infer which paths are authoritative without additional guidance.

## Gap Analysis Against a Credible Benchmark

For a benchmark claim to be credible and independently repeatable, the workspace should provide:

- installable runtime
- accessible dataset roots or a documented acquisition step
- valid split files for the local workspace
- stage 1 checkpoint path
- stage 2 checkpoint path
- canonical inference commands
- emitted benchmark artifacts for both stages

Current status:

- Runtime: missing in active interpreter
- Dataset: missing
- Local split integrity: failing
- Checkpoint inventory: partially present
- Canonical inference paths: ambiguous
- Benchmark artifacts: not generated in this workspace

## Added Support Artifact

`benchmarking/preflight_audit.py` was added to provide a stdlib-first reproducibility check before attempting a multi-hour setup or benchmark run.

Recommended usage:

```bash
python3 benchmarking/preflight_audit.py
python3 benchmarking/preflight_audit.py --json
```

## Recommended Next Steps

1. Create the Python 3.10 environment and install the runtime stack described in `INSTALL.md`.
2. Restore SkullBreak and SkullFix datasets into the workspace and regenerate the split CSVs locally.
3. Decide the canonical checkpoint locations and update docs/configs so stage 2 references checked-in or documented paths.
4. Run stage 1 point-cloud inference and archive `benchmark_stage_report.json`.
5. Run stage 2 voxelization and archive its `benchmark_stage_report.json`.
6. Compare reproduced metrics against README claims only after both stage reports and referenced checkpoints are preserved together.
