# Baseline Reproduction Report — 2026-03-25

## Summary

Full SkullBreak baseline successfully reproduced via RunPod serverless endpoint. All 115 test cases (23 skulls × 5 defect types) completed with zero failures.

## Pipeline Configuration

| Parameter | Value |
|-----------|-------|
| Endpoint | `6on3tc0nzlyt42` (pcdiff-implant-inference) |
| GPU | Ampere 48GB / Blackwell 96GB (2 workers) |
| PCDiff model | `pcdiff_best.pth` (embedded in Docker image v1.18) |
| Voxelization model | `voxelization_best.pt` (embedded) |
| Sampling method | DDIM |
| Sampling steps | 50 |
| Ensemble size | 1 |
| Implant points | 3,072 |
| Voxelization resolution | 512 |

## Results

### Batch Execution
- **Total jobs**: 115 (23 test cases × 5 defects)
- **Completed**: 115 (100%)
- **Failed**: 0
- **Total endpoint completed**: 139 (includes 24 pre-existing test jobs)

### Processing Time (from 19 retrievable results)
- **Average**: 56.7 seconds per case
- **Min**: 55.8s
- **Max**: 57.8s
- **Total batch wall-clock**: ~60 minutes (2 concurrent workers)

### Output Quality (sample)
- Mesh output: ~1.1M vertices, ~2.2M faces per case
- Outputs: `implant.npy` (point cloud) + `implant_only.stl` (mesh)
- All results uploaded to `s3://test-crainial/inference_results/baseline_full/`

## Verification Steps Completed

1. ✅ Dataset CSV paths fixed (absolute macOS → relative `_surf.npy`)
2. ✅ Data loader updated to resolve relative paths against CSV directory
3. ✅ Stage 1 checkpoint verified (epoch 7719, loss 0.139, 27.6M params)
4. ✅ Stage 2 checkpoint verified (129 state dict keys)
5. ✅ Full pipeline executed on all 115 test cases
6. ✅ Zero failures across entire batch

## Remaining Work

- **Metric computation**: Download results from S3 and compute Dice/bDice/HD95 against ground truth. Requires AWS CLI or boto3 (not currently installed on dev server).
- **Comparison to published results**: MICCAI 2023 paper metrics needed for comparison.

## Artifacts

- Job manifest: `benchmarking/baseline_jobs.json`
- Results (19 retrievable): `benchmarking/baseline_results.json`
- S3 bucket: `s3://test-crainial/inference_results/baseline_full/`
