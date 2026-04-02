# Stage-1 Ablation Workflow

Date: 2026-03-22

## Scope

This runbook defines how to execute `DIM-6` once the blocked baseline environment is restored. It keeps the ablation credible by forcing every retained configuration through the same stage-1 and stage-2 artifact path before selection.

## Decision Rule

Select the fastest stage-1 configuration whose downstream stage-2 aggregate metrics do not regress against the locked baseline:

- mean `dice` must be greater than or equal to baseline
- mean `bdice_10mm` must be greater than or equal to baseline
- mean `hd95_mm` must be less than or equal to baseline

Only use archived `benchmark_summary.json` files for the decision.

## Planned Sweep

- sampling methods:
  - `ddpm`
  - `ddim`
- DDPM steps:
  - `1000`
- DDIM steps:
  - `250`
  - `100`
  - `50`
  - `25`
- ensemble sizes:
  - `1`
  - `3`
  - `5`

This yields 15 runs per dataset.

## Directory Contract

For each `<dataset>/<run-id>` under `benchmarking/runs/stage1_ablation/`:

- `stage1/`
  - direct output root for `pcdiff/test_completion.py`
  - must contain `benchmark_cases.csv`, `benchmark_summary.json`, `benchmark_stage_report.json`
- `stage2/`
  - direct output root for `voxelization/generate.py`
  - must contain `benchmark_cases.csv`, `benchmark_summary.json`, `benchmark_stage_report.json`

## Execution

1. Generate the sweep manifest and shell script:

```bash
python benchmarking/stage1_ablation_matrix.py
```

2. For each planned run, place stage-1 outputs under:

```text
benchmarking/runs/stage1_ablation/<dataset>/<run-id>/stage1
```

3. Point the voxelization config `data.path` at the matching `stage1/syn` directory and write stage-2 outputs under:

```text
benchmarking/runs/stage1_ablation/<dataset>/<run-id>/stage2
```

4. After all runs finish, compare them with the locked baseline stage-2 summary:

```bash
python benchmarking/select_stage1_candidate.py \
  --baseline SkullBreak=path/to/skullbreak_locked_baseline_stage2/benchmark_summary.json \
  --runs-root benchmarking/runs/stage1_ablation \
  --output benchmarking/runs/stage1_ablation/selection_report.json
```

## Notes

- Do not treat stage-1 speedups as wins unless the matching stage-2 metrics are archived.
- Candidate reranking should be evaluated as a separate stage-2 policy on top of the same archived stage-1 outputs.
- If the baseline is rerun, keep the old baseline summary and record the new one separately so selection remains auditable.
