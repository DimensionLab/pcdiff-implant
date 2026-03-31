#!/usr/bin/env bash
# Stage-1 Ablation Sweep for DIM-6: run on RunPod GPU pod
# Generated for the 15-run ablation matrix (DDPM/DDIM x steps x ensemble sizes)
# After stage-1 completes, runs stage-2 voxelization for each run to get final metrics
set -euo pipefail

REPO_ROOT="/workspace/pcdiff-implant"
cd "$REPO_ROOT"

# Ensure we're on latest main
git fetch origin main && git checkout main && git pull origin main

# Install dependencies if needed
pip install -q -r requirements.txt 2>/dev/null || true

ABLATION_ROOT="benchmarking/runs/stage1_ablation"
DATASET="SkullBreak"
MODEL="pcdiff/checkpoints/model_best.pth"
DATASET_CSV="pcdiff/datasets/SkullBreak/test.csv"
GPU=0
WORKERS=8

# Verify checkpoint and dataset exist
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Stage-1 checkpoint not found at $MODEL"
    exit 1
fi
if [ ! -f "$DATASET_CSV" ]; then
    echo "ERROR: Dataset CSV not found at $DATASET_CSV"
    exit 1
fi

echo "=== DIM-6 Stage-1 Ablation Sweep ==="
echo "Starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "15 runs planned: 3 DDPM (1000 steps x ens 1,3,5) + 12 DDIM (25,50,100,250 steps x ens 1,3,5)"
echo ""

# --- STAGE 1: Point-cloud completion ---

run_stage1() {
    local method=$1 steps=$2 ens=$3
    local run_id="${method}-steps${steps}-ens${ens}"
    local stage1_dir="${ABLATION_ROOT}/${DATASET}/${run_id}/stage1"

    if [ -f "${stage1_dir}/benchmark_summary.json" ]; then
        echo "[SKIP] Stage-1 ${run_id} already complete"
        return 0
    fi

    echo "[RUN] Stage-1 ${run_id} ..."
    mkdir -p "$stage1_dir"
    python pcdiff/test_completion.py \
        --path "$DATASET_CSV" \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --eval_path "$stage1_dir" \
        --sampling_method "$method" \
        --sampling_steps "$steps" \
        --num_ens "$ens" \
        --gpu "$GPU" \
        --workers "$WORKERS" \
        2>&1 | tee "${stage1_dir}/run.log"
    echo "[DONE] Stage-1 ${run_id}"
}

# DDPM runs (1000 steps only)
for ens in 1 3 5; do
    run_stage1 ddpm 1000 "$ens"
done

# DDIM runs (25, 50, 100, 250 steps)
for steps in 25 50 100 250; do
    for ens in 1 3 5; do
        run_stage1 ddim "$steps" "$ens"
    done
done

echo ""
echo "=== Stage 1 complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""

# --- STAGE 2: Voxelization for each stage-1 run ---

run_stage2() {
    local method=$1 steps=$2 ens=$3
    local run_id="${method}-steps${steps}-ens${ens}"
    local stage1_dir="${ABLATION_ROOT}/${DATASET}/${run_id}/stage1"
    local stage2_dir="${ABLATION_ROOT}/${DATASET}/${run_id}/stage2"

    if [ -f "${stage2_dir}/benchmark_summary.json" ]; then
        echo "[SKIP] Stage-2 ${run_id} already complete"
        return 0
    fi

    if [ ! -d "${stage1_dir}/syn" ]; then
        echo "[SKIP] Stage-2 ${run_id}: no stage-1 outputs at ${stage1_dir}/syn"
        return 0
    fi

    echo "[RUN] Stage-2 ${run_id} ..."
    mkdir -p "$stage2_dir"
    # Stage 2 uses config file; override data.path to point at stage-1 outputs
    PCDIFF_SKULLBREAK_RESULTS="${stage1_dir}/syn" \
    python voxelization/generate.py voxelization/configs/gen_skullbreak.yaml \
        2>&1 | tee "${stage2_dir}/run.log"
    # Move generated outputs to the stage2 dir if needed
    if [ -d "voxelization/gen_skullbreak" ] && [ ! -f "${stage2_dir}/benchmark_summary.json" ]; then
        cp -r voxelization/gen_skullbreak/* "${stage2_dir}/" 2>/dev/null || true
    fi
    echo "[DONE] Stage-2 ${run_id}"
}

# DDPM
for ens in 1 3 5; do
    run_stage2 ddpm 1000 "$ens"
done

# DDIM
for steps in 25 50 100 250; do
    for ens in 1 3 5; do
        run_stage2 ddim "$steps" "$ens"
    done
done

echo ""
echo "=== Stage 2 complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""

# --- SELECTION ---
echo "=== Running candidate selection ==="
# Note: baseline summary path must be provided - check if we have one
BASELINE_SUMMARY=""
for candidate in \
    "${ABLATION_ROOT}/${DATASET}/ddpm-steps1000-ens1/stage2/benchmark_summary.json" \
    "benchmarking/baseline_locked/stage2/benchmark_summary.json"; do
    if [ -f "$candidate" ]; then
        BASELINE_SUMMARY="$candidate"
        break
    fi
done

if [ -n "$BASELINE_SUMMARY" ]; then
    python benchmarking/select_stage1_candidate.py \
        --baseline "SkullBreak=${BASELINE_SUMMARY}" \
        --runs-root "${ABLATION_ROOT}" \
        --output "${ABLATION_ROOT}/selection_report.json"
    echo "Selection report written to ${ABLATION_ROOT}/selection_report.json"
else
    echo "WARNING: No baseline summary found. Run selection manually after establishing baseline."
    echo "Use: python benchmarking/select_stage1_candidate.py --baseline SkullBreak=<path> --runs-root ${ABLATION_ROOT} --output ${ABLATION_ROOT}/selection_report.json"
fi

echo ""
echo "=== DIM-6 ablation sweep finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
