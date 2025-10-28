#!/bin/bash
set -e  # Exit on first error

# --------------------------------------------------
# Configuration
# --------------------------------------------------
INPUT_SKULL="pcdiff/datasets/SkullBreak/defective_skull/random_2/077.npy"
PCDIFF_MODEL="pcdiff/output/train_completion/latest/best.pth"
VOX_MODEL="voxelization/out/skullbreak/model_best.pt"
OUTPUT_DIR="single_inference_runs"
RUN_NAME="random2_077_demo"
GPU_ID=7

# Evaluation settings
EVAL_DATASET_CSV="pcdiff/datasets/SkullBreak/test.csv"
EVAL_NUM_SAMPLES=4
EVAL_SEED=1234

# --------------------------------------------------
# Sanity checks
# --------------------------------------------------
echo "================================================"
echo "PCDiff Single Inference + Evaluation"
echo "================================================"
echo "Input skull:         ${INPUT_SKULL}"
echo "PCDiff checkpoint:   ${PCDIFF_MODEL}"
echo "Voxelization ckpt:   ${VOX_MODEL}"
echo "Output directory:    ${OUTPUT_DIR}/${RUN_NAME}"
echo "GPU:                 ${GPU_ID}"
echo "Evaluation CSV:      ${EVAL_DATASET_CSV}"
echo "Evaluation samples:  ${EVAL_NUM_SAMPLES}"
echo "================================================"
echo ""

if [ ! -f "${INPUT_SKULL}" ]; then
    echo "ERROR: Input skull not found at ${INPUT_SKULL}"
    exit 1
fi

if [ ! -f "${PCDIFF_MODEL}" ]; then
    echo "ERROR: PCDiff checkpoint not found at ${PCDIFF_MODEL}"
    exit 1
fi

if [ ! -f "${VOX_MODEL}" ]; then
    echo "ERROR: Voxelization checkpoint not found at ${VOX_MODEL}"
    exit 1
fi

if [ ! -f "${EVAL_DATASET_CSV}" ]; then
    echo "ERROR: Evaluation CSV not found at ${EVAL_DATASET_CSV}"
    exit 1
fi

# --------------------------------------------------
# Run single inference
# --------------------------------------------------
echo "[1/2] Running single inference..."
python3 run_single_inference.py \
    --input "${INPUT_SKULL}" \
    --pcdiff_model "${PCDIFF_MODEL}" \
    --vox_model "${VOX_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --name "${RUN_NAME}" \
    --num_ens 1 \
    --sampling_method ddpm \
    --sampling_steps 1000 \
    --dataset SkullBreak \
    --gpu "${GPU_ID}" \
    --export_ply \
    --export_stl \
    --export_nrrd

echo ""
echo "Single inference results stored in ${OUTPUT_DIR}/${RUN_NAME}"
echo ""

# --------------------------------------------------
# Run evaluation
# --------------------------------------------------
METRICS_REPORT="${OUTPUT_DIR}/${RUN_NAME}_metrics.yaml"
echo "[2/2] Running evaluation (report -> ${METRICS_REPORT})..."

python3 run_skullbreak_eval.py \
    --pcdiff-model "${PCDIFF_MODEL}" \
    --vox-model "${VOX_MODEL}" \
    --dataset-csv "${EVAL_DATASET_CSV}" \
    --num-samples "${EVAL_NUM_SAMPLES}" \
    --seed "${EVAL_SEED}" \
    --sampling-method ddpm \
    --sampling-steps 1000 \
    --device "cuda:${GPU_ID}" \
    --output-report "${METRICS_REPORT}"

echo ""
echo "================================================"
echo "Pipeline finished successfully!"
echo " - Inference output: ${OUTPUT_DIR}/${RUN_NAME}"
echo " - Metrics report:   ${METRICS_REPORT}"
echo "================================================"
echo ""
