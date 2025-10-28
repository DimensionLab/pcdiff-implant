#!/bin/bash
set -e  # Exit on error

# Configuration
DATASET_PATH="pcdiff/datasets/SkullBreak/train.csv"  # Train split CSV from pre-processing
DATASET_NAME="SkullBreak"
CHECKPOINT="" #"pcdiff/output/train_completion/2025-10-23-19-35-15/epoch_1999_remapped.pth"
NUM_GPUS=8                   # Using GPUs 0-7 for full data-parallel run
BATCH_SIZE=64                # Global batch ⇒ 8 samples per GPU when NUM_GPUS=8
LEARNING_RATE=0.0002         # Baseline LR (scaled automatically inside training script)
LR_BASE_BATCH=8              # Baseline global batch size for LR scaling
LR_WARMUP_EPOCHS=1000        # Linear warmup steps before switching to exponential decay
LR_WARMUP_START_FACTOR=0.01  # Warmup start factor (relative to scaled LR)
PREFETCH_FACTOR=4            # Matches paper-trained dataloader behaviour
MATMUL_PRECISION="high"      # Paper uses FP32 training; keep TF32 enabled for speed
AMP_FLAG="--no-amp"          # Paper baseline is full precision; remove flag to enable AMP later
COMPILE_BACKEND="inductor"
COMPILE_MODE="default"
COMPILE_FULLGRAPH_FLAG=""    # Set to '--compile-fullgraph' only if custom ops support full graphs
FUSED_ADAM_FLAG=""           # Leave empty to use fused Adam; set to '--no-fused-adam' for plain Adam

COMPILE_FLAGS=( --compile-backend "${COMPILE_BACKEND}" --compile-mode "${COMPILE_MODE}" )
EXTRA_FLAGS=( --matmul-precision "${MATMUL_PRECISION}" )
# EXTRA_FLAGS+=( "${COMPILE_FLAGS[@]}" )
[ -n "${COMPILE_FULLGRAPH_FLAG}" ] && EXTRA_FLAGS+=( "${COMPILE_FULLGRAPH_FLAG}" )
[ -n "${AMP_FLAG}" ] && EXTRA_FLAGS+=( "${AMP_FLAG}" )
[ -n "${FUSED_ADAM_FLAG}" ] && EXTRA_FLAGS+=( "${FUSED_ADAM_FLAG}" )

PER_DEVICE_BATCH=$(( BATCH_SIZE / NUM_GPUS ))
if [ ${PER_DEVICE_BATCH} -lt 1 ]; then
  PER_DEVICE_BATCH=1
fi
EFFECTIVE_BATCH=$((PER_DEVICE_BATCH * NUM_GPUS))

# Print configuration
echo "================================================"
echo "Starting PCDiff Training (Resumed)"
echo "================================================"
echo "Dataset: ${DATASET_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "GPUs: ${NUM_GPUS} (0-7)"
echo "Batch Size: ${PER_DEVICE_BATCH} per GPU (effective: ${EFFECTIVE_BATCH})"
echo "Base Learning Rate: ${LEARNING_RATE} (scaled by factor ${BATCH_SIZE}/${LR_BASE_BATCH})"
echo "Warmup: epochs=${LR_WARMUP_EPOCHS}, start_factor=${LR_WARMUP_START_FACTOR}"
echo "================================================"
echo ""

# Verify checkpoint exists
if [ -n "${CHECKPOINT}" ] && [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

# Start training
# Using GPUs 0-7 for training
# --model ${CHECKPOINT} \ for checkpoint resume training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    pcdiff/train_completion.py \
    --path ${DATASET_PATH} \
    --dataset ${DATASET_NAME} \
    --bs ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --lr-base-batch ${LR_BASE_BATCH} \
    --lr-warmup-epochs ${LR_WARMUP_EPOCHS} \
    --lr-warmup-start-factor ${LR_WARMUP_START_FACTOR} \
    --niter 15000 \
    --num_points 30720 \
    --num_nn 3072 \
    --workers 21 \
    --prefetch-factor ${PREFETCH_FACTOR} \
    --nc 3 \
    --attention True \
    --dropout 0.1 \
    --embed_dim 64 \
    --beta_start 0.0001 \
    --beta_end 0.02 \
    --schedule_type linear \
    --time_num 1000 \
    --loss_type mse \
    --model_mean_type eps \
    --model_var_type fixedsmall \
    --vox_res_mult 1.0 \
    --width_mult 1.0 \
    --saveIter 1000 \
    --diagIter 2000 \
    --vizIter 2000 \
    --print_freq 10 \
    --manualSeed 1234 \
    --dist-backend nccl \
    --augment False \
    --disable-compile \
    "${EXTRA_FLAGS[@]}"
    #--wandb-project pcdiff
    # Keep fused Adam enabled by default; set to --no-fused-adam to revert to paper baseline


echo ""
echo "PCDiff training completed or interrupted."
