#!/bin/bash
set -e  # Exit on error

# Configuration
DATASET_PATH="pcdiff/datasets/SkullBreak/train.csv"  # Train split CSV from pre-processing
DATASET_NAME="SkullBreak"
CHECKPOINT=""                # Path to checkpoint for resume training (leave empty for fresh start)
NUM_GPUS=1                   # Single A100 GPU (matches paper setup)
BATCH_SIZE=8                 # Paper uses batch size 8
LEARNING_RATE=0.0002         # Paper LR: 2×10⁻⁴
LR_BASE_BATCH=8              # Baseline global batch size for LR scaling (no scaling with bs=8)
LR_WARMUP_EPOCHS=500         # Linear warmup epochs
LR_WARMUP_START_FACTOR=0.01  # Warmup start factor (relative to scaled LR)
NUM_EPOCHS=15000             # Paper trains for 15,000 epochs
CHECKPOINT_FREQ=500          # Save checkpoint every 500 epochs
KEEP_LAST_N=3                # Keep latest 3 checkpoints + best model
PREFETCH_FACTOR=4            # Dataloader prefetch factor
MATMUL_PRECISION="high"      # Paper uses FP32 training; keep TF32 enabled for speed
AMP_FLAG="--no-amp"          # Paper baseline is full precision; remove flag to enable AMP later
COMPILE_BACKEND="inductor"
COMPILE_MODE="default"
COMPILE_FULLGRAPH_FLAG=""    # Set to '--compile-fullgraph' only if custom ops support full graphs
FUSED_ADAM_FLAG=""           # Leave empty to use fused Adam; set to '--no-fused-adam' for plain Adam
WANDB_PROJECT="pcdiff"       # Weights & Biases project name

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
echo "PCDiff Training - Paper Parity Configuration"
echo "================================================"
echo "Dataset: ${DATASET_PATH}"
echo "Checkpoint: ${CHECKPOINT:-'None (fresh start)'}"
echo "GPUs: ${NUM_GPUS}"
echo "Batch Size: ${BATCH_SIZE} (paper: 8)"
echo "Learning Rate: ${LEARNING_RATE} (paper: 2e-4)"
echo "Epochs: ${NUM_EPOCHS} (paper: 15000)"
echo "Checkpoint Frequency: every ${CHECKPOINT_FREQ} epochs"
echo "Keep Last N Checkpoints: ${KEEP_LAST_N}"
echo "Warmup: epochs=${LR_WARMUP_EPOCHS}, start_factor=${LR_WARMUP_START_FACTOR}"
echo "W&B Project: ${WANDB_PROJECT}"
echo "================================================"
echo ""

# Verify checkpoint exists
if [ -n "${CHECKPOINT}" ] && [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

# Build checkpoint argument if provided
CHECKPOINT_ARG=""
if [ -n "${CHECKPOINT}" ]; then
    CHECKPOINT_ARG="--model ${CHECKPOINT}"
fi

# Start training
# Single GPU training (paper configuration)
CUDA_VISIBLE_DEVICES=0 torchrun \
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
    --niter ${NUM_EPOCHS} \
    --num_points 30720 \
    --num_nn 3072 \
    --workers 8 \
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
    --checkpoint_dir pcdiff/checkpoints \
    --checkpoint_freq ${CHECKPOINT_FREQ} \
    --keep_last_n ${KEEP_LAST_N} \
    --saveIter ${CHECKPOINT_FREQ} \
    --diagIter 1000 \
    --vizIter 1000 \
    --print_freq 10 \
    --manualSeed 1234 \
    --dist-backend nccl \
    --augment False \
    --gating-enabled False \
    --proxy-eval-enabled False \
    --disable-compile \
    --wandb-project ${WANDB_PROJECT} \
    ${CHECKPOINT_ARG} \
    "${EXTRA_FLAGS[@]}"


echo ""
echo "PCDiff training completed or interrupted."
