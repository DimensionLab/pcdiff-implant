#!/bin/bash
# Multi-GPU PCDiff Training Script for 6x H100 GPUs
# Task 12: Full multi-GPU training run with DDPM-1000 evaluation

set -e

# Configuration
NUM_GPUS=6
BATCH_SIZE=8  # per-GPU batch size (paper default)
# sqrt LR scaling: 2e-4 * sqrt(6) ≈ 4.9e-4
LEARNING_RATE=4.9e-4
WARMUP_EPOCHS=100
MAX_EPOCHS=2000
DECISION_EPOCHS="200,500,1000,1500,2000"
PROXY_EVAL_FREQ=200

# Timestamp for run tag
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_TAG="6xH100-sqrt-lr-${TIMESTAMP}"

# Set CUDA arch for H100 (sm_90)
export TORCH_CUDA_ARCH_LIST="9.0"

# Activate virtual environment
cd /workspace/pcdiff-implant
source .venv/bin/activate

echo "=== Starting 6-GPU PCDiff Training ==="
echo "Experiment: ${EXPERIMENT_TAG}"
echo "GPUs: ${NUM_GPUS}x H100"
echo "Batch size: ${BATCH_SIZE}/GPU × ${NUM_GPUS} = $((BATCH_SIZE * NUM_GPUS)) global"
echo "Learning rate: ${LEARNING_RATE} (sqrt scaling)"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Decision epochs: ${DECISION_EPOCHS}"
echo "Proxy eval freq: ${PROXY_EVAL_FREQ} epochs"
echo ""

# Run multi-GPU training with torchrun
torchrun --nproc_per_node=${NUM_GPUS} --master_port=29503 \
    pcdiff/train_completion.py \
    --path datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --lr-warmup-epochs ${WARMUP_EPOCHS} \
    --niter 15000 \
    --gating-enabled True \
    --gating-max-epochs ${MAX_EPOCHS} \
    --gating-decision-epochs "${DECISION_EPOCHS}" \
    --gating-proxy-eval-freq ${PROXY_EVAL_FREQ} \
    --proxy-eval-enabled True \
    --proxy-eval-num-ens 1 \
    --proxy-eval-sampling-method ddim \
    --proxy-eval-sampling-steps 50 \
    --amp \
    --amp-dtype bfloat16 \
    --wandb-project pcdiff-implant \
    --experiment-tag "${EXPERIMENT_TAG}" \
    --print_freq 5

echo "=== Training Complete ==="
