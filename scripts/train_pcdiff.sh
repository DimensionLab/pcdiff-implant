#!/bin/bash
set -e  # Exit on error

# Configuration
DATASET_PATH="pcdiff/datasets/SkullBreak"  # Path to dataset directory
DATASET_NAME="SkullBreak"
CHECKPOINT="pcdiff/output/train_completion/2025-10-23-19-35-15/epoch_1999_remapped.pth"
NUM_GPUS=7
BATCH_SIZE=56
LEARNING_RATE=0.00122

# Print configuration
echo "================================================"
echo "Starting PCDiff Training (Resumed)"
echo "================================================"
echo "Dataset: ${DATASET_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "GPUs: ${NUM_GPUS} (0-6)"
echo "Batch Size: ${BATCH_SIZE} per GPU (effective: $((BATCH_SIZE * NUM_GPUS)))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "================================================"
echo ""

# Verify checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

# Start training
# Using only GPUs 0-6 (7 total) for training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    pcdiff/train_completion.py \
    --path ${DATASET_PATH} \
    --dataset ${DATASET_NAME} \
    --model ${CHECKPOINT} \
    --bs ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --niter 15000 \
    --num_points 30720 \
    --num_nn 3072 \
    --workers 21 \
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
    --augment False #\
    #--wandb-project pcdiff

echo ""
echo "PCDiff training completed or interrupted."

