#!/bin/bash
# Experiment Matrix Runner for PCDiff Training (Task 8)
# Runs E0, E1, E2 experiments within the 700-epoch gating budget
#
# E0: Paper parity (lr=2e-4, bs=8, single GPU)
# E1: Sqrt LR scaling (bs=16, scaled_lr = 2e-4 × sqrt(2) ≈ 2.83e-4)
# E2: Linear LR scaling + warmup (bs=16, scaled_lr = 2e-4 × 2 = 4e-4)

set -e

# Configuration
DATASET_PATH="datasets/SkullBreak/train.csv"
DATASET_NAME="SkullBreak"
VOX_CONFIG="voxelization/configs/gen_skullbreak.yaml"
VOX_CHECKPOINT="voxelization/checkpoints/model_best.pt"
PROXY_SUBSET="pcdiff/proxy_validation_subset.json"

# Detect available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Common training arguments
COMMON_ARGS="
    --dataset ${DATASET_NAME}
    --niter 15000
    --gating-enabled True
    --gating-max-epochs 700
    --gating-decision-epochs 50,100,200,500,700
    --gating-proxy-eval-freq 50
    --proxy-eval-enabled True
    --proxy-eval-sampling-method ddim
    --proxy-eval-sampling-steps 50
    --proxy-eval-num-ens 1
    --proxy-eval-vox-config ${VOX_CONFIG}
    --proxy-eval-vox-checkpoint ${VOX_CHECKPOINT}
    --proxy-eval-subset ${PROXY_SUBSET}
    --checkpoint_freq 10
    --keep_last_n 5
    --no-amp
    --matmul-precision high
    --dist-backend nccl
    --print_freq 5
    --workers 16
    --prefetch-factor 4
"

# Timestamp for this experiment run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to run an experiment
run_experiment() {
    local exp_name=$1
    local exp_tag=$2
    local batch_size=$3
    local learning_rate=$4
    local warmup_epochs=$5
    local use_multi_gpu=$6
    local extra_args=$7

    echo ""
    echo "=============================================="
    echo "Starting experiment: ${exp_name}"
    echo "Tag: ${exp_tag}"
    echo "Batch size: ${batch_size}"
    echo "Learning rate: ${learning_rate}"
    echo "Warmup epochs: ${warmup_epochs}"
    echo "Multi-GPU: ${use_multi_gpu}"
    echo "=============================================="
    echo ""

    if [ "$use_multi_gpu" = "true" ]; then
        # Multi-GPU with torchrun
        torchrun --nproc_per_node=${NUM_GPUS} \
            --master_port=29500 \
            pcdiff/train_completion.py \
            --path ${DATASET_PATH} \
            --bs ${batch_size} \
            --lr ${learning_rate} \
            --lr-base-batch 8 \
            --lr-warmup-epochs ${warmup_epochs} \
            --lr-warmup-start-factor 0.01 \
            --experiment-tag "${exp_tag}" \
            --wandb-project pcdiff-implant \
            ${COMMON_ARGS} \
            ${extra_args}
    else
        # Single GPU
        CUDA_VISIBLE_DEVICES=0 python pcdiff/train_completion.py \
            --path ${DATASET_PATH} \
            --bs ${batch_size} \
            --lr ${learning_rate} \
            --lr-base-batch 8 \
            --lr-warmup-epochs ${warmup_epochs} \
            --lr-warmup-start-factor 0.01 \
            --experiment-tag "${exp_tag}" \
            --wandb-project pcdiff-implant \
            ${COMMON_ARGS} \
            ${extra_args}
    fi

    echo ""
    echo "Experiment ${exp_name} completed."
    echo ""
}

# Parse command line arguments
EXPERIMENT="all"
if [ $# -ge 1 ]; then
    EXPERIMENT=$1
fi

case $EXPERIMENT in
    "E0"|"e0")
        # E0: Paper parity - single GPU, bs=8, lr=2e-4
        # scaled_lr = 2e-4 × (8/8) = 2e-4
        run_experiment "E0" "E0-paper-parity-${TIMESTAMP}" 8 2e-4 0 "false"
        ;;
    "E1"|"e1")
        # E1: Sqrt LR scaling - multi-GPU, bs=16, lr=1.414e-4
        # scaled_lr = 1.414e-4 × (16/8) = 2.828e-4 ≈ 2e-4 × sqrt(2)
        PER_GPU_BATCH=$((16 / NUM_GPUS))
        GLOBAL_BATCH=$((PER_GPU_BATCH * NUM_GPUS))
        # For sqrt scaling with 2 GPUs: sqrt(2) ≈ 1.414
        # lr = 2e-4 × sqrt(2) / (16/8) = 2e-4 × 1.414 / 2 = 1.414e-4
        run_experiment "E1" "E1-sqrt-scaling-${TIMESTAMP}" ${GLOBAL_BATCH} 1.414e-4 0 "true"
        ;;
    "E2"|"e2")
        # E2: Linear LR scaling + warmup - multi-GPU, bs=16, lr=2e-4
        # scaled_lr = 2e-4 × (16/8) = 4e-4 (linear scaling)
        # Warmup: 100 epochs (adapted from 1000 for 700-epoch budget)
        PER_GPU_BATCH=$((16 / NUM_GPUS))
        GLOBAL_BATCH=$((PER_GPU_BATCH * NUM_GPUS))
        run_experiment "E2" "E2-linear-warmup-${TIMESTAMP}" ${GLOBAL_BATCH} 2e-4 100 "true"
        ;;
    "all")
        echo "Running all experiments sequentially: E0 -> E1 -> E2"
        echo "This will take a while..."
        echo ""

        # E0: Paper parity
        run_experiment "E0" "E0-paper-parity-${TIMESTAMP}" 8 2e-4 0 "false"

        # E1: Sqrt LR scaling
        PER_GPU_BATCH=$((16 / NUM_GPUS))
        GLOBAL_BATCH=$((PER_GPU_BATCH * NUM_GPUS))
        run_experiment "E1" "E1-sqrt-scaling-${TIMESTAMP}" ${GLOBAL_BATCH} 1.414e-4 0 "true"

        # E2: Linear LR scaling + warmup
        run_experiment "E2" "E2-linear-warmup-${TIMESTAMP}" ${GLOBAL_BATCH} 2e-4 100 "true"

        echo ""
        echo "=============================================="
        echo "All experiments completed!"
        echo "=============================================="
        ;;
    *)
        echo "Usage: $0 [E0|E1|E2|all]"
        echo ""
        echo "Experiments:"
        echo "  E0  - Paper parity (bs=8, lr=2e-4, single GPU)"
        echo "  E1  - Sqrt LR scaling (bs=16, scaled_lr ≈ 2.83e-4)"
        echo "  E2  - Linear LR + warmup (bs=16, scaled_lr = 4e-4)"
        echo "  all - Run all experiments sequentially"
        exit 1
        ;;
esac

echo ""
echo "Experiment run complete. Check pcdiff/runs/SkullBreak/ for results."
echo "Run E2E evaluation with: python pcdiff/eval_e2e.py --pcdiff-checkpoint <best_checkpoint>"
