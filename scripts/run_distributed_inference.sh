#!/bin/bash
# Run multi-GPU distributed inference for PCDiff
#
# Usage:
#   ./scripts/run_distributed_inference.sh <checkpoint_path> <output_dir> [sampling_method] [sampling_steps] [num_ens]
#
# Examples:
#   # DDIM-50 (fast) with 5 ensemble samples
#   ./scripts/run_distributed_inference.sh pcdiff/runs/SkullBreak/best/checkpoints/model_best.pth eval_ddim50 ddim 50 5
#
#   # DDPM-1000 (full) with 5 ensemble samples
#   ./scripts/run_distributed_inference.sh pcdiff/runs/SkullBreak/best/checkpoints/model_best.pth eval_ddpm1000 ddpm 1000 5

set -euo pipefail

# Parse arguments
CHECKPOINT="${1:-}"
OUTPUT_DIR="${2:-}"
SAMPLING_METHOD="${3:-ddim}"
SAMPLING_STEPS="${4:-50}"
NUM_ENS="${5:-5}"

# Validate arguments
if [[ -z "$CHECKPOINT" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 <checkpoint_path> <output_dir> [sampling_method] [sampling_steps] [num_ens]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path   Path to model checkpoint (.pth file)"
    echo "  output_dir        Output directory for inference results"
    echo "  sampling_method   'ddim' (default) or 'ddpm'"
    echo "  sampling_steps    Number of sampling steps (default: 50 for DDIM, use 1000 for DDPM)"
    echo "  num_ens           Number of ensemble samples (default: 5)"
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
else
    NUM_GPUS=1
fi

echo "=============================================="
echo "PCDiff Distributed Inference"
echo "=============================================="
echo "Checkpoint:      $CHECKPOINT"
echo "Output:          $OUTPUT_DIR"
echo "Sampling:        $SAMPLING_METHOD ($SAMPLING_STEPS steps)"
echo "Ensemble:        $NUM_ENS samples"
echo "GPUs:            $NUM_GPUS"
echo "=============================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Run distributed inference
if [[ "$NUM_GPUS" -gt 1 ]]; then
    echo "Running with torchrun (multi-GPU)..."
    torchrun --nproc_per_node="$NUM_GPUS" \
        --master_port=29501 \
        pcdiff/test_completion_distributed.py \
        --path datasets/SkullBreak/test.csv \
        --dataset SkullBreak \
        --model "$CHECKPOINT" \
        --eval_path "$OUTPUT_DIR" \
        --sampling_method "$SAMPLING_METHOD" \
        --sampling_steps "$SAMPLING_STEPS" \
        --num_ens "$NUM_ENS" \
        --verify
else
    echo "Running single-GPU inference..."
    python pcdiff/test_completion_distributed.py \
        --path datasets/SkullBreak/test.csv \
        --dataset SkullBreak \
        --model "$CHECKPOINT" \
        --eval_path "$OUTPUT_DIR" \
        --sampling_method "$SAMPLING_METHOD" \
        --sampling_steps "$SAMPLING_STEPS" \
        --num_ens "$NUM_ENS" \
        --verify
fi

echo ""
echo "=============================================="
echo "Inference complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
