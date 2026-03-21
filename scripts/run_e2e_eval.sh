#!/bin/bash
#
# Run E2E evaluation comparing DDIM-50 vs DDPM-1000
#
# Usage:
#   ./scripts/run_e2e_eval.sh <checkpoint> [output_dir] [num_ens] [gpus]
#
# Example:
#   ./scripts/run_e2e_eval.sh pcdiff/runs/SkullBreak/best_run/checkpoints/model_best.pth
#   ./scripts/run_e2e_eval.sh path/to/model.pth pcdiff/eval/my_eval 5 0,1,2,3

set -e

CHECKPOINT="${1:?Usage: $0 <checkpoint> [output_dir] [num_ens] [gpus]}"
OUTPUT_DIR="${2:-pcdiff/eval/e2e_$(date +%Y%m%d_%H%M%S)}"
NUM_ENS="${3:-5}"
GPUS="${4:-0,1}"

# Activate environment if not already
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

echo "=============================================="
echo "E2E Evaluation: DDIM-50 vs DDPM-1000"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "Ensemble:   $NUM_ENS"
echo "GPUs:       $GPUS"
echo "=============================================="

# Run the evaluation
python pcdiff/eval_e2e.py \
    --pcdiff-checkpoint "$CHECKPOINT" \
    --vox-checkpoint voxelization/checkpoints/model_best.pt \
    --vox-config voxelization/configs/gen_skullbreak.yaml \
    --dataset-csv datasets/SkullBreak/test.csv \
    --output-dir "$OUTPUT_DIR" \
    --num-ens "$NUM_ENS" \
    --gpus "$GPUS"

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results: $OUTPUT_DIR"
echo "=============================================="
