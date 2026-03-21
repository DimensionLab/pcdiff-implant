#!/bin/bash
set -e

echo "================================================"
echo "Training Setup Verification"
echo "================================================"
echo ""

# Check if scripts exist
echo "1. Checking scripts..."
SCRIPTS=(
    "scripts/train_pcdiff.sh"
    "scripts/train_voxelization.sh"
    "scripts/launch_both.sh"
    "scripts/monitor_training.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "  ✓ $script exists"
    else
        echo "  ✗ $script missing!"
        exit 1
    fi
done
echo ""

# Check PCDiff checkpoint
echo "2. Checking PCDiff checkpoint..."
CHECKPOINT="pcdiff/output/train_completion/2025-10-23-19-35-15/epoch_1999.pth"
if [ -f "${CHECKPOINT}" ]; then
    echo "  ✓ Checkpoint found: ${CHECKPOINT}"
else
    echo "  ✗ Checkpoint not found: ${CHECKPOINT}"
    echo "    Please update CHECKPOINT in scripts/train_pcdiff.sh"
fi
echo ""

# Check voxelization config
echo "3. Checking voxelization config..."
CONFIG="voxelization/configs/train_skullbreak.yaml"
if [ -f "${CONFIG}" ]; then
    echo "  ✓ Config found: ${CONFIG}"
    GPU_IN_CONFIG=$(grep "gpu:" "${CONFIG}" | awk '{print $2}')
    echo "    Current GPU setting: ${GPU_IN_CONFIG}"
    if [ "${GPU_IN_CONFIG}" == "0" ]; then
        echo "  ✓ GPU correctly set to 0 (will be GPU 7 via CUDA_VISIBLE_DEVICES)"
    else
        echo "  ⚠ GPU is set to ${GPU_IN_CONFIG}, should be 0"
        echo "    Run: sed -i 's/gpu: ${GPU_IN_CONFIG}/gpu: 0/' ${CONFIG}"
    fi
else
    echo "  ✗ Config not found: ${CONFIG}"
fi
echo ""

# Check dataset path
echo "4. Checking dataset..."
DATASET_PATH=$(grep "DATASET_PATH=" scripts/train_pcdiff.sh | head -1 | cut -d'"' -f2)
echo "  Dataset path in script: ${DATASET_PATH}"
if [ -d "${DATASET_PATH}" ]; then
    echo "  ✓ Dataset directory exists"
else
    echo "  ✗ Dataset not found!"
    echo "    Please update DATASET_PATH in scripts/train_pcdiff.sh"
fi
echo ""

# Check CUDA availability
echo "5. Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "  ✓ nvidia-smi available"
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "    Available GPUs: ${NUM_GPUS}"
    if [ ${NUM_GPUS} -ge 8 ]; then
        echo "  ✓ Sufficient GPUs for training (need 8, have ${NUM_GPUS})"
    else
        echo "  ⚠ Only ${NUM_GPUS} GPUs available (need 8)"
    fi
else
    echo "  ✗ nvidia-smi not found!"
fi
echo ""

# Check tmux
echo "6. Checking tmux..."
if command -v tmux &> /dev/null; then
    echo "  ✓ tmux available"
else
    echo "  ✗ tmux not found! Install with: sudo apt install tmux"
fi
echo ""

echo "================================================"
echo "Setup verification complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. If any issues above, fix them first"
echo "  2. Launch training with: bash scripts/launch_both.sh"
echo "  3. Or manually in tmux panes (see scripts/README.md)"
echo ""

