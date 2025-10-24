#!/bin/bash
set -e  # Exit on error

# Configuration
CONFIG_FILE="voxelization/configs/train_skullbreak.yaml"
GPU_ID=7

# Print configuration
echo "================================================"
echo "Starting Voxelization Training"
echo "================================================"
echo "Config: ${CONFIG_FILE}"
echo "GPU: ${GPU_ID}"
echo "================================================"
echo ""

# Verify config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found at ${CONFIG_FILE}"
    exit 1
fi

# Check if GPU setting in config is correct
GPU_IN_CONFIG=$(grep "gpu:" "${CONFIG_FILE}" | awk '{print $2}')
if [ "${GPU_IN_CONFIG}" != "${GPU_ID}" ]; then
    echo "WARNING: Config file has gpu: ${GPU_IN_CONFIG}, but script expects gpu: ${GPU_ID}"
    echo "Please update ${CONFIG_FILE} and set 'gpu: ${GPU_ID}' in the train section"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set CUDA device - IMPORTANT: This overrides any inherited CUDA_VISIBLE_DEVICES
# Make GPU 7 appear as GPU 0 to this process
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Since we're making GPU 7 visible as device 0, update the config temporarily
# Or we need to tell the script to use device 0 (which is actually GPU 7)
echo "Note: GPU ${GPU_ID} will be mapped to device 0 for this process"

# Start training - the config should use gpu: 0 when CUDA_VISIBLE_DEVICES=7
python voxelization/train.py "${CONFIG_FILE}" #--wandb-project pcdiff

echo ""
echo "Voxelization training completed or interrupted."

