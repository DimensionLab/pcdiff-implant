#!/bin/bash
set -e

# Quick fix for checkpoint device mismatch
# This remaps the checkpoint to CPU so it can be loaded with any GPU configuration

CHECKPOINT="pcdiff/output/train_completion/2025-10-23-19-35-15/epoch_1999.pth"
OUTPUT="${CHECKPOINT%.pth}_remapped.pth"

echo "================================================"
echo "Remapping Checkpoint to CPU"
echo "================================================"
echo "Input:  ${CHECKPOINT}"
echo "Output: ${OUTPUT}"
echo ""

# Run Python script with torch available
python3 << 'PYTHON_SCRIPT'
import torch
import sys

checkpoint_path = "pcdiff/output/train_completion/2025-10-23-19-35-15/epoch_1999.pth"
output_path = "pcdiff/output/train_completion/2025-10-23-19-35-15/epoch_1999_remapped.pth"

print(f"Loading checkpoint from: {checkpoint_path}")
try:
    # Load with map_location='cpu' to avoid device issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"✓ Checkpoint loaded successfully")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Keys: {list(checkpoint.keys())}")
    
    # Save the remapped checkpoint
    print(f"\nSaving remapped checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    
    print("✓ Checkpoint successfully remapped to CPU!")
    print(f"\n✓ Done! Use this checkpoint:")
    print(f"  {output_path}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
PYTHON_SCRIPT

echo ""
echo "================================================"
echo "Update your training script to use:"
echo "  CHECKPOINT=\"pcdiff/output/train_completion/2025-10-23-19-35-15/epoch_1999_remapped.pth\""
echo "================================================"

