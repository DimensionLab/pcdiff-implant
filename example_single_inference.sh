#!/bin/bash

# Example: Quick Single Inference
# This script demonstrates how to run a single inference with both PCDiff and Voxelization

# Configuration
INPUT_SKULL="pcdiff/datasets/SkullBreak/defective_skull/random_2/077.npy"
PCDIFF_MODEL="pcdiff/output/train_completion/2025-10-26-21-57-58/epoch_14999.pth"
VOX_MODEL="voxelization/out/skullbreak/model_best.pt"
OUTPUT_DIR="single_inference_demo"
NAME="test_random2_077"
GPU=7  # Use GPU 7 (has lowest utilization during training)

echo "=================================="
echo "Single Inference Example"
echo "=================================="
echo "Input: $INPUT_SKULL"
echo "Output: $OUTPUT_DIR/$NAME"
echo "GPU: $GPU"
echo ""

# Check if input exists
if [ ! -f "$INPUT_SKULL" ]; then
    echo "Error: Input file not found: $INPUT_SKULL"
    echo "Please provide a valid input skull file."
    exit 1
fi

# Check if models exist
if [ ! -f "$PCDIFF_MODEL" ]; then
    echo "Error: PCDiff model not found: $PCDIFF_MODEL"
    echo "Please train or download a PCDiff model first."
    exit 1
fi

if [ ! -f "$VOX_MODEL" ]; then
    echo "Error: Voxelization model not found: $VOX_MODEL"
    echo "Please train or download a voxelization model first."
    exit 1
fi

# Run inference
python3 run_single_inference.py \
    --input "$INPUT_SKULL" \
    --pcdiff_model "$PCDIFF_MODEL" \
    --vox_model "$VOX_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --name "$NAME" \
    --num_ens 1 \
    --sampling_method ddpm \
    --sampling_steps 1000 \
    --dataset SkullBreak \
    --gpu $GPU \
    --export_ply \
    --export_stl \
    --export_nrrd

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Inference Complete!"
    echo "=================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR/$NAME/"
    echo ""
    echo "To view in web browser:"
    echo "  1. python3 pcdiff/utils/convert_to_web.py $OUTPUT_DIR/$NAME"
    echo "  2. cd web_viewer && ./start_dev.sh"
    echo "  3. Open http://localhost:5173"
    echo ""
    echo "To view in MeshLab:"
    echo "  meshlab $OUTPUT_DIR/$NAME/skull_complete.ply"
    echo ""
    echo "For 3D printing:"
    echo "  Open $OUTPUT_DIR/$NAME/skull_complete.stl in your slicer"
    echo ""
else
    echo ""
    echo "=================================="
    echo "Inference Failed!"
    echo "=================================="
    echo "Check the error messages above."
    exit 1
fi

