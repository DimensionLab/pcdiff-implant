#!/bin/bash

# Enhanced monitoring script
echo "================================================"
echo "Training Monitor"
echo "================================================"
echo ""

# Check if both trainings are running
PCDIFF_RUNNING=$(pgrep -f "train_completion.py" | wc -l)
VOXEL_RUNNING=$(pgrep -f "voxelization/train.py" | wc -l)

echo "Status:"
echo "  PCDiff:       $([[ ${PCDIFF_RUNNING} -gt 0 ]] && echo '✓ Running' || echo '✗ Not running')"
echo "  Voxelization: $([[ ${VOXEL_RUNNING} -gt 0 ]] && echo '✓ Running' || echo '✗ Not running')"
echo ""

# Show GPU usage
echo "GPU Usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s%% | %sMB/%sMB | %s°C\n", $1, $3, $4, $5, $6}'
echo ""

# Show latest PCDiff log
PCDIFF_LOG=$(find /home/michaltakac/pcdiff-implant/pcdiff/output/train_completion -name "train.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
if [ -n "${PCDIFF_LOG}" ] && [ -f "${PCDIFF_LOG}" ]; then
    echo "Latest PCDiff log (last 5 lines):"
    tail -n 5 "${PCDIFF_LOG}" | sed 's/^/  /'
    echo ""
fi

# Show latest Voxelization log
VOXEL_LOG=$(find /home/michaltakac/pcdiff-implant/voxelization/out -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
if [ -n "${VOXEL_LOG}" ] && [ -f "${VOXEL_LOG}" ]; then
    echo "Latest Voxelization log (last 5 lines):"
    tail -n 5 "${VOXEL_LOG}" | sed 's/^/  /'
    echo ""
fi

# Show disk usage for output directories
echo "Disk Usage:"
if [ -d "/home/michaltakac/pcdiff-implant/pcdiff/output" ]; then
    PCDIFF_SIZE=$(du -sh /home/michaltakac/pcdiff-implant/pcdiff/output 2>/dev/null | cut -f1)
    echo "  PCDiff output:   ${PCDIFF_SIZE}"
fi
if [ -d "/home/michaltakac/pcdiff-implant/voxelization/out" ]; then
    VOXEL_SIZE=$(du -sh /home/michaltakac/pcdiff-implant/voxelization/out 2>/dev/null | cut -f1)
    echo "  Voxel output:    ${VOXEL_SIZE}"
fi

