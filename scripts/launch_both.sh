#!/bin/bash

# Launch both trainings in separate tmux windows
SESSION_NAME="skull_training"

echo "================================================"
echo "Launching training in tmux session: ${SESSION_NAME}"
echo "================================================"
echo ""

# Check if tmux session already exists
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "Tmux session '${SESSION_NAME}' already exists."
    read -p "Kill and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t ${SESSION_NAME}
    else
        echo "Exiting. Attach with: tmux attach -t ${SESSION_NAME}"
        exit 0
    fi
fi

# Create new tmux session with first window for PCDiff
echo "Creating tmux session..."
tmux new-session -d -s ${SESSION_NAME} -n "pcdiff"

# Run PCDiff training in first window
echo "Starting PCDiff training (window: pcdiff)..."
tmux send-keys -t ${SESSION_NAME}:pcdiff "cd /home/michaltakac/pcdiff-implant && bash scripts/train_pcdiff_resume.sh" C-m

# Create second window for Voxelization
echo "Starting Voxelization training (window: voxel)..."
tmux new-window -t ${SESSION_NAME} -n "voxel"
tmux send-keys -t ${SESSION_NAME}:voxel "cd /home/michaltakac/pcdiff-implant && bash scripts/train_voxelization.sh" C-m

# Create third window for monitoring
echo "Creating monitoring window (window: monitor)..."
tmux new-window -t ${SESSION_NAME} -n "monitor"
tmux send-keys -t ${SESSION_NAME}:monitor "watch -n 1 nvidia-smi" C-m

# Split the monitor window to show logs
tmux split-window -t ${SESSION_NAME}:monitor -h
tmux send-keys -t ${SESSION_NAME}:monitor "echo 'Waiting for logs...'; sleep 3; tail -f /home/michaltakac/pcdiff-implant/pcdiff/output/train_completion/*/train.log 2>/dev/null || echo 'No log file yet'" C-m

echo ""
echo "================================================"
echo "Tmux session '${SESSION_NAME}' created!"
echo "================================================"
echo ""
echo "Windows:"
echo "  - pcdiff:  PCDiff training (GPUs 0-6)"
echo "  - voxel:   Voxelization training (GPU 7)"
echo "  - monitor: GPU monitoring + logs"
echo ""
echo "Attach with:    tmux attach -t ${SESSION_NAME}"
echo "Switch windows: Ctrl+b then 'w' (list) or '0-2' (direct)"
echo "Detach:         Ctrl+b then 'd'"
echo ""

