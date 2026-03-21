# Training Scripts

This directory contains scripts for training both PCDiff and Voxelization models simultaneously on the 8×H100 server.

## Scripts Overview

### 1. `train_pcdiff_resume.sh`
Resumes PCDiff training from epoch 1999 using GPUs 0-6.

**Configuration:**
- GPUs: 0-6 (7 GPUs total)
- Batch size: 56 per GPU (effective: 392)
- Learning rate: 0.00122
- Workers: 21 (3 per GPU)

**Before running:**
Update `DATASET_PATH` in the script to point to your SkullBreak dataset.

### 2. `train_voxelization.sh`
Trains the voxelization model on GPU 7.

**Configuration:**
- GPU: 7
- Batch size: Configured in yaml file
- Workers: Configured in yaml file

**Before running:**
Update `voxelization/configs/train_skullbreak.yaml`:
```yaml
train:
  gpu: 7
  batch_size: 4
  n_workers: 8
```

### 3. `launch_both.sh`
Automatically launches both trainings in separate tmux windows with monitoring.

Creates a tmux session called `skull_training` with 3 windows:
- `pcdiff`: PCDiff training
- `voxel`: Voxelization training  
- `monitor`: GPU monitoring and logs

### 4. `monitor_training.sh`
Standalone monitoring script that shows:
- Training status (running/not running)
- GPU utilization and memory
- Latest log entries
- Disk usage

## Usage

### Option 1: Manual Launch (Separate Tmux Panes)

```bash
# Create tmux session
tmux new -s training

# In first pane
bash scripts/train_pcdiff_resume.sh

# Split horizontally (Ctrl+b then ")
# In second pane
bash scripts/train_voxelization.sh

# Split again for monitoring
# In third pane
watch -n 5 bash scripts/monitor_training.sh
```

### Option 2: Automated Launch

```bash
# Launch everything automatically
bash scripts/launch_both.sh

# Attach to the session
tmux attach -t skull_training
```

## Tmux Cheat Sheet

```bash
# Navigation
Ctrl+b then 0-2     # Switch to window 0, 1, or 2
Ctrl+b then w       # List all windows
Ctrl+b then n       # Next window
Ctrl+b then p       # Previous window

# Pane management
Ctrl+b then "       # Split horizontally
Ctrl+b then %       # Split vertically
Ctrl+b then arrows  # Navigate between panes

# Session management
Ctrl+b then d       # Detach (keeps training running)
tmux attach -t skull_training  # Reattach
tmux ls             # List all sessions
tmux kill-session -t skull_training  # Kill session
```

## Monitoring

### Quick Status Check
```bash
bash scripts/monitor_training.sh
```

### Continuous Monitoring
```bash
watch -n 5 bash scripts/monitor_training.sh
```

### GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

### Log Files
- PCDiff: `pcdiff/output/train_completion/2025-10-23-19-35-15/train.log`
- Voxelization: `voxelization/out/skullbreak*/train.log`

## Stopping Training

### Stop One Training
In the tmux window, press `Ctrl+C`

### Stop All Training
```bash
# Kill specific processes
pkill -f train_completion.py
pkill -f "voxelization/train.py"

# Or kill entire tmux session
tmux kill-session -t skull_training
```

## Troubleshooting

### PCDiff: Checkpoint Not Found
Update `CHECKPOINT` path in `train_pcdiff_resume.sh`

### PCDiff: Dataset Not Found
Update `DATASET_PATH` in `train_pcdiff_resume.sh`

### Voxelization: GPU Conflict
Check `voxelization/configs/train_skullbreak.yaml` and ensure `gpu: 7`

### Port Already in Use (PCDiff)
Change `--master_port=29500` to another port (e.g., 29501)

### Out of Memory
- PCDiff: Reduce `BATCH_SIZE` in script (try 48 or 40)
- Voxelization: Reduce `batch_size` in config (try 2 or 3)

## Expected Performance

### PCDiff
- Time per epoch: ~10-12 seconds (with 7 GPUs)
- GPU utilization: 60-80% per GPU (0-6)
- Memory usage: ~25-40GB per GPU

### Voxelization
- Time per epoch: Variable (depends on dataset size)
- GPU utilization: 60-80% (GPU 7)
- Memory usage: ~15-25GB

### GPU 7 (Shared)
- Total utilization: 80-90%
- Total memory: ~50-65GB / 80GB available

