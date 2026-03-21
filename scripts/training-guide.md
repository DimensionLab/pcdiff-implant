# Training Scripts Guide

Comprehensive guide for running PCDiff and Voxelization training simultaneously on multi-GPU systems.

## 📁 Quick Reference

### Scripts Available
- `train_pcdiff.sh` - PCDiff diffusion model training (resumes from checkpoint)
- `train_voxelization.sh` - Voxelization network training
- `launch_both.sh` - Automated launcher for both models in tmux
- `monitor_training.sh` - Real-time monitoring script
- `setup_verify.sh` - Verify configuration before training
- `remap_checkpoint.py` - Remap checkpoints for different GPU configurations

### Documentation
- This file - Complete training guide
- `README.md` - Detailed technical documentation

## 🚀 Quick Start

### Prerequisites
```bash
# Verify setup
bash scripts/setup_verify.sh
```

### Launch Training

**Option 1: Automated (Recommended)**
```bash
bash scripts/launch_both.sh
tmux attach -t skull_training
```

**Option 2: Manual (Separate terminals)**
```bash
# Terminal 1: PCDiff
bash scripts/train_pcdiff.sh

# Terminal 2: Voxelization  
bash scripts/train_voxelization.sh

# Terminal 3: Monitor
watch -n 5 bash scripts/monitor_training.sh
```

## ⚙️ Configuration

### PCDiff Training
```yaml
Script:        scripts/train_pcdiff.sh
GPUs:          0-6 (7 GPUs)
               CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
Batch Size:    56 per GPU (392 effective)
Learning Rate: 0.00122 (scaled for 7 GPUs)
Checkpoint:    epoch_1999_remapped.pth
Dataset:       pcdiff/datasets/SkullBreak
Resumes from:  Epoch 2000
Target:        Epoch 15000
Duration:      ~36-43 hours (~2 days)
```

### Voxelization Training
```yaml
Script:        scripts/train_voxelization.sh
GPU:           7 (mapped to device 0)
               CUDA_VISIBLE_DEVICES=7
Batch Size:    4 (paper used 2)
Learning Rate: 1×10⁻³ (scaled from 5×10⁻⁴ due to 2x batch)
Epochs:        1300 (paper specification)
Workers:       8
Duration:      ~36-48 hours (~2 days)
```

## 🎯 GPU Allocation Strategy

### Physical GPU Mapping
```
Physical GPUs    PCDiff Sees        Voxel Sees
─────────────    ───────────        ──────────
GPU 0            device 0           (hidden)
GPU 1            device 1           (hidden)
GPU 2            device 2           (hidden)
GPU 3            device 3           (hidden)
GPU 4            device 4           (hidden)
GPU 5            device 5           (hidden)
GPU 6            device 6           (hidden)
GPU 7            (hidden)           device 0
```

### How It Works

**PCDiff (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`):**
- Sees GPUs 0-6 as devices 0-6
- GPU 7 is completely hidden
- Distributed across 7 GPUs with torchrun

**Voxelization (`CUDA_VISIBLE_DEVICES=7`):**
- Sees only GPU 7 as device 0
- All other GPUs hidden
- Config uses `gpu: 0` (which is physical GPU 7)

**Key Insight:** Both processes can use "device 0" in their configs without conflict because `CUDA_VISIBLE_DEVICES` remaps the physical GPUs differently for each process.

## 🔧 Common Issues & Solutions

### Issue 1: Checkpoint Device Mismatch

**Error:**
```
RuntimeError: Attempting to deserialize object on CUDA device 7 
but torch.cuda.device_count() is 7.
```

**Solution:**
Checkpoint was trained on 8 GPUs but loading with 7 visible. Use the remapped checkpoint:

```bash
# Remap checkpoint (already done)
python3 scripts/remap_checkpoint.py checkpoint.pth

# Update script to use remapped checkpoint
CHECKPOINT="epoch_1999_remapped.pth"
```

### Issue 2: GPU Device Ordinal Error

**Error:**
```
RuntimeError: CUDA error: invalid device ordinal
```

**Solution:**
Voxelization script must explicitly set `CUDA_VISIBLE_DEVICES=7`:

```bash
# In train_voxelization.sh (already configured)
export CUDA_VISIBLE_DEVICES=7
python voxelization/train.py config.yaml
```

### Issue 3: Dataset Not Found

**Solution:**
Update dataset path in `scripts/train_pcdiff.sh`:

```bash
DATASET_PATH="pcdiff/datasets/SkullBreak"  # Directory, not CSV
```

### Issue 4: Port Already in Use

**Solution:**
Change master port in `scripts/train_pcdiff.sh`:

```bash
torchrun --master_port=29501 ...  # Change from 29500
```

### Issue 5: Out of Memory

**Solution:**
Reduce batch sizes:

```bash
# PCDiff
BATCH_SIZE=48  # Instead of 56

# Voxelization
batch_size: 2  # Instead of 4 in config
```

## 📊 Monitoring

### Quick Status Check
```bash
bash scripts/monitor_training.sh
```

Shows:
- Training status (running/stopped)
- GPU utilization and memory
- Latest log entries
- Disk usage

### Continuous Monitoring
```bash
watch -n 5 bash scripts/monitor_training.sh
```

### GPU Watch
```bash
# Simple
nvidia-smi

# Continuous
watch -n 1 nvidia-smi

# Detailed monitoring
nvidia-smi dmon -s u -d 1
```

### Log Files
```bash
# PCDiff
tail -f pcdiff/output/train_completion/2025-10-23-19-35-15/train.log

# Voxelization
tail -f voxelization/out/skullbreak*/train.log
```

## 🎛️ Tmux Controls

### Basic Commands
```bash
# Create session
tmux new -s training

# Detach (training continues)
Ctrl+b then d

# Reattach
tmux attach -t training

# List sessions
tmux ls

# Kill session
tmux kill-session -t training
```

### Window Management
```bash
# Switch windows
Ctrl+b then 0-2     # Direct number
Ctrl+b then w       # List windows
Ctrl+b then n       # Next window
Ctrl+b then p       # Previous window

# Rename window
Ctrl+b then ,       # Then type new name
```

### Pane Management
```bash
# Split panes
Ctrl+b then "       # Horizontal split
Ctrl+b then %       # Vertical split

# Navigate panes
Ctrl+b then arrows  # Move between panes
```

## 📈 Expected Performance

### GPU Utilization
- **GPUs 0-6**: 60-80% utilization (PCDiff distributed)
- **GPU 7**: 60-80% utilization (Voxelization)
- **Total**: All 8 GPUs actively training

### Memory Usage
- **PCDiff**: ~30-40GB per GPU (0-6)
- **Voxelization**: ~15-25GB (GPU 7)
- **Total GPU 7**: ~45-65GB (shared between PCDiff process and Voxelization)

### Training Duration
- **PCDiff**: ~36-43 hours (epoch 2000 → 15000)
- **Voxelization**: ~36-48 hours (1300 epochs)
- **Both complete**: ~2 days simultaneously

### Checkpoints
**PCDiff:**
- Saved every 1000 epochs
- `epoch_2000.pth`, `epoch_3000.pth`, etc.
- Location: `pcdiff/output/train_completion/*/`

**Voxelization:**
- Saved every 5 epochs (last checkpoint)
- `model.pt` (latest), `model_best.pt` (best validation)
- Location: `voxelization/out/skullbreak*/`

## 🔄 Resuming Training

### PCDiff
Already configured to resume from epoch 1999:

```bash
CHECKPOINT="pcdiff/output/train_completion/.../epoch_1999_remapped.pth"
# Will start at epoch 2000
```

### Voxelization
If interrupted, modify config:

```python
# In voxelization/train.py (line 98-108)
try:
    state_dict = torch.load(os.path.join(cfg['train']['out_dir'], 'model.pt'))
    # Automatically resumes from last checkpoint
except:
    # Starts fresh
    state_dict = dict()
```

## 📚 Learning Rate Scaling

### Why Scale?

When increasing batch size, scale learning rate proportionally:

```
new_lr = old_lr × (new_batch / old_batch)
```

### PCDiff
```
Paper:      8 GPUs × 8 batch = 64 total, lr = 2×10⁻⁴
Your setup: 7 GPUs × 56 batch = 392 total
Scaled lr:  2×10⁻⁴ × (392/64) = 0.00122
```

### Voxelization
```
Paper:      batch = 2, lr = 5×10⁻⁴
Your setup: batch = 4
Scaled lr:  5×10⁻⁴ × (4/2) = 1×10⁻³
```

**Rule of Thumb:** Double batch size → Double learning rate (for reasonable batch sizes)

## 🛑 Stopping Training

### Graceful Stop (Recommended)
In tmux window, press:
```bash
Ctrl+C
```

This allows:
- Current epoch to complete
- Checkpoint to be saved
- Clean shutdown

### Force Kill (Emergency)
```bash
# Kill specific training
pkill -f train_completion.py
pkill -f "voxelization/train.py"

# Kill entire tmux session
tmux kill-session -t skull_training
```

## 📝 Output Structure

```
pcdiff/output/train_completion/2025-10-23-19-35-15/
├── epoch_2000.pth              # Checkpoint at epoch 2000
├── epoch_3000.pth              # Checkpoint at epoch 3000
├── ...
├── syn/                        # Generated samples (if vizIter)
│   ├── epoch_2000_samples/
│   ├── epoch_2000_ground_truth/
│   └── epoch_2000_partial/
└── train.log                   # Training log

voxelization/out/skullbreak_YYYY_MM_DD_HH_MM_SS/
├── model.pt                    # Latest checkpoint
├── model_best.pt               # Best validation checkpoint
├── tensorboard_log/            # TensorBoard logs
└── train.log                   # Training log
```

## 🎓 Best Practices

1. **Always run `setup_verify.sh` first** - Catches configuration errors early
2. **Use tmux for long training** - Survives SSH disconnections
3. **Monitor first 100 epochs** - Verify loss is decreasing
4. **Check GPU memory** - Ensure no OOM issues
5. **Keep checkpoints** - Disk space is cheaper than retraining
6. **Use wandb/tensorboard** - Track experiments systematically
7. **Test on small dataset first** - Verify pipeline works end-to-end

## 🆘 Getting Help

### Check Documentation
- This guide (training scripts overview)
- `scripts/README.md` (detailed technical docs)
- `pcdiff/distributed-training.md` (distributed training concepts)

### Verify Configuration
```bash
bash scripts/setup_verify.sh
```

### Check Logs
```bash
# Recent errors
tail -100 pcdiff/output/train_completion/*/train.log

# Search for errors
grep -i error pcdiff/output/train_completion/*/train.log
```

### Monitor Resources
```bash
# GPU
nvidia-smi

# CPU/Memory
htop

# Disk space
df -h
```

## 📖 Related Documentation

- [Distributed Training Guide](../pcdiff/distributed-training.md) - Multi-GPU training concepts
- [Checkpoint Remapping](checkpoint-fix.md) - Details on fixing device mismatches  
- [GPU Allocation](gpu-fix.md) - Understanding CUDA_VISIBLE_DEVICES
- [Scripts README](README.md) - Technical implementation details

---

**Ready to train? Run:** `bash scripts/launch_both.sh` 🚀

