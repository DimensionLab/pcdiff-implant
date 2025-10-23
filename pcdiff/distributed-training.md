# Distributed Training Guide

This guide explains how to properly scale batch size and learning rate when training the point cloud diffusion model across multiple GPUs.

## Table of Contents
- [Quick Reference](#quick-reference)
- [Understanding Batch Size Distribution](#understanding-batch-size-distribution)
- [Learning Rate Scaling](#learning-rate-scaling)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Quick Reference

### Single GPU (Original Setup)
```bash
python pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 8 \
    --lr 2e-4
```
- **Global batch size:** 8
- **Per-GPU batch size:** 8
- **Training time:** ~baseline

### 8x GPU (Recommended Configuration)
```bash
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 \
    --lr 1.6e-3
```
- **Global batch size:** 64 (8× larger)
- **Per-GPU batch size:** 8 (same as original)
- **Training time:** ~8× faster
- **Learning rate:** 1.6e-3 (8× original)

### 8x GPU (NOT Recommended - Same Global Batch)
```bash
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 8 \
    --lr 2e-4
```
- **Global batch size:** 8 (same as original)
- **Per-GPU batch size:** 1 (very small!)
- ⚠️ **Problem:** Each GPU gets only 1 sample, leading to noisy gradients and poor convergence

## Understanding Batch Size Distribution

### How `--bs` Works with Multiple GPUs

The training script automatically distributes the global batch size across available GPUs:

```python
# From train_completion.py (lines 566-568)
global_batch = opt.bs
per_device_batch = max(global_batch // world_size, 1)
```

**Example calculations:**

| Configuration | `--bs` | GPUs | Global Batch | Per-GPU Batch |
|---------------|--------|------|--------------|---------------|
| Original (1× A100) | 8 | 1 | 8 | 8 |
| Multi-GPU ❌ | 8 | 8 | 8 | 1 |
| Multi-GPU ✅ | 64 | 8 | 64 | 8 |
| Multi-GPU | 128 | 8 | 128 | 16 |

### Why This Matters

**The model uses GroupNorm, not BatchNorm:**
- ✅ **GroupNorm** normalizes within each sample independently
- ✅ Works well even with per-GPU batch size of 1
- ✅ No "BatchNorm with small batch" issues

**However, per-GPU batch size still affects:**
- **Gradient variance:** Smaller batches → noisier gradients → less stable training
- **GPU utilization:** Very small batches underutilize GPU memory and compute
- **Convergence quality:** May require different hyperparameters to converge well

## Learning Rate Scaling

### Why Scale Learning Rate?

When you increase the global batch size, you're averaging gradients over more samples per update. This makes the gradient estimate more accurate but also changes the effective learning rate.

### Scaling Rules

**Linear Scaling (Recommended):**
```
new_lr = base_lr × (new_batch_size / base_batch_size)
```

For 8× batch size increase:
```
--lr 1.6e-3  # = 2e-4 × 8
```

**Square Root Scaling (More Conservative):**
```
new_lr = base_lr × sqrt(new_batch_size / base_batch_size)
```

For 8× batch size increase:
```
--lr 5.66e-4  # = 2e-4 × √8
```

### Recommendations by Number of GPUs

| GPUs | `--bs` | `--lr` (Linear) | `--lr` (Conservative) |
|------|--------|-----------------|----------------------|
| 1 | 8 | 2e-4 | 2e-4 |
| 2 | 16 | 4e-4 | 2.83e-4 |
| 4 | 32 | 8e-4 | 4e-4 |
| 8 | 64 | 1.6e-3 | 5.66e-4 |

### Learning Rate Warmup (Optional)

For very large batch sizes, consider adding learning rate warmup. Currently not implemented in the codebase, but you can:

1. Start with lower LR for first N epochs
2. Gradually increase to target LR
3. Then follow normal schedule

## Best Practices

### 1. Maintain Per-GPU Batch Size

**Rule of thumb:** Keep per-GPU batch size at 8 (the original setting)

```bash
# For N GPUs, use:
--bs $(( N * 8 ))
```

### 2. Scale Learning Rate Appropriately

Start with linear scaling, then fine-tune:

```bash
# Calculate: N × 2e-4
--lr $(python3 -c "print(f'{8 * 2e-4:.1e}')")  # For 8 GPUs
```

### 3. Monitor Training Metrics

Watch for these signs:

**Good training:**
- ✅ Smooth loss curves (less noisy than single GPU)
- ✅ Convergence to similar final loss as baseline
- ✅ Validation metrics comparable to single-GPU training

**Needs adjustment:**
- ⚠️ Loss diverging or not decreasing → reduce learning rate
- ⚠️ Very noisy loss curves → increase per-GPU batch size
- ⚠️ Slower convergence → increase learning rate or add warmup

### 4. Use Wandb for Monitoring

Enable experiment tracking to compare runs:

```bash
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 \
    --lr 1.6e-3 \
    --wandb-project pcdiff-implant \
    --wandb-name "skullbreak_8gpu_bs64_lr1.6e-3"
```

### 5. Persistent Training Sessions

Use `tmux` or `screen` to prevent training interruption:

**tmux (recommended):**
```bash
tmux new -s training
torchrun --nproc_per_node=8 pcdiff/train_completion.py [args]
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

**screen:**
```bash
screen -S training
torchrun --nproc_per_node=8 pcdiff/train_completion.py [args]
# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

**nohup (simple background process):**
```bash
nohup torchrun --nproc_per_node=8 pcdiff/train_completion.py [args] > training.log 2>&1 &
tail -f training.log
```

**systemd service (auto-restart on reboot):**

Create `/etc/systemd/system/pcdiff-training.service`:
```ini
[Unit]
Description=PCDiff Training Service
After=network.target

[Service]
Type=simple
User=michaltakac
WorkingDirectory=/home/michaltakac/pcdiff-implant
Environment="PATH=/home/michaltakac/.local/bin:/usr/local/bin:/usr/bin"
ExecStart=/usr/bin/torchrun --nproc_per_node=8 pcdiff/train_completion.py --path pcdiff/datasets/SkullBreak/train.csv --dataset SkullBreak --bs 64 --lr 1.6e-3
Restart=on-failure
RestartSec=10s
StandardOutput=append:/home/michaltakac/pcdiff-implant/training.log
StandardError=append:/home/michaltakac/pcdiff-implant/training_error.log

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable pcdiff-training.service
sudo systemctl start pcdiff-training.service
sudo systemctl status pcdiff-training.service
```

## Troubleshooting

### Issue: "FileExistsError" on Resume

**Solution:** Already fixed in `pcdiff/utils/file_utils.py` with `exist_ok=True`

### Issue: "Missing key(s) in state_dict" or "module." prefix mismatch

**Error message:**
```
RuntimeError: Error(s) in loading state_dict for Model:
    Missing key(s) in state_dict: "model.sa_layers.0.0.voxel_layers.0.weight"...
    Unexpected key(s) in state_dict: "model.module.sa_layers.0.0.voxel_layers.0.weight"...
```

**Cause:** Checkpoint was saved during distributed training with `DistributedDataParallel`, which adds a `module.` prefix to all parameters. When loading on single GPU for inference, the model expects keys without this prefix.

**Solution:** Already fixed in `pcdiff/test_completion.py`. The script now automatically detects and removes the `module.` prefix when loading DDP-saved checkpoints.

If you're using an older version or custom script, add this before `load_state_dict()`:
```python
state_dict = torch.load(checkpoint_path)['model_state']
if list(state_dict.keys())[0].startswith('model.module.'):
    state_dict = {k.replace('model.module.', 'model.'): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
```

### Issue: Loss is too noisy

**Cause:** Per-GPU batch size is too small (e.g., 1)

**Solution:** Increase `--bs` to maintain per-GPU batch of 8+:
```bash
--bs 64  # For 8 GPUs
```

### Issue: Loss diverges or doesn't decrease

**Cause:** Learning rate is too high for large batch size

**Solution:** Use more conservative LR scaling:
```bash
--lr 5.66e-4  # Square root scaling instead of linear
```

Or reduce batch size:
```bash
--bs 32 --lr 8e-4  # For 8 GPUs, 4 per-GPU batch
```

### Issue: Out of memory (OOM)

**Cause:** Per-GPU batch size is too large for GPU memory

**Solution:** Reduce per-GPU batch:
```bash
--bs 32  # For 8 GPUs, 4 per-GPU batch
--lr 8e-4  # Adjust LR accordingly
```

Or use gradient accumulation (not currently implemented in codebase).

### Issue: Training is slower than expected

**Possible causes:**
1. Per-GPU batch too small → increase `--bs`
2. Data loading bottleneck → increase `--workers`
3. Network/communication overhead → check NCCL backend

**Debug:**
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Check data loading
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    [...] \
    --workers 24  # Adjust based on CPU cores
```

### Issue: "NCCL error" or Communication Timeout

**Cause:** Network issues or misconfigured distributed setup

**Solution:**
```bash
# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # If InfiniBand is not available
export NCCL_SOCKET_IFNAME=eth0  # Adjust to your network interface

torchrun --nproc_per_node=8 pcdiff/train_completion.py [args]
```

### Issue: Different GPUs have different utilization

**Cause:** Imbalanced data distribution or model parallelism issues

**Solution:** This is normal for the last batch if `drop_last=False`. The training script uses `drop_last=True` (line 576), so this shouldn't happen. If it persists, check data sampler.

## Technical Details

### Model Architecture
- Uses **GroupNorm** (not BatchNorm), so normalization is independent of batch size
- Uses **Adam optimizer** with β₁=0.5, β₂=0.999
- Default LR schedule: Exponential decay with γ=1.0 (no decay by default)

### Distributed Training Setup
- Backend: NCCL (default, optimized for NVIDIA GPUs)
- Uses PyTorch `DistributedDataParallel` (DDP)
- Automatic environment variable detection from `torchrun`:
  - `WORLD_SIZE`: Total number of GPUs
  - `RANK`: Global rank of current process
  - `LOCAL_RANK`: Local rank on current node

### Loss Computation
```python
# From train_completion.py (line 677)
loss = model.get_loss_iter(pc_in, noises_batch).mean()
```

The loss is computed per-sample and averaged across the batch:
- With larger batches: Lower variance, more stable gradients
- With smaller batches: Higher variance, noisier gradients

### Gradient Synchronization
DDP automatically:
1. Computes gradients independently on each GPU
2. All-reduces gradients across GPUs (averages them)
3. Updates model parameters identically on all GPUs

This means the effective batch size for gradient computation is the **global batch size** (sum across all GPUs).

### Data Loading
```python
# From train_completion.py (lines 570-578)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=per_device_batch,
    sampler=train_sampler,
    shuffle=train_sampler is None,
    num_workers=num_workers,
    drop_last=True,
    pin_memory=True,
)
```

- `DistributedSampler` ensures each GPU gets different data
- `drop_last=True` ensures all GPUs get same number of batches
- `pin_memory=True` for faster GPU transfer

## Example Training Runs

### Baseline (1× A100, batch=8, lr=2e-4)
```bash
python pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 8 \
    --lr 2e-4 \
    --niter 15000
```
Expected: ~15,000 epochs to convergence

### Scaled (8× H100, batch=64, lr=1.6e-3)
```bash
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 \
    --lr 1.6e-3 \
    --niter 15000 \
    --wandb-name "8gpu_bs64_linear_scaling"
```
Expected: Similar final metrics, ~8× faster wall-clock time

### Conservative Scaling (8× H100, batch=64, lr=5.66e-4)
```bash
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 \
    --lr 5.66e-4 \
    --niter 15000 \
    --wandb-name "8gpu_bs64_sqrt_scaling"
```
Expected: Slower convergence per epoch, but more stable training

### Smaller Batch (8× H100, batch=32, lr=8e-4)
```bash
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 32 \
    --lr 8e-4 \
    --niter 15000 \
    --wandb-name "8gpu_bs32_4per_gpu"
```
Per-GPU batch size: 4 (lower memory usage)

## Performance Expectations

### Training Speed Comparison

| Setup | Global BS | Per-GPU BS | Relative Speed | Wall-Clock Time (15k epochs) |
|-------|-----------|------------|----------------|------------------------------|
| 1× A100 | 8 | 8 | 1.0× | ~baseline |
| 8× H100 (bs=8) | 8 | 1 | ~1.0× | ~baseline (no benefit!) |
| 8× H100 (bs=64) | 64 | 8 | ~8.0× | ~baseline/8 |
| 8× H100 (bs=128) | 128 | 16 | ~8.0× | ~baseline/8 |

**Note:** Actual speedup depends on:
- Data loading efficiency (use sufficient `--workers`)
- Network bandwidth between GPUs (NVLink vs PCIe)
- Model architecture (communication overhead)

### Memory Usage

Per-GPU memory usage increases with per-GPU batch size:

| Per-GPU Batch | Estimated Memory | Compatible GPUs |
|---------------|------------------|-----------------|
| 1 | ~6 GB | Most modern GPUs |
| 4 | ~12 GB | RTX 3090, A100, H100 |
| 8 | ~20 GB | A100-40GB, H100 |
| 16 | ~35 GB | A100-80GB, H100-80GB |

**Monitoring memory:**
```bash
nvidia-smi dmon -s u -c 100  # Monitor GPU utilization
watch -n 1 nvidia-smi  # Real-time monitoring
```

## Multi-Node Training

For training across multiple nodes (machines), use:

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    pcdiff/train_completion.py [args]

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    pcdiff/train_completion.py [args]
```

For 16 GPUs total (2 nodes × 8 GPUs):
```bash
--bs 128  # 16 × 8
--lr 3.2e-3  # 16 × 2e-4
```

## References

- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) - Linear LR scaling
- [Train longer, generalize better](https://arxiv.org/abs/1705.08741) - Batch size effects
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)

