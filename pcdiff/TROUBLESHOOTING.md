# Distributed Training Troubleshooting Guide

## NCCL Timeout / Hang Issues

### Symptom
Training progresses normally for a while, then hangs for 10+ minutes before crashing with errors like:
```
[Rank N] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=XXXXX, OpType=ALLREDUCE, ...)
```

### Root Causes & Fixes

#### 1. Rank Divergence (Most Common)

**Problem**: One or more ranks stop participating in collective operations while others wait indefinitely.

**Common triggers**:
- Uneven batch counts across ranks (dataset size not divisible by world_size)
- Rank-conditional code paths (e.g., `if rank == 0: model.forward()`)
- DataLoader worker crashes on one rank

**Fixes** (already implemented in the latest code):

1. **DistributedSampler with `drop_last=True`**:
```python
# In get_dataloader() - lines 553-559
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True,  # ← Critical: ensures equal batch counts
)
```

2. **Distributed barriers around rank-conditional operations**:
```python
# Diagnostics at epoch milestones - lines 707-741
if (epoch + 1) % opt.diagIter == 0:
    if is_distributed:
        dist.barrier()  # ← All ranks wait here
    
    if should_diag:  # Only rank 0 does expensive ops
        logger.info('Diagnosis:')
        kl_stats = model.all_kl(pc_in)
        # ... logging ...
    
    if is_distributed:
        dist.barrier()  # ← All ranks wait again
```

3. **Automatic step count validation** (lines 722-737):
```python
# At end of each epoch, verify all ranks ran same number of steps
if is_distributed:
    step_count_tensor = torch.tensor([epoch_step_count], ...)
    gathered_counts = [torch.zeros_like(step_count_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_counts, step_count_tensor)
    
    if should_diag:
        counts = [t.item() for t in gathered_counts]
        if len(set(counts)) > 1:
            logger.error(f"STEP COUNT DIVERGENCE! Counts: {counts}")
            raise RuntimeError(f"Mismatch: {counts}")
```

This will **fail fast** if ranks diverge, instead of hanging for 10 minutes.

---

#### 2. GPU Hardware Issues

**Problem**: ECC errors, Xid faults, or GPU resets on one GPU can cause that rank to stall.

**Diagnosis**:
```bash
# Check for GPU errors in system logs
dmesg -T | grep -Ei 'nvrm|xid|cuda|gpu'

# Or on systemd systems
journalctl -k --since "1 hour ago" | grep -Ei 'xid|nvrm|cuda'

# Monitor GPU health during training
nvidia-smi dmon -s pucvmet -d 5
```

**Fixes**:
- If you see Xid errors (especially Xid 79, 63, 48): GPU hardware fault → contact admin
- High ECC error rates: may need GPU replacement
- Temperature throttling: improve cooling/airflow

---

#### 3. Network / NVLink Issues

**Problem**: NCCL communication fails or becomes very slow due to network topology issues.

**Diagnosis**:
```bash
# Check GPU interconnect topology
nvidia-smi topo -m

# All GPUs should show "NV#" (NVLink) or "PIX" (PCIe) between them
# "PHB" or "NODE" indicates poor connectivity → very slow collective ops
```

**Fixes**:
- Ensure all GPUs are on the same node or have proper NVLink/InfiniBand
- Set `NCCL_DEBUG=INFO` to see which operations are slow
- For multi-node: verify InfiniBand/RoCE is configured correctly

---

#### 4. Debugging Environment Variables

Enable detailed logging to pinpoint the issue:

```bash
# Maximum verbosity (warning: lots of output)
export NCCL_DEBUG=INFO                    # NCCL communication details
export TORCH_DISTRIBUTED_DEBUG=DETAIL     # PyTorch DDP details
export NCCL_ASYNC_ERROR_HANDLING=1        # Fail fast on NCCL errors
export NCCL_BLOCKING_WAIT=1               # Synchronous waits (easier to debug)

# Shorter timeout for faster failure (default: 30 min in our script)
export TORCH_DIST_TIMEOUT=5               # 5 minutes

# Run training
torchrun --nproc_per_node=8 pcdiff/train_completion.py [args]
```

**What to look for in logs**:
- Last successful collective operation number (SeqNum)
- Which rank(s) didn't report a timeout (likely the one that diverged)
- Any Python exceptions or warnings just before the hang

---

## Out of Memory (OOM) Errors

### Symptom
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

### Fixes

1. **Reduce per-GPU batch size**:
```bash
# Instead of bs=64 for 8 GPUs (8 per GPU)
torchrun --nproc_per_node=8 pcdiff/train_completion.py --bs 32  # 4 per GPU

# Or bs=48 (6 per GPU)
```

2. **Enable gradient checkpointing** (not yet implemented, but can be added):
- Trades compute for memory by recomputing activations during backward pass
- See `torch.utils.checkpoint.checkpoint()` for implementation

3. **Reduce point cloud resolution**:
```bash
python pcdiff/train_completion.py --num_points 2048  # default is 2048
```

4. **Check GPU memory usage**:
```bash
nvidia-smi dmon -s um -d 1
```

---

## DataLoader / Worker Crashes

### Symptom
Training hangs at random iterations, or you see warnings about worker crashes.

### Diagnosis
```python
# In your terminal, check for worker process errors
ps aux | grep python | grep multiprocessing
```

### Fixes

1. **Reduce `num_workers`** (already computed automatically in the script):
```bash
python pcdiff/train_completion.py --workers 8  # or lower
```

2. **Disable persistent workers** (if using):
```python
# In get_dataloader(), line 570-578
train_dataloader = torch.utils.data.DataLoader(
    ...,
    num_workers=num_workers,
    persistent_workers=False,  # ← Change to False if crashes occur
)
```

3. **Add timeout to DataLoader**:
```python
train_dataloader = torch.utils.data.DataLoader(
    ...,
    timeout=60,  # seconds to wait before worker is considered dead
)
```

---

## Checkpoint Loading Issues

### Symptom
```
KeyError: 'model_state_dict'
# or
RuntimeError: Error(s) in loading state_dict
```

### Fixes

1. **Check checkpoint structure**:
```python
import torch
ckpt = torch.load('path/to/checkpoint.pth', map_location='cpu')
print(ckpt.keys())  # Should show: dict_keys(['epoch', 'model_state', 'optimizer_state'])
```

2. **Ensure DDP model compatibility**:
```python
# If checkpoint was saved with DDP, keys will have 'module.' prefix
# Our script handles this automatically via model.multi_gpu_wrapper()
```

---

## Logging / WandB Issues

### WandB authentication errors

```bash
# Login to wandb before training
wandb login

# Or disable wandb entirely
python pcdiff/train_completion.py --no-wandb [other args]
```

### Missing logs from non-rank-0 processes

This is **expected behavior**. Only rank 0 logs to files and WandB to avoid conflicts.
If you need per-rank logs for debugging:

```python
# Temporarily modify setup_logging() to log per-rank
logger = logging.getLogger(f'rank_{rank}')
handler = logging.FileHandler(f'{output_dir}/rank_{rank}.log')
```

---

## Quick Diagnostic Checklist

When your training hangs or crashes:

1. **Check the last logged epoch/step**:
   - Does it correlate with `diagIter`, `vizIter`, or `saveIter`? → likely a barrier issue
   - Random iteration? → likely dataloader/worker or GPU issue

2. **Check system logs**:
   ```bash
   dmesg -T | tail -100
   ```

3. **Check GPU health**:
   ```bash
   nvidia-smi
   ```

4. **Enable debug logs and retry**:
   ```bash
   export NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL
   ```

5. **Try on fewer GPUs first**:
   ```bash
   torchrun --nproc_per_node=2 pcdiff/train_completion.py [args]
   ```

6. **Check dataset size**:
   ```python
   import pandas as pd
   df = pd.read_csv('pcdiff/datasets/SkullBreak/train.csv')
   print(f"Dataset size: {len(df)}")
   print(f"Per rank (8 GPUs): {len(df) // 8}")
   print(f"Dropped samples (drop_last): {len(df) % 8}")
   ```

---

## Getting Help

If none of the above fixes your issue, please provide:

1. **Full error traceback** (not just the last few lines)
2. **Last 100 lines of training logs** from rank 0
3. **Environment info**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, NCCL: {torch.cuda.nccl.version()}')"
   nvidia-smi
   ```
4. **Command used to launch training**
5. **Dataset size and number of GPUs**

Open an issue on GitHub with these details for assistance.

---

## Prevention: Best Practices

1. **Always use `DistributedSampler` with `drop_last=True`** for training
2. **Always use `dist.barrier()` around rank-conditional model operations**
3. **Start with fewer GPUs/epochs to validate setup** before full-scale runs
4. **Monitor GPU memory and utilization** during initial epochs
5. **Use tmux/screen** to prevent SSH disconnects from killing long runs
6. **Enable automatic step count validation** (already in latest code)
7. **Set reasonable timeout values** via `TORCH_DIST_TIMEOUT` during development


