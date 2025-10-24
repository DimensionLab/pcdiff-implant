# Quick Start: Resume Your Training

Your training crashed at epoch 1999 due to NCCL timeout. The issue has been fixed. Here's how to resume:

## Option 1: Resume from Checkpoint (Recommended)

```bash
cd /home/michaltakac/pcdiff-implant

# Find your checkpoint
ls -lh pcdiff/output/train_completion/*/epoch_*.pth

# Resume training with the fixed code
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 \
    --lr 1.6e-3 \
    --niter 15000 \
    --model pcdiff/output/train_completion/<YOUR_RUN_ID>/epoch_1500.pth
```

Replace `<YOUR_RUN_ID>` with your actual run directory and use the latest checkpoint available.

---

## Option 2: Test Fix with Short Run First

```bash
# Test on 2 GPUs for faster validation
export TORCH_DIST_TIMEOUT=5  # Fail fast if issues remain

torchrun --nproc_per_node=2 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 16 \
    --niter 2010 \
    --diagIter 2000 \
    --print_freq 1
```

This will test:
- Step count validation every epoch
- Diagnostic barriers at epoch 2000
- Should complete in ~30 minutes

If successful, scale to 8 GPUs.

---

## Option 3: Start Fresh with Full Debugging

```bash
# Enable debug logging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1

# Run training
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 \
    --lr 1.6e-3 \
    --niter 15000
```

**Warning**: Debug mode generates lots of logs. Only use for initial validation.

---

## What Was Fixed

### Critical Changes:
1. ✅ `DistributedSampler` now has `drop_last=True` → equal batch counts
2. ✅ Distributed barriers around diagnostic code → prevents rank divergence  
3. ✅ Automatic step count validation → catches issues early
4. ✅ Configurable timeout (default 30 min) → faster failure detection

### Files Modified:
- `pcdiff/train_completion.py` - Core fixes
- `pcdiff/README.md` - Added debugging section
- `pcdiff/TROUBLESHOOTING.md` - Comprehensive guide
- `pcdiff/NCCL-TIMEOUT-FIX.md` - Detailed explanation

---

## Monitoring Your Run

### In another terminal:
```bash
# Watch logs in real-time
tail -f pcdiff/output/train_completion/*/output.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Expected log output every 100 epochs:
```
Epoch 100: All ranks completed 7 steps ✓
Epoch 200: All ranks completed 7 steps ✓
...
```

### At epoch 2000, 4000, etc. (diagIter milestones):
```
Diagnosis:
      [2000/15000]    x_range: [-0.5, 0.5],   total_bpd_b: ...
```

**No hanging or timeouts!**

---

## If Problems Persist

See the full troubleshooting guide:
```bash
cat pcdiff/TROUBLESHOOTING.md
```

Or check for hardware issues:
```bash
dmesg -T | grep -Ei 'nvrm|xid|cuda'
nvidia-smi topo -m
```

---

## Background Training (Persistent Session)

To prevent SSH disconnects from killing your training:

```bash
# Start tmux
tmux new -s training

# Run training command
torchrun --nproc_per_node=8 pcdiff/train_completion.py [your args]

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t training
```

---

## Need Help?

All documentation is in `pcdiff/`:
- `NCCL-TIMEOUT-FIX.md` - Detailed explanation of what was fixed
- `TROUBLESHOOTING.md` - Comprehensive debugging guide  
- `README.md` - General usage and debugging tips

Good luck with your training! 🚀

