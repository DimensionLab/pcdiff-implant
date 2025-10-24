# Checkpoint & GPU Configuration Fixes

Technical reference for resolving common GPU and checkpoint issues when training with different hardware configurations.

## Checkpoint Device Mismatch

### Problem
```
RuntimeError: Attempting to deserialize object on CUDA device 7 
but torch.cuda.device_count() is 7.
```

### Root Cause
- Checkpoint was saved on system with 8 GPUs (devices 0-7)
- Checkpoint contains tensors that were on GPU 7
- Loading with only 7 visible GPUs (devices 0-6)
- PyTorch can't find device 7 to load the tensors

### Solution: Remap Checkpoint to CPU

**Method 1: Using Script**
```bash
python3 scripts/remap_checkpoint.py checkpoint.pth
```

**Method 2: Manual (Python)**
```python
import torch

# Load with map_location='cpu' to avoid device issues
checkpoint = torch.load('checkpoint.pth', map_location='cpu')

# Save remapped checkpoint
torch.save(checkpoint, 'checkpoint_remapped.pth')
```

### Why This Works
- `map_location='cpu'` loads all tensors to CPU regardless of original device
- PyTorch automatically moves tensors to correct GPUs during model initialization
- No data loss or performance impact
- Checkpoint becomes device-agnostic

## GPU Device Ordinal Error

### Problem
```
RuntimeError: CUDA error: invalid device ordinal
```

### Root Cause
When running multiple trainings:
1. PCDiff sets `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`
2. Voxelization inherits this environment variable
3. Voxelization config says `gpu: 7`
4. But GPU 7 isn't in the visible devices list
5. PyTorch throws "invalid device ordinal"

### Solution: Independent CUDA_VISIBLE_DEVICES

Each training script must explicitly set its own device visibility:

**PCDiff Script:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 train.py
```

**Voxelization Script:**
```bash
export CUDA_VISIBLE_DEVICES=7  # Overrides any inherited value
python train.py config.yaml
```

### GPU Remapping

When `CUDA_VISIBLE_DEVICES=7`:
- Physical GPU 7 becomes logical device 0
- PyTorch sees only 1 GPU (device 0)
- Config should use `gpu: 0` (not `gpu: 7`)
- This maps to physical GPU 7 ✓

**Configuration:**
```yaml
# voxelization/configs/train_skullbreak.yaml
train:
  gpu: 0  # Logical device 0 = physical GPU 7
```

## GPU Allocation Architecture

### Physical vs Logical Mapping

```
┌─────────────────────────────────────────────┐
│ PCDiff Process                              │
│ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6         │
├──────────────────────────────────────────── │
│ Physical → Logical                          │
│ GPU 0    → device 0                         │
│ GPU 1    → device 1                         │
│ GPU 2    → device 2                         │
│ GPU 3    → device 3                         │
│ GPU 4    → device 4                         │
│ GPU 5    → device 5                         │
│ GPU 6    → device 6                         │
│ GPU 7    → (hidden)                         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Voxelization Process                        │
│ CUDA_VISIBLE_DEVICES=7                      │
├─────────────────────────────────────────────│
│ Physical → Logical                          │
│ GPU 0    → (hidden)                         │
│ GPU 1    → (hidden)                         │
│ GPU 2    → (hidden)                         │
│ GPU 3    → (hidden)                         │
│ GPU 4    → (hidden)                         │
│ GPU 5    → (hidden)                         │
│ GPU 6    → (hidden)                         │
│ GPU 7    → device 0                         │
└─────────────────────────────────────────────┘
```

### Key Benefits

1. **Process Isolation**: Each process only sees its assigned GPUs
2. **No Conflicts**: Both can use "device 0" without collision
3. **Resource Control**: Clear GPU ownership per process
4. **Easy Management**: Simple environment variable control

## Learning Rate Scaling

### Linear Scaling Rule

When changing batch size, scale learning rate proportionally:

```
new_lr = old_lr × (new_batch / old_batch)
```

### Why Scale?

**Larger batch size:**
- Averages gradients over more samples
- Fewer optimization steps per epoch
- Each step uses more computation

**Scaling LR compensates:**
- Maintains effective learning rate
- Same convergence speed
- Similar final performance

### Examples

**PCDiff:**
```
Paper:      8 GPUs × 8 batch = 64,  lr = 2×10⁻⁴
Your setup: 7 GPUs × 56 batch = 392, lr = 0.00122

Calculation: 2×10⁻⁴ × (392/64) = 2×10⁻⁴ × 6.125 = 0.001225 ≈ 0.00122
```

**Voxelization:**
```
Paper:      batch = 2, lr = 5×10⁻⁴
Your setup: batch = 4, lr = 1×10⁻³

Calculation: 5×10⁻⁴ × (4/2) = 5×10⁻⁴ × 2 = 1×10⁻³
```

### When to Apply

✅ **Apply scaling:**
- Batch size changes are moderate (2x - 8x)
- Using well-tested optimizer (Adam, SGD)
- Training from scratch or resuming with same batch
- Goal is faster training with equivalent results

⚠️ **Be cautious:**
- Very large batch sizes (>1024)
- Fine-tuning pretrained models
- Batch size changes are extreme (>10x)
- Training is already unstable

### Monitoring

Watch first 100-200 epochs for:

**Good signs (LR appropriate):**
- ✅ Steady loss decrease
- ✅ Smooth training curves
- ✅ Validation improving
- ✅ No NaN/inf values

**Bad signs (LR too high):**
- ❌ Loss oscillates wildly
- ❌ Sudden spikes
- ❌ NaN or inf appears
- ❌ Training diverges

**Slow signs (LR too low):**
- 📉 Very slow progress
- 📉 Flat loss curves
- 📉 Takes many epochs to improve

## Practical Guidelines

### Before Training

1. **Remap checkpoints if needed**
   ```bash
   python3 scripts/remap_checkpoint.py checkpoint.pth
   ```

2. **Verify GPU visibility**
   ```bash
   # In PCDiff terminal
   echo $CUDA_VISIBLE_DEVICES  # Should show 0,1,2,3,4,5,6
   
   # In Voxelization terminal
   echo $CUDA_VISIBLE_DEVICES  # Should show 7
   ```

3. **Check configuration**
   ```bash
   bash scripts/setup_verify.sh
   ```

### During Training

1. **Monitor GPU usage**
   ```bash
   watch -n 1 nvidia-smi
   ```
   
   Expect:
   - GPUs 0-6: PCDiff processes (60-80% util)
   - GPU 7: Voxelization process (60-80% util)

2. **Watch for errors**
   ```bash
   tail -f pcdiff/output/*/train.log
   tail -f voxelization/out/*/train.log
   ```

3. **Track loss curves**
   - PCDiff: Should be similar to previous training
   - Voxelization: Should decrease steadily

### Troubleshooting Checklist

**Checkpoint won't load:**
- [ ] Remap checkpoint to CPU
- [ ] Verify checkpoint path exists
- [ ] Check file isn't corrupted

**GPU device errors:**
- [ ] Set CUDA_VISIBLE_DEVICES explicitly in script
- [ ] Update config to use logical device 0
- [ ] Verify no other processes using GPUs

**Training unstable:**
- [ ] Reduce learning rate by 30-50%
- [ ] Check batch size isn't too large
- [ ] Verify data loading works correctly
- [ ] Monitor GPU memory usage

**Slow training:**
- [ ] Increase learning rate (if stable)
- [ ] Check GPU utilization is high
- [ ] Verify not CPU-bound (data loading)
- [ ] Consider increasing batch size

## References

### Research Papers

- Goyal et al. 2017: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
- You et al. 2017: "Large Batch Training of Convolutional Networks"  
- Smith et al. 2018: "Don't Decay the Learning Rate, Increase the Batch Size"

### PyTorch Documentation

- [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Saving & Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

---

For practical usage, see [training-guide.md](training-guide.md)

