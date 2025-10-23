# Installation Guide (Python 3.10 + PyTorch 2.5)

## Quick Install

```bash
# 1. Install Python 3.10 via uv (if not already available)
uv python install 3.10

# 2. Create environment
uv venv --python 3.10
source .venv/bin/activate

# 3. Install PyTorch 2.5.0 with CUDA 12.4 (forward compatible with CUDA 13)
uv pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install "torchvision==0.20.0" --index-url https://download.pytorch.org/whl/cu124

# 4. Install project dependencies
uv pip install -e .

# 5. Install PyTorch3D (builds from source, ~5-10 min)
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# 6. Install PyTorch Scatter (pre-built wheel)
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

**Total time: ~8-12 minutes** (PyTorch3D compilation ~5-10 min, torch-scatter is pre-built wheel)

## Or use the automated script

```bash
./setup_uv.sh
```

## Multi-GPU Training (8x H100)

```bash
# Point cloud diffusion model
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64

# Voxelization model
torchrun --nproc_per_node=8 voxelization/train.py \
    voxelization/configs/train_skullbreak.yaml
```

## Why This Stack?

- ✅ **Python 3.10**: Managed by uv (no system package conflicts)
- ✅ **PyTorch 2.5.0**: Stable release with CUDA 13 compatibility
- ✅ **CUDA 12.4 binaries**: Forward-compatible with CUDA 13 drivers, pre-built torch-scatter wheels
- ✅ **All multi-GPU features**: torchrun DDP works perfectly

**Install time**: 
- Python 3.10 download: ~1 min (one-time)
- PyTorch3D: ~5-10 min (builds from source)
- torch-scatter: instant (pre-built wheel)

## Next Steps

1. Download SkullBreak dataset → `datasets/SkullBreak/`
2. Preprocess: `python3 pcdiff/utils/preproc_skullbreak.py`
3. Split: `python3 pcdiff/utils/split_skullbreak.py`  
4. Train: See commands above

