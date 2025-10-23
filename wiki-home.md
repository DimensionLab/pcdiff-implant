# PCDiff-Implant Wiki

Welcome to the Point Cloud Diffusion Models for Automatic Implant Generation wiki!

## 📚 Documentation Index

### Getting Started
- [Installation Guide](../INSTALL.md)
- [Setup Guide](../SETUP.md)
- [Quick Start](../QUICKSTART.md)

### Training
- [Point Cloud Diffusion Model](pcdiff/README.md)
  - [Single GPU Training](pcdiff/README.md#single-gpu-training)
  - [Multi-GPU Training](pcdiff/README.md#multi-gpu-training-distributed)
  - [Background Training Sessions](pcdiff/README.md#background-training-persistent-sessions)
- [Distributed Training Guide](pcdiff/distributed-training.md) ⭐ **Detailed Guide**
  - [Quick Reference](pcdiff/distributed-training.md#quick-reference)
  - [Batch Size Distribution](pcdiff/distributed-training.md#understanding-batch-size-distribution)
  - [Learning Rate Scaling](pcdiff/distributed-training.md#learning-rate-scaling)
  - [Best Practices](pcdiff/distributed-training.md#best-practices)
  - [Troubleshooting](pcdiff/distributed-training.md#troubleshooting)
  - [Multi-Node Training](pcdiff/distributed-training.md#multi-node-training)
- [Voxelization Network](voxelization/README.md)

### Data
- [Dataset Information](README.md#data)
- [Data Preprocessing](pcdiff/README.md#data-preprocessing)
- [Train/Test Split](pcdiff/README.md#traintest-split)

### Deployment & Usage
- [Using Trained Models](pcdiff/README.md#use-the-model)
- [Evaluation Metrics](README.md#results)

### Advanced Topics
- [Multi-GPU Configuration](pcdiff/distributed-training.md#quick-reference)
- [Learning Rate Scaling Strategies](pcdiff/distributed-training.md#learning-rate-scaling)
- [Performance Optimization](pcdiff/distributed-training.md#performance-expectations)
- [Persistent Training Sessions (tmux/screen)](pcdiff/distributed-training.md#5-persistent-training-sessions)

## 🚀 Quick Links

### Common Training Commands

**Single GPU:**
```bash
python pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 8
```

**8x GPU (Recommended):**
```bash
torchrun --nproc_per_node=8 pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 \
    --lr 1.6e-3
```

**Background Training (tmux):**
```bash
tmux new -s training
torchrun --nproc_per_node=8 pcdiff/train_completion.py [your args]
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

## 🐛 Troubleshooting

Common issues and solutions:

| Issue | Solution Link |
|-------|---------------|
| FileExistsError on training | Fixed in latest version |
| Loss too noisy | [Batch size guidance](pcdiff/distributed-training.md#issue-loss-is-too-noisy) |
| Loss diverges | [Learning rate tuning](pcdiff/distributed-training.md#issue-loss-diverges-or-doesnt-decrease) |
| Out of memory | [Memory optimization](pcdiff/distributed-training.md#issue-out-of-memory-oom) |
| SSH disconnection | [Persistent sessions](pcdiff/distributed-training.md#5-persistent-training-sessions) |
| NCCL errors | [Communication issues](pcdiff/distributed-training.md#issue-nccl-error-or-communication-timeout) |

## 📊 Performance Guidelines

| Setup | Global Batch | Per-GPU Batch | Learning Rate | Speedup |
|-------|--------------|---------------|---------------|---------|
| 1× A100 | 8 | 8 | 2e-4 | 1.0× |
| 2× GPU | 16 | 8 | 4e-4 | ~2.0× |
| 4× GPU | 32 | 8 | 8e-4 | ~4.0× |
| 8× H100 | 64 | 8 | 1.6e-3 | ~8.0× |

See [complete performance table](pcdiff/distributed-training.md#performance-expectations) for details.

## 📖 Paper & Citation

This implementation is based on the MICCAI 2023 paper:

**[Point Cloud Diffusion Models for Automatic Implant Generation](https://pfriedri.github.io/pcdiff-implant-io/)**

If you use this code, please cite:
```bibtex
@InProceedings{10.1007/978-3-031-43996-4_11,
    author="Friedrich, Paul and Wolleb, Julia and Bieder, Florentin 
            and Thieringer, Florian M. and Cattin, Philippe C.",
    title="Point Cloud Diffusion Models for Automatic Implant Generation",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
    year="2023",
    pages="112--122",
}
```

## 🔗 External Resources

- [Project Homepage](https://pfriedri.github.io/pcdiff-implant-io/)
- [arXiv Paper](https://arxiv.org/abs/2303.08061)
- [SkullBreak Dataset](https://www.fit.vutbr.cz/~ikodym/skullbreak_training.zip)
- [SkullFix Dataset](https://files.icg.tugraz.at/f/2c5f458e781a42c6a916/?dl=1)

## 💡 Contributing

Issues and pull requests are welcome! Please see the main [README](README.md) for more information.

