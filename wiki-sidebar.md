## Navigation

**[🏠 Home](wiki-home.md)**

---

### Getting Started
- [Installation](../INSTALL.md)
- [Setup](../SETUP.md)
- [Quick Start](../QUICKSTART.md)

---

### Training Guides
- [Training Overview](pcdiff/README.md)
- **[Distributed Training](pcdiff/distributed-training.md)** ⭐
  - [Quick Reference](pcdiff/distributed-training.md#quick-reference)
  - [Batch Size Guide](pcdiff/distributed-training.md#understanding-batch-size-distribution)
  - [Learning Rate Scaling](pcdiff/distributed-training.md#learning-rate-scaling)
  - [Best Practices](pcdiff/distributed-training.md#best-practices)
  - [Troubleshooting](pcdiff/distributed-training.md#troubleshooting)

---

### Data
- [Datasets](README.md#data)
- [Preprocessing](pcdiff/README.md#data-preprocessing)
- [Split Data](pcdiff/README.md#traintest-split)

---

### Inference
- [Use Trained Models](pcdiff/README.md#use-the-model)
- [Voxelization](voxelization/README.md)

---

### Results
- [SkullBreak Results](README.md#results-on-the-skullbreak-dataset)
- [SkullFix Results](README.md#results-on-the-skullfix-dataset)
- [Performance Stats](README.md#runtime--gpu-memory-requirement-information)

---

### Quick Commands

**Single GPU:**
```bash
python pcdiff/train_completion.py \
  --path pcdiff/datasets/SkullBreak/train.csv \
  --dataset SkullBreak --bs 8
```

**8× GPU:**
```bash
torchrun --nproc_per_node=8 \
  pcdiff/train_completion.py \
  --path pcdiff/datasets/SkullBreak/train.csv \
  --dataset SkullBreak --bs 64 --lr 1.6e-3
```

**tmux Session:**
```bash
tmux new -s training
# Detach: Ctrl+B, then D
tmux attach -t training
```

---

### External Links
- [📄 Paper](https://arxiv.org/abs/2303.08061)
- [🌐 Project Page](https://pfriedri.github.io/pcdiff-implant-io/)
- [📊 SkullBreak Data](https://www.fit.vutbr.cz/~ikodym/skullbreak_training.zip)

