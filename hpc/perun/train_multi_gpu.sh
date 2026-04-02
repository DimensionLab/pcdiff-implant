#!/bin/bash
#SBATCH --job-name=pcdiff-ddp
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/mamuke588/pcdiff/logs/ddp_%j.out
#SBATCH --error=/scratch/mamuke588/pcdiff/logs/ddp_%j.err
#
# Multi-GPU (8× H200) DDP PCDiff training on PERUN
# Uses all 8 H200 GPUs on a single node with NVLink interconnect.
#
# Usage:
#   sbatch hpc/perun/train_multi_gpu.sh
#   sbatch --nodes=2 hpc/perun/train_multi_gpu.sh  # 16 GPUs across 2 nodes

set -euo pipefail

echo "=== PCDiff Multi-GPU DDP Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE:-8}"
echo "Total tasks: $SLURM_NTASKS"
echo "Start: $(date)"
echo ""

# Environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff

# Wandb config
export WANDB_PROJECT="pcdiff-implant-perun"
export WANDB_RUN_GROUP="multi-gpu-ddp"
export WANDB_TAGS="perun,h200,ddp,8gpu"
export WANDB_NOTES="SLURM_JOB_ID=$SLURM_JOB_ID nodes=$SLURM_NODELIST ngpu=$SLURM_NTASKS"

# DDP environment (set by Slurm)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# NCCL optimizations for H200 + NVLink + InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=WARN

# Scratch paths
SCRATCH="/scratch/mamuke588/pcdiff"
CHECKPOINT_DIR="$SCRATCH/checkpoints/$SLURM_JOB_ID"
mkdir -p "$CHECKPOINT_DIR"

cd ~/pcdiff-implant

# Build CUDA extensions
if [ -d "pcdiff/modules/functional" ]; then
    cd pcdiff/modules/functional
    python setup.py build_ext --inplace 2>/dev/null || true
    cd ~/pcdiff-implant
fi

nvidia-smi

# Launch DDP training with torchrun
# torchrun handles LOCAL_RANK assignment per GPU
srun torchrun \
    --nproc_per_node=8 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    autoresearch/train_pcdiff.py \
    --time-budget 82800 \
    2>&1 | tee "$SCRATCH/logs/ddp_${SLURM_JOB_ID}_console.log"

# Copy results
cp -r autoresearch/results/* "$SCRATCH/results/" 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "End: $(date)"
