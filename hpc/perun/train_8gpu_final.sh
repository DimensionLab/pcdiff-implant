#!/bin/bash
#SBATCH --job-name=pcdiff-8gpu-final
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/8gpu_final/train_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/8gpu_final/train_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

set -eo pipefail
echo "=== PCDiff 8x H200 Final Training ==="
echo "Job: $SLURM_JOB_ID, Node: $SLURM_NODELIST, Start: $(date)"

# Environment setup
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="9.0"

# CUDA includes for JIT compilation
NVIDIA_BASE=$(python -c "import nvidia; import os; print(os.path.dirname(nvidia.__file__))")
INCLUDE_DIRS=""
for pkg in cuda_runtime cublas cusparse cusolver cufft curand nvtx cuda_nvrtc; do
  d="$NVIDIA_BASE/${pkg}/include"
  [ -d "$d" ] && INCLUDE_DIRS="$d:$INCLUDE_DIRS"
done
export CPLUS_INCLUDE_PATH="$INCLUDE_DIRS${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$INCLUDE_DIRS${C_INCLUDE_PATH:-}"

# NCCL configuration for single-node multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

cd /mnt/data/home/mamuke588/pcdiff-implant
export PYTHONPATH=/mnt/data/home/mamuke588/pcdiff-implant/pcdiff:/mnt/data/home/mamuke588/pcdiff-implant:$PYTHONPATH

# Create output directory
RESULTS_DIR=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/8gpu_final
mkdir -p "$RESULTS_DIR"

nvidia-smi

# 8x H200 training with V14 winning config
# Global batch = 64 (8 per GPU * 8 GPUs)
# LR: sqrt scaling = 2e-4 * sqrt(8) ≈ 5.66e-4
# Cosine LR schedule with 50-epoch warmup
# CosineAnnealingLR T_max = 5000 (full training)
torchrun --nproc_per_node=8 \
    pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 \
    --lr 5.66e-4 \
    --niter 5000 \
    --schedule_type cosine \
    --loss_type mse_minsnr \
    --min_snr_gamma 5.0 \
    --embed_dim 96 \
    --width_mult 1.5 \
    --dropout 0.1 \
    --ema_decay 0.9999 \
    --grad_clip 1.0 \
    --lr_scheduler cosine \
    --cosine_T_max 5000 \
    --cosine_eta_min 1e-6 \
    --lr_warmup_epochs 50 \
    --lr_warmup_start_factor 0.0177 \
    --workers 16 \
    --save_dir "$RESULTS_DIR/checkpoints" \
    --save_interval 100 \
    --val_interval 50 2>&1

echo "=== Training Complete at $(date) ==="
