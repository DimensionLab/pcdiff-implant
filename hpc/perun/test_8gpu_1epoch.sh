#!/bin/bash
#SBATCH --job-name=test-8gpu-1ep
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=0-01:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/8gpu_test/test_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/8gpu_test/test_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

set -eo pipefail
echo "=== PCDiff 8x H200 DDP Test (1 epoch) ==="
echo "Job: $SLURM_JOB_ID, Node: $SLURM_NODELIST, Start: $(date)"

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="9.0"

NVIDIA_BASE=$(python -c "import nvidia; import os; print(os.path.dirname(nvidia.__file__))")
INCLUDE_DIRS=""
for pkg in cuda_runtime cublas cusparse cusolver cufft curand nvtx cuda_nvrtc; do
  d="$NVIDIA_BASE/${pkg}/include"
  [ -d "$d" ] && INCLUDE_DIRS="$d:$INCLUDE_DIRS"
done
export CPLUS_INCLUDE_PATH="$INCLUDE_DIRS${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$INCLUDE_DIRS${C_INCLUDE_PATH:-}"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

cd /mnt/data/home/mamuke588/pcdiff-implant
export PYTHONPATH=/mnt/data/home/mamuke588/pcdiff-implant/pcdiff:/mnt/data/home/mamuke588/pcdiff-implant:$PYTHONPATH

RESULTS_DIR=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/8gpu_test
mkdir -p "$RESULTS_DIR"

nvidia-smi

# Quick 2-epoch test to verify DDP + all V14 features work
torchrun --nproc_per_node=8 \
    pcdiff/train_completion.py \
    --path pcdiff/datasets/SkullBreak/train.csv \
    --dataset SkullBreak \
    --bs 64 \
    --lr 5.66e-4 \
    --niter 2 \
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
    --lr_warmup_epochs 1 \
    --lr_warmup_start_factor 0.01 \
    --workers 4 \
    --save_dir "$RESULTS_DIR/checkpoints" 2>&1

echo "=== Test Complete at $(date) ==="
