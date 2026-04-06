#!/bin/bash
#SBATCH --job-name=pcdiff-6gpu-15k
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:6
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/8gpu_final/train_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/8gpu_final/train_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174
#SBATCH --requeue

echo "=== PCDiff 6x H200 Extended Training (15k epochs, 3 days) ==="
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
export OMP_NUM_THREADS=8

cd /mnt/data/home/mamuke588/pcdiff-implant
export PYTHONPATH=/mnt/data/home/mamuke588/pcdiff-implant/pcdiff:/mnt/data/home/mamuke588/pcdiff-implant:$PYTHONPATH

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"
nvidia-smi

# Resume from latest checkpoint of 8-GPU run
RESUME_CKPT="/mnt/data/home/mamuke588/pcdiff-implant/pcdiff/runs/SkullBreak/20260406_093136/checkpoints/model_latest.pth"
if [ -f "$RESUME_CKPT" ]; then
    echo "Resuming from checkpoint: $RESUME_CKPT"
    RESUME_ARG="--model $RESUME_CKPT"
else
    echo "WARNING: No checkpoint found, starting fresh"
    RESUME_ARG=""
fi

# 6 GPUs x 8 per GPU = 48 effective batch (close to 64 target)
# LR scaled: base 5.66e-4 for batch 64, scale to 48 -> 4.245e-4
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500     pcdiff/train_completion.py     --path pcdiff/datasets/SkullBreak/train.csv     --dataset SkullBreak     --bs 48     --lr 5.66e-4     --lr-base-batch 48     --niter 15000     --gating-max-epochs 15000     --schedule_type cosine     --loss_type mse_minsnr     --min_snr_gamma 5.0     --embed_dim 96     --width_mult 1.5     --attention True     --dropout 0.1     --vox_res_mult 1.0     --ema_decay 0.9999     --beta1 0.5     --grad_clip 1.0     --checkpoint_freq 100     --keep_last_n 5     --lr_scheduler cosine     --cosine_T_max 15000     --cosine_eta_min 1e-6     --lr-warmup-epochs 50     --lr-warmup-start-factor 0.0177     --workers 12     --prefetch-factor 4     --no-wandb     --disable-compile     --gating-enabled True     --gating-decision-epochs 500,1000,2000,5000,10000,15000     --proxy-eval-enabled True     --proxy-eval-subset pcdiff/proxy_validation_subset.json     --proxy-eval-vox-config voxelization/configs/gen_skullbreak.yaml     --proxy-eval-vox-checkpoint voxelization/checkpoints/model_best.pt     --gating-proxy-eval-freq 200 $RESUME_ARG

echo "=== Training Complete ==="
