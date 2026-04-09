#!/bin/bash
#SBATCH --job-name=cons-distill
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=autoresearch/results/perun/8gpu_final/consistency_distill_%j.out
#SBATCH --error=autoresearch/results/perun/8gpu_final/consistency_distill_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

set -eo pipefail

echo "=== Consistency Distillation Training (DIM-50) ==="
echo "Job: $SLURM_JOB_ID, Node: $SLURM_NODELIST, Start: $(date)"

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="9.0"

NVIDIA_BASE=$(python -c "import nvidia; import os; print(os.path.dirname(nvidia.__file__))")
INCLUDE_DIRS=""
for pkg in cuda_runtime cublas cusparse cusolver cufft curand nvtx cuda_nvrtc; do
  d="$NVIDIA_BASE/$pkg/include"
  [ -d "$d" ] && INCLUDE_DIRS="$d:$INCLUDE_DIRS"
done
export CPLUS_INCLUDE_PATH="$INCLUDE_DIRS${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$INCLUDE_DIRS${C_INCLUDE_PATH:-}"

cd /mnt/data/home/mamuke588/pcdiff-implant
export PYTHONPATH=/mnt/data/home/mamuke588/pcdiff-implant/pcdiff:/mnt/data/home/mamuke588/pcdiff-implant:/mnt/data/home/mamuke588/pcdiff-implant/voxelization:$PYTHONPATH

nvidia-smi

# Run consistency distillation training
python autoresearch/consistency_distillation.py \
    --teacher_checkpoint pcdiff/runs/SkullBreak/20260407_181433/checkpoints/model_best.pth \
    --dataset SkullBreak \
    --data_path pcdiff/datasets/SkullBreak/train.csv \
    --output_dir autoresearch/results/perun/8gpu_final/consistency_distill_${SLURM_JOB_ID} \
    --schedule_type cosine \
    --embed_dim 96 \
    --width_mult 1.5 \
    --attention True \
    --dropout 0.1 \
    --loss_type mse_minsnr \
    --num_points 30720 \
    --num_nn 3072 \
    --nc 3 \
    --target_steps 4 \
    --n_epochs 5000 \
    --lr 1e-4 \
    --batch_size 16 \
    --eval_every 500 \
    --eval_path pcdiff/datasets/SkullBreak/test.csv \
    --workers 8 \
    --seed 42 \
    2>&1 | tee autoresearch/results/perun/8gpu_final/consistency_distill_${SLURM_JOB_ID}/training.log

echo "=== Consistency Distillation Complete at $(date) ==="
