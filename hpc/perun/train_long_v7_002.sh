#!/bin/bash
#SBATCH --job-name=long-train
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/long_v7_002/train_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/long_v7_002/train_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

set -eo pipefail

echo "=== Long Training: v7_002 best config ==="
echo "Config: BS=2, no warmup, 1.5x wider, embed96, sigmoid, beta1=0.9, AMP bf16"
echo "Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST, Start: $(date)"

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
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

cd /mnt/data/home/mamuke588/pcdiff-implant
export PYTHONPATH=/mnt/data/home/mamuke588/pcdiff-implant/pcdiff:/mnt/data/home/mamuke588/pcdiff-implant/autoresearch:/mnt/data/home/mamuke588/pcdiff-implant:$PYTHONPATH
nvidia-smi

python -u autoresearch/results/perun/long_v7_002/train_pcdiff_modified.py \
  --time-budget 259200 \
  2>&1

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
