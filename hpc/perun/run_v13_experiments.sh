#!/bin/bash
#SBATCH --job-name=v13-exp
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/exp_v13_%a/train_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/exp_v13_%a/train_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174
#SBATCH --array=1-5

set -eo pipefail

TASK=$SLURM_ARRAY_TASK_ID
EXPERIMENTS=(
  ""  # placeholder for 0
  "001_cosine_muon"
  "002_cosine_ema_gradaccum"
  "003_cosine_warmrestarts"
  "004_minsnr_cosine_ema"
  "005_cosine_ema_muon"
)
EXP_NAME="${EXPERIMENTS[$TASK]}"

echo "=== V13 Experiment: $EXP_NAME ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $TASK, Node: $SLURM_NODELIST"
echo "Start: $(date)"

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
export PYTHONPATH=$PWD/pcdiff:$PWD/autoresearch:$PWD:$PYTHONPATH

RESULTS_DIR="autoresearch/results/perun/exp_v13_${EXP_NAME}"
mkdir -p "$RESULTS_DIR"

nvidia-smi

python -u "$RESULTS_DIR/train_pcdiff_modified.py" --time-budget 14400 2>&1

echo ""
echo "=== Experiment Complete ==="
echo "End: $(date)"
