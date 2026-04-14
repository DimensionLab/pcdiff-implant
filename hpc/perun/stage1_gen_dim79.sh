#!/bin/bash
#SBATCH --job-name=s1-dim79
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

# Stage-1 generation for DIM-79: ddim_250_ens1 and ddim_50_ens3
# These configs don't have existing stage-1 outputs

set -euo pipefail
source .activate_scratch

echo "=== Stage-1 Generation for DIM-79 (ddim_250_ens1 + ddim_50_ens3) ==="
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

CHECKPOINT="pcdiff/runs/SkullBreak/20260407_181433/checkpoints/model_best.pth"
RESULTS_BASE="autoresearch/results/perun/8gpu_final/s1_dim79_${SLURM_JOB_ID}"

echo ""
echo "=============================="
echo "=== Config 1: DDIM-250 ens1 ==="
echo "=============================="

OUT_DIR_250="${RESULTS_BASE}/ddim_250_ens1"
mkdir -p "$OUT_DIR_250"

python pcdiff/test_completion_distributed.py \
    --model "$CHECKPOINT" \
    --dataset SkullBreak \
    --path pcdiff/datasets/SkullBreak/test.csv \
    --eval_path "$OUT_DIR_250" \
    --sampling_method ddim \
    --sampling_steps 250 \
    --num_ens 1 \
    --schedule_type cosine \
    --embed_dim 96 \
    --width_mult 1.5 \
    --attention True \
    --dropout 0.1 \
    --loss_type mse_minsnr \
    --num_points 30720 \
    --num_nn 3072 \
    --nc 3 \
    --workers 4 \
    --verbose \
    2>&1 | tee "$OUT_DIR_250/generation.log"

NUM_250=$(ls "$OUT_DIR_250/syn/" 2>/dev/null | wc -l || echo 0)
echo "DDIM-250-ens1: $NUM_250 samples generated"

echo ""
echo "=============================="
echo "=== Config 2: DDIM-50 ens3 ==="
echo "=============================="

OUT_DIR_50E3="${RESULTS_BASE}/ddim_50_ens3"
mkdir -p "$OUT_DIR_50E3"

python pcdiff/test_completion_distributed.py \
    --model "$CHECKPOINT" \
    --dataset SkullBreak \
    --path pcdiff/datasets/SkullBreak/test.csv \
    --eval_path "$OUT_DIR_50E3" \
    --sampling_method ddim \
    --sampling_steps 50 \
    --num_ens 3 \
    --schedule_type cosine \
    --embed_dim 96 \
    --width_mult 1.5 \
    --attention True \
    --dropout 0.1 \
    --loss_type mse_minsnr \
    --num_points 30720 \
    --num_nn 3072 \
    --nc 3 \
    --workers 4 \
    --verbose \
    2>&1 | tee "$OUT_DIR_50E3/generation.log"

NUM_50E3=$(ls "$OUT_DIR_50E3/syn/" 2>/dev/null | wc -l || echo 0)
echo "DDIM-50-ens3: $NUM_50E3 samples generated"

echo ""
echo "=== Stage-1 Generation Complete at $(date) ==="
echo "DDIM-250-ens1: $NUM_250 samples in $OUT_DIR_250/syn/"
echo "DDIM-50-ens3: $NUM_50E3 samples in $OUT_DIR_50E3/syn/"
echo ""
echo "NOTE: Stage-2 voxelization still needed for these 2 configs after generation completes."
