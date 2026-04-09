#!/bin/bash
#SBATCH --job-name=ddpm-ens-swp
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=autoresearch/results/perun/8gpu_final/ddpm_ens_sweep_%j.out
#SBATCH --error=autoresearch/results/perun/8gpu_final/ddpm_ens_sweep_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

# DDPM-1000 comparison + ensemble sweep (DIM-6)
# 1. DDPM-1000 ens=1 (for DDPM vs DDIM comparison)
# 2. DDIM-50 ens=3 and ens=5 (best step count from sweep, ensemble variants)

set -eo pipefail
echo "=== DDPM + Ensemble Sweep (DIM-6) ==="
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
nvidia-smi

CHECKPOINT="pcdiff/runs/SkullBreak/20260407_181433/checkpoints/model_best.pth"
RESULTS_BASE="autoresearch/results/perun/8gpu_final/ddpm_ens_sweep_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_BASE"

export PYTHONPATH=/mnt/data/home/mamuke588/pcdiff-implant/pcdiff:/mnt/data/home/mamuke588/pcdiff-implant:/mnt/data/home/mamuke588/pcdiff-implant/voxelization:$PYTHONPATH

# --- Run 1: DDPM-1000, ens=1 ---
echo ""
echo "=============================="
echo "=== DDPM-1000, ens=1 ==="
echo "=============================="

DDPM_DIR="$RESULTS_BASE/ddpm_1000_ens1"
mkdir -p "$DDPM_DIR"

python pcdiff/test_completion_distributed.py \
    --model "$CHECKPOINT" \
    --dataset SkullBreak \
    --path pcdiff/datasets/SkullBreak/test.csv \
    --eval_path "$DDPM_DIR" \
    --sampling_method ddpm \
    --sampling_steps 1000 \
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
    2>&1 | tee "$DDPM_DIR/generation.log"

echo "DDPM-1000 ens1 complete"

# --- Run 2: DDIM-50, ens=3 ---
echo ""
echo "=============================="
echo "=== DDIM-50, ens=3 ==="
echo "=============================="

ENS3_DIR="$RESULTS_BASE/ddim_50_ens3"
mkdir -p "$ENS3_DIR"

python pcdiff/test_completion_distributed.py \
    --model "$CHECKPOINT" \
    --dataset SkullBreak \
    --path pcdiff/datasets/SkullBreak/test.csv \
    --eval_path "$ENS3_DIR" \
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
    2>&1 | tee "$ENS3_DIR/generation.log"

echo "DDIM-50 ens3 complete"

# --- Run 3: DDIM-50, ens=5 ---
echo ""
echo "=============================="
echo "=== DDIM-50, ens=5 ==="
echo "=============================="

ENS5_DIR="$RESULTS_BASE/ddim_50_ens5"
mkdir -p "$ENS5_DIR"

python pcdiff/test_completion_distributed.py \
    --model "$CHECKPOINT" \
    --dataset SkullBreak \
    --path pcdiff/datasets/SkullBreak/test.csv \
    --eval_path "$ENS5_DIR" \
    --sampling_method ddim \
    --sampling_steps 50 \
    --num_ens 5 \
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
    2>&1 | tee "$ENS5_DIR/generation.log"

echo "DDIM-50 ens5 complete"

echo ""
echo "=== DDPM + Ensemble Sweep Complete at $(date) ==="
echo "Results: $RESULTS_BASE"

# Summary
for d in "$RESULTS_BASE"/*/; do
    name=$(basename "$d")
    if [ -f "$d/inference_summary.json" ]; then
        python3 -c "
import json
with open('$d/inference_summary.json') as f:
    s = json.load(f)
wall = s.get('wall_clock_time_seconds', 0)
n = s.get('total_processed', 0)
print(f'$name: {n} samples, {wall:.1f}s total, {wall/n:.2f}s/sample')
" 2>/dev/null || echo "$name: summary parse error"
    else
        echo "$name: no inference summary"
    fi
done
