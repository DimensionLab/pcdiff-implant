#!/bin/bash
#SBATCH --job-name=ddim-sweep
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=autoresearch/results/perun/8gpu_final/ddim_sweep_%j.out
#SBATCH --error=autoresearch/results/perun/8gpu_final/ddim_sweep_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

set -eo pipefail

echo "=== DDIM Step Sweep for Inference Speedup (DIM-50) ==="
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
RESULTS_BASE="autoresearch/results/perun/8gpu_final/ddim_sweep_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_BASE"

# DDIM step sweep: test 5, 10, 15, 20, 25, 35, 50
STEPS=(5 10 15 20 25 35 50)
ENSEMBLE=1  # Single ensemble for speed comparison (quality baseline is DDIM-50)

echo "steps,time_per_sample_sec,total_samples" > "$RESULTS_BASE/summary.csv"

for N_STEPS in "${STEPS[@]}"; do
    echo ""
    echo "=============================="
    echo "=== DDIM-${N_STEPS} ==="
    echo "=============================="

    OUTPUT_DIR="$RESULTS_BASE/ddim_${N_STEPS}"
    mkdir -p "$OUTPUT_DIR"

    START_TIME=$(date +%s)

    python pcdiff/test_completion_distributed.py \
        --model "$CHECKPOINT" \
        --dataset SkullBreak \
        --path pcdiff/datasets/SkullBreak/test.csv \
        --eval_path "$OUTPUT_DIR" \
        --sampling_method ddim \
        --sampling_steps $N_STEPS \
        --num_ens $ENSEMBLE \
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
        2>&1 | tee "$OUTPUT_DIR/generation.log"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    # Count generated samples
    NUM_SAMPLES=$(ls "$OUTPUT_DIR"/*.ply 2>/dev/null | wc -l || echo 0)
    TIME_PER_SAMPLE=0
    if [ "$NUM_SAMPLES" -gt 0 ]; then
        TIME_PER_SAMPLE=$(echo "scale=2; $ELAPSED / $NUM_SAMPLES" | bc)
    fi

    echo "DDIM-${N_STEPS}: ${ELAPSED}s total, ${NUM_SAMPLES} samples, ${TIME_PER_SAMPLE}s/sample" | tee -a "$RESULTS_BASE/summary.log"
    echo "${N_STEPS},${TIME_PER_SAMPLE},${NUM_SAMPLES}" >> "$RESULTS_BASE/summary.csv"
done

echo ""
echo "=== DDIM Step Sweep Complete ==="
echo "Results in: $RESULTS_BASE"
cat "$RESULTS_BASE/summary.csv"
