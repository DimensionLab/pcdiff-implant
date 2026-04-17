#!/bin/bash
#SBATCH --job-name=pareto-bench
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=benchmarking/runs/pareto_bench_%j.out
#SBATCH --error=benchmarking/runs/pareto_bench_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

set -eo pipefail

echo "=== DIM-96 Pareto Inference Benchmark ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi || true

source /mnt/data/home/mamuke588/miniconda3/etc/profile.d/conda.sh
conda activate pcdiff
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="9.0"

REPO=/mnt/data/home/mamuke588/pcdiff-implant
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/pcdiff:$REPO/voxelization:${PYTHONPATH:-}"

# ---- Checkpoint paths (override by exporting before sbatch) -------------------
PCDIFF_CKPT=${PCDIFF_CKPT:-pcdiff/checkpoints/model_best.pth}
VOX_CKPT=${VOX_CKPT:-voxelization/checkpoints/model_best.pt}
VOX_CFG=${VOX_CFG:-voxelization/configs/gen_skullbreak.yaml}
WV3_CKPT=${WV3_CKPT:-wodzinski_v3_nosym/model_best.pt}
RF_CKPT=${RF_CKPT:-pcdiff/runs/rectified_flow/model_best.pt}   # best-loss snapshot (epoch-3, loss=2.214 per DIM-96)

# ---- Knobs ---------------------------------------------------------------------
NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-42}
METHODS=${METHODS:-"pcdiff wv3 rf"}
OUT=${OUT:-benchmarking/runs/pareto_bench_${SLURM_JOB_ID:-$(date +%s)}}

mkdir -p "$OUT" benchmarking/runs

echo "PCDIFF_CKPT=$PCDIFF_CKPT"
echo "VOX_CKPT=$VOX_CKPT"
echo "WV3_CKPT=$WV3_CKPT"
echo "RF_CKPT=$RF_CKPT"
echo "OUT=$OUT"

python benchmarking/pareto_bench.py \
    --dataset-csv datasets/SkullBreak/test.csv \
    --output-dir "$OUT" \
    --num-cases "$NUM_CASES" \
    --seed "$SEED" \
    --device cuda:0 \
    --methods $METHODS \
    --pcdiff-checkpoint "$PCDIFF_CKPT" \
    --vox-checkpoint "$VOX_CKPT" \
    --vox-config "$VOX_CFG" \
    --wv3-checkpoint "$WV3_CKPT" \
    --rf-checkpoint "$RF_CKPT" \
    --pcdiff-weights raw \
    --wv3-weights raw \
    --rf-weights ema \
    2>&1 | tee "$OUT/pareto_bench.stdout.log"

echo ""
echo "=== Plotting ==="
python benchmarking/pareto_plot.py \
    --cases-csv "$OUT/pareto_cases.csv" \
    --output-dir "$OUT" \
    --quality dice

echo ""
echo "=== DONE: results in $OUT ==="
ls -la "$OUT"
