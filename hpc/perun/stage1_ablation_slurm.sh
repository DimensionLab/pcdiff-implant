#!/bin/bash
#SBATCH --job-name=s1-ablate
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/logs/ablation_%a_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/logs/ablation_%a_%j.err
#SBATCH --array=0-14
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

set -eo pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

# 15 configurations: method-steps-ensemble
CONFIGS=(
  "ddpm 1000 1"
  "ddpm 1000 3"
  "ddpm 1000 5"
  "ddim 250 1"
  "ddim 250 3"
  "ddim 250 5"
  "ddim 100 1"
  "ddim 100 3"
  "ddim 100 5"
  "ddim 50 1"
  "ddim 50 3"
  "ddim 50 5"
  "ddim 25 1"
  "ddim 25 3"
  "ddim 25 5"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r METHOD STEPS ENS <<< "$CONFIG"
RUN_ID="${METHOD}-steps${STEPS}-ens${ENS}"

echo "=== Stage-1 Ablation: $RUN_ID ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: method=$METHOD steps=$STEPS ensemble=$ENS"
echo "Start: $(date)"
echo ""

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff
# CUDA toolkit setup — use nvidia pip package headers (CUDA 12.4) instead of
# conda's mixed 13.2 headers to avoid driver version mismatch
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="9.0"

# Build include path from nvidia pip packages (all CUDA 12.4 compatible)
NVIDIA_BASE=$(python -c "import nvidia; import os; print(os.path.dirname(nvidia.__file__))")
INCLUDE_DIRS=""
for pkg in cuda_runtime cublas cusparse cusolver cufft curand nvtx cuda_nvrtc; do
  d="$NVIDIA_BASE/${pkg}/include"
  [ -d "$d" ] && INCLUDE_DIRS="$d:$INCLUDE_DIRS"
done
export CPLUS_INCLUDE_PATH="$INCLUDE_DIRS${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$INCLUDE_DIRS${C_INCLUDE_PATH:-}" 

cd /mnt/data/home/mamuke588/pcdiff-implant

# Build CUDA extensions if needed
if [ -d "pcdiff/modules/functional" ]; then
    cd pcdiff/modules/functional
    python setup.py build_ext --inplace 2>/dev/null || true
    cd /mnt/data/home/mamuke588/pcdiff-implant
fi

nvidia-smi

MODEL="/mnt/data/home/mamuke588/pcdiff-implant/pcdiff/checkpoints/pcdiff_model_best.pth"
DATASET_CSV="/mnt/data/home/mamuke588/pcdiff-implant/pcdiff/datasets/SkullBreak/test.csv"
EVAL_PATH="/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/SkullBreak/${RUN_ID}/stage1"

mkdir -p "$EVAL_PATH"

echo ""
echo "Model: $MODEL"
echo "Dataset CSV: $DATASET_CSV"
echo "Output: $EVAL_PATH"
echo ""

START_TIME=$(date +%s)

python -u pcdiff/test_completion.py \
  --path "$DATASET_CSV" \
  --dataset SkullBreak \
  --model "$MODEL" \
  --eval_path "$EVAL_PATH" \
  --sampling_method "$METHOD" \
  --sampling_steps "$STEPS" \
  --num_ens "$ENS" \
  --gpu 0 \
  --workers 16 \
  --schedule_type linear

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=== Ablation $RUN_ID Complete ==="
echo "Elapsed: ${ELAPSED}s"
echo "End: $(date)"

# Write timing metadata
python3 -c "
import json
meta = {'run_id': '$RUN_ID', 'method': '$METHOD', 'steps': $STEPS, 'ensemble': $ENS, 'elapsed_seconds': $ELAPSED, 'node': '$SLURM_NODELIST'}
with open('$EVAL_PATH/timing.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
"
