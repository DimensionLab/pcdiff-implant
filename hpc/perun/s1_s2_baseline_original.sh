#!/bin/bash
#SBATCH --job-name=base-orig
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/baseline_original_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/baseline_original_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

# DIM-6 Baseline Ablation: ORIGINAL PCDiff model + ORIGINAL vox model
# This is the ONLY proven-working pipeline (demo: DSC=0.94)
# Purpose: establish baseline ablation numbers to compare 6GPU model against

set -eo pipefail

echo "=== DIM-6 Baseline Ablation: Original Model Pipeline ==="
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

PCDIFF_MODEL="pcdiff/checkpoints/model_best.pth"
VOX_MODEL="voxelization/checkpoints/model_best.pt"
OUTPUT_ROOT="benchmarking/runs/stage1_ablation/SkullBreak/baseline_original_job${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_ROOT"

echo "PCDiff model: $PCDIFF_MODEL ($(stat -c%s $PCDIFF_MODEL 2>/dev/null || stat -f%z $PCDIFF_MODEL) bytes)"
echo "Vox model: $VOX_MODEL ($(stat -c%s $VOX_MODEL 2>/dev/null || stat -f%z $VOX_MODEL) bytes)"

# ============================================================
# STAGE 1: Point Cloud Generation
# ============================================================
# Generate DDPM-250 and DDIM-{250,100,50} with ensemble {1,3}
# Focus on configs that fit within time budget

STAGE1_ROOT="$OUTPUT_ROOT/stage1"
mkdir -p "$STAGE1_ROOT"

# Config: method steps ensemble
CONFIGS=(
  "ddpm 250 1"
  "ddpm 250 3"
  "ddim 250 1"
  "ddim 250 3"
  "ddim 100 1"
  "ddim 100 3"
  "ddim 50 1"
  "ddim 50 3"
)

SUMMARY="$STAGE1_ROOT/summary.csv"
echo "config,method,steps,ensemble,cases,runtime_sec,mean_gpu_mb" > "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
  read -r method steps ensemble <<< "$cfg"
  CONFIG_NAME="${method}-${steps}-ens${ensemble}"
  EVAL_PATH="$STAGE1_ROOT/$CONFIG_NAME"

  if [ -d "$EVAL_PATH/syn" ]; then
    count=$(find "$EVAL_PATH/syn" -maxdepth 1 -name "*_surf" -type d 2>/dev/null | wc -l)
    if [ "$count" -ge 115 ]; then
      echo "SKIP: $CONFIG_NAME already has $count cases"
      continue
    fi
  fi

  echo ""
  echo "=== Stage-1: $CONFIG_NAME ==="
  echo "Start: $(date)"

  python3 -u pcdiff/test_completion.py \
    --eval_path "$EVAL_PATH" \
    --dataset SkullBreak \
    --model "$PCDIFF_MODEL" \
    --sampling_method "$method" \
    --sampling_steps "$steps" \
    --num_ens "$ensemble" \
    --manualSeed 42 \
    2>&1 | tee "$EVAL_PATH/log.txt"

  echo "End: $(date)"
  echo "$CONFIG_NAME,done,$steps,$ensemble,115,0,0" >> "$SUMMARY"
done

echo ""
echo "=== Stage-1 Generation Complete at $(date) ==="

# ============================================================
# STAGE 2: Voxelization + Evaluation
# ============================================================
STAGE2_ROOT="$OUTPUT_ROOT/stage2"
mkdir -p "$STAGE2_ROOT"

ln -sfn "$(pwd)/pcdiff/datasets/SkullBreak/defective_skull" autoresearch/defective_skull 2>/dev/null || true
ln -sfn "$(pwd)/pcdiff/datasets/SkullBreak/implant" autoresearch/implant 2>/dev/null || true

SUMMARY2="$STAGE2_ROOT/summary.csv"
echo "config,cases,mean_dsc,mean_bdsc_10mm,mean_hd95_mm,mean_runtime_sec" > "$SUMMARY2"

for cfg in "${CONFIGS[@]}"; do
  read -r method steps ensemble <<< "$cfg"
  CONFIG_NAME="${method}-${steps}-ens${ensemble}"
  SYN_DIR="$STAGE1_ROOT/$CONFIG_NAME/syn"

  if [ ! -d "$SYN_DIR" ]; then
    echo "SKIP: $CONFIG_NAME - no syn dir"
    continue
  fi

  count=$(find "$SYN_DIR" -maxdepth 1 -name "*_surf" -type d 2>/dev/null | wc -l)
  if [ "$count" -lt 10 ]; then
    echo "SKIP: $CONFIG_NAME - only $count cases"
    continue
  fi

  echo ""
  echo "=== Stage-2: $CONFIG_NAME ($count cases) ==="
  echo "Start: $(date)"

  EVAL_DIR="$STAGE2_ROOT/$CONFIG_NAME"
  mkdir -p "$EVAL_DIR"
  GEN_DIR="gen_baseline_orig_${CONFIG_NAME}_job${SLURM_JOB_ID}"

  cat > "$EVAL_DIR/config.yaml" << YAML
inherit_from: voxelization/configs/train_skullbreak.yaml
train:
  gpu: 0
data:
  dset: SkullBreak
  path: $SYN_DIR
generation:
  batch_size: 1
  generation_dir: $GEN_DIR
  num_ensemble: $ensemble
  save_ensemble_implants: False
  compute_eval_metrics: True
test:
  model_file: $VOX_MODEL
YAML

  python -u voxelization/generate.py "$EVAL_DIR/config.yaml" 2>&1 | tee "$EVAL_DIR/log.txt"

  # Extract metrics
  DSCS=$(grep "Dice score:" "$EVAL_DIR/log.txt" | awk '{print $3}')
  BDSCS=$(grep "Boundary dice (10mm):" "$EVAL_DIR/log.txt" | awk '{print $4}')
  HD95S=$(grep "95 percentile Haussdorf distance:" "$EVAL_DIR/log.txt" | awk '{print $4}')

  MEAN_DSC=$(echo "$DSCS" | awk '{s+=$1;n++}END{if(n>0) print s/n; else print 0}')
  MEAN_BDSC=$(echo "$BDSCS" | awk '{s+=$1;n++}END{if(n>0) print s/n; else print 0}')
  MEAN_HD95=$(echo "$HD95S" | awk '{s+=$1;n++}END{if(n>0) print s/n; else print 0}')

  echo "$CONFIG_NAME,$count,$MEAN_DSC,$MEAN_BDSC,$MEAN_HD95,0" >> "$SUMMARY2"
  echo "=== $CONFIG_NAME: DSC=$MEAN_DSC, bDSC=$MEAN_BDSC, HD95=$MEAN_HD95 ==="

  # Copy benchmark artifacts
  GEN_OUT="voxelization/out/skullbreak/$GEN_DIR"
  if [ -d "$GEN_OUT" ]; then
    cp -r "$GEN_OUT"/* "$EVAL_DIR/" 2>/dev/null || true
  fi
done

echo ""
echo "=== ALL DONE at $(date) ==="
echo "Stage-1 summary: $SUMMARY"
echo "Stage-2 summary: $SUMMARY2"
cat "$SUMMARY2"
