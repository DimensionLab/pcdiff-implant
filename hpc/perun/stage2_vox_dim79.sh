#!/bin/bash
#SBATCH --job-name=s2-dim79
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

# Stage-2 voxelization for DIM-79: 3 ready configs (25/50/100 step, ens=1)
# ddim_5 was excluded because ALL samples contain NaN (too few diffusion steps)

set -euo pipefail
source .activate_scratch

echo "=== Stage-2 Voxelization for DIM-79 (configs: 25/50/100 ens1) ==="
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

VOX_MODEL="voxelization/checkpoints/model_best.pt"
DATASET_ROOT="pcdiff/datasets/SkullBreak"
OUTPUT_BASE="autoresearch/results/perun/8gpu_final/stage2_dim79_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_BASE"

# Create symlinks for ground truth data
if [ ! -L "autoresearch/defective_skull" ]; then
    ln -sfn "$(pwd)/$DATASET_ROOT/defective_skull" autoresearch/defective_skull
fi
if [ ! -L "autoresearch/implant" ]; then
    ln -sfn "$(pwd)/$DATASET_ROOT/implant" autoresearch/implant
fi

echo "steps,samples,mean_dsc,mean_bdsc_10mm,mean_hd95_mm,wall_sec" > "$OUTPUT_BASE/summary.csv"

# Config 1: ddim-steps25-ens1
run_vox() {
    local CONFIG_NAME="$1"
    local SYN_SRC="$2"
    local N_STEPS="$3"

    echo ""
    echo "=============================="
    echo "=== Evaluating $CONFIG_NAME ==="
    echo "=============================="

    if [ ! -d "$SYN_SRC" ]; then
        echo "SKIP: $SYN_SRC not found"
        echo "$CONFIG_NAME,0,MISSING,n/a,n/a,0" >> "$OUTPUT_BASE/summary.csv"
        return
    fi

    local COUNT=$(find "$SYN_SRC" -maxdepth 1 -name "*_surf" -type d 2>/dev/null | wc -l)
    if [ "$COUNT" -lt 1 ]; then
        echo "SKIP: no sample directories"
        echo "$CONFIG_NAME,0,EMPTY,n/a,n/a,0" >> "$OUTPUT_BASE/summary.csv"
        return
    fi
    echo "Found $COUNT samples"

    local GEN_DIR="gen_dim79_${CONFIG_NAME}_${SLURM_JOB_ID}"
    local CONFIG_FILE="$OUTPUT_BASE/${CONFIG_NAME}_config.yaml"

    cat > "$CONFIG_FILE" << YAML
inherit_from: voxelization/configs/train_skullbreak.yaml

train:
  gpu: 0

data:
  dset: SkullBreak
  path: $SYN_SRC

generation:
  batch_size: 1
  generation_dir: $GEN_DIR
  num_ensemble: 1
  save_ensemble_implants: False
  compute_eval_metrics: True

test:
  model_file: $VOX_MODEL
YAML

    local START=$(date +%s)
    python -u voxelization/generate.py "$CONFIG_FILE" 2>&1 | tee "$OUTPUT_BASE/${CONFIG_NAME}_log.txt"
    local END=$(date +%s)
    local ELAPSED=$((END - START))

    local GEN_OUT="voxelization/out/skullbreak/$GEN_DIR"
    local RESULT_DIR="$OUTPUT_BASE/${CONFIG_NAME}_results"
    if [ -d "$GEN_OUT" ]; then
        mkdir -p "$RESULT_DIR"
        cp -r "$GEN_OUT"/* "$RESULT_DIR/" 2>/dev/null || true
    fi

    if [ -f "$RESULT_DIR/benchmark_summary.json" ]; then
        METRICS=$(python3 -c "
import json
with open('$RESULT_DIR/benchmark_summary.json') as f:
    d = json.load(f)
dsc = d.get('mean_dsc', d.get('dice_mean', 0))
bdsc = d.get('mean_bdsc_10mm', d.get('bdice_10mm_mean', 0))
hd95 = d.get('mean_hd95_mm', d.get('hd95_mm_mean', 0))
print(f'DSC={dsc:.4f} bDSC={bdsc:.4f} HD95={hd95:.2f}mm')
print(f'$CONFIG_NAME,$COUNT,{dsc},{bdsc},{hd95},$ELAPSED')
")
        echo "$CONFIG_NAME: $ELAPSED sec | $METRICS"
        echo "$METRICS" | tail -1 >> "$OUTPUT_BASE/summary.csv"
    else
        echo "$CONFIG_NAME: NO METRICS ($ELAPSED sec)"
        echo "$CONFIG_NAME,$COUNT,n/a,n/a,n/a,$ELAPSED" >> "$OUTPUT_BASE/summary.csv"
    fi
}

# Run the 3 ready configs
run_vox "ddim_25_ens1" "autoresearch/results/perun/8gpu_final/ddim_sweep_19586/ddim_25/syn" 25
run_vox "ddim_50_ens1" "autoresearch/results/perun/8gpu_final/ddim_sweep_19586/ddim_50/syn" 50
run_vox "ddim_100_ens1" "autoresearch/results/perun/8gpu_final/ddim_sweep_100_200_19413/ddim_100/syn" 100

echo ""
echo "=== Stage-2 DIM-79 Complete at $(date) ==="
echo "Results: $OUTPUT_BASE"
echo ""
echo "=== Summary ==="
cat "$OUTPUT_BASE/summary.csv"
