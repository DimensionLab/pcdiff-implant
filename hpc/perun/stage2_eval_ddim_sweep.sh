#!/bin/bash
#SBATCH --job-name=s2-ddim-swp
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=autoresearch/results/perun/8gpu_final/stage2_ddim_sweep_%j.out
#SBATCH --error=autoresearch/results/perun/8gpu_final/stage2_ddim_sweep_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

# Stage-2 voxelization eval for the DDIM step sweep (DIM-6)
# Evaluates DSC/bDSC/HD95 for each DDIM step count (5,10,15,20,25,35,50)

set -eo pipefail
echo "=== Stage-2 Voxelization Eval: DDIM Step Sweep (DIM-6) ==="
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

SWEEP_BASE="autoresearch/results/perun/8gpu_final/ddim_sweep_19586"
VOX_MODEL="voxelization/checkpoints/model_best.pt"
DATASET_ROOT="pcdiff/datasets/SkullBreak"
OUTPUT_BASE="autoresearch/results/perun/8gpu_final/stage2_ddim_sweep_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_BASE"

# Create symlinks for ground truth data in each sweep dir
# The generate.py code resolves GT as: (path before "/results")/defective_skull/<defect_type>/<id>.nrrd
# So we need: autoresearch/defective_skull -> dataset/defective_skull
# and autoresearch/implant -> dataset/implant
if [ ! -L "autoresearch/defective_skull" ]; then
    ln -sfn "$(pwd)/$DATASET_ROOT/defective_skull" autoresearch/defective_skull
fi
if [ ! -L "autoresearch/implant" ]; then
    ln -sfn "$(pwd)/$DATASET_ROOT/implant" autoresearch/implant
fi
echo "GT symlinks: autoresearch/defective_skull -> $DATASET_ROOT/defective_skull"
echo "GT symlinks: autoresearch/implant -> $DATASET_ROOT/implant"

# Evaluate all completed DDIM step configs
STEPS=(5 10 15 20 25 35 50)

echo ""
echo "steps,samples,mean_dsc,mean_bdsc_10mm,mean_hd95_mm,wall_sec" > "$OUTPUT_BASE/summary.csv"

for N_STEPS in "${STEPS[@]}"; do
    CONFIG="ddim_${N_STEPS}"
    SYN_SRC="$SWEEP_BASE/$CONFIG/syn"

    echo ""
    echo "=============================="
    echo "=== Evaluating DDIM-${N_STEPS} ==="
    echo "=============================="

    if [ ! -d "$SYN_SRC" ]; then
        echo "SKIP: $SYN_SRC not found"
        continue
    fi

    COUNT=$(find "$SYN_SRC" -maxdepth 1 -name "*_surf" -type d 2>/dev/null | wc -l)
    if [ "$COUNT" -lt 1 ]; then
        echo "SKIP: no sample directories"
        continue
    fi
    echo "Found $COUNT samples"

    GEN_DIR="gen_ddim_sweep_${N_STEPS}_${SLURM_JOB_ID}"
    CONFIG_FILE="$OUTPUT_BASE/${CONFIG}_config.yaml"

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

    START=$(date +%s)
    python -u voxelization/generate.py "$CONFIG_FILE" 2>&1 | tee "$OUTPUT_BASE/${CONFIG}_log.txt"
    END=$(date +%s)
    ELAPSED=$((END - START))

    # Copy results
    GEN_OUT="voxelization/out/skullbreak/$GEN_DIR"
    RESULT_DIR="$OUTPUT_BASE/${CONFIG}_results"
    if [ -d "$GEN_OUT" ]; then
        mkdir -p "$RESULT_DIR"
        cp -r "$GEN_OUT"/* "$RESULT_DIR/" 2>/dev/null || true
    fi

    # Extract metrics
    if [ -f "$RESULT_DIR/benchmark_summary.json" ]; then
        METRICS=$(python3 -c "
import json
with open('$RESULT_DIR/benchmark_summary.json') as f:
    d = json.load(f)
dsc = d.get('mean_dsc', d.get('dice_mean', 0))
bdsc = d.get('mean_bdsc_10mm', d.get('bdice_10mm_mean', 0))
hd95 = d.get('mean_hd95_mm', d.get('hd95_mm_mean', 0))
print(f'DSC={dsc:.4f} bDSC={bdsc:.4f} HD95={hd95:.2f}mm')
print(f'$N_STEPS,$COUNT,{dsc},{bdsc},{hd95},$ELAPSED')
")
        echo "DDIM-${N_STEPS}: $ELAPSED sec | $METRICS"
        echo "$METRICS" | tail -1 >> "$OUTPUT_BASE/summary.csv"
    else
        echo "DDIM-${N_STEPS}: NO METRICS ($ELAPSED sec)"
        echo "$N_STEPS,$COUNT,n/a,n/a,n/a,$ELAPSED" >> "$OUTPUT_BASE/summary.csv"
    fi
done

echo ""
echo "=== Stage-2 Eval Complete at $(date) ==="
echo "Results: $OUTPUT_BASE"
echo ""
echo "=== Summary ==="
cat "$OUTPUT_BASE/summary.csv"
