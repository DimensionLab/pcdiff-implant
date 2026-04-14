#!/bin/bash
#SBATCH --job-name=s2-d79-now
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

# Stage-2 voxelization for DIM-79: 5 key DDIM configs
# Uses current best 6GPU retrained voxelization model (psr_l2=0.2184 at epoch 340)
# Stage-1 data: 115 SkullBreak test cases per config

set -eo pipefail

echo "=== Stage-2 Voxelization DIM-79: 5 Key DDIM Configs ==="
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

ln -sfn "$(pwd)/pcdiff/datasets/SkullBreak/defective_skull" autoresearch/defective_skull 2>/dev/null || true
ln -sfn "$(pwd)/pcdiff/datasets/SkullBreak/implant" autoresearch/implant 2>/dev/null || true

nvidia-smi

VOX_MODEL="voxelization/out/skullbreak_6gpu/model_best.pt"
if [ ! -f "$VOX_MODEL" ]; then
    echo "ERROR: 6GPU vox model not found at $VOX_MODEL"
    exit 1
fi
echo "Using vox model: $VOX_MODEL"
ls -la $VOX_MODEL

ABLATION_ROOT="benchmarking/runs/stage1_ablation/SkullBreak"
mkdir -p "$ABLATION_ROOT"

OUTPUT_BASE="$ABLATION_ROOT/dim79_5configs_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_BASE"

SUMMARY_FILE="$OUTPUT_BASE/summary.csv"
echo "config,samples,mean_dsc,mean_bdsc_10mm,mean_hd95_mm,wall_sec" > "$SUMMARY_FILE"

run_vox() {
    local CONFIG_NAME="$1"
    local SYN_SRC="$2"
    local ENS_NUM="${3:-1}"
    local syn_dir="$SYN_SRC/syn"

    echo ""
    echo "=============================="
    echo "=== Evaluating $CONFIG_NAME (ens=$ENS_NUM) ==="
    echo "=============================="

    if [ ! -d "$syn_dir" ]; then
        echo "SKIP: $syn_dir not found"
        echo "$CONFIG_NAME,0,MISSING,n/a,n/a,0" >> "$SUMMARY_FILE"
        return
    fi

    local count=$(find "$syn_dir" -maxdepth 1 -name "*_surf" -type d 2>/dev/null | wc -l)
    if [ "$count" -lt 1 ]; then
        echo "SKIP: no sample directories"
        echo "$CONFIG_NAME,0,EMPTY,n/a,n/a,0" >> "$SUMMARY_FILE"
        return
    fi
    echo "Found $count samples"

    local GEN_DIR="gen_dim79_${CONFIG_NAME}_${SLURM_JOB_ID}"
    local CONFIG_FILE="$OUTPUT_BASE/${CONFIG_NAME}_config.yaml"

    cat > "$CONFIG_FILE" << YAML
inherit_from: voxelization/configs/train_skullbreak_6gpu.yaml
train:
  gpu: 0
data:
  dset: SkullBreak
  path: $syn_dir
generation:
  batch_size: 1
  generation_dir: $GEN_DIR
  num_ensemble: $ENS_NUM
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

    # Parse metrics
    local MEAN_DSC=$(grep "Dice score:" "$OUTPUT_BASE/${CONFIG_NAME}_log.txt" | awk '{print $3}' | awk '{s+=$1;n++}END{if(n>0) printf "%.6f", s/n; else print "0"}')
    local MEAN_BDSC=$(grep "Boundary dice (10mm):" "$OUTPUT_BASE/${CONFIG_NAME}_log.txt" | awk '{print $4}' | awk '{s+=$1;n++}END{if(n>0) printf "%.6f", s/n; else print "0"}')
    local MEAN_HD95=$(grep "95 percentile Haussdorf distance:" "$OUTPUT_BASE/${CONFIG_NAME}_log.txt" | awk '{print $5}' | awk '{s+=$1;n++}END{if(n>0) printf "%.4f", s/n; else print "0"}')

    echo "$CONFIG_NAME: DSC=$MEAN_DSC, bDSC=$MEAN_BDSC, HD95=${MEAN_HD95}mm, time=${ELAPSED}s"
    echo "$CONFIG_NAME,$count,$MEAN_DSC,$MEAN_BDSC,$MEAN_HD95,$ELAPSED" >> "$SUMMARY_FILE"
}

# Run all 5 configs
run_vox "ddim_250_ens1" "autoresearch/results/perun/8gpu_final/s1_dim79_19888/ddim_250_ens1" 1
run_vox "ddim_100_ens1" "autoresearch/results/perun/8gpu_final/ddim_sweep_100_200_19413/ddim_100" 1
run_vox "ddim_50_ens1"  "autoresearch/results/perun/8gpu_final/ddim_sweep_19586/ddim_50" 1
run_vox "ddim_50_ens3"  "autoresearch/results/perun/8gpu_final/s1_dim79_19888/ddim_50_ens3" 3
run_vox "ddim_25_ens1"  "autoresearch/results/perun/8gpu_final/ddim_sweep_19586/ddim_25" 1

echo ""
echo "=== All 5 configs done at $(date) ==="
echo "Summary saved to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
