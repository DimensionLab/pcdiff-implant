#!/bin/bash
#SBATCH --job-name=s2-gen
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/logs/stage2_gen_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/logs/stage2_gen_%j.err

set -eo pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

echo "=== Stage-2 Generate + Quality Metrics ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

eval "$(/mnt/data/home/mamuke588/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

echo "Python: $(which python)"

cd /mnt/data/home/mamuke588/pcdiff-implant
nvidia-smi

ABLATION_ROOT="benchmarking/runs/stage1_ablation/SkullBreak"
CONFIG_FILE="voxelization/configs/gen_skullbreak.yaml"
VOX_MODEL="voxelization/checkpoints/model_best.pt"

for config_dir in "$ABLATION_ROOT"/*/; do
    config_name=$(basename "$config_dir")
    stage1_summary="$config_dir/stage1/benchmark_summary.json"
    results_syn="$config_dir/results/syn"
    
    # Skip if stage1 not complete
    if [ ! -f "$stage1_summary" ]; then
        echo "SKIP $config_name: stage1 not complete"
        continue
    fi
    
    # Skip if already has voxelized output (mean_impl.nrrd in any case dir)
    existing=$(find "$results_syn" -name "mean_impl.nrrd" 2>/dev/null | wc -l)
    total=$(find "$results_syn" -maxdepth 1 -type d -name "*_surf" 2>/dev/null | wc -l)
    if [ "$existing" -ge "$total" ] && [ "$total" -gt 0 ]; then
        echo "SKIP $config_name: already voxelized ($existing/$total)"
        continue
    fi
    
    echo ""
    echo "=== Processing $config_name ($existing/$total done so far) ==="
    
    # Extract ensemble size from config name (e.g., ddim-steps25-ens3 -> 3)
    ens_size=$(echo "$config_name" | grep -oP "ens\\K[0-9]+")
    PCDIFF_SKULLBREAK_RESULTS="$results_syn" \
    PCDIFF_SKULLBREAK_MODEL="$VOX_MODEL" \
    PCDIFF_NUM_ENSEMBLE="$ens_size" \
    python -u voxelization/generate.py "$CONFIG_FILE" 2>&1 || {
        echo "ERROR: generate.py failed for $config_name, continuing..."
        continue
    }
    
    echo "DONE $config_name at $(date)"
done

echo ""
echo "=== All Stage-2 generation complete ==="
echo "End: $(date)"
