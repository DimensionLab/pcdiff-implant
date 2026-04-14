#!/bin/bash
#SBATCH --job-name=s2-dim80
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/logs/stage2_dim80_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/logs/stage2_dim80_%j.err
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

set -eo pipefail

echo "=== Stage-2 DIM-80: Voxelization for 5 key DDIM configs ==="
echo "Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST, Start: $(date)"

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="9.0"
NVIDIA_BASE=$(python -c "import nvidia; import os; print(os.path.dirname(nvidia.__file__))")
INCLUDE_DIRS=""
for pkg in cuda_runtime cublas cusparse cusolver cufft curand nvtx cuda_nvrtc; do
  d="$NVIDIA_BASE/${pkg}/include"
  [ -d "$d" ] && INCLUDE_DIRS="$d:$INCLUDE_DIRS"
done
export CPLUS_INCLUDE_PATH="$INCLUDE_DIRS${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$INCLUDE_DIRS${C_INCLUDE_PATH:-}"

cd /mnt/data/home/mamuke588/pcdiff-implant
nvidia-smi

# Optional: activate scratch for Lustre fast I/O
if [ -f .activate_scratch ]; then
  source .activate_scratch
  echo "Scratch activated"
else
  echo "No .activate_scratch found, running without scratch"
fi

ABLATION_ROOT="benchmarking/runs/stage1_ablation/SkullBreak"
VOX_MODEL="voxelization/checkpoints/model_best.pt"
DATASET_ROOT="pcdiff/datasets/SkullBreak"
LOG_DIR="$ABLATION_ROOT/logs"
mkdir -p "$LOG_DIR"

# DIM-80: 5 key DDIM configs
CONFIGS=("ddim-steps250-ens1" "ddim-steps100-ens1" "ddim-steps50-ens1" "ddim-steps50-ens3" "ddim-steps25-ens1")

for config in "${CONFIGS[@]}"; do
  config_dir="$ABLATION_ROOT/$config"
  syn_dir="$config_dir/stage1/syn"

  # Check if stage-1 has results
  count=$(find "$syn_dir" -name "sample.npy" 2>/dev/null | wc -l)
  if [ "$count" -lt 1 ]; then
    echo "SKIP $config: no stage-1 cases"
    continue
  fi

  # Check if already done with valid results
  if [ -f "$config_dir/stage2_dim80/benchmark_summary.json" ]; then
    has_error=$(python3 -c "import json; d=json.load(open('$config_dir/stage2_dim80/benchmark_summary.json')); print('yes' if 'error' in d else 'no')" 2>/dev/null || echo "yes")
    if [ "$has_error" = "no" ]; then
      echo "SKIP $config: already has valid results"
      continue
    fi
  fi

  echo ""
  echo "=== Evaluating $config ($count cases) ==="

  # Create symlink wrapper structure (required for generate.py path resolution)
  eval_wrapper="$config_dir/SkullBreak_eval_dim80"
  mkdir -p "$eval_wrapper/results"
  ln -sfn "$(realpath $syn_dir)" "$eval_wrapper/results/syn"
  ln -sfn "$(realpath $DATASET_ROOT/defective_skull)" "$eval_wrapper/defective_skull"
  ln -sfn "$(realpath $DATASET_ROOT/implant)" "$eval_wrapper/implant"

  # Determine ensemble count from config name
  ENS=$(echo "$config" | grep -oP 'ens\K\d+')

  # Create temporary config
  tmp_config="/tmp/gen_dim80_${config}.yaml"
  cat > "$tmp_config" << YAMLEOF
inherit_from: voxelization/configs/train_skullbreak.yaml

train:
  gpu: 0

data:
  dset: SkullBreak
  path: $eval_wrapper/results/syn

generation:
  batch_size: 1
  generation_dir: gen_dim80_${config}
  num_ensemble: ${ENS}
  save_ensemble_implants: False
  compute_eval_metrics: True

test:
  model_file: $VOX_MODEL
YAMLEOF

  echo "Config: $tmp_config"
  cat "$tmp_config"

  # Run generate.py
  python -u voxelization/generate.py "$tmp_config" 2>&1 | tee "$config_dir/stage2_dim80_log.txt"

  # Copy results
  gen_dir="voxelization/out/skullbreak/gen_dim80_${config}"
  if [ -d "$gen_dir" ]; then
    mkdir -p "$config_dir/stage2_dim80"
    cp -r "$gen_dir"/* "$config_dir/stage2_dim80/" 2>/dev/null
    echo "Results copied to $config_dir/stage2_dim80/"

    # Show summary
    if [ -f "$config_dir/stage2_dim80/benchmark_summary.json" ]; then
      echo "--- Summary for $config ---"
      cat "$config_dir/stage2_dim80/benchmark_summary.json"
      echo ""
    fi
  else
    echo "WARNING: no output directory found at $gen_dir"
    ls voxelization/out/skullbreak/ 2>/dev/null
  fi
done

echo ""
echo "=== All DIM-80 configs processed ==="
echo "End: $(date)"
