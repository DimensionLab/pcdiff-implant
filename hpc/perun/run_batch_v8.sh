#!/bin/bash
#SBATCH --job-name=pcdiff-v8
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/logs/v8_exp_%a_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/logs/v8_exp_%a_%j.err
#SBATCH --array=1-8

set -euo pipefail

EXPERIMENTS=(perun_v8_001 perun_v8_002 perun_v8_003 perun_v8_004 perun_v8_005 perun_v8_006 perun_v8_007 perun_v8_008)
EXP_ID="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID - 1]}"

echo "=== PCDiff V8 Experiment: $EXP_ID ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff

cd /mnt/data/home/mamuke588/pcdiff-implant

# Build CUDA extensions
if [ -d "pcdiff/modules/functional" ]; then
    cd pcdiff/modules/functional
    python setup.py build_ext --inplace 2>/dev/null || true
    cd /mnt/data/home/mamuke588/pcdiff-implant
fi

nvidia-smi

CONFIG_FILE="/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/perun_experiments/$EXP_ID.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config: $(cat $CONFIG_FILE)"
echo ""

python -u /mnt/data/home/mamuke588/pcdiff-implant/hpc/perun/run_single_experiment.py --config "$CONFIG_FILE"

echo ""
echo "=== Experiment $EXP_ID Complete ==="
echo "End: $(date)"
