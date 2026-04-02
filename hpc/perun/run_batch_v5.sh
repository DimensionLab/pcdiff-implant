#!/bin/bash
#SBATCH --job-name=pcdiff-v5
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/logs/v5_exp_%a_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/logs/v5_exp_%a_%j.err
#SBATCH --array=1-10

set -euo pipefail

EXPERIMENTS=(perun_v5_001 perun_v5_002 perun_v5_003 perun_v5_004 perun_v5_005 perun_v5_006 perun_v5_007 perun_v5_008 perun_v5_009 perun_v5_010)
EXP_ID="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID - 1]}"

echo "=== PCDiff V5 Experiment: $EXP_ID ==="
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

python /mnt/data/home/mamuke588/pcdiff-implant/hpc/perun/run_single_experiment.py --config "$CONFIG_FILE"

echo ""
echo "=== Experiment $EXP_ID Complete ==="
echo "End: $(date)"
