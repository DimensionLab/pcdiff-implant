#!/bin/bash
#SBATCH --job-name=pcdiff-batch
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/logs/exp_%a_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/results/perun/logs/exp_%a_%j.err
#SBATCH --array=1-16

set -euo pipefail

# Map array task ID to experiment
EXPERIMENTS=(perun_001 perun_002 perun_003 perun_004 perun_005 perun_006 perun_007 perun_008 perun_009 perun_010 perun_011 perun_012 perun_013 perun_014 perun_015 perun_016)
EXP_ID="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID - 1]}"

echo "=== PCDiff Experiment: $EXP_ID ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Environment
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

# Read experiment config
CONFIG_FILE="/mnt/data/home/mamuke588/pcdiff-implant/autoresearch/perun_experiments/$EXP_ID.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config: $(cat $CONFIG_FILE)"
echo ""

# Run the experiment with config overrides
python /mnt/data/home/mamuke588/pcdiff-implant/hpc/perun/run_single_experiment.py --config "$CONFIG_FILE"

echo ""
echo "=== Experiment $EXP_ID Complete ==="
echo "End: $(date)"
