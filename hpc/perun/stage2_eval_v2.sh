#!/bin/bash
#SBATCH --job-name=s2-eval
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/logs/stage2_eval_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/logs/stage2_eval_%j.err

set -euo pipefail

echo "=== Stage-2 Evaluation: Voxelization + Metrics ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

eval "$(/mnt/data/home/mamuke588/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

echo "Python: $(which python)"
python -c "from skimage import measure; print('skimage OK')"
python -c "from plyfile import PlyData; print('plyfile OK')"

cd /mnt/data/home/mamuke588/pcdiff-implant
nvidia-smi

# Run the external script instead of heredoc
python -u benchmarking/run_stage2_eval.py

echo ""
echo "=== Done ==="
echo "End: $(date)"
