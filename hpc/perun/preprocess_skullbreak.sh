#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --output=preproc_%j.out
#SBATCH --error=preproc_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

export PYTHONUNBUFFERED=1

source /mnt/data/home/mamuke588/miniconda3/etc/profile.d/conda.sh
conda activate pcdiff

cd /mnt/data/home/mamuke588/pcdiff-implant

python pcdiff/utils/preproc_skullbreak.py \
    --root datasets/SkullBreak \
    --target-points 400000 \
    --threads 16
