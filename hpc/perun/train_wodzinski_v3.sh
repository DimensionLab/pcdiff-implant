#!/bin/bash
#SBATCH --job-name=wv3-full
#SBATCH --output=wv3-full_%j.out
#SBATCH --error=wv3-full_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:6
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

source /mnt/data/home/mamuke588/miniconda3/etc/profile.d/conda.sh
conda activate pcdiff

cd /mnt/data/home/mamuke588/pcdiff-implant

torchrun --nproc_per_node=6 \
    scripts/train_wodzinski_v3.py \
    --epochs 500 \
    --batch_size 1 \
    --lr 1e-3 \
    --base_filters 48 \
    --dropout 0.1 \
    --dice_w 0.5 \
    --bce_w 0.2 \
    --boundary_w 0.15 \
    --symmetry_w 0.15 \
    --val_every 5 \
    --checkpoint_every 50
