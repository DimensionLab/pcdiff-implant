#!/bin/bash
#SBATCH --job-name=wv3-nosym
#SBATCH --output=wv3-nosym_%j.out
#SBATCH --error=wv3-nosym_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

source /mnt/data/home/mamuke588/miniconda3/etc/profile.d/conda.sh
conda activate pcdiff

cd /mnt/data/home/mamuke588/pcdiff-implant

torchrun --nproc_per_node=4 \
    scripts/train_wodzinski_v3.py \
    --results_dir /mnt/data/home/mamuke588/pcdiff-implant/wodzinski_v3_nosym \
    --epochs 500 \
    --batch_size 1 \
    --lr 1e-3 \
    --base_filters 32 \
    --dropout 0.1 \
    --dice_w 0.6 \
    --bce_w 0.25 \
    --boundary_w 0.15 \
    --symmetry_w 0.0 \
    --val_every 5 \
    --checkpoint_every 50
