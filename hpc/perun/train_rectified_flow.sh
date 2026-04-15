#!/bin/bash
#SBATCH --job-name=rf-train
#SBATCH --output=rf-train_%j.out
#SBATCH --error=rf-train_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:6
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

export PYTHONUNBUFFERED=1

source /mnt/data/home/mamuke588/miniconda3/etc/profile.d/conda.sh
conda activate pcdiff

cd /mnt/data/home/mamuke588/pcdiff-implant

torchrun --nproc_per_node=6 \
    rectified_flow.py \
    --data-dir datasets/SkullBreak/pcdiff_train.csv \
    --dataset SkullBreak \
    --save-dir pcdiff/runs/rectified_flow \
    --epochs 10000 \
    --batch-size 8 \
    --lr 1e-4 \
    --ema-decay 0.9999 \
    --num-points 30720 \
    --num-nn 3072 \
    --embed-dim 64 \
    --width-mult 1.0 \
    --gpus 6 \
    --seed 42 \
    --log-interval 50 \
    --save-interval 50 \
    --sample-interval 200 \
    --augment
