#!/bin/bash
#SBATCH --job-name=wv3-eval
#SBATCH --output=wv3-eval_%j.out
#SBATCH --error=wv3-eval_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --account=perun2501174
#SBATCH --qos=perun2501174

source /mnt/data/home/mamuke588/miniconda3/etc/profile.d/conda.sh
conda activate pcdiff

cd /mnt/data/home/mamuke588/pcdiff-implant

# Evaluate wv3-full (with symmetry loss)
python scripts/eval_wodzinski_v3.py \
    --checkpoint wodzinski_v3/model_best.pt \
    --output_dir wodzinski_v3/full_eval \
    --splits_json wodzinski_v3/splits.json \
    --base_filters 32

echo "=== DONE wv3-full ==="

# Evaluate wv3-nosym
python scripts/eval_wodzinski_v3.py \
    --checkpoint wodzinski_v3_nosym/model_best.pt \
    --output_dir wodzinski_v3_nosym/full_eval \
    --splits_json wodzinski_v3_nosym/splits.json \
    --base_filters 32

echo "=== DONE wv3-nosym ==="
