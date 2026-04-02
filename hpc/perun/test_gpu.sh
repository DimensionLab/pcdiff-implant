#!/bin/bash
#SBATCH --job-name=pcdiff-test
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/hpc/perun/test_gpu_%j.out

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff

echo "=== GPU Test ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    # Quick tensor test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x.T)
    print(f'Matrix multiply test: OK (result shape {y.shape})')
print('=== ALL TESTS PASSED ===')
"
