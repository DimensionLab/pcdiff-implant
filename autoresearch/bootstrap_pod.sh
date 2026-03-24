#!/bin/bash
# bootstrap_pod.sh — Fully automated pod setup for autoresearch
# Run this on a fresh RunPod pod after SSH'ing in.
# Usage: bash /workspace/bootstrap_pod.sh
# Or remotely: ssh root@<ip> -p <port> 'bash -s' < autoresearch/bootstrap_pod.sh

set -euo pipefail

echo "=== Autoresearch Pod Bootstrap ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not detected')"

# 1. Clone or update repo
REPO_DIR="/workspace/pcdiff-implant"
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo exists, updating..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout main
    git pull origin main
else
    echo "Cloning repo..."
    cd /workspace
    git clone https://github.com/DimensionLab/pcdiff-implant.git
    cd "$REPO_DIR"
fi

# 2. Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null || true
# Extra deps for autoresearch
pip install open3d 2>/dev/null || true

# 3. Check if SkullBreak data exists, if not run preprocessing
DATASET_DIR="$REPO_DIR/pcdiff/datasets/SkullBreak"
if [ -d "$DATASET_DIR/defective_skull" ]; then
    # Check if .npy files exist (preprocessed)
    NPY_COUNT=$(find "$DATASET_DIR" -name "*_surf.npy" 2>/dev/null | wc -l)
    if [ "$NPY_COUNT" -gt 0 ]; then
        echo "SkullBreak preprocessed data found ($NPY_COUNT .npy files)"
    else
        echo "SkullBreak raw data found but not preprocessed. Running preprocessing..."
        cd "$REPO_DIR"
        python pcdiff/utils/preproc_skullbreak.py --root "$DATASET_DIR" || echo "Preprocessing failed - may need manual intervention"
    fi
else
    echo "WARNING: SkullBreak dataset not found at $DATASET_DIR"
    echo "You need to download/copy the SkullBreak dataset to this location."
    echo "If on network volume, check /runpod-volume/ for existing data."
fi

# 4. Verify CUDA extensions compile
echo "Testing CUDA extension compilation..."
cd "$REPO_DIR/pcdiff"
python -c "from modules import PVConv; print('CUDA extensions: OK')" 2>/dev/null || {
    echo "CUDA extensions need compilation. This may take a few minutes..."
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
}

# 5. Verify autoresearch data access
echo "Verifying autoresearch data access..."
cd "$REPO_DIR"
python autoresearch/prepare_pcdiff.py 2>/dev/null && echo "Data access: OK" || echo "Data access: NEEDS SETUP"

# 6. Print summary
echo ""
echo "=== Bootstrap Complete ==="
echo "Repo: $REPO_DIR"
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'not available')"
echo ""
echo "To run autoresearch:"
echo "  cd $REPO_DIR/autoresearch"
echo "  OPENROUTER_API_KEY=<key> python run_experiments.py --time-budget 900 --max-experiments 50"
