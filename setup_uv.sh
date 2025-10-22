#!/bin/bash
# Quick setup script for pcdiff-implant using uv and Python 3.14
# Usage: ./setup_uv.sh [cuda_version]
# Example: ./setup_uv.sh cu130  # CUDA 13.0 (default)
# Example: ./setup_uv.sh cu124  # CUDA 12.4
# Example: ./setup_uv.sh cpu    # CPU only

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}pcdiff-implant Setup (Python 3.14 + uv)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Determine CUDA version
CUDA_VERSION="${1:-cu130}"  # Default to CUDA 13.0
echo -e "${YELLOW}Using CUDA version: ${CUDA_VERSION}${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "Install it with: pip install uv"
    echo "Or visit: https://github.com/astral-sh/uv"
    exit 1
fi

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${YELLOW}Python version: ${PYTHON_VERSION}${NC}"

if [[ "$PYTHON_VERSION" < "3.14" ]]; then
    echo -e "${RED}Warning: Python 3.14+ is recommended, you have ${PYTHON_VERSION}${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo -e "${GREEN}[1/6] Creating virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Remove it? (y/N)${NC}"
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        uv venv --python python3.14
    fi
else
    uv venv --python python3.14
fi

# Activate virtual environment
echo ""
echo -e "${GREEN}[2/6] Activating virtual environment...${NC}"
source .venv/bin/activate

# Install PyTorch with CUDA
echo ""
echo -e "${GREEN}[3/6] Installing PyTorch with ${CUDA_VERSION}...${NC}"
if [ "$CUDA_VERSION" = "cpu" ]; then
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    uv pip install torch --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
    uv pip install torchvision --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
fi

# Install main dependencies
echo ""
echo -e "${GREEN}[4/6] Installing project dependencies...${NC}"
uv pip install -e .

# Install PyTorch3D
echo ""
echo -e "${GREEN}[5/6] Installing PyTorch3D...${NC}"
echo -e "${YELLOW}(This may take a few minutes and might require compilation)${NC}"
uv pip install pytorch3d || {
    echo -e "${RED}PyTorch3D installation failed. You may need to install it manually.${NC}"
    echo "See: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"
}

# Install PyTorch Scatter
echo ""
echo -e "${GREEN}[6/6] Installing PyTorch Scatter...${NC}"
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html || {
    echo -e "${RED}PyTorch Scatter installation failed. You may need to install it manually.${NC}"
    echo "See: https://github.com/rusty1s/pytorch_scatter"
}

# Verify installation
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Verifying installation...${NC}"
echo -e "${GREEN}========================================${NC}"

python3 << 'EOF'
import sys

def check_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {display_name:20} {version}")
        return True
    except ImportError:
        print(f"✗ {display_name:20} NOT FOUND")
        return False

print("")
all_ok = True
all_ok &= check_import('torch', 'PyTorch')
all_ok &= check_import('torchvision', 'TorchVision')
all_ok &= check_import('trimesh', 'Trimesh')
all_ok &= check_import('open3d', 'Open3D')
all_ok &= check_import('numpy', 'NumPy')
all_ok &= check_import('matplotlib', 'Matplotlib')
all_ok &= check_import('tqdm', 'tqdm')
all_ok &= check_import('yaml', 'PyYAML')
all_ok &= check_import('scipy', 'SciPy')

# Voxelization-specific
all_ok &= check_import('pytorch3d', 'PyTorch3D')
all_ok &= check_import('torch_scatter', 'PyTorch Scatter')

print("")
if all_ok:
    print("✓ All core dependencies installed successfully!")
else:
    print("✗ Some dependencies are missing. See errors above.")
    sys.exit(1)

# Check CUDA
import torch
print("")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("(Running in CPU mode)")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Setup completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}To activate this environment in the future:${NC}"
    echo -e "  source .venv/bin/activate"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Download datasets (see README.md)"
    echo -e "  2. Preprocess data: python3 pcdiff/utils/preproc_skullbreak.py"
    echo -e "  3. Train model: python3 pcdiff/train_completion.py --help"
    echo ""
    echo -e "${YELLOW}For more information, see:${NC}"
    echo -e "  - SETUP.md (comprehensive setup guide)"
    echo -e "  - MIGRATION.md (migration from conda)"
    echo -e "  - pcdiff/README.md (point cloud diffusion model)"
    echo -e "  - voxelization/README.md (voxelization network)"
else
    echo ""
    echo -e "${RED}Setup completed with errors.${NC}"
    echo "Please check the error messages above."
    exit 1
fi

