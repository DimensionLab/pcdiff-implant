#!/bin/bash
# Quick setup script for pcdiff-implant using uv and Python 3.10
# Usage: ./setup_uv.sh [cuda_version] [--wandb]
# Example: ./setup_uv.sh cu124         # CUDA 12.4 (default)
# Example: ./setup_uv.sh cu124 --wandb # CUDA 12.4 with wandb
# Example: ./setup_uv.sh cpu           # CPU only

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}pcdiff-implant Setup (Python 3.10 + uv)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Parse arguments
CUDA_VERSION="cu124"  # Default to CUDA 12.4 (compatible with CUDA 13 drivers)
INSTALL_WANDB=false

for arg in "$@"; do
    if [[ "$arg" == "--wandb" ]]; then
        INSTALL_WANDB=true
    elif [[ "$arg" == cu* ]] || [[ "$arg" == "cpu" ]]; then
        CUDA_VERSION="$arg"
    else
        echo -e "${YELLOW}Unknown argument: $arg${NC}"
        echo "Usage: ./setup_uv.sh [cuda_version] [--wandb]"
        exit 1
    fi
done

echo -e "${YELLOW}Using CUDA version: ${CUDA_VERSION}${NC}"
if [ "$INSTALL_WANDB" = true ]; then
    echo -e "${YELLOW}Wandb will be installed for experiment tracking${NC}"
fi

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

if [[ "$PYTHON_VERSION" < "3.10" ]]; then
    echo -e "${RED}Warning: Python 3.10+ is recommended, you have ${PYTHON_VERSION}${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check and install GCC-9 for CUDA compilation
echo ""
echo -e "${GREEN}[1/8] Checking GCC-9 compiler...${NC}"
if ! command -v gcc-9 &> /dev/null; then
    echo -e "${YELLOW}GCC-9 not found. Installing gcc-9 and g++-9 (required for CUDA extensions)...${NC}"
    echo -e "${YELLOW}This requires sudo privileges.${NC}"
    sudo apt-get update && sudo apt-get install -y gcc-9 g++-9
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ GCC-9 installed successfully${NC}"
    else
        echo -e "${RED}Failed to install gcc-9. You may need to install it manually.${NC}"
        echo -e "${RED}Run: sudo apt-get install gcc-9 g++-9${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ GCC-9 already installed${NC}"
fi

# Create virtual environment
echo ""
echo -e "${GREEN}[2/8] Creating virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Remove it? (y/N)${NC}"
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        uv venv --python 3.10
    fi
else
    uv venv --python 3.10
fi

# Activate virtual environment
echo ""
echo -e "${GREEN}[3/8] Activating virtual environment...${NC}"
source .venv/bin/activate

# Install PyTorch with CUDA
echo ""
echo -e "${GREEN}[4/8] Installing PyTorch 2.5.0 with ${CUDA_VERSION}...${NC}"
if [ "$CUDA_VERSION" = "cpu" ]; then
    uv pip install "torch==2.5.0" "torchvision==0.20.0" --index-url https://download.pytorch.org/whl/cpu
else
    uv pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
    uv pip install "torchvision==0.20.0" --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
fi

# Install main dependencies
echo ""
echo -e "${GREEN}[5/8] Installing project dependencies...${NC}"
uv pip install -e .

# Install Wandb (optional, for experiment tracking)
if [ "$INSTALL_WANDB" = true ]; then
    echo ""
    echo -e "${GREEN}[6/8] Installing Weights & Biases...${NC}"
    uv pip install wandb
fi

# Install PyTorch3D
echo ""
echo -e "${GREEN}[7/8] Installing PyTorch3D...${NC}"

# First, try pre-built wheels (much faster, no compilation needed)
echo -e "${YELLOW}Trying pre-built PyTorch3D wheels...${NC}"
if uv pip install --no-deps "fvcore" "iopath" 2>/dev/null; then
    if uv pip install "pytorch3d" 2>/dev/null; then
        echo -e "${GREEN}✓ Installed PyTorch3D from pre-built wheels${NC}"
        PYTORCH3D_INSTALLED=true
    else
        PYTORCH3D_INSTALLED=false
    fi
else
    PYTORCH3D_INSTALLED=false
fi

# If pre-built wheels failed, try building from source with CUDA compatibility fix
if [ "$PYTORCH3D_INSTALLED" = false ]; then
    echo -e "${YELLOW}Pre-built wheels not available, building from source...${NC}"
    echo -e "${YELLOW}(This takes ~5-10 minutes)${NC}"
    
    # Force PyTorch3D to use the same CUDA version as PyTorch
    export FORCE_CUDA="1"
    export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"  # Common GPU architectures
    
    # Get PyTorch's CUDA version to ensure compatibility
    PYTORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
    
    if [ -n "$PYTORCH_CUDA" ]; then
        echo -e "${YELLOW}PyTorch was built with CUDA ${PYTORCH_CUDA}${NC}"
        echo -e "${YELLOW}Setting CUDA_HOME to match PyTorch's CUDA version...${NC}"
        
        # Try to find matching CUDA installation
        if [ -d "/usr/local/cuda-${PYTORCH_CUDA}" ]; then
            export CUDA_HOME="/usr/local/cuda-${PYTORCH_CUDA}"
            export PATH="${CUDA_HOME}/bin:${PATH}"
            export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
        fi
    fi
    
    # Install dependencies first
    uv pip install --no-deps "fvcore" "iopath"
    
    # Try building PyTorch3D
    if uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"; then
        echo -e "${GREEN}✓ Built PyTorch3D from source${NC}"
    else
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}PyTorch3D installation failed!${NC}"
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo -e "${YELLOW}This is usually due to CUDA version mismatch.${NC}"
        echo -e "${YELLOW}Your system has CUDA $(nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//' || echo 'unknown')${NC}"
        echo -e "${YELLOW}PyTorch was built with CUDA ${PYTORCH_CUDA}${NC}"
        echo ""
        echo -e "${YELLOW}Options to fix this:${NC}"
        echo -e "  1. Install CUDA ${PYTORCH_CUDA} toolkit and set CUDA_HOME:"
        echo -e "     export CUDA_HOME=/usr/local/cuda-${PYTORCH_CUDA}"
        echo -e "     Then re-run this script"
        echo ""
        echo -e "  2. Use a different PyTorch version matching your system CUDA"
        echo ""
        echo -e "  3. Skip PyTorch3D (only needed for voxelization)"
        echo ""
        echo -e "${YELLOW}Manual installation:${NC}"
        echo -e "  See: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"
        echo ""
    fi
fi

# Install PyTorch Scatter (pre-built wheel for PyTorch 2.5)
echo ""
echo -e "${GREEN}[8/8] Installing PyTorch Scatter...${NC}"
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+${CUDA_VERSION}.html || {
    echo -e "${YELLOW}Pre-built wheel not found, building from source (~2-3 min)...${NC}"
    uv pip install --no-build-isolation "git+https://github.com/rusty1s/pytorch_scatter.git" || {
        echo -e "${RED}PyTorch Scatter installation failed. You may need to install it manually.${NC}"
        echo "See: https://github.com/rusty1s/pytorch_scatter"
    }
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
print("Core dependencies:")
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

print("")
print("Optional (voxelization only):")
pytorch3d_ok = check_import('pytorch3d', 'PyTorch3D')
scatter_ok = check_import('torch_scatter', 'PyTorch Scatter')

print("")
if all_ok:
    print("✓ All core dependencies installed successfully!")
    if not pytorch3d_ok:
        print("⚠ PyTorch3D not installed (only needed for voxelization)")
        print("  The main point cloud diffusion model will work fine without it.")
else:
    print("✗ Some core dependencies are missing. See errors above.")
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

# Check GCC-9
import subprocess
print("")
try:
    result = subprocess.run(['gcc-9', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        gcc_version = result.stdout.split('\n')[0].split()[-1]
        print(f"✓ GCC-9 available: {gcc_version} (required for CUDA extensions)")
    else:
        print("✗ GCC-9 not found")
except FileNotFoundError:
    print("✗ GCC-9 not found (needed for training with CUDA extensions)")
    print("  Install with: sudo apt-get install gcc-9 g++-9")
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
    echo -e "${YELLOW}Before training, set these environment variables:${NC}"
    echo -e "  export CC=/usr/bin/gcc-9"
    echo -e "  export CXX=/usr/bin/g++-9"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Download datasets (see README.md)"
    echo -e "  2. Preprocess data: python3 pcdiff/utils/preproc_skullbreak.py"
    echo -e "  3. Train model: python3 pcdiff/train_completion.py --help"
    echo ""
    echo -e "${YELLOW}For more information, see:${NC}"
    echo -e "  - QUICKSTART.md (quick start guide)"
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

