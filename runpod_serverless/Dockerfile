# Runpod Serverless Dockerfile for PCDiff + Voxelization Inference
# 
# Build from project root:
#   docker build --platform linux/amd64 -t YOUR_USERNAME/pcdiff-implant-serverless:v1.0 -f runpod_serverless/Dockerfile .
#
# Push:
#   docker push YOUR_USERNAME/pcdiff-implant-serverless:v1.0
#
# Model Loading Priority:
#   1. Network Volume (/runpod-volume/models/) - Best for frequent model updates
#   2. Embedded in Docker image (/app/models/) - Default fallback

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies including ninja for CUDA JIT compilation
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CUDA environment variables - MUST be set BEFORE building CUDA extensions
# The Runpod base image has CUDA in /usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Set TORCH_CUDA_ARCH_LIST for JIT compilation BEFORE building extensions
# Include newer architectures: 8.9 (Ada Lovelace - RTX 4090), 9.0 (Hopper - H100), 10.0, 12.0 (Blackwell)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"

# Force CUDA build for PyTorch extensions
ENV FORCE_CUDA=1

# Limit parallel jobs to avoid OOM during compilation
ENV MAX_JOBS=1

# Copy requirements first for better caching
COPY runpod_serverless/requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
# Use --ignore-installed to handle distutils-installed packages in base image
# Pin numpy<2.0 and scipy<1.14 for compatibility
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy>=1.24.0,<2.0.0" "scipy>=1.11.0,<1.14.0" && \
    pip install --no-cache-dir --ignore-installed -r /app/requirements.txt

# Install ninja Python package (needed for CUDA compilation)
RUN pip install --no-cache-dir ninja

# Install PyTorch3D from source with CUDA support
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install torch-scatter from source with CUDA support
RUN pip install --no-cache-dir --no-build-isolation torch-scatter

# Copy the entire project
COPY . /app/

# Create models directory
RUN mkdir -p /app/models

# Copy model checkpoints from the correct locations
# PCDiff models from pcdiff/checkpoints/ (both best and latest for model switching)
COPY pcdiff/checkpoints/model_best.pth /app/models/pcdiff_best.pth
COPY pcdiff/checkpoints/model_latest.pth /app/models/pcdiff_latest.pth

# Voxelization model from voxelization/checkpoints/
COPY voxelization/checkpoints/model_best.pt /app/models/voxelization_best.pt

# Set environment variables
ENV PYTHONPATH=/app:/app/pcdiff:/app/voxelization
ENV PYTHONUNBUFFERED=1

# Network volume mount point (Runpod mounts network volumes here)
# Models in /runpod-volume/models/ will take priority over embedded models
ENV RUNPOD_NETWORK_VOLUME_PATH=/runpod-volume

# Pre-compile Python files for faster startup
RUN python -m compileall /app/runpod_serverless /app/pcdiff /app/voxelization || true

# Set the entrypoint
CMD ["python", "-u", "/app/runpod_serverless/rp_handler.py"]
