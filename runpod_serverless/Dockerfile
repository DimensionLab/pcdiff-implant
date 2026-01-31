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

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY runpod_serverless/requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
# Use --ignore-installed to handle distutils-installed packages in base image
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r /app/requirements.txt

# Install PyTorch3D from source (required for voxelization)
# Use --no-build-isolation to use the existing torch from base image
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install torch-scatter from PyG wheels (required for point cloud operations)
RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install ninja for CUDA JIT compilation (after cached layers)
RUN apt-get update && apt-get install -y ninja-build build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir ninja

# Copy the entire project
COPY . /app/

# Create models directory
RUN mkdir -p /app/models

# Copy model checkpoints from the correct locations
# PCDiff model from pcdiff/checkpoints/
COPY pcdiff/checkpoints/model_best.pth /app/models/pcdiff_best.pth

# Voxelization model from voxelization/checkpoints/
COPY voxelization/checkpoints/model_best.pt /app/models/voxelization_best.pt

# Set environment variables
ENV PYTHONPATH=/app:/app/pcdiff:/app/voxelization
ENV PYTHONUNBUFFERED=1

# CUDA environment variables - critical for PVCNN CUDA kernels
# The Runpod base image has CUDA in /usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
# Set TORCH_CUDA_ARCH_LIST for JIT compilation
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Network volume mount point (Runpod mounts network volumes here)
# Models in /runpod-volume/models/ will take priority over embedded models
ENV RUNPOD_NETWORK_VOLUME_PATH=/runpod-volume

# Pre-compile Python files for faster startup
RUN python -m compileall /app/runpod_serverless /app/pcdiff /app/voxelization || true

# Set the entrypoint
CMD ["python", "-u", "/app/runpod_serverless/rp_handler.py"]
