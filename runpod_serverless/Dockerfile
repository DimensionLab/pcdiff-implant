# Runpod Serverless Dockerfile for PCDiff + Voxelization Inference
# 
# Build:
#   docker build --platform linux/amd64 -t YOUR_USERNAME/pcdiff-implant-serverless:v1.0 .
#
# Push:
#   docker push YOUR_USERNAME/pcdiff-implant-serverless:v1.0

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

# Copy the entire project
COPY . /app/

# Create models directory
RUN mkdir -p /app/models

# Copy model checkpoints (these should be added during build or via network volume)
# Option 1: Copy from local build context
COPY output_m1_test/2025-12-28_22-22-17/best.pth /app/models/pcdiff_best.pth
COPY voxelization/checkpoints/model_best.pt /app/models/voxelization_best.pt

# Option 2: Download from cloud storage (uncomment if using this approach)
# ARG PCDIFF_MODEL_URL
# ARG VOX_MODEL_URL
# RUN wget -O /app/models/pcdiff_best.pth "${PCDIFF_MODEL_URL}" && \
#     wget -O /app/models/voxelization_best.pt "${VOX_MODEL_URL}"

# Set environment variables
ENV PYTHONPATH=/app:/app/pcdiff:/app/voxelization
ENV PYTHONUNBUFFERED=1

# Pre-compile Python files for faster startup
RUN python -m compileall /app/runpod_serverless /app/pcdiff /app/voxelization || true

# Set the entrypoint
CMD ["python", "-u", "/app/runpod_serverless/rp_handler.py"]

