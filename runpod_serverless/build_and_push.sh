#!/bin/bash
# Build and push PCDiff Serverless Docker image to Docker Hub
#
# Usage:
#   ./build_and_push.sh YOUR_DOCKERHUB_USERNAME [TAG]
#
# Example:
#   ./build_and_push.sh myusername v1.0

set -e

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <dockerhub_username> [tag]"
    echo "Example: $0 myusername v1.0"
    exit 1
fi

DOCKERHUB_USERNAME=$1
TAG=${2:-latest}
IMAGE_NAME="pcdiff-implant-serverless"
FULL_IMAGE_NAME="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  Building PCDiff Serverless Docker Image"
echo "=============================================="
echo "Image: ${FULL_IMAGE_NAME}"
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Check if model files exist
if [ ! -f "output_m1_test/2025-12-28_22-22-17/best.pth" ]; then
    echo "Error: PCDiff model not found at output_m1_test/2025-12-28_22-22-17/best.pth"
    exit 1
fi

if [ ! -f "voxelization/checkpoints/model_best.pt" ]; then
    echo "Error: Voxelization model not found at voxelization/checkpoints/model_best.pt"
    exit 1
fi

echo "✓ Model files found"
echo ""

# Build the image
echo "Building Docker image..."
docker build \
    --platform linux/amd64 \
    -t "${FULL_IMAGE_NAME}" \
    -f runpod_serverless/Dockerfile \
    .

echo ""
echo "✓ Build complete!"
echo ""

# Ask to push
read -p "Push image to Docker Hub? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing to Docker Hub..."
    docker push "${FULL_IMAGE_NAME}"
    echo ""
    echo "✓ Push complete!"
    echo ""
    echo "Image available at: docker.io/${FULL_IMAGE_NAME}"
    echo ""
    echo "Next steps:"
    echo "1. Go to https://www.runpod.io/console/serverless"
    echo "2. Click 'New Endpoint'"
    echo "3. Click 'Import from Docker Registry'"
    echo "4. Enter: docker.io/${FULL_IMAGE_NAME}"
    echo "5. Configure GPU, environment variables, and deploy"
else
    echo "Skipping push."
    echo ""
    echo "To push later, run:"
    echo "  docker push ${FULL_IMAGE_NAME}"
fi

