#!/bin/bash

# PCDiff Web Viewer - Start Script
# This script starts both the FastAPI backend and serves the frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================"
echo "PCDiff Web Viewer - Starting"
echo "================================"

# Configuration
HOST="${PCDIFF_HOST:-0.0.0.0}"
PORT="${PCDIFF_PORT:-8080}"
WORKERS="${PCDIFF_WORKERS:-4}"

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing backend dependencies..."
    pip install -r "$SCRIPT_DIR/backend/requirements.txt"
fi

# Check if trimesh is available (needed for conversion)
if ! python3 -c "import trimesh" 2>/dev/null; then
    echo "Installing trimesh (required for file conversion)..."
    pip install trimesh
fi

# Start backend
echo ""
echo "Starting FastAPI backend on ${HOST}:${PORT}..."
echo "Access the API at: http://${HOST}:${PORT}/api"
echo ""

cd "$SCRIPT_DIR/backend"
python3 -m uvicorn main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"

