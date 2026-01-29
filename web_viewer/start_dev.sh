#!/bin/bash

# PCDiff Web Viewer - Development Script
# Starts both backend and frontend in development mode with hot-reload

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================"
echo "DimensionLab CrAInial - Data Viewer"
echo "Development Mode"
echo "================================"

# -------------------------------------------------------------------
# Find and activate virtual environment
# -------------------------------------------------------------------
VENV_DIR="$PROJECT_ROOT/.venv"
if [ -d "$VENV_DIR" ]; then
    echo "Using venv: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using active venv: $VIRTUAL_ENV"
else
    echo "Warning: No virtual environment found at $VENV_DIR"
    echo "Falling back to system Python."
fi

# Check Python3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi
echo "Python: $(python3 --version) ($(which python3))"

# Check Node.js/npm
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed. Please install Node.js and npm first."
    echo "Visit: https://nodejs.org/"
    exit 1
fi

# -------------------------------------------------------------------
# Backend dependencies
# -------------------------------------------------------------------
echo ""
echo "Checking backend dependencies..."
cd "$SCRIPT_DIR/backend"
if command -v uv &> /dev/null; then
    uv pip install -q -r requirements.txt 2>/dev/null || true
else
    python3 -m pip install -q -r requirements.txt 2>/dev/null || true
fi

# Create data directory for SQLite
DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"
echo "Data directory: $DATA_DIR"

# Database init and seeding are handled by the app lifespan handler
# in main.py (init_db + _seed_defaults_if_needed) on every startup.

# -------------------------------------------------------------------
# Frontend dependencies
# -------------------------------------------------------------------
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd "$SCRIPT_DIR/frontend"
    npm install
fi

# -------------------------------------------------------------------
# Start servers
# -------------------------------------------------------------------
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup EXIT INT TERM

# Start backend in background
echo ""
echo "Starting backend on http://localhost:8080..."
cd "$SCRIPT_DIR/backend"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8080 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend in background
echo "Starting frontend on http://localhost:5173..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "================================"
echo "Development servers started!"
echo "Backend:  http://localhost:8080"
echo "Frontend: http://localhost:5173"
echo "API docs: http://localhost:8080/docs"
echo "================================"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
