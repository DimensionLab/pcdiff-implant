#!/bin/bash
# sync_results.sh — Sync experiment results between PERUN and local server
#
# Usage (from local server):
#   bash hpc/perun/sync_results.sh pull   # Pull results from PERUN
#   bash hpc/perun/sync_results.sh push   # Push code to PERUN
#   bash hpc/perun/sync_results.sh status # Check PERUN job status

set -euo pipefail

PERUN_USER="mamuke588"
PERUN_HOST="login01.perun.tuke.sk"
SSH_KEY="$HOME/.ssh/id_ed25519"
LOCAL_DIR="$HOME/pcdiff-implant"
REMOTE_DIR="~/pcdiff-implant"
SCRATCH="/scratch/$PERUN_USER/pcdiff"

SSH_CMD="ssh -i $SSH_KEY $PERUN_USER@$PERUN_HOST"
RSYNC_OPTS="-avz --progress -e 'ssh -i $SSH_KEY'"

case "${1:-help}" in
    push)
        echo ">> Pushing code to PERUN..."
        rsync $RSYNC_OPTS \
            --exclude='__pycache__' \
            --exclude='.git' \
            --exclude='*.pyc' \
            --exclude='datasets/' \
            --exclude='my_results/' \
            --exclude='*.npy' \
            --exclude='*.npz' \
            --exclude='*.nrrd' \
            --exclude='*.pth' \
            --exclude='*.pt' \
            "$LOCAL_DIR/" \
            "$PERUN_USER@$PERUN_HOST:$REMOTE_DIR/"
        echo ">> Code synced to PERUN."
        ;;
    pull)
        echo ">> Pulling results from PERUN..."
        # Pull autoresearch results
        rsync $RSYNC_OPTS \
            "$PERUN_USER@$PERUN_HOST:$REMOTE_DIR/autoresearch/results/" \
            "$LOCAL_DIR/autoresearch/results/perun/"

        # Pull scratch results
        rsync $RSYNC_OPTS \
            "$PERUN_USER@$PERUN_HOST:$SCRATCH/results/" \
            "$LOCAL_DIR/my_results/perun/"

        echo ">> Results pulled from PERUN."
        ;;
    push-data)
        echo ">> Pushing dataset to PERUN scratch..."
        rsync $RSYNC_OPTS \
            "$LOCAL_DIR/datasets/" \
            "$PERUN_USER@$PERUN_HOST:$SCRATCH/data/"
        echo ">> Dataset synced to PERUN."
        ;;
    status)
        echo ">> PERUN job status:"
        $SSH_CMD "squeue -u $PERUN_USER -o '%.10i %.20j %.8T %.10M %.9l %.6D %R' 2>/dev/null" || \
            echo "Cannot connect to PERUN (VPN required)"
        ;;
    logs)
        JOB_ID="${2:?'Usage: sync_results.sh logs <JOB_ID>'}"
        echo ">> Fetching logs for job $JOB_ID..."
        $SSH_CMD "cat $SCRATCH/logs/*${JOB_ID}*.log 2>/dev/null | tail -50" || \
            echo "Cannot connect or log not found"
        ;;
    help|*)
        echo "Usage: $0 {push|pull|push-data|status|logs <JOB_ID>}"
        echo ""
        echo "Commands:"
        echo "  push      - Sync code to PERUN"
        echo "  pull      - Pull experiment results from PERUN"
        echo "  push-data - Push dataset to PERUN scratch storage"
        echo "  status    - Check running jobs on PERUN"
        echo "  logs ID   - Fetch logs for a specific job"
        ;;
esac
