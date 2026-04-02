#!/bin/bash
# Auto-submit stage-1 ablation when queue has capacity
set -euo pipefail

ABLATION_SCRIPT=~/pcdiff-implant/hpc/perun/stage1_ablation_slurm.sh
LOG=~/pcdiff-implant/benchmarking/runs/stage1_ablation/auto_submit.log

echo "[$(date)] Waiting for queue capacity..." >> $LOG

while true; do
    RUNNING=$(squeue -u mamuke588 -h | wc -l)
    echo "[$(date)] Running jobs: $RUNNING" >> $LOG
    
    if [ "$RUNNING" -le 5 ]; then
        echo "[$(date)] Queue has capacity ($RUNNING jobs). Submitting ablation..." >> $LOG
        cd ~/pcdiff-implant
        RESULT=$(sbatch $ABLATION_SCRIPT 2>&1)
        echo "[$(date)] sbatch result: $RESULT" >> $LOG
        if echo "$RESULT" | grep -q 'Submitted'; then
            echo "[$(date)] SUCCESS: $RESULT" >> $LOG
            exit 0
        else
            echo "[$(date)] FAILED: $RESULT - will retry in 60s" >> $LOG
        fi
    fi
    
    sleep 60
done
