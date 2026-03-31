# PERUN HPC Experiment Infrastructure

Scripts and configuration for running PCDiff experiments on the PERUN supercomputer
at the Technical University of Košice.

## PERUN System Specs (GPU Partition)
- 26 nodes × HPE ProLiant XD685
- 8 × NVIDIA H200 per node (141 GB HBM3e each)
- 128 CPU cores per node (2 × AMD EPYC 9535)
- 2,304 GB DDR5 ECC RAM per node
- 4 × 400 Gb/s NDR InfiniBand
- NVLink GPU interconnect (900 GB/s GPU-to-GPU)

## Quick Start

### 1. VPN Connection
```bash
# Import VPN profile (one-time)
pritunl-client add <PROFILE_URI>

# Connect
pritunl-client start <PROFILE_ID>

# Verify
pritunl-client list   # should show "Connected"
```

### 2. SSH Access
```bash
ssh mamuke588@login01.perun.tuke.sk -i ~/.ssh/id_ed25519
```

### 3. First-Time Setup on PERUN
```bash
# On login node:
bash ~/pcdiff-implant/hpc/perun/setup_perun_env.sh
```

### 4. Submit Jobs
```bash
# Single GPU training
sbatch hpc/perun/train_single_gpu.sh

# Multi-GPU DDP training (8 GPUs on 1 node)
sbatch hpc/perun/train_multi_gpu.sh

# Full experiment campaign
sbatch hpc/perun/run_campaign.sh
```

### 5. Monitor
```bash
# Job status
squeue -u mamuke588

# Cancel a job
scancel <JOB_ID>

# Check GPU utilization on allocated node
srun --jobid=<JOB_ID> nvidia-smi

# Wandb dashboard
# https://wandb.ai/ (logged to project: pcdiff-implant-perun)
```

## File Structure
```
perun/
├── README.md                   # This file
├── setup_perun_env.sh          # One-time environment setup
├── train_single_gpu.sh         # Slurm: single H200 training
├── train_multi_gpu.sh          # Slurm: 8-GPU DDP training
├── run_campaign.sh             # Slurm: automated experiment campaign
└── sync_results.sh             # Sync results back from PERUN
```
