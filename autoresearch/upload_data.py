#!/usr/bin/env python3
"""
upload_data.py — Upload SkullBreak dataset to a RunPod pod via SSH/SCP.

For community cloud pods without network storage, this script:
  1. Tars the local SkullBreak dataset
  2. Uploads it to the pod via SCP
  3. Extracts and verifies on the pod

Usage:
    python upload_data.py --host <ip> --port <ssh_port>
    python upload_data.py --host <ip> --port <ssh_port> --key ~/.ssh/id_ed25519

Requires: SSH key access to the pod (set up via RunPod PUBLIC_KEY env var)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Dataset location
LOCAL_DATASET = Path("/home/mike/pcdiff-implant/pcdiff/datasets/SkullBreak")
REMOTE_DEST = "/workspace/pcdiff-implant/pcdiff/datasets/SkullBreak"

# SSH defaults
DEFAULT_KEY = str(Path.home() / ".ssh" / "id_ed25519")
SSH_OPTS = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"


def ssh_cmd(host: str, port: int, key: str, cmd: str) -> subprocess.CompletedProcess:
    """Run a command on the remote pod via SSH."""
    full_cmd = f"ssh {SSH_OPTS} -i {key} -p {port} root@{host} '{cmd}'"
    return subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=300)


def scp_upload(host: str, port: int, key: str, local: str, remote: str) -> subprocess.CompletedProcess:
    """Upload a file to the remote pod via SCP."""
    full_cmd = f"scp {SSH_OPTS} -i {key} -P {port} {local} root@{host}:{remote}"
    return subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=1800)


def upload_dataset(host: str, port: int, key: str):
    """Upload SkullBreak dataset to pod."""

    print(f"Target: root@{host}:{port}")
    print(f"SSH key: {key}")
    print(f"Local dataset: {LOCAL_DATASET}")

    # Verify local dataset exists
    if not LOCAL_DATASET.exists():
        print(f"ERROR: Local dataset not found at {LOCAL_DATASET}")
        sys.exit(1)

    # Check what files we need to upload (preprocessed .npy files)
    npy_files = list(LOCAL_DATASET.rglob("*_surf.npy"))
    csv_files = list(LOCAL_DATASET.glob("*.csv"))
    print(f"Found {len(npy_files)} .npy files and {len(csv_files)} CSV files locally")

    if not npy_files:
        print("WARNING: No preprocessed .npy files found. Uploading raw .nrrd files instead.")

    # Test SSH connection
    print("\nTesting SSH connection...")
    result = ssh_cmd(host, port, key, "echo ok && nvidia-smi --query-gpu=name --format=csv,noheader")
    if result.returncode != 0:
        print(f"SSH connection failed: {result.stderr}")
        sys.exit(1)
    print(f"Connected. GPU: {result.stdout.strip().split(chr(10))[-1]}")

    # Create remote directory
    print("\nCreating remote directories...")
    ssh_cmd(host, port, key, f"mkdir -p {REMOTE_DEST}")

    # Check if data already exists remotely
    result = ssh_cmd(host, port, key, f"find {REMOTE_DEST} -name '*_surf.npy' 2>/dev/null | wc -l")
    remote_npy_count = int(result.stdout.strip()) if result.returncode == 0 else 0
    if remote_npy_count > 0:
        print(f"Remote already has {remote_npy_count} .npy files. Skipping upload.")
        return

    # Create tarball of dataset
    tar_path = "/tmp/skullbreak_data.tar.gz"
    print(f"\nCreating tarball at {tar_path}...")
    subprocess.run(
        f"tar -czf {tar_path} -C {LOCAL_DATASET.parent} {LOCAL_DATASET.name}",
        shell=True, check=True, timeout=600,
    )
    tar_size = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"Tarball size: {tar_size:.1f} MB")

    # Upload
    print(f"\nUploading to pod (this may take several minutes)...")
    result = scp_upload(host, port, key, tar_path, "/tmp/skullbreak_data.tar.gz")
    if result.returncode != 0:
        print(f"Upload failed: {result.stderr}")
        sys.exit(1)
    print("Upload complete.")

    # Extract on remote
    print("Extracting on pod...")
    result = ssh_cmd(host, port, key,
                     f"cd {Path(REMOTE_DEST).parent} && tar -xzf /tmp/skullbreak_data.tar.gz && rm /tmp/skullbreak_data.tar.gz")
    if result.returncode != 0:
        print(f"Extraction failed: {result.stderr}")
        sys.exit(1)

    # Verify
    result = ssh_cmd(host, port, key, f"find {REMOTE_DEST} -name '*_surf.npy' 2>/dev/null | wc -l")
    remote_count = int(result.stdout.strip()) if result.returncode == 0 else 0
    print(f"\nVerification: {remote_count} .npy files on pod")

    # Clean up local tarball
    os.remove(tar_path)
    print("Done. Dataset uploaded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload SkullBreak dataset to RunPod pod")
    parser.add_argument("--host", required=True, help="Pod SSH IP address")
    parser.add_argument("--port", type=int, required=True, help="Pod SSH port")
    parser.add_argument("--key", default=DEFAULT_KEY, help=f"SSH key path (default: {DEFAULT_KEY})")
    args = parser.parse_args()

    upload_dataset(args.host, args.port, args.key)
