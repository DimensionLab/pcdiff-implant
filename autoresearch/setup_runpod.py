#!/usr/bin/env python3
"""
setup_runpod.py — Create and configure a RunPod pod for autoresearch experiments.

This script:
  1. Creates a RunPod pod on SECURE cloud (required for SSH TCP access)
  2. Adds SSH key for access
  3. Waits for pod to be ready
  4. Prints SSH connection info

Usage:
    python setup_runpod.py                # Create pod (secure cloud, no network volume)
    python setup_runpod.py --stop         # Stop running pods
    python setup_runpod.py --status       # Show pod status

Requires RUNPOD_API_KEY in environment or crainial_app/.env
See RUNPOD_RUNBOOK.md for operational knowledge.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RUNPOD_API_URL = "https://api.runpod.io/graphql"
NETWORK_VOLUME_ID = "wzcc20z2mg"  # crainial-implant, CA-MTL-3, 60GB
SSH_KEY_PATH = Path("/home/mike/.ssh/id_ed25519.pub")

# GPU preferences in order (CA-MTL-3 compatible, cost-effective)
GPU_PREFERENCES = [
    "NVIDIA A40",  # $0.35/hr, 48GB — best value
    "NVIDIA GeForce RTX 4090",  # $0.34/hr, 24GB — good alternative
    "NVIDIA GeForce RTX 3090",  # $0.22/hr, 24GB — budget option
]


def get_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        env_file = Path(__file__).resolve().parent.parent / "crainial_app" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("RUNPOD_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise RuntimeError("RUNPOD_API_KEY not found")
    return key


def graphql(query: str, variables: dict = None) -> dict:
    api_key = get_api_key()
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        RUNPOD_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    if "errors" in result:
        raise RuntimeError(f"GraphQL errors: {json.dumps(result['errors'], indent=2)}")
    return result["data"]


def get_ssh_key() -> str:
    if SSH_KEY_PATH.exists():
        return SSH_KEY_PATH.read_text().strip()
    # Try default locations
    for path in [Path.home() / ".ssh" / "id_ed25519.pub", Path.home() / ".ssh" / "id_rsa.pub"]:
        if path.exists():
            return path.read_text().strip()
    return ""


def get_pods() -> list:
    data = graphql(
        "{ myself { pods { id name desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } gpus { id gpuUtilPercent } } machine { gpuDisplayName } } } }"
    )
    return data["myself"]["pods"]


def create_pod(gpu_id: str = None) -> dict:
    ssh_key = get_ssh_key()

    # Docker image: PyTorch with CUDA, matching the existing Dockerfile
    docker_image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

    # Startup script: update repo, install deps, start SSH
    docker_args = ""

    # Try GPUs in preference order
    gpu_ids_to_try = [gpu_id] if gpu_id else GPU_PREFERENCES

    for gid in gpu_ids_to_try:
        print(f"Attempting to create pod with {gid}...")
        try:
            mutation = """
            mutation createPod($input: PodFindAndDeployOnDemandInput!) {
                podFindAndDeployOnDemand(input: $input) {
                    id
                    name
                    desiredStatus
                    imageName
                    machine { gpuDisplayName }
                }
            }
            """
            variables = {
                "input": {
                    "name": "autoresearch-pcdiff",
                    "imageName": docker_image,
                    "gpuTypeId": gid,
                    "gpuCount": 1,
                    "volumeInGb": 50,
                    "containerDiskInGb": 30,
                    "cloudType": "SECURE",
                    "minVcpuCount": 4,
                    "minMemoryInGb": 32,
                    "ports": "22/tcp,8888/http",
                    "dockerArgs": docker_args,
                    "startSsh": True,
                    "env": [
                        {"key": "PUBLIC_KEY", "value": ssh_key},
                    ]
                    if ssh_key
                    else [],
                }
            }

            data = graphql(mutation, variables)
            pod = data["podFindAndDeployOnDemand"]
            print(f"Pod created: {pod['id']} ({pod.get('machine', {}).get('gpuDisplayName', gid)})")
            return pod

        except RuntimeError as e:
            print(f"  Failed with {gid}: {e}")
            continue

    raise RuntimeError("Could not create pod with any preferred GPU")


def wait_for_pod(pod_id: str, timeout: int = 300) -> dict:
    """Wait for pod to be running and return connection info."""
    print(f"Waiting for pod {pod_id} to be ready...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        pods = get_pods()
        for pod in pods:
            if pod["id"] == pod_id:
                status = pod.get("desiredStatus", "")
                runtime = pod.get("runtime")
                if runtime and runtime.get("ports"):
                    # Find SSH port
                    for port in runtime["ports"]:
                        if port.get("privatePort") == 22:
                            ssh_ip = port.get("ip", "")
                            ssh_port = port.get("publicPort", 22)
                            print("\nPod ready!")
                            print(f"  SSH: ssh root@{ssh_ip} -p {ssh_port} -i ~/.ssh/id_ed25519")
                            print(f"  GPU: {pod.get('machine', {}).get('gpuDisplayName', 'unknown')}")
                            return {"ssh_ip": ssh_ip, "ssh_port": ssh_port, "pod_id": pod_id}
                print(f"  Status: {status} (waiting...)")
                break
        time.sleep(10)

    raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")


def stop_pods():
    """Stop all running pods."""
    pods = get_pods()
    for pod in pods:
        if pod.get("desiredStatus") == "RUNNING":
            print(f"Stopping pod {pod['id']} ({pod['name']})...")
            graphql(
                "mutation stopPod($input: PodStopInput!) { podStop(input: $input) { id desiredStatus } }",
                {"input": {"podId": pod["id"]}},
            )
            print("  Stopped.")


def show_status():
    """Show current pod status and balance."""
    data = graphql(
        "{ myself { clientBalance pods { id name desiredStatus machine { gpuDisplayName } runtime { uptimeInSeconds ports { ip privatePort publicPort } } } } }"
    )
    balance = data["myself"]["clientBalance"]
    pods = data["myself"]["pods"]

    print(f"Balance: ${balance:.2f}")
    print(f"Pods: {len(pods)}")
    for pod in pods:
        gpu = pod.get("machine", {}).get("gpuDisplayName", "?")
        status = pod.get("desiredStatus", "?")
        uptime = pod.get("runtime", {}).get("uptimeInSeconds", 0) if pod.get("runtime") else 0
        print(f"  {pod['id']} | {pod['name']} | {gpu} | {status} | uptime: {uptime // 3600}h{(uptime % 3600) // 60}m")
        if pod.get("runtime") and pod["runtime"].get("ports"):
            for port in pod["runtime"]["ports"]:
                if port.get("privatePort") == 22:
                    print(f"    SSH: ssh root@{port['ip']} -p {port['publicPort']} -i ~/.ssh/id_ed25519")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RunPod pod management for autoresearch")
    parser.add_argument("--stop", action="store_true", help="Stop all running pods")
    parser.add_argument("--status", action="store_true", help="Show pod status")
    parser.add_argument("--gpu", type=str, default=None, help="Specific GPU type ID")
    args = parser.parse_args()

    if args.stop:
        stop_pods()
    elif args.status:
        show_status()
    else:
        pod = create_pod(args.gpu)
        conn = wait_for_pod(pod["id"])
        print("\n=== NEXT STEPS ===")
        print("1. SSH into the pod:")
        print(f"   ssh root@{conn['ssh_ip']} -p {conn['ssh_port']} -i ~/.ssh/id_ed25519")
        print("2. Update the repo on networked storage:")
        print("   cd /workspace/pcdiff-implant && git checkout main && git pull")
        print("3. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("4. Run autoresearch:")
        print("   cd /workspace/pcdiff-implant/autoresearch")
        print("   OPENROUTER_API_KEY=<key> python run_experiments.py --time-budget 900 --max-experiments 50")
