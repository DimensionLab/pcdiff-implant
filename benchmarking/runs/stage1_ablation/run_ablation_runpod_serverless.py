#!/usr/bin/env python3
"""
DIM-6: Stage-1 Ablation Sweep via RunPod Serverless
====================================================

Submits 15 ablation configurations (DDPM/DDIM x steps x ensemble sizes)
to the RunPod serverless endpoint. Each test case x config = 1 job.

Usage:
    # Submit all ablation jobs
    python benchmarking/runs/stage1_ablation/run_ablation_runpod_serverless.py submit

    # Check status of running jobs
    python benchmarking/runs/stage1_ablation/run_ablation_runpod_serverless.py status

    # Collect completed results
    python benchmarking/runs/stage1_ablation/run_ablation_runpod_serverless.py collect

Environment Variables:
    RUNPOD_API_KEY: RunPod API key
    RUNPOD_ENDPOINT_ID: RunPod serverless endpoint ID
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
ABLATION_ROOT = REPO_ROOT / "benchmarking" / "runs" / "stage1_ablation"
DATASET_ROOT = REPO_ROOT / "pcdiff" / "datasets" / "SkullBreak"
JOBS_FILE = ABLATION_ROOT / "serverless_jobs.json"

# Ablation matrix: 15 configurations
ABLATION_CONFIGS = [
    # DDPM: 1000 steps only
    {"sampling_method": "ddpm", "sampling_steps": 1000, "num_ensemble": 1},
    {"sampling_method": "ddpm", "sampling_steps": 1000, "num_ensemble": 3},
    {"sampling_method": "ddpm", "sampling_steps": 1000, "num_ensemble": 5},
    # DDIM: 250, 100, 50, 25 steps
    {"sampling_method": "ddim", "sampling_steps": 250, "num_ensemble": 1},
    {"sampling_method": "ddim", "sampling_steps": 250, "num_ensemble": 3},
    {"sampling_method": "ddim", "sampling_steps": 250, "num_ensemble": 5},
    {"sampling_method": "ddim", "sampling_steps": 100, "num_ensemble": 1},
    {"sampling_method": "ddim", "sampling_steps": 100, "num_ensemble": 3},
    {"sampling_method": "ddim", "sampling_steps": 100, "num_ensemble": 5},
    {"sampling_method": "ddim", "sampling_steps": 50, "num_ensemble": 1},
    {"sampling_method": "ddim", "sampling_steps": 50, "num_ensemble": 3},
    {"sampling_method": "ddim", "sampling_steps": 50, "num_ensemble": 5},
    {"sampling_method": "ddim", "sampling_steps": 25, "num_ensemble": 1},
    {"sampling_method": "ddim", "sampling_steps": 25, "num_ensemble": 3},
    {"sampling_method": "ddim", "sampling_steps": 25, "num_ensemble": 5},
]

DEFECT_FAMILIES = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]


def run_id_for(cfg):
    return f"{cfg['sampling_method']}-steps{cfg['sampling_steps']}-ens{cfg['num_ensemble']}"


def load_test_cases():
    """Load SkullBreak test cases from CSV."""
    test_csv = DATASET_ROOT / "test.csv"
    cases = []
    with open(test_csv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # CSV contains relative paths like complete_skull/077_surf.npy
            rel_path = line.split(",")[0] if "," in line else line
            # Extract case number from filename
            fname = Path(rel_path).stem  # e.g. "077_surf"
            case_id = fname.replace("_surf", "")
            cases.append(case_id)
    return cases


def encode_npy(file_path):
    """Encode a .npy file as base64."""
    data = np.load(str(file_path))
    buf = io.BytesIO()
    np.save(buf, data)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def submit_jobs(args):
    """Submit all ablation jobs to RunPod."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not api_key or not endpoint_id:
        print("ERROR: Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")
        sys.exit(1)

    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases")
    print(f"Submitting {len(ABLATION_CONFIGS)} configs x {len(test_cases)} cases x {len(DEFECT_FAMILIES)} defects")
    total = len(ABLATION_CONFIGS) * len(test_cases) * len(DEFECT_FAMILIES)
    print(f"Total jobs: {total}")

    # Load existing jobs file if resuming
    jobs = {}
    if JOBS_FILE.exists():
        with open(JOBS_FILE) as f:
            jobs = json.load(f)
        print(f"Resuming: {len(jobs)} jobs already tracked")

    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    submitted = 0
    skipped = 0

    for cfg in ABLATION_CONFIGS:
        rid = run_id_for(cfg)
        for case_id in test_cases:
            for defect in DEFECT_FAMILIES:
                job_key = f"{rid}/{case_id}_{defect}"
                if job_key in jobs and jobs[job_key].get("status") in ("submitted", "completed"):
                    skipped += 1
                    continue

                # Find defective skull file
                defective_path = DATASET_ROOT / "defective_skull" / defect / f"{case_id}_surf.npy"
                if not defective_path.exists():
                    print(f"  SKIP {job_key}: {defective_path} not found")
                    continue

                payload = {
                    "input": {
                        "defective_skull": encode_npy(defective_path),
                        "input_format": "base64",
                        "num_ensemble": cfg["num_ensemble"],
                        "sampling_method": cfg["sampling_method"],
                        "sampling_steps": cfg["sampling_steps"],
                        "output_prefix": f"ablation/{rid}/{case_id}_{defect}",
                    }
                }

                try:
                    resp = requests.post(url, json=payload, headers=headers, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    runpod_id = data.get("id")
                    jobs[job_key] = {
                        "runpod_id": runpod_id,
                        "status": "submitted",
                        "config": cfg,
                        "case_id": case_id,
                        "defect": defect,
                        "submitted_at": datetime.now(timezone.utc).isoformat(),
                    }
                    submitted += 1
                    if submitted % 10 == 0:
                        print(f"  Submitted {submitted} jobs...")
                        # Save progress periodically
                        with open(JOBS_FILE, "w") as f:
                            json.dump(jobs, f, indent=2)
                except Exception as e:
                    print(f"  ERROR submitting {job_key}: {e}")
                    jobs[job_key] = {
                        "status": "error",
                        "error": str(e),
                        "config": cfg,
                        "case_id": case_id,
                        "defect": defect,
                    }

                # Small delay to avoid rate limits
                time.sleep(0.1)

    # Save final state
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"\nDone: submitted={submitted}, skipped={skipped}, total_tracked={len(jobs)}")


def check_status(args):
    """Check status of submitted jobs."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not api_key or not endpoint_id:
        print("ERROR: Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")
        sys.exit(1)

    if not JOBS_FILE.exists():
        print("No jobs file found. Run 'submit' first.")
        return

    with open(JOBS_FILE) as f:
        jobs = json.load(f)

    headers = {"Authorization": f"Bearer {api_key}"}
    updated = 0

    for job_key, job in jobs.items():
        if job.get("status") not in ("submitted",):
            continue

        runpod_id = job.get("runpod_id")
        if not runpod_id:
            continue

        url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{runpod_id}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            data = resp.json()
            rp_status = data.get("status", "unknown")

            if rp_status == "COMPLETED":
                job["status"] = "completed"
                job["output"] = data.get("output", {})
                job["completed_at"] = datetime.now(timezone.utc).isoformat()
                updated += 1
            elif rp_status == "FAILED":
                job["status"] = "failed"
                job["error"] = data.get("error", "unknown")
                updated += 1
            # else still IN_QUEUE or IN_PROGRESS

        except Exception as e:
            print(f"  Error polling {job_key}: {e}")

        time.sleep(0.05)

    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)

    # Summary by config
    by_config = {}
    for job_key, job in jobs.items():
        rid = job_key.split("/")[0]
        if rid not in by_config:
            by_config[rid] = {"submitted": 0, "completed": 0, "failed": 0, "error": 0}
        by_config[rid][job.get("status", "unknown")] = by_config[rid].get(job.get("status", "unknown"), 0) + 1

    total_completed = sum(1 for j in jobs.values() if j.get("status") == "completed")
    total_submitted = sum(1 for j in jobs.values() if j.get("status") == "submitted")
    total_failed = sum(1 for j in jobs.values() if j.get("status") == "failed")

    print(f"Total: {len(jobs)} jobs")
    print(f"  Completed: {total_completed}")
    print(f"  Submitted (pending): {total_submitted}")
    print(f"  Failed: {total_failed}")
    print(f"  Updated this round: {updated}")
    print()
    for rid in sorted(by_config.keys()):
        counts = by_config[rid]
        print(f"  {rid:30s} {counts}")


def collect_results(args):
    """Collect completed results into per-config benchmark summaries."""
    if not JOBS_FILE.exists():
        print("No jobs file found. Run 'submit' first.")
        return

    with open(JOBS_FILE) as f:
        jobs = json.load(f)

    # Group by config
    by_config = {}
    for job_key, job in jobs.items():
        if job.get("status") != "completed":
            continue

        rid = job_key.split("/")[0]
        if rid not in by_config:
            by_config[rid] = []

        output = job.get("output", {})
        metadata = output.get("metadata", {})
        by_config[rid].append({
            "case_id": job["case_id"],
            "defect": job["defect"],
            "processing_time_seconds": metadata.get("processing_time_seconds"),
            "mesh_vertices": metadata.get("mesh_vertices"),
            "mesh_faces": metadata.get("mesh_faces"),
            "num_implant_points": metadata.get("num_implant_points"),
            "sampling_method": metadata.get("sampling_method"),
            "sampling_steps": metadata.get("sampling_steps"),
            "num_ensemble": metadata.get("num_ensemble"),
            "results": output.get("results", {}),
        })

    print(f"Configs with completed results: {len(by_config)}")
    for rid, results in sorted(by_config.items()):
        out_dir = ABLATION_ROOT / "SkullBreak" / rid
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write results manifest
        manifest_path = out_dir / "serverless_results.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2)

        times = [r["processing_time_seconds"] for r in results if r.get("processing_time_seconds")]
        avg_time = sum(times) / len(times) if times else 0

        print(f"  {rid:30s} cases={len(results):3d} avg_time={avg_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIM-6 Stage-1 Ablation via RunPod Serverless")
    parser.add_argument("action", choices=["submit", "status", "collect"])
    args = parser.parse_args()

    if args.action == "submit":
        submit_jobs(args)
    elif args.action == "status":
        check_status(args)
    elif args.action == "collect":
        collect_results(args)
