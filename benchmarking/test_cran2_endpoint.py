#!/usr/bin/env python3
"""
Batch test script for cran-2 (Wodzinski) RunPod Serverless endpoint.

Sends 15 SkullBreak test cases (3 per defect type) to the endpoint,
measures timing (total, inference, download), and computes evaluation
metrics (DSC, bDSC@10mm, HD95) against ground truth.
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import nrrd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voxelization.eval_metrics import dc

# --- memory-efficient HD95 + bDSC for low-RAM hosts (avoids 512^3 distance transforms) ---
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from scipy.spatial import cKDTree


def _surface_points(mask, voxelspacing):
    fp = generate_binary_structure(mask.ndim, 1)
    border = mask & ~binary_erosion(mask, structure=fp, iterations=1, border_value=0)
    pts = np.argwhere(border).astype(np.float32)
    if voxelspacing is not None:
        pts *= np.asarray(voxelspacing, dtype=np.float32)
    return pts


def hd95_kdtree(pred_bin, gt_bin, voxelspacing=None):
    p = _surface_points(pred_bin.astype(bool), voxelspacing)
    g = _surface_points(gt_bin.astype(bool), voxelspacing)
    if len(p) == 0 or len(g) == 0:
        return float("nan")
    d_pg, _ = cKDTree(g).query(p)
    d_gp, _ = cKDTree(p).query(g)
    return float(np.percentile(np.concatenate([d_pg, d_gp]), 95))


def bdc_cropped(pred_bin, gt_bin, def_bin, voxelspacing=None, distance=10):
    """Crop to bbox around implants + margin so distance transform fits in RAM."""
    union = (pred_bin | gt_bin).astype(bool)
    if not union.any():
        return 0.0
    coords = np.argwhere(union)
    mn = coords.min(axis=0); mx = coords.max(axis=0) + 1
    if voxelspacing is None:
        margin = int(distance) + 2
        margins = np.array([margin] * pred_bin.ndim, dtype=int)
    else:
        margins = np.ceil(np.array(distance) / np.array(voxelspacing, dtype=float)).astype(int) + 2
    mn = np.maximum(mn - margins, 0)
    mx = np.minimum(mx + margins, np.array(pred_bin.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(mn, mx))
    p_c = pred_bin[sl]; g_c = gt_bin[sl]; d_c = def_bin[sl]
    dt = distance_transform_edt(~(d_c > 0), sampling=voxelspacing)
    p_m = p_c.copy(); g_m = g_c.copy()
    p_m[dt > distance] = 0
    g_m[dt > distance] = 0
    return dc(p_m, g_m)

ENDPOINT_ID = "wferq1g3i1hhqd"
DATASET_DIR = PROJECT_ROOT / "datasets" / "SkullBreak"
OUTPUT_DIR = PROJECT_ROOT / "benchmarking" / "cran2_endpoint_test"

TEST_CASES = [
    ("bilateral", "109"),
    ("bilateral", "112"),
    ("bilateral", "051"),
    ("frontoorbital", "091"),
    ("frontoorbital", "062"),
    ("frontoorbital", "095"),
    ("parietotemporal", "009"),
    ("parietotemporal", "058"),
    ("parietotemporal", "026"),
    ("random_1", "080"),
    ("random_1", "088"),
    ("random_1", "024"),
    ("random_2", "000"),
    ("random_2", "013"),
    ("random_2", "062"),
]


def load_env():
    env_path = PROJECT_ROOT / "crainial_app" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def encode_nrrd_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def submit_job(api_key, encoded_data, output_prefix):
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "input": {
            "job_type": "cran2",
            "defective_skull": encoded_data,
            "input_format": "base64",
            "threshold": 0.5,
            "output_prefix": output_prefix,
        }
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


def check_status(api_key, job_id):
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def wait_for_completion(api_key, job_id, timeout=300, poll_interval=3):
    start = time.time()
    while time.time() - start < timeout:
        result = check_status(api_key, job_id)
        status = result.get("status")
        if status == "COMPLETED":
            return result
        elif status == "FAILED":
            raise Exception(f"Job failed: {result.get('error', result)}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def get_s3_client():
    import boto3
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_S3_REGION", "eu-central-1"),
    )


def download_nrrd_from_s3(url, local_path):
    start = time.time()
    bucket = os.environ.get("AWS_S3_BUCKET", "test-crainial")
    prefix = f"https://{bucket}.s3.{os.environ.get('AWS_S3_REGION', 'eu-central-1')}.amazonaws.com/"
    if url.startswith(prefix):
        s3_key = url[len(prefix):]
        s3 = get_s3_client()
        s3.download_file(bucket, s3_key, str(local_path))
    else:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    return time.time() - start


def compute_metrics(pred_path, gt_implant_path, defective_skull_path):
    pred_data, pred_header = nrrd.read(str(pred_path))
    gt_data, gt_header = nrrd.read(str(gt_implant_path))
    defective_data, def_header = nrrd.read(str(defective_skull_path))

    spacing = None
    if "space directions" in gt_header:
        sd = gt_header["space directions"]
        spacing = tuple(abs(sd[i][i]) for i in range(3))
    elif "spacings" in gt_header:
        spacing = tuple(gt_header["spacings"])

    pred_bin = (pred_data > 0).astype(np.uint8)
    gt_bin = (gt_data > 0).astype(np.uint8)
    def_bin = (defective_data > 0).astype(np.uint8)

    if pred_bin.shape != gt_bin.shape:
        from scipy.ndimage import zoom
        factors = tuple(g / p for g, p in zip(gt_bin.shape, pred_bin.shape))
        pred_bin = (zoom(pred_bin.astype(np.float32), factors, order=0) > 0.5).astype(np.uint8)

    dice_val = dc(pred_bin, gt_bin)
    hd95_val = hd95_kdtree(pred_bin, gt_bin, voxelspacing=spacing)
    bdice_val = bdc_cropped(pred_bin, gt_bin, def_bin, voxelspacing=spacing, distance=10)

    return {
        "dice": float(dice_val),
        "hd95_mm": float(hd95_val),
        "bdice_10mm": float(bdice_val),
        "pred_shape": list(pred_data.shape),
        "gt_shape": list(gt_data.shape),
        "spacing": list(spacing) if spacing else None,
    }


def main():
    load_env()
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"Testing cran-2 endpoint {ENDPOINT_ID} with {len(TEST_CASES)} cases")
    print("=" * 80)

    for i, (defect_type, case_id) in enumerate(TEST_CASES):
        label = f"{defect_type}/{case_id}"
        print(f"\n[{i+1}/{len(TEST_CASES)}] {label}")

        defective_path = DATASET_DIR / "defective_skull" / defect_type / f"{case_id}.nrrd"
        gt_implant_path = DATASET_DIR / "implant" / defect_type / f"{case_id}.nrrd"

        if not defective_path.exists():
            print(f"  SKIP: {defective_path} not found")
            continue
        if not gt_implant_path.exists():
            print(f"  SKIP: ground truth {gt_implant_path} not found")
            continue

        case_dir = OUTPUT_DIR / defect_type / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Encoding NRRD ({defective_path.stat().st_size / 1024:.0f} KB)...")
        encoded = encode_nrrd_file(defective_path)

        prefix = f"cran2_bench_{defect_type}_{case_id}"

        print("  Submitting job...")
        t_submit = time.time()
        job_result = submit_job(api_key, encoded, prefix)
        job_id = job_result.get("id")
        print(f"  Job ID: {job_id}")

        print("  Waiting for completion...")
        job_timeout = 600 if i == 0 else 300
        try:
            result = wait_for_completion(api_key, job_id, timeout=job_timeout)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "defect_type": defect_type, "case_id": case_id,
                "status": "failed", "error": str(e)
            })
            continue

        t_total_runpod = time.time() - t_submit
        output = result.get("output", {})
        metadata = output.get("metadata", {})
        s3_results = output.get("results", {})

        inference_time = metadata.get("inference_time_seconds", -1)
        processing_time = metadata.get("processing_time_seconds", -1)

        print(f"  RunPod total: {t_total_runpod:.1f}s | inference: {inference_time:.3f}s | processing: {processing_time:.3f}s")

        nrrd_url = s3_results.get("implant_volume_nrrd")
        if not nrrd_url:
            print("  WARN: No NRRD URL in results, trying implant_nrrd...")
            nrrd_url = s3_results.get("implant_nrrd")

        download_time = 0
        pred_nrrd_path = case_dir / "pred_implant.nrrd"

        if nrrd_url:
            print(f"  Downloading NRRD from S3...")
            try:
                download_time = download_nrrd_from_s3(nrrd_url, pred_nrrd_path)
                print(f"  Download time: {download_time:.2f}s ({pred_nrrd_path.stat().st_size / 1024:.0f} KB)")
            except Exception as e:
                print(f"  Download failed: {e}")
                results.append({
                    "defect_type": defect_type, "case_id": case_id,
                    "status": "download_failed", "error": str(e),
                    "t_total_runpod": t_total_runpod,
                    "inference_time": inference_time,
                    "processing_time": processing_time,
                })
                continue
        else:
            print(f"  ERROR: No NRRD result URL found. Keys: {list(s3_results.keys())}")
            results.append({
                "defect_type": defect_type, "case_id": case_id,
                "status": "no_nrrd_url",
                "s3_keys": list(s3_results.keys()),
                "t_total_runpod": t_total_runpod,
            })
            continue

        t_end_to_end = t_total_runpod + download_time

        print("  Computing metrics...")
        try:
            metrics = compute_metrics(pred_nrrd_path, gt_implant_path, defective_path)
            print(f"  DSC={metrics['dice']:.4f} | bDSC@10mm={metrics['bdice_10mm']:.4f} | HD95={metrics['hd95_mm']:.2f}mm")
        except Exception as e:
            print(f"  Metrics failed: {e}")
            metrics = {"error": str(e)}

        case_result = {
            "defect_type": defect_type,
            "case_id": case_id,
            "status": "success",
            "t_total_runpod_s": round(t_total_runpod, 2),
            "t_inference_s": round(inference_time, 4),
            "t_processing_s": round(processing_time, 4),
            "t_download_s": round(download_time, 2),
            "t_end_to_end_s": round(t_end_to_end, 2),
            "s3_urls": s3_results,
            **{k: v for k, v in metrics.items() if k != "error"},
        }
        if "error" in metrics:
            case_result["metrics_error"] = metrics["error"]

        results.append(case_result)

        with open(case_dir / "result.json", "w") as f:
            json.dump(case_result, f, indent=2)

        partial_path = OUTPUT_DIR / "results_partial.json"
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r.get("status") == "success"]

    if successful:
        print(f"\n{'Case':<30} {'DSC':>6} {'bDSC':>6} {'HD95mm':>8} {'Infer':>7} {'Total':>7} {'DL':>5}")
        print("-" * 80)
        for r in successful:
            label = f"{r['defect_type']}/{r['case_id']}"
            print(
                f"{label:<30} {r.get('dice',0):.4f} {r.get('bdice_10mm',0):.4f} "
                f"{r.get('hd95_mm',0):>8.2f} {r.get('t_inference_s',0):>6.3f}s "
                f"{r.get('t_total_runpod_s',0):>6.1f}s {r.get('t_download_s',0):>4.1f}s"
            )

        dsc_vals = [r["dice"] for r in successful if "dice" in r]
        bdsc_vals = [r["bdice_10mm"] for r in successful if "bdice_10mm" in r]
        hd95_vals = [r["hd95_mm"] for r in successful if "hd95_mm" in r]
        infer_vals = [r["t_inference_s"] for r in successful]
        total_vals = [r["t_total_runpod_s"] for r in successful]
        dl_vals = [r["t_download_s"] for r in successful]
        e2e_vals = [r["t_end_to_end_s"] for r in successful]

        print("-" * 80)
        print(f"{'MEAN':<30} {np.mean(dsc_vals):.4f} {np.mean(bdsc_vals):.4f} "
              f"{np.mean(hd95_vals):>8.2f} {np.mean(infer_vals):>6.3f}s "
              f"{np.mean(total_vals):>6.1f}s {np.mean(dl_vals):>4.1f}s")
        print(f"{'STD':<30} {np.std(dsc_vals):.4f} {np.std(bdsc_vals):.4f} "
              f"{np.std(hd95_vals):>8.2f} {np.std(infer_vals):>6.3f}s "
              f"{np.std(total_vals):>6.1f}s {np.std(dl_vals):>4.1f}s")

        summary = {
            "endpoint_id": ENDPOINT_ID,
            "num_cases": len(successful),
            "num_failed": len(results) - len(successful),
            "metrics": {
                "dice": {"mean": float(np.mean(dsc_vals)), "std": float(np.std(dsc_vals)),
                         "min": float(np.min(dsc_vals)), "max": float(np.max(dsc_vals))},
                "bdice_10mm": {"mean": float(np.mean(bdsc_vals)), "std": float(np.std(bdsc_vals)),
                               "min": float(np.min(bdsc_vals)), "max": float(np.max(bdsc_vals))},
                "hd95_mm": {"mean": float(np.mean(hd95_vals)), "std": float(np.std(hd95_vals)),
                            "min": float(np.min(hd95_vals)), "max": float(np.max(hd95_vals))},
            },
            "timing": {
                "inference_s": {"mean": float(np.mean(infer_vals)), "std": float(np.std(infer_vals))},
                "runpod_total_s": {"mean": float(np.mean(total_vals)), "std": float(np.std(total_vals))},
                "download_s": {"mean": float(np.mean(dl_vals)), "std": float(np.std(dl_vals))},
                "end_to_end_s": {"mean": float(np.mean(e2e_vals)), "std": float(np.std(e2e_vals))},
            },
            "per_case": results,
        }

        summary_path = OUTPUT_DIR / "benchmark_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")
    else:
        print("No successful cases to report.")

    failed = [r for r in results if r.get("status") != "success"]
    if failed:
        print(f"\nFailed cases ({len(failed)}):")
        for r in failed:
            print(f"  {r['defect_type']}/{r['case_id']}: {r.get('status')} - {r.get('error', 'N/A')}")


if __name__ == "__main__":
    main()
