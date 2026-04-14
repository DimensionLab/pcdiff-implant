#!/usr/bin/env python3
"""
DIM-88: Evaluate Wodzinski baseline on SkullBreak test set.
Computes DSC, boundary DSC (10mm), HD95, and inference timing.
"""
import os
import sys
import json
import time
import numpy as np
import nrrd
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

# Add scripts dir to import model
sys.path.insert(0, os.path.dirname(__file__))
from train_wodzinski_baseline import ResidualUNet3D

BASE_DIR = "/mnt/data/home/mamuke588/pcdiff-implant/datasets/SkullBreak"
RESULTS_DIR = "/mnt/data/home/mamuke588/pcdiff-implant/wodzinski_baseline"
SPLITS_FILE = os.path.join(RESULTS_DIR, "splits.json")
CHECKPOINT = os.path.join(RESULTS_DIR, "model_best.pt")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "test_results")

RESOLUTION = 256
DEFECT_TYPE = "bilateral"


def load_volume(path):
    """Load a NRRD volume and binarize."""
    data, header = nrrd.read(path)
    return (data > 0).astype(np.float32)


def binarize_pred(pred, threshold=0.5):
    """Binarize model prediction."""
    return (pred > threshold).astype(np.float32)


def compute_dice(pred, target):
    """Compute Dice Score Coefficient."""
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    if union == 0:
        return 1.0 if np.sum(pred) == 0 and np.sum(target) == 0 else 0.0
    return 2.0 * intersection / union


def compute_boundary_dice(pred, target, border_mm=10, spacing=(1.0, 1.0, 1.0)):
    """Compute boundary Dice at given border distance in mm."""
    # Convert mm to voxels (approximate)
    avg_spacing = np.mean(spacing)
    border_voxels = int(border_mm / avg_spacing)
    
    # Get boundary of target
    if np.sum(target) == 0 or np.sum(pred) == 0:
        return 0.0
    
    from scipy.ndimage import binary_erosion
    target_border = target - binary_erosion(target, iterations=border_voxels)
    pred_border = pred - binary_erosion(pred, iterations=border_voxels)
    
    if np.sum(target_border) == 0:
        target_border = target
    if np.sum(pred_border) == 0:
        pred_border = pred
    
    intersection = np.sum(target_border * pred_border)
    total = np.sum(target_border) + np.sum(pred_border)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def compute_hd95(pred, target, spacing=(1.0, 1.0, 1.0)):
    """Compute 95th percentile Hausdorff distance in mm."""
    from scipy.spatial.distance import cdist
    
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')
    
    # Subsample if too many points
    max_points = 10000
    if len(pred_points) > max_points:
        idx = np.random.choice(len(pred_points), max_points, replace=False)
        pred_points = pred_points[idx]
    if len(target_points) > max_points:
        idx = np.random.choice(len(target_points), max_points, replace=False)
        target_points = target_points[idx]
    
    # Scale by spacing
    pred_points = pred_points * np.array(spacing)
    target_points = target_points * np.array(spacing)
    
    dists = cdist(pred_points, target_points)
    
    hd_pred_to_target = np.max(np.min(dists, axis=1))
    hd_target_to_pred = np.max(np.min(dists, axis=0))
    
    hd95 = max(
        np.percentile(np.min(dists, axis=1), 95),
        np.percentile(np.min(dists, axis=0), 95)
    )
    return hd95


def get_voxel_spacing(nrrd_path):
    """Extract voxel spacing from NRRD header."""
    header = nrrd.read_header(nrrd_path)
    space = header.get('space directions', None)
    if space is not None:
        # Diagonal elements give spacing
        spacing = [abs(space[i][i]) for i in range(3)]
        return spacing
    return [1.0, 1.0, 1.0]


def resize_volume(vol, target_shape):
    """Resize volume to target shape using scipy zoom."""
    from scipy.ndimage import zoom
    if vol.shape == target_shape:
        return vol
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, factors, order=1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load splits
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    test_ids = splits["test"]
    print(f"Test set: {len(test_ids)} cases")
    
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet3D(in_channels=1, base_filters=32)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {CHECKPOINT}")
    
    # Evaluate
    results = []
    total_inference_time = 0.0
    
    with torch.no_grad():
        for i, case_id in enumerate(test_ids):
            print(f"\n[{i+1}/{len(test_ids)}] Evaluating case {case_id}...")
            
            # Load defective skull
            defect_path = os.path.join(BASE_DIR, "defective_skull", DEFECT_TYPE, f"{case_id}.nrrd")
            implant_path = os.path.join(BASE_DIR, "implant", DEFECT_TYPE, f"{case_id}.nrrd")
            
            if not os.path.exists(defect_path) or not os.path.exists(implant_path):
                print(f"  SKIP: missing files")
                continue
            
            # Get spacing
            spacing = get_voxel_spacing(defect_path)
            
            # Load and preprocess
            defect_vol = load_volume(defect_path)
            implant_vol = load_volume(implant_path)
            
            # Resize to model resolution
            defect_vol = resize_volume(defect_vol, (RESOLUTION, RESOLUTION, RESOLUTION))
            implant_vol = resize_volume(implant_vol, (RESOLUTION, RESOLUTION, RESOLUTION))
            
            # Run inference
            input_tensor = torch.from_numpy(defect_vol).unsqueeze(0).unsqueeze(0).to(device)
            
            start = time.time()
            output = model(input_tensor)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            total_inference_time += elapsed
            
            # Post-process
            pred_vol = output.squeeze().cpu().numpy()
            pred_binary = binarize_pred(pred_vol)
            
            # Compute metrics
            dsc = compute_dice(pred_binary, implant_vol)
            bdsc = compute_boundary_dice(pred_binary, implant_vol, border_mm=10, spacing=spacing)
            hd95 = compute_hd95(pred_binary, implant_vol, spacing=spacing)
            
            result = {
                "case_id": case_id,
                "dsc": float(dsc),
                "bdsc_10mm": float(bdsc),
                "hd95_mm": float(hd95),
                "inference_time_s": float(elapsed),
            }
            results.append(result)
            print(f"  DSC={dsc:.4f} bDSC={bdsc:.4f} HD95={hd95:.2f}mm time={elapsed:.3f}s")
    
    # Summary
    if results:
        dscs = [r["dsc"] for r in results]
        bdscs = [r["bdsc_10mm"] for r in results]
        hd95s = [r["hd95_mm"] for r in results]
        times = [r["inference_time_s"] for r in results]
        
        summary = {
            "n_cases": len(results),
            "mean_dsc": float(np.mean(dscs)),
            "std_dsc": float(np.std(dscs)),
            "mean_bdsc_10mm": float(np.mean(bdscs)),
            "mean_hd95_mm": float(np.mean(hd95s)),
            "mean_inference_time_s": float(np.mean(times)),
            "total_inference_time_s": float(total_inference_time),
            "per_case": results,
        }
        
        summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"SUMMARY ({len(results)} test cases)")
        print(f"{'='*50}")
        print(f"Mean DSC: {np.mean(dscs):.4f} +/- {np.std(dscs):.4f}")
        print(f"Mean bDSC (10mm): {np.mean(bdscs):.4f}")
        print(f"Mean HD95: {np.mean(hd95s):.2f} mm")
        print(f"Mean inference time: {np.mean(times):.3f}s/case")
        print(f"Total inference time: {total_inference_time:.1f}s")
        print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
