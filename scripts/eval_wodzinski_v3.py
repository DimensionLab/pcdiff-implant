#!/usr/bin/env python3
"""
DIM-88: Full evaluation of Wodzinski v3 models on ALL 115 SkullBreak test cases.
Evaluates per-defect-type and overall metrics (DSC, BDSC, HD95).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import nrrd
import torch
from pathlib import Path
from scipy.ndimage import binary_erosion, zoom
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(__file__))
from train_wodzinski_v3 import ResidualUNet3D, DEFECT_TYPES, DEFECT_TYPE_MAP

BASE_DIR = Path("/mnt/data/home/mamuke588/pcdiff-implant/datasets/SkullBreak")
RESOLUTION = 256


def load_volume(path):
    data, _ = nrrd.read(str(path))
    return (data > 0).astype(np.float32)


def get_voxel_spacing(path):
    try:
        header = nrrd.read_header(str(path))
        space = header.get('space directions', None)
        if space is not None:
            return [abs(space[i][i]) for i in range(3)]
    except:
        pass
    return [1.0, 1.0, 1.0]


def resize_volume(vol, target_shape):
    if vol.shape == target_shape:
        return vol
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, factors, order=1)


def compute_dice(pred, target):
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


def compute_boundary_dice(pred, target, border_mm=10, spacing=(1.0, 1.0, 1.0)):
    avg_spacing = np.mean(spacing)
    border_voxels = max(1, int(border_mm / avg_spacing))
    if np.sum(target) == 0 or np.sum(pred) == 0:
        return 0.0
    target_border = target - binary_erosion(target, iterations=border_voxels).astype(np.float32)
    pred_border = pred - binary_erosion(pred, iterations=border_voxels).astype(np.float32)
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
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')
    max_points = 10000
    if len(pred_points) > max_points:
        idx = np.random.choice(len(pred_points), max_points, replace=False)
        pred_points = pred_points[idx]
    if len(target_points) > max_points:
        idx = np.random.choice(len(target_points), max_points, replace=False)
        target_points = target_points[idx]
    pred_points = pred_points * np.array(spacing)
    target_points = target_points * np.array(spacing)
    dists = cdist(pred_points, target_points)
    hd95 = max(
        np.percentile(np.min(dists, axis=1), 95),
        np.percentile(np.min(dists, axis=0), 95)
    )
    return hd95


def get_all_test_cases(base_dir):
    base_dir = Path(base_dir)
    complete_dir = base_dir / "complete_skull"
    complete_ids = {f.stem for f in complete_dir.glob("*.nrrd")}

    cases = []
    for dt in DEFECT_TYPES:
        defect_dir = base_dir / "defective_skull" / dt
        if not defect_dir.exists():
            continue
        for f in sorted(defect_dir.glob("*.nrrd")):
            base_id = f.stem.split("_")[0] if "_" in f.stem else f.stem
            if base_id in complete_ids:
                cases.append((base_id, dt))
    return cases


def evaluate(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    in_channels = 2 if args.defect_type_channel else 1
    model = ResidualUNet3D(in_channels=in_channels, base_filters=args.base_filters, dropout=0.0)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {args.checkpoint}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    all_cases = get_all_test_cases(args.base_dir)

    if args.splits_json:
        with open(args.splits_json) as f:
            splits = json.load(f)
        test_ids = set(splits.get("test", []))
        if test_ids:
            all_cases = [(cid, dt) for cid, dt in all_cases if cid in test_ids]
            print(f"Using test split: {len(all_cases)} cases")
        else:
            print(f"No test split found, evaluating ALL {len(all_cases)} cases")
    else:
        print(f"No splits file, evaluating ALL {len(all_cases)} cases")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    by_defect = {dt: [] for dt in DEFECT_TYPES}

    for i, (case_id, defect_type) in enumerate(all_cases):
        complete_path = Path(args.base_dir) / "complete_skull" / f"{case_id}.nrrd"
        defect_path = Path(args.base_dir) / "defective_skull" / defect_type / f"{case_id}.nrrd"

        spacing = get_voxel_spacing(complete_path)
        complete = load_volume(complete_path)
        defective = load_volume(defect_path)
        gt_implant = np.clip(complete - defective, 0, 1)

        original_shape = defective.shape
        target_shape = (RESOLUTION,) * 3

        defective_resized = resize_volume(defective, target_shape)
        defective_resized = (defective_resized > 0.5).astype(np.float32)

        channels = [defective_resized[np.newaxis]]
        if args.defect_type_channel:
            dt_channel = np.full_like(defective_resized, DEFECT_TYPE_MAP[defect_type] / (len(DEFECT_TYPES) - 1))
            channels.append(dt_channel[np.newaxis])

        input_tensor = np.concatenate(channels, axis=0)
        input_tensor = torch.from_numpy(input_tensor[np.newaxis]).to(device)

        t0 = time.time()
        with torch.no_grad():
            pred = model(input_tensor)
            pred = torch.sigmoid(pred)
        inference_time = time.time() - t0

        pred_np = pred.cpu().numpy()[0, 0]
        pred_np = (pred_np > 0.5).astype(np.float32)

        if pred_np.shape != original_shape:
            pred_np = resize_volume(pred_np, original_shape)
            pred_np = (pred_np > 0.5).astype(np.float32)

        gt_bin = (gt_implant > 0.5).astype(np.float32)

        dsc = compute_dice(pred_np, gt_bin)
        bdsc = compute_boundary_dice(pred_np, gt_bin, spacing=spacing)
        hd95 = compute_hd95(pred_np, gt_bin, spacing=spacing)

        result = {
            "case_id": case_id,
            "defect_type": defect_type,
            "dsc": float(dsc),
            "bdsc": float(bdsc),
            "hd95": float(hd95),
            "inference_time": float(inference_time),
        }
        results.append(result)
        by_defect[defect_type].append(result)

        print(f"[{i+1}/{len(all_cases)}] {defect_type}/{case_id}: DSC={dsc:.4f} BDSC={bdsc:.4f} HD95={hd95:.2f}mm t={inference_time:.2f}s")

    print("\n" + "="*70)
    print("RESULTS BY DEFECT TYPE")
    print("="*70)

    summary = {}
    for dt in DEFECT_TYPES:
        if not by_defect[dt]:
            continue
        dscs = [r["dsc"] for r in by_defect[dt]]
        bdscs = [r["bdsc"] for r in by_defect[dt]]
        hd95s = [r["hd95"] for r in by_defect[dt] if r["hd95"] < float('inf')]
        times = [r["inference_time"] for r in by_defect[dt]]

        summary[dt] = {
            "n": len(dscs),
            "mean_dsc": float(np.mean(dscs)),
            "std_dsc": float(np.std(dscs)),
            "mean_bdsc": float(np.mean(bdscs)),
            "mean_hd95": float(np.mean(hd95s)) if hd95s else float('inf'),
            "mean_time": float(np.mean(times)),
        }

        print(f"\n{dt} (n={len(dscs)}):")
        print(f"  DSC:  {np.mean(dscs):.4f} ± {np.std(dscs):.4f}")
        print(f"  BDSC: {np.mean(bdscs):.4f}")
        print(f"  HD95: {np.mean(hd95s):.2f} mm" if hd95s else "  HD95: N/A")
        print(f"  Time: {np.mean(times):.3f}s")

    all_dscs = [r["dsc"] for r in results]
    all_bdscs = [r["bdsc"] for r in results]
    all_hd95s = [r["hd95"] for r in results if r["hd95"] < float('inf')]
    all_times = [r["inference_time"] for r in results]

    summary["overall"] = {
        "n": len(all_dscs),
        "mean_dsc": float(np.mean(all_dscs)),
        "std_dsc": float(np.std(all_dscs)),
        "mean_bdsc": float(np.mean(all_bdscs)),
        "mean_hd95": float(np.mean(all_hd95s)) if all_hd95s else float('inf'),
        "mean_time": float(np.mean(all_times)),
    }

    print(f"\n{'='*70}")
    print(f"OVERALL (n={len(all_dscs)}):")
    print(f"  DSC:  {np.mean(all_dscs):.4f} ± {np.std(all_dscs):.4f}")
    print(f"  BDSC: {np.mean(all_bdscs):.4f}")
    print(f"  HD95: {np.mean(all_hd95s):.2f} mm" if all_hd95s else "  HD95: N/A")
    print(f"  Time: {np.mean(all_times):.3f}s")

    with open(output_dir / "results.json", "w") as f:
        json.dump({"cases": results, "summary": summary}, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_dir", default=str(BASE_DIR))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits_json", default=None)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--defect_type_channel", action="store_true", default=True)
    parser.add_argument("--no_defect_type_channel", dest="defect_type_channel", action="store_false")
    args = parser.parse_args()
    evaluate(args)
