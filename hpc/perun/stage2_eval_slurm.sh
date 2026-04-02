#!/bin/bash
#SBATCH --job-name=s2-eval
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/logs/stage2_eval_%j.out
#SBATCH --error=/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/logs/stage2_eval_%j.err

set -eo pipefail

echo "=== Stage-2 Evaluation: Voxelization + Metrics ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate pcdiff
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="9.0"

# Build include path from nvidia pip packages (CUDA 12.4)
NVIDIA_BASE=$(python -c "import nvidia; import os; print(os.path.dirname(nvidia.__file__))")
INCLUDE_DIRS=""
for pkg in cuda_runtime cublas cusparse cusolver cufft curand nvtx cuda_nvrtc; do
  d="$NVIDIA_BASE/${pkg}/include"
  [ -d "$d" ] && INCLUDE_DIRS="$d:$INCLUDE_DIRS"
done
export CPLUS_INCLUDE_PATH="$INCLUDE_DIRS${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$INCLUDE_DIRS${C_INCLUDE_PATH:-}"
export PATH=$CUDA_HOME/bin:$PATH

cd /mnt/data/home/mamuke588/pcdiff-implant

# Build CUDA extensions if needed
if [ -d "pcdiff/modules/functional" ]; then
    cd pcdiff/modules/functional
    python setup.py build_ext --inplace 2>/dev/null || true
    cd /mnt/data/home/mamuke588/pcdiff-implant
fi

nvidia-smi

ABLATION_ROOT="/mnt/data/home/mamuke588/pcdiff-implant/benchmarking/runs/stage1_ablation/SkullBreak"
VOX_CHECKPOINT="voxelization/checkpoints/model_best.pt"
VOX_CONFIG="voxelization/configs/gen_skullbreak.yaml"
DATASET_CSV="pcdiff/datasets/SkullBreak/test.csv"

# Run stage-2 eval on all configs that have completed stage-1 sampling
python -u - <<'PYEOF'
import json
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("/mnt/data/home/mamuke588/pcdiff-implant")
sys.path.insert(0, str(ROOT / "pcdiff"))
sys.path.insert(0, str(ROOT / "voxelization"))
sys.path.insert(0, str(ROOT))

from voxelization.src.model import Encode2Points
from voxelization.src.utils import load_config, load_model_manual, filter_voxels_within_radius
from voxelization.src import config as vox_config
from voxelization.eval_metrics import bdc, dc, hd95

import diplib as dip
import nrrd

ABLATION_ROOT = ROOT / "benchmarking" / "runs" / "stage1_ablation" / "SkullBreak"
VOX_CHECKPOINT = ROOT / "voxelization" / "checkpoints" / "model_best.pt"
VOX_CONFIG_PATH = ROOT / "voxelization" / "configs" / "gen_skullbreak.yaml"
DEFAULT_CONFIG = ROOT / "voxelization" / "configs" / "default.yaml"
DATASET_ROOT = ROOT / "pcdiff" / "datasets" / "SkullBreak"

device = torch.device("cuda:0")

# Load voxelization model
print("Loading voxelization model...")
cfg = load_config(str(VOX_CONFIG_PATH), str(DEFAULT_CONFIG))
cfg["test"]["model_file"] = str(VOX_CHECKPOINT)
vox_model = Encode2Points(cfg).to(device)
state_dict = torch.load(str(VOX_CHECKPOINT), map_location="cpu", weights_only=False)
load_model_manual(state_dict["state_dict"], vox_model)
vox_model.eval()
generator = vox_config.get_generator(vox_model, cfg, device=device)
print("Voxelization model loaded.")

def voxelize_and_eval(sample_npy_path, shift_path, scale_path, gt_implant_nrrd_path):
    """Convert point cloud to voxels and compute metrics against ground truth."""
    sample = np.load(sample_npy_path)  # (num_ens, num_points, 3)
    shift = np.load(shift_path)
    scale = np.load(scale_path)
    
    # Use first ensemble member (or majority vote if ens>1)
    if sample.ndim == 3:
        # Take mean of ensemble
        pts = sample.mean(axis=0)  # (num_points, 3)
    else:
        pts = sample
    
    # Denormalize
    pts = pts * scale + shift
    
    # Convert to torch
    pts_tensor = torch.from_numpy(pts).float().unsqueeze(0).to(device)
    
    # Generate voxel volume using the voxelization model
    with torch.no_grad():
        data = {"inputs": pts_tensor}
        vox_output = generator.generate_mesh(data)
    
    # Get the predicted voxel grid
    if isinstance(vox_output, dict):
        pred_volume = vox_output.get("voxels", vox_output.get("occ", None))
        if pred_volume is None:
            # Try mesh-based approach
            pred_volume = vox_output
    
    # Load ground truth
    gt_volume, gt_header = nrrd.read(str(gt_implant_nrrd_path))
    gt_binary = (gt_volume > 0).astype(np.bool_)
    
    # Convert prediction to binary
    if isinstance(pred_volume, torch.Tensor):
        pred_volume = pred_volume.cpu().numpy()
    if pred_volume.ndim > 3:
        pred_volume = pred_volume.squeeze()
    pred_binary = (pred_volume > 0.5).astype(np.bool_)
    
    # Ensure same shape
    if pred_binary.shape != gt_binary.shape:
        # Resize pred to match gt
        from scipy.ndimage import zoom
        zoom_factors = [g/p for g, p in zip(gt_binary.shape, pred_binary.shape)]
        pred_binary = zoom(pred_binary.astype(float), zoom_factors, order=1) > 0.5
    
    # Compute metrics
    try:
        dice = dc(pred_binary, gt_binary)
    except:
        dice = 0.0
    try:
        bdice = bdc(pred_binary, gt_binary, border_dist=10)
    except:
        bdice = 0.0
    try:
        hd95_val = hd95(pred_binary, gt_binary)
    except:
        hd95_val = float("inf")
    
    return {"dice": dice, "bdice_10mm": bdice, "hd95_mm": hd95_val}


# Find all completed configs (all 25 case dirs have sample.npy)
configs = sorted([d.name for d in ABLATION_ROOT.iterdir() if d.is_dir()])
print(f"Found {len(configs)} config directories")

for config_name in configs:
    config_dir = ABLATION_ROOT / config_name / "stage1" / "syn"
    stage2_dir = ABLATION_ROOT / config_name / "stage2"
    
    if not config_dir.exists():
        print(f"SKIP {config_name}: no stage1/syn directory")
        continue
    
    case_dirs = sorted([d for d in config_dir.iterdir() if d.is_dir()])
    # Check all cases have sample.npy
    complete = all((d / "sample.npy").exists() for d in case_dirs)
    if not complete or len(case_dirs) < 20:
        print(f"SKIP {config_name}: only {len(case_dirs)} cases, incomplete")
        continue
    
    # Check if stage2 already done
    if (stage2_dir / "benchmark_summary.json").exists():
        print(f"SKIP {config_name}: stage2 already computed")
        continue
    
    print(f"\n=== Evaluating {config_name} ({len(case_dirs)} cases) ===")
    stage2_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    for case_dir in case_dirs:
        case_name = case_dir.name
        # Parse defect type and skull ID from case name
        # e.g., "bilateral027_surf" -> defect="bilateral", skull="027"
        
        # Find ground truth implant nrrd
        # Pattern: DATASET_ROOT/implant/<defect_type>/<skull_id>_surf.nrrd
        parts = case_name.replace("_surf", "")
        # Extract defect type and number
        for defect in ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]:
            if parts.startswith(defect):
                skull_num = parts[len(defect):]
                gt_path = DATASET_ROOT / "implant" / defect / f"{skull_num}.nrrd"
                break
        else:
            print(f"  WARN: cannot parse case {case_name}")
            continue
        
        if not gt_path.exists():
            print(f"  WARN: GT not found: {gt_path}")
            continue
        
        sample_path = case_dir / "sample.npy"
        shift_path = case_dir / "shift.npy"
        scale_path = case_dir / "scale.npy"
        
        try:
            t0 = time.time()
            metrics = voxelize_and_eval(sample_path, shift_path, scale_path, gt_path)
            elapsed = time.time() - t0
            metrics["case_name"] = case_name
            metrics["runtime_sec"] = round(elapsed, 2)
            all_metrics.append(metrics)
            print(f"  {case_name}: DSC={metrics['dice']:.4f} bDSC={metrics['bdice_10mm']:.4f} HD95={metrics['hd95_mm']:.2f}mm ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  ERROR {case_name}: {e}")
            all_metrics.append({"case_name": case_name, "dice": None, "bdice_10mm": None, "hd95_mm": None, "error": str(e)})
    
    # Summarize
    valid = [m for m in all_metrics if m.get("dice") is not None]
    if valid:
        summary = {
            "dice": {"count": len(valid), "mean": mean(m["dice"] for m in valid), "std": stdev(m["dice"] for m in valid) if len(valid)>1 else 0},
            "bdice_10mm": {"count": len(valid), "mean": mean(m["bdice_10mm"] for m in valid), "std": stdev(m["bdice_10mm"] for m in valid) if len(valid)>1 else 0},
            "hd95_mm": {"count": len(valid), "mean": mean(m["hd95_mm"] for m in valid), "std": stdev(m["hd95_mm"] for m in valid) if len(valid)>1 else 0},
            "case_count": len(valid),
            "config": config_name,
        }
    else:
        summary = {"error": "no valid results", "config": config_name}
    
    with open(stage2_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(stage2_dir / "per_case_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n  SUMMARY {config_name}:")
    if "error" not in summary:
        print(f"    DSC:  {summary['dice']['mean']:.4f} ± {summary['dice']['std']:.4f}")
        print(f"    bDSC: {summary['bdice_10mm']['mean']:.4f} ± {summary['bdice_10mm']['std']:.4f}")
        print(f"    HD95: {summary['hd95_mm']['mean']:.2f} ± {summary['hd95_mm']['std']:.2f} mm")

print("\n=== Stage-2 evaluation complete ===")
PYEOF

echo ""
echo "=== Done ==="
echo "End: $(date)"
