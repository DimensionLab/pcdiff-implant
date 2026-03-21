# Single Inference Pipeline - Complete Skull Implant Generation

This script runs the complete PCDiff + Voxelization pipeline on a single input skull to generate an implant.

## Quick Start

```bash
# Example: Generate implant for a test skull
CUDA_VISIBLE_DEVICES=7 python3 run_single_inference.py \
    --input pcdiff/datasets/SkullBreak/defective_skull/random_2/077.npy \
    --pcdiff_model pcdiff/output/train_completion/LATEST/model_best.pth \
    --vox_model voxelization/out/skullbreak/model_best.pt \
    --output_dir single_inference_results \
    --name test_skull_077 \
    --gpu 0
```

## What It Does

The pipeline performs these steps:

1. **PCDiff Inference**: Generates implant point cloud from defective skull
   - Input: Defective skull point cloud (.npy or .nrrd)
   - Output: Generated implant point cloud (3072 points)
   - Method: Diffusion model with DDIM sampling

2. **Voxelization**: Converts point clouds to volumetric mesh
   - Input: Combined defective skull + implant point cloud
   - Output: Watertight mesh (vertices + faces)
   - Method: Neural implicit surface reconstruction

3. **Export**: Generates visualization and 3D printing files
   - `skull_complete.ply/stl` - Complete skull with implant
   - `implant_only.ply/stl` - Just the generated implant
   - `*_pc.ply` - Point clouds for visualization
   - `*.nrrd` - Volume files for medical imaging software
   - `*.npy` - Raw numpy arrays for further processing

## Usage

### Basic Usage

```bash
python3 run_single_inference.py \
    --input <path_to_defective_skull> \
    --pcdiff_model <path_to_pcdiff_checkpoint> \
    --vox_model <path_to_vox_checkpoint>
```

### Advanced Options

```bash
python3 run_single_inference.py \
    --input pcdiff/datasets/SkullBreak/defective_skull/bilateral/064.npy \
    --pcdiff_model pcdiff/output/train_completion/2025-10-23-19-35-15/epoch_5000.pth \
    --vox_model voxelization/out/skullbreak/model_best.pt \
    --output_dir my_inference_results \
    --name bilateral_064 \
    --num_ens 3 \
    --sampling_method ddim \
    --sampling_steps 50 \
    --dataset SkullBreak \
    --gpu 7 \
    --export_ply \
    --export_stl \
    --export_nrrd
```

## Command-Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--input` | Path to input defective skull (.npy or .nrrd) |
| `--pcdiff_model` | Path to trained PCDiff model checkpoint |
| `--vox_model` | Path to trained voxelization model checkpoint |

### Optional Arguments

**Input/Output:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `pipeline_output` | Output directory for results |
| `--name` | (from input) | Name for this inference run |

**PCDiff Settings:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--num_ens` | 1 | Number of ensemble samples |
| `--sampling_method` | `ddim` | Sampling method (ddpm/ddim) |
| `--sampling_steps` | 50 | Number of diffusion steps |

**Voxelization Settings:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--vox_config` | `voxelization/configs/gen_skullbreak.yaml` | Config file |

**Dataset Settings:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `SkullBreak` | Dataset type (SkullBreak/SkullFix) |
| `--num_points` | 30720 | Number of points in defective skull |
| `--num_nn` | 3072 | Number of points in implant |

**Export Options:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--export_ply` | True | Export PLY files for visualization |
| `--export_stl` | True | Export STL files for 3D printing |
| `--export_nrrd` | True | Export NRRD volume files |

**GPU Settings:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu` | 7 | GPU ID to use (use GPU 7 if training is running) |

## Input Formats

### Option 1: Point Cloud (.npy)

```python
# Load existing point cloud
points = np.load('defective_skull.npy')  # Shape: (N, 3)
```

### Option 2: Volume (.nrrd)

```python
# Script will automatically extract surface points
volume, header = nrrd.read('defective_skull.nrrd')
```

## Output Files

The script generates a directory structure:

```
output_dir/
└── <name>/
    ├── skull_complete.ply       # Complete skull mesh (for viewing)
    ├── skull_complete.stl       # Complete skull mesh (for 3D printing)
    ├── implant_only.ply         # Implant mesh (for viewing)
    ├── implant_only.stl         # Implant mesh (for 3D printing)
    ├── defective_skull_pc.ply   # Input point cloud (gray)
    ├── implant_pc.ply           # Generated implant point cloud (red)
    ├── skull_complete.nrrd      # Volume file for medical imaging
    ├── defective_skull.npy      # Raw input points
    ├── implant.npy              # Raw implant points
    ├── shift.npy                # Normalization shift
    └── scale.npy                # Normalization scale
```

## Viewing Results

### Option 1: Web Viewer (Recommended)

```bash
# Convert for web viewing
python3 pcdiff/utils/convert_to_web.py pipeline_output/test_skull_077

# Start web viewer
cd web_viewer
./start_dev.sh

# Open http://localhost:5173
```

### Option 2: MeshLab / CloudCompare

```bash
# Open PLY files directly
meshlab skull_complete.ply
cloudcompare defective_skull_pc.ply implant_pc.ply
```

### Option 3: 3D Printing Slicer

```bash
# Import STL files
prusaslicer skull_complete.stl
# or
cura implant_only.stl
```

## Example Workflows

### 1. Quick Test with Single Sample

```bash
# Fast inference with DDIM (50 steps, ~30 seconds on H100)
CUDA_VISIBLE_DEVICES=7 python3 run_single_inference.py \
    --input pcdiff/datasets/SkullBreak/defective_skull/random_2/077.npy \
    --pcdiff_model pcdiff/output/train_completion/latest/model_best.pth \
    --vox_model voxelization/out/skullbreak/model_best.pt \
    --gpu 0
```

### 2. High-Quality with Ensemble

```bash
# Generate 5 samples, use best one
CUDA_VISIBLE_DEVICES=7 python3 run_single_inference.py \
    --input pcdiff/datasets/SkullBreak/defective_skull/bilateral/064.npy \
    --pcdiff_model pcdiff/output/train_completion/latest/model_best.pth \
    --vox_model voxelization/out/skullbreak/model_best.pt \
    --num_ens 5 \
    --sampling_steps 100 \
    --gpu 0
```

### 3. Clinical Use Case

```bash
# Generate implant for patient CT scan
python3 run_single_inference.py \
    --input patient_data/skull_defect.nrrd \
    --pcdiff_model models/pcdiff_trained.pth \
    --vox_model models/vox_trained.pt \
    --name patient_001 \
    --export_stl \
    --export_nrrd \
    --gpu 0
```

## GPU Usage During Training

If you have training jobs running on GPUs 0-6 and voxelization on GPU 7:

```bash
# Use GPU 7 (has lowest utilization per nvidia-smi)
CUDA_VISIBLE_DEVICES=7 python3 run_single_inference.py \
    --input ... \
    --gpu 0  # This becomes GPU 7 due to CUDA_VISIBLE_DEVICES
```

OR directly specify:

```bash
# Let script use GPU 7
python3 run_single_inference.py \
    --input ... \
    --gpu 7  # Use GPU 7 directly
```

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Expected memory usage:**
- PCDiff inference: ~3-5GB
- Voxelization: ~2-3GB
- Total: ~5-8GB on GPU 7

## Performance

Typical inference times on NVIDIA H100:

| Stage | DDIM (50 steps) | DDPM (1000 steps) |
|-------|-----------------|-------------------|
| PCDiff | ~20-30s | ~5-8min |
| Voxelization | ~5-10s | ~5-10s |
| Export | ~5s | ~5s |
| **Total** | **~30-45s** | **~6-9min** |

## Troubleshooting

### OOM Error on GPU

```bash
# Reduce batch size or use different GPU
CUDA_VISIBLE_DEVICES=0 python3 run_single_inference.py ...
```

### Model Loading Error

```bash
# Check model paths exist
ls pcdiff/output/train_completion/
ls voxelization/out/skullbreak/
```

### Import Errors

```bash
# Make sure you're in project root
cd /home/michaltakac/pcdiff-implant
python3 run_single_inference.py ...
```

### Poor Quality Results

```bash
# Increase sampling steps and use ensemble
python3 run_single_inference.py \
    --num_ens 5 \
    --sampling_steps 100 \
    ...
```

## Integration with Web Viewer

After running inference, you can immediately view results in the web viewer:

```bash
# 1. Run inference
python3 run_single_inference.py --input ... --output_dir results --name test_001

# 2. Copy to web viewer directory structure
mkdir -p inference_results_ddim50/syn/test_001
cp -r results/test_001/* inference_results_ddim50/syn/test_001/

# 3. Convert for web (if needed)
python3 pcdiff/utils/convert_to_web.py inference_results_ddim50/syn/test_001

# 4. Start web viewer
cd web_viewer && ./start_dev.sh

# 5. Open http://localhost:5173 and select test_001
```

## Advanced: Batch Processing

To process multiple skulls:

```bash
#!/bin/bash
# process_batch.sh

for skull in pcdiff/datasets/SkullBreak/defective_skull/random_2/*.npy; do
    name=$(basename "$skull" .npy)
    echo "Processing $name..."
    
    CUDA_VISIBLE_DEVICES=7 python3 run_single_inference.py \
        --input "$skull" \
        --pcdiff_model pcdiff/output/train_completion/latest/model_best.pth \
        --vox_model voxelization/out/skullbreak/model_best.pt \
        --output_dir batch_results \
        --name "$name" \
        --gpu 0
    
    echo "Completed $name"
done
```

## Notes

- The script uses GPU 7 by default (safest if training is running)
- All outputs are saved in the specified output directory
- Point clouds are automatically normalized/denormalized
- Ensemble mode generates multiple samples and uses the first one for voxelization
- STL files use convex hull for mesh reconstruction (fast but simplified)
- For production use, consider implementing more advanced surface reconstruction

## Related Documentation

- **Web Viewer:** `web_viewer/README.md`
- **PCDiff Training:** `pcdiff/README.md`
- **Voxelization Training:** `voxelization/README.md`
- **Deployment:** `web_viewer/DEPLOYMENT.md`

