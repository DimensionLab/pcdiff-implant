#!/usr/bin/env python3
"""
Runpod Serverless Handler for PCDiff + Voxelization Inference Pipeline

This handler processes skull implant generation requests:
1. Downloads defective skull point cloud from input (base64 or S3 URL)
2. Runs PCDiff to generate implant point cloud
3. Runs Voxelization to convert to volumetric mesh
4. Uploads results to AWS S3
5. Returns URLs to the generated files

Model Loading Priority:
1. Network Volume (/runpod-volume/models/) - Best for frequent model updates
2. Embedded in Docker image (/app/models/) - Default fallback

Environment Variables Required:
- AWS_ACCESS_KEY_ID: AWS access key for S3 uploads
- AWS_SECRET_ACCESS_KEY: AWS secret key for S3 uploads
- AWS_S3_BUCKET: S3 bucket name for results
- AWS_S3_REGION: AWS region (default: us-east-1)

Optional Environment Variables:
- PCDIFF_MODEL_PATH: Override path to PCDiff model
- VOXELIZATION_MODEL_PATH: Override path to voxelization model
"""

import os
import sys
import json
import base64
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
import uuid

import runpod
import numpy as np
import torch
import boto3
from botocore.exceptions import ClientError

# Add project paths
PROJECT_ROOT = Path("/app")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pcdiff"))
sys.path.insert(0, str(PROJECT_ROOT / "voxelization"))

# Network volume path (Runpod mounts network volumes here)
NETWORK_VOLUME_PATH = Path("/runpod-volume")

# Model paths with priority: Network Volume > Environment Variable > Embedded
def get_model_paths():
    """
    Determine model paths with priority:
    1. Network Volume (/runpod-volume/models/) - for easy updates
    2. Environment variable override
    3. Embedded in Docker image (/app/models/)
    """
    # Network volume paths
    nv_pcdiff = NETWORK_VOLUME_PATH / "models" / "pcdiff_best.pth"
    nv_vox = NETWORK_VOLUME_PATH / "models" / "voxelization_best.pt"
    
    # Embedded paths (fallback)
    embedded_pcdiff = Path("/app/models/pcdiff_best.pth")
    embedded_vox = Path("/app/models/voxelization_best.pt")
    
    # Determine PCDiff model path
    if os.environ.get('PCDIFF_MODEL_PATH'):
        pcdiff_path = Path(os.environ['PCDIFF_MODEL_PATH'])
    elif nv_pcdiff.exists():
        pcdiff_path = nv_pcdiff
        print(f"✓ Using PCDiff model from network volume: {nv_pcdiff}")
    else:
        pcdiff_path = embedded_pcdiff
        print(f"✓ Using embedded PCDiff model: {embedded_pcdiff}")
    
    # Determine Voxelization model path
    if os.environ.get('VOXELIZATION_MODEL_PATH'):
        vox_path = Path(os.environ['VOXELIZATION_MODEL_PATH'])
    elif nv_vox.exists():
        vox_path = nv_vox
        print(f"✓ Using Voxelization model from network volume: {nv_vox}")
    else:
        vox_path = embedded_vox
        print(f"✓ Using embedded Voxelization model: {embedded_vox}")
    
    return str(pcdiff_path), str(vox_path)

VOXELIZATION_CONFIG_PATH = "/app/voxelization/configs/gen_skullbreak.yaml"

# Global model instances (loaded once at startup)
pcdiff_model = None
voxelization_model = None
voxelization_generator = None
device = None
current_model_paths = None  # Track which models are loaded


def get_s3_client():
    """Create and return an S3 client using environment variables."""
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_S3_REGION', 'us-east-1')
    )


def upload_to_s3(local_path, s3_key):
    """Upload a file to S3 and return the URL."""
    bucket = os.environ.get('AWS_S3_BUCKET')
    if not bucket:
        raise ValueError("AWS_S3_BUCKET environment variable not set")
    
    s3_client = get_s3_client()
    s3_client.upload_file(str(local_path), bucket, s3_key)
    
    region = os.environ.get('AWS_S3_REGION', 'us-east-1')
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"
    return url


def download_from_s3(s3_url, local_path):
    """Download a file from S3 URL to local path."""
    # Parse S3 URL: https://bucket.s3.region.amazonaws.com/key
    # or s3://bucket/key
    if s3_url.startswith('s3://'):
        parts = s3_url[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
    else:
        # Parse HTTPS URL
        from urllib.parse import urlparse
        parsed = urlparse(s3_url)
        bucket = parsed.netloc.split('.')[0]
        key = parsed.path.lstrip('/')
    
    s3_client = get_s3_client()
    s3_client.download_file(bucket, key, str(local_path))


def load_models():
    """Load PCDiff and Voxelization models into GPU memory."""
    global pcdiff_model, voxelization_model, voxelization_generator, device, current_model_paths
    
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)
    
    # Get model paths (with network volume priority)
    pcdiff_path, vox_path = get_model_paths()
    current_model_paths = (pcdiff_path, vox_path)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ Warning: CUDA not available, using CPU (this will be slow)")
    
    # Load PCDiff model
    print(f"\nLoading PCDiff model from: {pcdiff_path}")
    if not Path(pcdiff_path).exists():
        raise FileNotFoundError(f"PCDiff model not found: {pcdiff_path}")
    pcdiff_model = load_pcdiff_model(pcdiff_path, device)
    print("✓ PCDiff model loaded successfully")
    
    # Load Voxelization model
    print(f"\nLoading Voxelization model from: {vox_path}")
    if not Path(vox_path).exists():
        raise FileNotFoundError(f"Voxelization model not found: {vox_path}")
    voxelization_model, voxelization_generator = load_voxelization_model(
        vox_path, 
        VOXELIZATION_CONFIG_PATH,
        device
    )
    print("✓ Voxelization model loaded successfully")
    
    print("\n" + "=" * 60)
    print("All models loaded successfully!")
    print("=" * 60)


def load_pcdiff_model(model_path, device):
    """Load the PCDiff diffusion model."""
    # Import from the pcdiff package
    # The test_completion module contains the Model class
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "pcdiff"))
    from test_completion import Model
    
    # Model configuration
    class ModelArgs:
        def __init__(self):
            self.nc = 3
            self.num_points = 30720
            self.num_nn = 3072
            self.attention = True
            self.dropout = 0.1
            self.embed_dim = 64
            self.sampling_method = 'ddpm'
            self.sampling_steps = 1000
            self.time_num = 1000
    
    args = ModelArgs()
    
    # Define betas for diffusion
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    
    model = Model(args, betas, 'mse', 'eps', 'fixedsmall',
                  width_mult=1.0, vox_res_mult=1.0)
    model = model.to(device)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state']
    
    # Handle DDP checkpoints
    if list(state_dict.keys())[0].startswith('model.module.'):
        state_dict = {k.replace('model.module.', 'model.'): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    return model


def load_voxelization_model(model_path, config_path, device):
    """Load the voxelization model."""
    from voxelization.src.model import Encode2Points
    from voxelization.src.utils import load_config, load_model_manual
    from voxelization.src import config as vox_config
    
    # Load config
    default_config = str(PROJECT_ROOT / 'voxelization/configs/default.yaml')
    cfg = load_config(config_path, default_config)
    cfg['test']['model_file'] = model_path
    
    # Load model
    model = Encode2Points(cfg).to(device)
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_manual(state_dict['state_dict'], model)
    model.eval()
    
    # Get generator
    generator = vox_config.get_generator(model, cfg, device=device)
    
    return model, generator


def run_pcdiff_inference(input_points, num_ens=1, sampling_steps=1000):
    """Run PCDiff to generate implant point cloud."""
    global pcdiff_model, device
    
    num_points = 30720
    num_nn = 3072
    sv_points = num_points - num_nn
    
    # Sample points from input
    if input_points.shape[0] < sv_points:
        raise ValueError(
            f"Input point cloud has {input_points.shape[0]} points, "
            f"but the model expects at least {sv_points} defective points."
        )
    
    idx = np.random.choice(input_points.shape[0], sv_points, replace=False)
    partial_points_raw = input_points[idx]
    
    # Normalize
    pc_min = partial_points_raw.min(axis=0)
    pc_max = partial_points_raw.max(axis=0)
    shift = (pc_min + pc_max) / 2.0
    scale = (pc_max - pc_min).max() / 2.0
    if scale <= 0:
        raise ValueError("Invalid scale computed from input point cloud bounding box.")
    scale = scale / 3.0
    
    partial_points = (partial_points_raw - shift) / scale
    
    # Convert to torch
    pc_input = torch.from_numpy(partial_points).float().unsqueeze(0)
    pc_input = pc_input.transpose(1, 2).to(device)
    pc_input = pc_input.repeat(num_ens, 1, 1)
    noise_shape = torch.Size([num_ens, 3, num_nn])
    
    # Generate implant
    with torch.no_grad():
        sample = pcdiff_model.gen_samples(
            pc_input, noise_shape, device,
            clip_denoised=False,
            sampling_method='ddpm',
            sampling_steps=sampling_steps
        )
        sample = sample.detach().cpu().numpy()
    
    # Separate defective and implant points
    completed_points = sample.transpose(0, 2, 1)
    implant_normalized = completed_points[:, sv_points:, :]
    implant_world = implant_normalized * scale + shift
    
    return implant_world, input_points, shift, scale, implant_normalized


def run_voxelization(implant_points, defective_points, implant_points_normalized):
    """Run voxelization to convert point cloud to mesh."""
    global voxelization_generator, device
    
    # Get first ensemble sample
    implant_pc = implant_points[0]
    implant_pc_normalized = implant_points_normalized[0]
    
    # Normalize to [0, 1]
    defective_normalized = (defective_points / 512.0).astype(np.float32)
    implant_pc_normalized = implant_pc_normalized.astype(np.float32)
    combined_points = np.concatenate([defective_normalized, implant_pc_normalized], axis=0).astype(np.float32)
    
    # Convert to torch
    inputs = torch.from_numpy(combined_points).float().unsqueeze(0).to(device)
    
    # Generate mesh
    with torch.no_grad():
        vertices, faces, points, normals, psr_grid = voxelization_generator.generate_mesh(inputs)
    
    # Convert PSR grid to binary volume
    psr_grid_np = psr_grid.detach().cpu().numpy()[0, :, :, :]
    volume_complete = np.zeros((512, 512, 512), dtype=np.uint8)
    volume_complete[psr_grid_np <= 0] = 1
    
    # Generate volume for defective skull alone for boolean subtraction
    defective_tensor = torch.from_numpy(defective_normalized).float().unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, _, _, psr_defective = voxelization_generator.generate_mesh(defective_tensor)
    psr_defective_np = psr_defective.detach().cpu().numpy()[0, :, :, :]
    volume_defective = np.zeros_like(volume_complete)
    volume_defective[psr_defective_np <= 0] = 1
    
    volume_implant = np.clip(
        volume_complete.astype(np.int16) - volume_defective.astype(np.int16), 
        0, 1
    ).astype(np.uint8)
    
    return vertices, faces, volume_complete, volume_implant, implant_pc


def export_results(output_dir, vertices, faces, volume_complete, volume_implant, 
                   implant_pc, defective_points, shift, scale, implant_points_normalized):
    """Export all output formats to output directory."""
    import trimesh
    import nrrd
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Export complete skull mesh
    vertices_scaled = vertices * 512.0
    mesh_complete = trimesh.Trimesh(vertices=vertices_scaled, faces=faces)
    
    ply_path = output_dir / 'skull_complete.ply'
    mesh_complete.export(str(ply_path))
    results['skull_complete_ply'] = ply_path
    
    stl_path = output_dir / 'skull_complete.stl'
    mesh_complete.export(str(stl_path))
    results['skull_complete_stl'] = stl_path
    
    # 2. Export implant point cloud
    implant_cloud = trimesh.PointCloud(vertices=implant_pc)
    implant_cloud.colors = np.tile([255, 100, 100, 255], (len(implant_pc), 1))
    ply_path = output_dir / 'implant_pc.ply'
    implant_cloud.export(str(ply_path))
    results['implant_pc_ply'] = ply_path
    
    # 3. Try to create implant mesh from convex hull
    try:
        implant_mesh = implant_cloud.convex_hull
        ply_path = output_dir / 'implant_only.ply'
        implant_mesh.export(str(ply_path))
        results['implant_only_ply'] = ply_path
        
        stl_path = output_dir / 'implant_only.stl'
        implant_mesh.export(str(stl_path))
        results['implant_only_stl'] = stl_path
    except Exception as e:
        print(f"Warning: Could not create implant mesh: {e}")
    
    # 4. Export NRRD volumes
    nrrd_path = output_dir / 'skull_complete.nrrd'
    nrrd.write(str(nrrd_path), volume_complete.astype(np.float32))
    results['skull_complete_nrrd'] = nrrd_path
    
    implant_nrrd_path = output_dir / 'implant_volume.nrrd'
    nrrd.write(str(implant_nrrd_path), volume_implant.astype(np.float32))
    results['implant_volume_nrrd'] = implant_nrrd_path
    
    # 5. Save numpy arrays
    np.save(output_dir / 'implant.npy', implant_pc)
    results['implant_npy'] = output_dir / 'implant.npy'
    
    np.save(output_dir / 'implant_normalized.npy', implant_points_normalized)
    results['implant_normalized_npy'] = output_dir / 'implant_normalized.npy'
    
    np.save(output_dir / 'shift.npy', shift)
    np.save(output_dir / 'scale.npy', np.array(scale))
    
    return results


def handler(event):
    """
    Runpod Serverless Handler Function
    
    Input format:
    {
        "input": {
            "defective_skull": <base64 encoded .npy file OR S3 URL>,
            "input_format": "base64" | "s3_url",
            "num_ensemble": 1,  # optional, default 1
            "sampling_steps": 1000,  # optional, default 1000
            "output_prefix": "job_123"  # optional, for S3 key prefix
        }
    }
    
    Output format:
    {
        "status": "success",
        "results": {
            "skull_complete_ply": "https://...",
            "skull_complete_stl": "https://...",
            "implant_only_ply": "https://...",
            "implant_only_stl": "https://...",
            "implant_pc_ply": "https://...",
            "skull_complete_nrrd": "https://...",
            "implant_volume_nrrd": "https://...",
            "implant_npy": "https://..."
        },
        "metadata": {
            "processing_time_seconds": 123.45,
            "num_implant_points": 3072,
            "num_ensemble": 1,
            "model_source": "network_volume" | "embedded"
        }
    }
    """
    global pcdiff_model, voxelization_model, current_model_paths
    
    print(f"Handler started with event: {json.dumps(event, default=str)[:500]}...")
    
    start_time = datetime.now()
    
    try:
        # Ensure models are loaded
        if pcdiff_model is None or voxelization_model is None:
            load_models()
        
        # Parse input
        job_input = event.get('input', {})
        
        defective_skull_data = job_input.get('defective_skull')
        input_format = job_input.get('input_format', 'base64')
        num_ensemble = job_input.get('num_ensemble', 1)
        sampling_steps = job_input.get('sampling_steps', 1000)
        output_prefix = job_input.get('output_prefix', str(uuid.uuid4())[:8])
        
        if not defective_skull_data:
            return {"error": "Missing 'defective_skull' in input"}
        
        # Create temp directory for this job
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            input_path = temp_dir / 'input.npy'
            output_dir = temp_dir / 'output'
            output_dir.mkdir()
            
            # Load input point cloud
            if input_format == 'base64':
                # Decode base64 to numpy file
                npy_bytes = base64.b64decode(defective_skull_data)
                with open(input_path, 'wb') as f:
                    f.write(npy_bytes)
            elif input_format == 's3_url':
                # Download from S3
                download_from_s3(defective_skull_data, input_path)
            else:
                return {"error": f"Unknown input_format: {input_format}"}
            
            # Load point cloud
            input_points = np.load(str(input_path))
            if len(input_points.shape) == 3:
                input_points = input_points[0]
            
            print(f"Loaded input point cloud: {input_points.shape}")
            
            # Step 1: Run PCDiff
            print("Running PCDiff inference...")
            implant_points, defective_points, shift, scale, implant_normalized = run_pcdiff_inference(
                input_points, 
                num_ens=num_ensemble,
                sampling_steps=sampling_steps
            )
            print(f"Generated implant: {implant_points.shape}")
            
            # Step 2: Run Voxelization
            print("Running voxelization...")
            vertices, faces, volume_complete, volume_implant, implant_pc = run_voxelization(
                implant_points, defective_points, implant_normalized
            )
            print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
            
            # Step 3: Export results
            print("Exporting results...")
            local_results = export_results(
                output_dir, vertices, faces, volume_complete, volume_implant,
                implant_pc, defective_points, shift, scale, implant_normalized
            )
            
            # Step 4: Upload to S3 (only the files needed by frontend)
            print("Uploading to S3...")
            s3_results = {}
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Only upload the files that the frontend expects:
            # - implant_npy: Generated implant point cloud
            # - implant_only_stl: Generated implant mesh
            required_outputs = ['implant_npy', 'implant_only_stl']
            
            for key, local_path in local_results.items():
                if key not in required_outputs:
                    print(f"  Skipping {key} (not required by frontend)")
                    continue
                    
                s3_key = f"inference_results/{output_prefix}/{timestamp}/{local_path.name}"
                try:
                    url = upload_to_s3(local_path, s3_key)
                    s3_results[key] = url
                    print(f"  Uploaded: {key} -> {url}")
                except Exception as e:
                    print(f"  Failed to upload {key}: {e}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Determine model source
            model_source = "network_volume" if current_model_paths and "runpod-volume" in current_model_paths[0] else "embedded"
            
            return {
                "status": "success",
                "results": s3_results,
                "metadata": {
                    "processing_time_seconds": processing_time,
                    "num_implant_points": implant_pc.shape[0],
                    "num_ensemble": num_ensemble,
                    "sampling_steps": sampling_steps,
                    "mesh_vertices": len(vertices),
                    "mesh_faces": len(faces),
                    "model_source": model_source,
                    "pcdiff_model": current_model_paths[0] if current_model_paths else "unknown",
                    "voxelization_model": current_model_paths[1] if current_model_paths else "unknown"
                }
            }
    
    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Load models at startup (cold start optimization)
print("=" * 60)
print("  PCDiff + Voxelization Serverless Worker")
print("=" * 60)

# CUDA diagnostics
print("=" * 60)
print("CUDA Environment Diagnostics:")
print(f"  CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")
print(f"  CUDA_PATH: {os.environ.get('CUDA_PATH', 'NOT SET')}")
print(f"  CUDA_VERSION: {os.environ.get('CUDA_VERSION', 'NOT SET')}")
print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  torch.version.cuda: {torch.version.cuda}")
    print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"  torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
print("=" * 60)

print(f"Network volume path: {NETWORK_VOLUME_PATH}")
print(f"Network volume exists: {NETWORK_VOLUME_PATH.exists()}")
if NETWORK_VOLUME_PATH.exists():
    print(f"Network volume contents: {list(NETWORK_VOLUME_PATH.iterdir()) if NETWORK_VOLUME_PATH.exists() else 'N/A'}")

try:
    load_models()
except Exception as e:
    print(f"⚠ Warning: Could not pre-load models: {e}")
    print("Models will be loaded on first request")

# Start the serverless worker
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
