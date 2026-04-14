#!/usr/bin/env python3
"""
Runpod Serverless Handler for CrAInial Implant Generation Pipeline

Supports two inference backends:
1. PCDiff + Voxelization (job_type: "full") — two-stage point cloud diffusion pipeline
2. Wodzinski "cran-2" (job_type: "cran2") — direct volumetric UNet (faster, higher quality)

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
- WODZINSKI_MODEL_PATH: Override path to Wodzinski (cran-2) model
"""

import base64
import json
import os
import sys
import tempfile
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import boto3
import numpy as np
import runpod
import torch
from botocore.exceptions import ClientError

# Add project paths
PROJECT_ROOT = Path("/app")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pcdiff"))
sys.path.insert(0, str(PROJECT_ROOT / "voxelization"))
sys.path.insert(0, str(PROJECT_ROOT / "runpod_serverless"))

# Network volume path (Runpod mounts network volumes here)
NETWORK_VOLUME_PATH = Path("/runpod-volume")

# Available PCDiff model variants
PCDIFF_MODEL_VARIANTS = {
    "best": "pcdiff_best.pth",
    "latest": "pcdiff_latest.pth",
}
DEFAULT_PCDIFF_MODEL = "best"

# Wodzinski (cran-2) model config
WODZINSKI_MODEL_FILENAME = "wodzinski_best.pt"
WODZINSKI_BASE_FILTERS = 32


# Model paths with priority: Network Volume > Environment Variable > Embedded
def get_model_paths(pcdiff_variant: str = None):
    """
    Determine model paths with priority:
    1. Network Volume (/runpod-volume/models/) - for easy updates
    2. Environment variable override
    3. Embedded in Docker image (/app/models/)

    Args:
        pcdiff_variant: Which PCDiff model to use ("best" or "latest").
                       Defaults to "best" if not specified.
    """
    # Resolve variant
    if pcdiff_variant is None:
        pcdiff_variant = DEFAULT_PCDIFF_MODEL

    if pcdiff_variant not in PCDIFF_MODEL_VARIANTS:
        print(f"⚠ Unknown PCDiff variant '{pcdiff_variant}', falling back to '{DEFAULT_PCDIFF_MODEL}'")
        pcdiff_variant = DEFAULT_PCDIFF_MODEL

    pcdiff_filename = PCDIFF_MODEL_VARIANTS[pcdiff_variant]

    # Network volume paths
    nv_pcdiff = NETWORK_VOLUME_PATH / "models" / pcdiff_filename
    nv_vox = NETWORK_VOLUME_PATH / "models" / "voxelization_best.pt"

    # Embedded paths (fallback)
    embedded_pcdiff = Path("/app/models") / pcdiff_filename
    embedded_vox = Path("/app/models/voxelization_best.pt")

    # Determine PCDiff model path
    if os.environ.get("PCDIFF_MODEL_PATH"):
        pcdiff_path = Path(os.environ["PCDIFF_MODEL_PATH"])
        print(f"✓ Using PCDiff model from env override: {pcdiff_path}")
    elif nv_pcdiff.exists():
        pcdiff_path = nv_pcdiff
        print(f"✓ Using PCDiff model '{pcdiff_variant}' from network volume: {nv_pcdiff}")
    elif embedded_pcdiff.exists():
        pcdiff_path = embedded_pcdiff
        print(f"✓ Using embedded PCDiff model '{pcdiff_variant}': {embedded_pcdiff}")
    else:
        # Fallback to best if requested variant not found
        fallback_path = Path("/app/models/pcdiff_best.pth")
        print(f"⚠ PCDiff model '{pcdiff_variant}' not found, falling back to: {fallback_path}")
        pcdiff_path = fallback_path

    # Determine Voxelization model path
    if os.environ.get("VOXELIZATION_MODEL_PATH"):
        vox_path = Path(os.environ["VOXELIZATION_MODEL_PATH"])
    elif nv_vox.exists():
        vox_path = nv_vox
        print(f"✓ Using Voxelization model from network volume: {nv_vox}")
    else:
        vox_path = embedded_vox
        print(f"✓ Using embedded Voxelization model: {embedded_vox}")

    return str(pcdiff_path), str(vox_path), pcdiff_variant


VOXELIZATION_CONFIG_PATH = "/app/voxelization/configs/gen_skullbreak.yaml"

# Global model instances (loaded once at startup)
pcdiff_model = None
voxelization_model = None
voxelization_generator = None
wodzinski_model = None  # Wodzinski (cran-2) model
device = None
current_model_paths = None  # Track which models are loaded
current_pcdiff_variant = None  # Track which PCDiff variant is loaded


def get_s3_client():
    """Create and return an S3 client using environment variables."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_S3_REGION", "us-east-1"),
    )


def upload_to_s3(local_path, s3_key):
    """Upload a file to S3 and return the URL."""
    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise ValueError("AWS_S3_BUCKET environment variable not set")

    s3_client = get_s3_client()
    s3_client.upload_file(str(local_path), bucket, s3_key)

    region = os.environ.get("AWS_S3_REGION", "us-east-1")
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"
    return url


def download_from_s3(s3_url, local_path):
    """Download a file from S3 URL to local path."""
    # Parse S3 URL: https://bucket.s3.region.amazonaws.com/key
    # or s3://bucket/key
    if s3_url.startswith("s3://"):
        parts = s3_url[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
    else:
        # Parse HTTPS URL
        from urllib.parse import urlparse

        parsed = urlparse(s3_url)
        bucket = parsed.netloc.split(".")[0]
        key = parsed.path.lstrip("/")

    s3_client = get_s3_client()
    s3_client.download_file(bucket, key, str(local_path))


def load_models(pcdiff_variant: str = None):
    """Load PCDiff and Voxelization models into GPU memory.

    Args:
        pcdiff_variant: Which PCDiff model to use ("best" or "latest").
                       If None, uses the default ("best").
    """
    global pcdiff_model, voxelization_model, voxelization_generator, device, current_model_paths, current_pcdiff_variant

    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    # Get model paths (with network volume priority)
    pcdiff_path, vox_path, resolved_variant = get_model_paths(pcdiff_variant)
    current_model_paths = (pcdiff_path, vox_path)
    current_pcdiff_variant = resolved_variant

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠ Warning: CUDA not available, using CPU (this will be slow)")

    # Load PCDiff model
    print(f"\nLoading PCDiff model from: {pcdiff_path}")
    if not Path(pcdiff_path).exists():
        raise FileNotFoundError(f"PCDiff model not found: {pcdiff_path}")
    pcdiff_model = load_pcdiff_model(pcdiff_path, device)
    print(f"✓ PCDiff model '{resolved_variant}' loaded successfully")

    # Load Voxelization model
    print(f"\nLoading Voxelization model from: {vox_path}")
    if not Path(vox_path).exists():
        raise FileNotFoundError(f"Voxelization model not found: {vox_path}")
    voxelization_model, voxelization_generator = load_voxelization_model(vox_path, VOXELIZATION_CONFIG_PATH, device)
    print("✓ Voxelization model loaded successfully")

    print("\n" + "=" * 60)
    print(f"All models loaded successfully! (PCDiff variant: {resolved_variant})")
    print("=" * 60)


def reload_pcdiff_model(pcdiff_variant: str):
    """Reload only the PCDiff model with a different variant.

    This is more efficient than reloading all models when only
    switching the PCDiff variant.
    """
    global pcdiff_model, current_model_paths, current_pcdiff_variant, device

    if pcdiff_variant not in PCDIFF_MODEL_VARIANTS:
        raise ValueError(f"Unknown PCDiff variant: {pcdiff_variant}. Available: {list(PCDIFF_MODEL_VARIANTS.keys())}")

    print(f"\n{'=' * 60}")
    print(f"Switching PCDiff model from '{current_pcdiff_variant}' to '{pcdiff_variant}'...")
    print("=" * 60)

    pcdiff_path, vox_path, resolved_variant = get_model_paths(pcdiff_variant)

    # Load new PCDiff model
    print(f"Loading PCDiff model from: {pcdiff_path}")
    if not Path(pcdiff_path).exists():
        raise FileNotFoundError(f"PCDiff model not found: {pcdiff_path}")

    pcdiff_model = load_pcdiff_model(pcdiff_path, device)
    current_model_paths = (pcdiff_path, current_model_paths[1])
    current_pcdiff_variant = resolved_variant

    print(f"✓ PCDiff model '{resolved_variant}' loaded successfully")
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
            self.sampling_method = "ddpm"
            self.sampling_steps = 1000
            self.time_num = 1000

    args = ModelArgs()

    # Define betas for diffusion
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)

    model = Model(args, betas, "mse", "eps", "fixedsmall", width_mult=1.0, vox_res_mult=1.0)
    model = model.to(device)
    model.eval()

    # Load checkpoint
    # weights_only=False is needed for PyTorch 2.6+ as checkpoints contain numpy arrays
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]

    # Handle DDP checkpoints
    if list(state_dict.keys())[0].startswith("model.module."):
        state_dict = {k.replace("model.module.", "model."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    return model


def load_voxelization_model(model_path, config_path, device, resolution=512):
    """Load the voxelization model with configurable resolution."""
    from voxelization.src import config as vox_config
    from voxelization.src.model import Encode2Points
    from voxelization.src.utils import load_config, load_model_manual

    # Load config
    default_config = str(PROJECT_ROOT / "voxelization/configs/default.yaml")
    cfg = load_config(config_path, default_config)
    cfg["test"]["model_file"] = model_path

    # Set voxelization resolution
    cfg["generation"]["psr_resolution"] = resolution
    cfg["generation"]["psr_sigma"] = 2 if resolution >= 512 else 1

    # Load model
    model = Encode2Points(cfg).to(device)
    # weights_only=False is needed for PyTorch 2.6+ as checkpoints contain numpy arrays
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    load_model_manual(state_dict["state_dict"], model)
    model.eval()

    # Get generator with custom resolution
    generator = vox_config.get_generator(model, cfg, device=device)

    return model, generator


# Cache for voxelization generators at different resolutions
voxelization_generators_cache = {}


def get_voxelization_generator(resolution=512):
    """Get or create a voxelization generator for a specific resolution."""
    global voxelization_generators_cache, device

    if resolution in voxelization_generators_cache:
        print(f"Using cached voxelization generator (resolution: {resolution}³)")
        return voxelization_generators_cache[resolution]

    print(f"Creating voxelization generator (resolution: {resolution}³)")
    _, vox_path, _ = get_model_paths()
    _, generator = load_voxelization_model(vox_path, VOXELIZATION_CONFIG_PATH, device, resolution=resolution)
    voxelization_generators_cache[resolution] = generator
    return generator


def run_pcdiff_inference(input_points, num_ens=1, sampling_steps=1000, sampling_method="ddpm"):
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
            pc_input,
            noise_shape,
            device,
            clip_denoised=False,
            sampling_method=sampling_method,
            sampling_steps=sampling_steps,
        )
        sample = sample.detach().cpu().numpy()

    # Separate defective and implant points
    completed_points = sample.transpose(0, 2, 1)
    implant_normalized = completed_points[:, sv_points:, :]
    implant_world = implant_normalized * scale + shift

    return implant_world, input_points, shift, scale, implant_normalized


def run_voxelization(implant_points, defective_points, implant_points_normalized, resolution=512):
    """Run voxelization to convert point cloud to mesh.

    Args:
        implant_points: World-space implant points (num_ensemble, N, 3)
        defective_points: World-space defective skull points (M, 3)
        implant_points_normalized: Normalized implant points (num_ensemble, N, 3)
        resolution: PSR grid resolution (128, 256, 512, or 1024)
    """
    global device

    # Get generator for this resolution
    generator = get_voxelization_generator(resolution)

    # Get first ensemble sample
    implant_pc = implant_points[0]
    implant_pc_normalized = implant_points_normalized[0]

    # Normalize to [0, 1]
    defective_normalized = (defective_points / 512.0).astype(np.float32)
    implant_pc_normalized = implant_pc_normalized.astype(np.float32)
    combined_points = np.concatenate([defective_normalized, implant_pc_normalized], axis=0).astype(np.float32)

    print(
        f"Voxelization input: {len(defective_normalized)} defective + {len(implant_pc_normalized)} implant = {len(combined_points)} total (resolution: {resolution}³)"
    )

    # Convert to torch
    inputs = torch.from_numpy(combined_points).float().unsqueeze(0).to(device)

    # Generate mesh
    with torch.no_grad():
        vertices, faces, points, normals, psr_grid = generator.generate_mesh(inputs)

    # Convert PSR grid to binary volume
    psr_grid_np = psr_grid.detach().cpu().numpy()[0, :, :, :]
    grid_shape = psr_grid_np.shape  # matches resolution³
    volume_complete = np.zeros(grid_shape, dtype=np.uint8)
    volume_complete[psr_grid_np <= 0] = 1

    # Generate volume for defective skull alone for boolean subtraction
    defective_tensor = torch.from_numpy(defective_normalized).float().unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, _, _, psr_defective = generator.generate_mesh(defective_tensor)
    psr_defective_np = psr_defective.detach().cpu().numpy()[0, :, :, :]
    volume_defective = np.zeros_like(volume_complete)
    volume_defective[psr_defective_np <= 0] = 1

    volume_implant = np.clip(volume_complete.astype(np.int16) - volume_defective.astype(np.int16), 0, 1).astype(
        np.uint8
    )

    return vertices, faces, volume_complete, volume_implant, implant_pc


def run_revoxelization_only(implant_points_world, defective_points_world, resolution=512):
    """Run voxelization only on existing implant point cloud.

    This skips PCDiff inference and only runs voxelization with custom resolution.
    Used for re-generating mesh with different detail level from existing point cloud.

    Args:
        implant_points_world: World-space implant points (N, 3)
        defective_points_world: World-space defective skull points (M, 3)
        resolution: PSR grid resolution (128, 256, 512, or 1024)

    Returns:
        vertices, faces, volume_complete, volume_implant, implant_pc
    """
    global device

    # Get generator for this resolution
    generator = get_voxelization_generator(resolution)

    # Compute normalization from combined bounding box
    all_points = np.concatenate([defective_points_world, implant_points_world], axis=0)
    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    scale = (bbox_max - bbox_min).max()
    shift = center - 0.5 * scale

    # Normalize both point clouds to [0, 1]
    defective_normalized = ((defective_points_world - shift) / scale).astype(np.float32)
    implant_normalized = ((implant_points_world - shift) / scale).astype(np.float32)

    combined_points = np.concatenate([defective_normalized, implant_normalized], axis=0).astype(np.float32)

    print(
        f"Re-voxelization input: {len(defective_normalized)} defective + {len(implant_normalized)} implant = {len(combined_points)} total (resolution: {resolution}³)"
    )

    # Convert to torch
    inputs = torch.from_numpy(combined_points).float().unsqueeze(0).to(device)

    # Generate mesh
    with torch.no_grad():
        vertices, faces, points, normals, psr_grid = generator.generate_mesh(inputs)

    # Scale vertices back to world space
    vertices_world = vertices * scale + shift

    # Convert PSR grid to binary volume
    psr_grid_np = psr_grid.detach().cpu().numpy()[0, :, :, :]
    grid_shape = psr_grid_np.shape  # matches resolution³
    volume_complete = np.zeros(grid_shape, dtype=np.uint8)
    volume_complete[psr_grid_np <= 0] = 1

    # Generate volume for defective skull alone for boolean subtraction
    defective_tensor = torch.from_numpy(defective_normalized).float().unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, _, _, psr_defective = generator.generate_mesh(defective_tensor)
    psr_defective_np = psr_defective.detach().cpu().numpy()[0, :, :, :]
    volume_defective = np.zeros_like(volume_complete)
    volume_defective[psr_defective_np <= 0] = 1

    volume_implant = np.clip(volume_complete.astype(np.int16) - volume_defective.astype(np.int16), 0, 1).astype(
        np.uint8
    )

    return vertices_world, faces, volume_complete, volume_implant, implant_points_world, shift, scale


def postprocess_mesh(mesh, smoothing_iterations=0, close_holes=False):
    """Apply post-processing to a trimesh mesh.

    Args:
        mesh: trimesh.Trimesh object
        smoothing_iterations: Laplacian smoothing iterations (0 = disabled, 1-100)
        close_holes: Whether to fill holes and repair the mesh

    Returns:
        Processed trimesh.Trimesh (modified in-place and returned)
    """
    import trimesh

    if close_holes:
        print(f"Closing holes in mesh ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        print(f"After hole closing: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    if smoothing_iterations > 0:
        iterations = max(1, min(smoothing_iterations, 100))
        print(f"Smoothing mesh ({iterations} iterations)")
        trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=iterations)
        print(f"After smoothing: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    return mesh


def export_results(
    output_dir,
    vertices,
    faces,
    volume_complete,
    volume_implant,
    implant_pc,
    defective_points,
    shift,
    scale,
    implant_points_normalized,
    smoothing_iterations=0,
    close_holes=False,
):
    """Export all output formats to output directory."""
    import nrrd
    import trimesh

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Export complete skull mesh
    vertices_scaled = vertices * 512.0
    mesh_complete = trimesh.Trimesh(vertices=vertices_scaled, faces=faces)

    ply_path = output_dir / "skull_complete.ply"
    mesh_complete.export(str(ply_path))
    results["skull_complete_ply"] = ply_path

    stl_path = output_dir / "skull_complete.stl"
    mesh_complete.export(str(stl_path))
    results["skull_complete_stl"] = stl_path

    # 2. Export implant point cloud
    implant_cloud = trimesh.PointCloud(vertices=implant_pc)
    implant_cloud.colors = np.tile([255, 100, 100, 255], (len(implant_pc), 1))
    ply_path = output_dir / "implant_pc.ply"
    implant_cloud.export(str(ply_path))
    results["implant_pc_ply"] = ply_path

    # 3. Try to create implant mesh from convex hull
    try:
        implant_mesh = implant_cloud.convex_hull

        # Apply post-processing to implant mesh
        implant_mesh = postprocess_mesh(
            implant_mesh, smoothing_iterations=smoothing_iterations, close_holes=close_holes
        )

        ply_path = output_dir / "implant_only.ply"
        implant_mesh.export(str(ply_path))
        results["implant_only_ply"] = ply_path

        stl_path = output_dir / "implant_only.stl"
        implant_mesh.export(str(stl_path))
        results["implant_only_stl"] = stl_path
    except Exception as e:
        print(f"Warning: Could not create implant mesh: {e}")

    # 4. Export NRRD volumes
    nrrd_path = output_dir / "skull_complete.nrrd"
    nrrd.write(str(nrrd_path), volume_complete.astype(np.float32))
    results["skull_complete_nrrd"] = nrrd_path

    implant_nrrd_path = output_dir / "implant_volume.nrrd"
    nrrd.write(str(implant_nrrd_path), volume_implant.astype(np.float32))
    results["implant_volume_nrrd"] = implant_nrrd_path

    # 5. Save numpy arrays
    np.save(output_dir / "implant.npy", implant_pc)
    results["implant_npy"] = output_dir / "implant.npy"

    np.save(output_dir / "implant_normalized.npy", implant_points_normalized)
    results["implant_normalized_npy"] = output_dir / "implant_normalized.npy"

    np.save(output_dir / "shift.npy", shift)
    np.save(output_dir / "scale.npy", np.array(scale))

    return results


# ======================= Wodzinski (cran-2) Functions =======================


def get_wodzinski_model_path():
    """Resolve Wodzinski model path with priority: env > network volume > embedded."""
    if os.environ.get("WODZINSKI_MODEL_PATH"):
        return Path(os.environ["WODZINSKI_MODEL_PATH"])

    nv_path = NETWORK_VOLUME_PATH / "models" / WODZINSKI_MODEL_FILENAME
    if nv_path.exists():
        print(f"Using Wodzinski model from network volume: {nv_path}")
        return nv_path

    embedded = Path("/app/models") / WODZINSKI_MODEL_FILENAME
    print(f"Using embedded Wodzinski model: {embedded}")
    return embedded


def load_wodzinski():
    """Load Wodzinski (cran-2) model into GPU memory."""
    global wodzinski_model, device

    from wodzinski_inference import load_wodzinski_model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = get_wodzinski_model_path()
    print(f"Loading Wodzinski (cran-2) model from: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Wodzinski model not found: {model_path}")

    wodzinski_model = load_wodzinski_model(str(model_path), device, base_filters=WODZINSKI_BASE_FILTERS)
    print(f"Wodzinski (cran-2) model loaded successfully ({sum(p.numel() for p in wodzinski_model.parameters()) / 1e6:.1f}M params)")


def run_cran2_inference(defective_volume, threshold=0.5, output_resolution=None):
    """Run Wodzinski (cran-2) direct volumetric inference.

    Args:
        defective_volume: numpy array (H, W, D) — binary defective skull volume.
        threshold: Binarization threshold for output.
        output_resolution: If set, resize output volume to this resolution.

    Returns:
        implant_volume: numpy uint8 array — predicted binary implant volume.
        inference_time: float — seconds.
    """
    global wodzinski_model, device

    from wodzinski_inference import preprocess_volume, postprocess_output

    input_tensor = preprocess_volume(defective_volume, target_resolution=256).to(device)

    import time
    start = time.time()
    with torch.no_grad():
        output = wodzinski_model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.time() - start

    implant_volume = postprocess_output(output, threshold=threshold)

    # Resize if requested
    if output_resolution and output_resolution != 256:
        from scipy.ndimage import zoom
        factors = [output_resolution / s for s in implant_volume.shape]
        implant_volume = zoom(implant_volume.astype(np.float32), factors, order=1)
        implant_volume = (implant_volume > 0.5).astype(np.uint8)

    return implant_volume, inference_time


def export_cran2_results(output_dir, implant_volume, defective_volume, spacing=(1.0, 1.0, 1.0),
                         smoothing_iterations=0, close_holes=False):
    """Export cran-2 results: implant STL mesh, NRRD volumes.

    Args:
        output_dir: Directory to write output files.
        implant_volume: numpy uint8 (R, R, R) binary implant prediction.
        defective_volume: numpy (H, W, D) original defective skull.
        spacing: Voxel spacing for mesh generation.
        smoothing_iterations: Laplacian smoothing iterations.
        close_holes: Whether to close mesh holes.

    Returns:
        dict mapping key names to local file paths.
    """
    import trimesh
    from skimage.measure import marching_cubes

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Export implant NRRD volume
    try:
        import nrrd
        nrrd_path = output_dir / "implant_volume.nrrd"
        nrrd.write(str(nrrd_path), implant_volume.astype(np.float32))
        results["implant_volume_nrrd"] = nrrd_path
    except ImportError:
        print("Warning: nrrd not available, skipping NRRD export")

    # 2. Generate implant mesh via marching cubes
    if np.sum(implant_volume) > 0:
        vertices, faces, normals, _ = marching_cubes(implant_volume, level=0.5, spacing=spacing)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

        mesh = postprocess_mesh(mesh, smoothing_iterations=smoothing_iterations, close_holes=close_holes)

        stl_path = output_dir / "implant.stl"
        mesh.export(str(stl_path))
        results["implant_stl"] = stl_path

        ply_path = output_dir / "implant.ply"
        mesh.export(str(ply_path))
        results["implant_ply"] = ply_path
    else:
        print("Warning: empty implant prediction, no mesh generated")

    # 3. Save numpy volume
    npy_path = output_dir / "implant_volume.npy"
    np.save(npy_path, implant_volume)
    results["implant_volume_npy"] = npy_path

    return results


def handler(event):
    """
    Runpod Serverless Handler Function

    Supports three job types:

    1. FULL GENERATION (job_type: "full" or omitted):
    {
        "input": {
            "job_type": "full",  # optional, default
            "defective_skull": <base64 encoded .npy file OR S3 URL>,
            "input_format": "base64" | "s3_url",
            "num_ensemble": 1,  # optional, default 1
            "sampling_steps": 1000,  # optional, default 1000
            "voxelization_resolution": 512,  # optional, default 512
            "output_prefix": "job_123",  # optional, for S3 key prefix
            "pcdiff_model": "best" | "latest"  # optional, default "best"
        }
    }

    2. RE-VOXELIZATION ONLY (job_type: "revoxelize"):
    {
        "input": {
            "job_type": "revoxelize",
            "implant_points": <base64 encoded .npy file>,  # existing implant point cloud
            "defective_skull": <base64 encoded .npy file>,  # defective skull for combined vox
            "input_format": "base64",
            "voxelization_resolution": 1024,  # new resolution
            "output_prefix": "revox_123"
        }
    }

    3. WODZINSKI CRAN-2 (job_type: "cran2"):
    {
        "input": {
            "job_type": "cran2",
            "defective_skull": <base64 encoded .npy/.nrrd OR S3 URL>,
            "input_format": "base64" | "s3_url",
            "threshold": 0.5,  # optional, binarization threshold
            "output_prefix": "cran2_123",
            "smoothing_iterations": 0,
            "close_holes": false
        }
    }

    Output format (all job types):
    {
        "status": "success",
        "results": {
            "implant_stl": "https://...",
            "implant_volume_nrrd": "https://..."
        },
        "metadata": {
            "processing_time_seconds": 0.2,
            "job_type": "cran2",
            "inference_time_seconds": 0.19,
            ...
        }
    }
    """
    global pcdiff_model, voxelization_model, current_model_paths, current_pcdiff_variant

    print(f"Handler started with event: {json.dumps(event, default=str)[:500]}...")

    start_time = datetime.now()

    try:
        # Parse input
        job_input = event.get("input", {})
        job_type = job_input.get("job_type", "full")
        input_format = job_input.get("input_format", "base64")
        output_prefix = job_input.get("output_prefix", str(uuid.uuid4())[:8])
        voxelization_resolution = job_input.get("voxelization_resolution", 512)
        smoothing_iterations = job_input.get("smoothing_iterations", 0)
        close_holes = job_input.get("close_holes", False)

        # Validate resolution
        valid_resolutions = [128, 256, 512, 1024]
        if voxelization_resolution not in valid_resolutions:
            return {"error": f"Invalid voxelization_resolution: {voxelization_resolution}. Valid: {valid_resolutions}"}

        # Create temp directory for this job
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            output_dir = temp_dir / "output"
            output_dir.mkdir()

            if job_type == "cran2":
                # WODZINSKI CRAN-2 — Direct volumetric UNet inference
                print("=== CRAN-2 (Wodzinski) JOB ===")

                defective_data = job_input.get("defective_skull")
                if not defective_data:
                    return {"error": "Missing 'defective_skull' in input"}

                threshold = job_input.get("threshold", 0.5)
                cran2_smoothing = job_input.get("smoothing_iterations", smoothing_iterations)
                cran2_close_holes = job_input.get("close_holes", close_holes)

                # Ensure Wodzinski model is loaded
                if wodzinski_model is None:
                    load_wodzinski()

                # Load input volume
                input_path = temp_dir / "input"
                if input_format == "base64":
                    raw_bytes = base64.b64decode(defective_data)
                    with open(input_path, "wb") as f:
                        f.write(raw_bytes)
                    # Try NRRD first, then numpy
                    try:
                        import nrrd
                        defective_volume, _ = nrrd.read(str(input_path))
                        defective_volume = (defective_volume > 0).astype(np.float32)
                    except Exception:
                        defective_volume = np.load(str(input_path))
                        if len(defective_volume.shape) == 4:
                            defective_volume = defective_volume[0]
                elif input_format == "s3_url":
                    download_from_s3(defective_data, input_path)
                    try:
                        import nrrd
                        defective_volume, _ = nrrd.read(str(input_path))
                        defective_volume = (defective_volume > 0).astype(np.float32)
                    except Exception:
                        defective_volume = np.load(str(input_path))
                        if len(defective_volume.shape) == 4:
                            defective_volume = defective_volume[0]
                else:
                    return {"error": f"Unknown input_format: {input_format}"}

                print(f"Loaded defective volume: {defective_volume.shape}")

                # Run inference
                implant_volume, inference_time = run_cran2_inference(
                    defective_volume, threshold=threshold
                )
                print(f"Inference complete: {inference_time:.3f}s, implant voxels: {np.sum(implant_volume)}")

                # Export results
                local_results = export_cran2_results(
                    output_dir, implant_volume, defective_volume,
                    smoothing_iterations=cran2_smoothing,
                    close_holes=cran2_close_holes,
                )

                # Upload to S3
                print("Uploading cran-2 results to S3...")
                s3_results = {}
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for key, local_path in local_results.items():
                    s3_key = f"inference_results/{output_prefix}/{timestamp}/{local_path.name}"
                    try:
                        url = upload_to_s3(local_path, s3_key)
                        s3_results[key] = url
                        print(f"  Uploaded: {key} -> {url}")
                    except Exception as e:
                        print(f"  Failed to upload {key}: {e}")

                processing_time = (datetime.now() - start_time).total_seconds()

                return {
                    "status": "success",
                    "results": s3_results,
                    "metadata": {
                        "processing_time_seconds": processing_time,
                        "inference_time_seconds": inference_time,
                        "job_type": "cran2",
                        "model": "wodzinski_residual_unet3d",
                        "input_shape": list(defective_volume.shape),
                        "output_shape": list(implant_volume.shape),
                        "threshold": threshold,
                        "smoothing_iterations": cran2_smoothing,
                        "close_holes": cran2_close_holes,
                        "implant_voxel_count": int(np.sum(implant_volume)),
                    },
                }

            elif job_type == "revoxelize":
                # RE-VOXELIZATION ONLY - skip PCDiff, just run voxelization
                print(f"=== RE-VOXELIZATION JOB (resolution: {voxelization_resolution}³) ===")

                implant_data = job_input.get("implant_points")
                defective_data = job_input.get("defective_skull")

                if not implant_data:
                    return {"error": "Missing 'implant_points' for revoxelize job"}
                if not defective_data:
                    return {"error": "Missing 'defective_skull' for revoxelize job"}

                # Load implant points
                implant_path = temp_dir / "implant.npy"
                if input_format == "base64":
                    npy_bytes = base64.b64decode(implant_data)
                    with open(implant_path, "wb") as f:
                        f.write(npy_bytes)
                else:
                    download_from_s3(implant_data, implant_path)

                implant_points = np.load(str(implant_path))
                if len(implant_points.shape) == 3:
                    implant_points = implant_points[0]

                # Load defective skull points
                defective_path = temp_dir / "defective.npy"
                if input_format == "base64":
                    npy_bytes = base64.b64decode(defective_data)
                    with open(defective_path, "wb") as f:
                        f.write(npy_bytes)
                else:
                    download_from_s3(defective_data, defective_path)

                defective_points = np.load(str(defective_path))
                if len(defective_points.shape) == 3:
                    defective_points = defective_points[0]

                print(f"Loaded implant: {implant_points.shape}, defective: {defective_points.shape}")

                # Ensure voxelization model is loaded
                if voxelization_model is None:
                    load_models()

                # Run re-voxelization
                print(f"Running re-voxelization at {voxelization_resolution}³...")
                vertices, faces, volume_complete, volume_implant, implant_pc, shift, scale = run_revoxelization_only(
                    implant_points, defective_points, resolution=voxelization_resolution
                )
                print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")

                # Export STL mesh
                import trimesh

                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                # Apply post-processing
                mesh = postprocess_mesh(mesh, smoothing_iterations=smoothing_iterations, close_holes=close_holes)

                stl_path = output_dir / f"implant_res{voxelization_resolution}.stl"
                mesh.export(str(stl_path))

                # Upload to S3
                print("Uploading to S3...")
                s3_results = {}
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                s3_key = f"inference_results/{output_prefix}/{timestamp}/implant_res{voxelization_resolution}.stl"
                url = upload_to_s3(stl_path, s3_key)
                s3_results["implant_only_stl"] = url
                print(f"  Uploaded: implant_only_stl -> {url}")

                processing_time = (datetime.now() - start_time).total_seconds()

                return {
                    "status": "success",
                    "results": s3_results,
                    "metadata": {
                        "processing_time_seconds": processing_time,
                        "job_type": "revoxelize",
                        "voxelization_resolution": voxelization_resolution,
                        "smoothing_iterations": smoothing_iterations,
                        "close_holes": close_holes,
                        "mesh_vertices": len(mesh.vertices),
                        "mesh_faces": len(mesh.faces),
                        "num_implant_points": len(implant_pc),
                    },
                }

            else:
                # FULL GENERATION - PCDiff + Voxelization
                print(f"=== FULL GENERATION JOB (resolution: {voxelization_resolution}³) ===")

                defective_skull_data = job_input.get("defective_skull")
                num_ensemble = job_input.get("num_ensemble", 1)
                sampling_steps = job_input.get("sampling_steps", 1000)
                sampling_method = job_input.get("sampling_method", "ddpm")
                if sampling_method not in ("ddpm", "ddim"):
                    return {"error": f"Invalid sampling_method: {sampling_method}. Must be 'ddpm' or 'ddim'."}
                requested_pcdiff_model = job_input.get("pcdiff_model", DEFAULT_PCDIFF_MODEL)

                # Ensure models are loaded
                if pcdiff_model is None or voxelization_model is None:
                    load_models(pcdiff_variant=requested_pcdiff_model)
                # Check if we need to switch PCDiff model variant
                elif current_pcdiff_variant != requested_pcdiff_model:
                    reload_pcdiff_model(requested_pcdiff_model)

                if not defective_skull_data:
                    return {"error": "Missing 'defective_skull' in input"}

                input_path = temp_dir / "input.npy"

                # Load input point cloud
                if input_format == "base64":
                    npy_bytes = base64.b64decode(defective_skull_data)
                    with open(input_path, "wb") as f:
                        f.write(npy_bytes)
                elif input_format == "s3_url":
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
                    input_points, num_ens=num_ensemble, sampling_steps=sampling_steps, sampling_method=sampling_method
                )
                print(f"Generated implant: {implant_points.shape}")

                # Step 2: Run Voxelization with specified resolution
                print(f"Running voxelization at {voxelization_resolution}³...")
                vertices, faces, volume_complete, volume_implant, implant_pc = run_voxelization(
                    implant_points, defective_points, implant_normalized, resolution=voxelization_resolution
                )
                print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")

                # Step 3: Export results
                print("Exporting results...")
                local_results = export_results(
                    output_dir,
                    vertices,
                    faces,
                    volume_complete,
                    volume_implant,
                    implant_pc,
                    defective_points,
                    shift,
                    scale,
                    implant_normalized,
                    smoothing_iterations=smoothing_iterations,
                    close_holes=close_holes,
                )

                # Step 4: Upload to S3 (only the files needed by frontend)
                print("Uploading to S3...")
                s3_results = {}
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                required_outputs = ["implant_npy", "implant_only_stl"]

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

                processing_time = (datetime.now() - start_time).total_seconds()
                model_source = (
                    "network_volume"
                    if current_model_paths and "runpod-volume" in current_model_paths[0]
                    else "embedded"
                )

                return {
                    "status": "success",
                    "results": s3_results,
                    "metadata": {
                        "processing_time_seconds": processing_time,
                        "job_type": "full",
                        "num_implant_points": implant_pc.shape[0],
                        "num_ensemble": num_ensemble,
                        "sampling_method": sampling_method,
                        "sampling_steps": sampling_steps,
                        "voxelization_resolution": voxelization_resolution,
                        "smoothing_iterations": smoothing_iterations,
                        "close_holes": close_holes,
                        "mesh_vertices": len(vertices),
                        "mesh_faces": len(faces),
                        "model_source": model_source,
                        "pcdiff_model": current_model_paths[0] if current_model_paths else "unknown",
                        "pcdiff_model_variant": current_pcdiff_variant,
                        "available_pcdiff_variants": list(PCDIFF_MODEL_VARIANTS.keys()),
                        "voxelization_model": current_model_paths[1] if current_model_paths else "unknown",
                    },
                }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


# Load models at startup (cold start optimization)
print("=" * 60)
print("  CrAInial Serverless Worker (PCDiff + Wodzinski cran-2)")
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

# Determine which models to load based on CRAN2_ENABLED env var
CRAN2_ENABLED = os.environ.get("CRAN2_ENABLED", "true").lower() in ("true", "1", "yes")
PCDIFF_ENABLED = os.environ.get("PCDIFF_ENABLED", "true").lower() in ("true", "1", "yes")

if PCDIFF_ENABLED:
    try:
        load_models()
    except Exception as e:
        print(f"Warning: Could not pre-load PCDiff models: {e}")
        print("PCDiff models will be loaded on first request")

if CRAN2_ENABLED:
    try:
        load_wodzinski()
    except Exception as e:
        print(f"Warning: Could not pre-load Wodzinski (cran-2) model: {e}")
        print("Wodzinski model will be loaded on first request")

# Start the serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
