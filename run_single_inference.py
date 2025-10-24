#!/usr/bin/env python3
"""
Complete PCDiff + Voxelization Inference Pipeline

This script runs the complete skull implant generation pipeline on a single input:
1. PCDiff: Generates implant point cloud from defective skull
2. Voxelization: Converts point cloud to volumetric mesh
3. Visualization: Exports PLY/STL files for 3D viewing

Output formats:
- implant_only.ply/stl - Just the generated implant
- skull_complete.ply/stl - Defective skull + implant together
- Point clouds and voxel data for further processing
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import nrrd
import trimesh
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "pcdiff"))
sys.path.insert(0, str(project_root / "voxelization"))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Complete PCDiff + Voxelization pipeline for single skull implant generation'
    )
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input defective skull (.npy point cloud or .nrrd volume)')
    parser.add_argument('--output_dir', type=str, default='pipeline_output',
                       help='Output directory for results')
    parser.add_argument('--name', type=str, default=None,
                       help='Name for this inference (default: derived from input filename)')
    
    # PCDiff Model
    parser.add_argument('--pcdiff_model', type=str, required=True,
                       help='Path to trained PCDiff model checkpoint')
    parser.add_argument('--num_ens', type=int, default=1,
                       help='Number of ensemble samples to generate')
    parser.add_argument('--sampling_method', type=str, default='ddim',
                       choices=['ddpm', 'ddim'],
                       help='Sampling method for diffusion')
    parser.add_argument('--sampling_steps', type=int, default=50,
                       help='Number of sampling steps')
    
    # Voxelization Model
    parser.add_argument('--vox_model', type=str, required=True,
                       help='Path to trained voxelization model checkpoint')
    parser.add_argument('--vox_config', type=str,
                       default='voxelization/configs/gen_skullbreak.yaml',
                       help='Path to voxelization config file')
    
    # Dataset Settings
    parser.add_argument('--dataset', type=str, default='SkullBreak',
                       choices=['SkullBreak', 'SkullFix'],
                       help='Dataset type')
    parser.add_argument('--num_points', type=int, default=30720,
                       help='Number of points in defective skull')
    parser.add_argument('--num_nn', type=int, default=3072,
                       help='Number of points in implant')
    
    # Export Options
    parser.add_argument('--export_ply', action='store_true', default=True,
                       help='Export PLY files for visualization')
    parser.add_argument('--export_stl', action='store_true', default=True,
                       help='Export STL files for 3D printing')
    parser.add_argument('--export_nrrd', action='store_true', default=True,
                       help='Export NRRD volume files')
    
    # GPU Settings
    parser.add_argument('--gpu', type=int, default=7,
                       help='GPU ID to use')
    
    return parser.parse_args()


def load_input_skull(input_path, num_points=30720):
    """Load defective skull from .npy or .nrrd file."""
    input_path = Path(input_path)
    
    if input_path.suffix == '.npy':
        # Already a point cloud
        points = np.load(input_path)
        if len(points.shape) == 3:
            points = points[0]  # Remove batch dimension
        
        # Sample to desired number of points
        if points.shape[0] != num_points:
            idx = np.random.choice(points.shape[0], num_points, replace=False)
            points = points[idx]
        
        return points, None
    
    elif input_path.suffix == '.nrrd':
        # Volume file - need to convert to point cloud
        volume, header = nrrd.read(str(input_path))
        
        # Extract surface points (simple thresholding)
        mask = volume > 0
        coords = np.argwhere(mask)
        
        # Sample points
        if len(coords) > num_points:
            idx = np.random.choice(len(coords), num_points, replace=False)
            coords = coords[idx]
        
        points = coords.astype(np.float32)
        
        return points, header
    
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}. Use .npy or .nrrd")


def run_pcdiff_inference(args, input_points):
    """Run PCDiff to generate implant point cloud."""
    print("\n" + "="*60)
    print("Step 1: PCDiff - Generating Implant Point Cloud")
    print("="*60)
    
    # Setup
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load model using the Model wrapper from test_completion (has DDIM support)
    # We need to import it from the test script location
    import sys
    sys.path.insert(0, str(project_root / "pcdiff"))
    
    # Import required modules
    from test_completion import Model
    
    # Load model
    print(f"Loading PCDiff model from: {args.pcdiff_model}")
    
    # Model arguments (matching test_completion config)
    class ModelArgs:
        def __init__(self):
            self.nc = 3
            self.num_points = args.num_points
            self.num_nn = args.num_nn
            self.attention = True
            self.dropout = 0.1
            self.embed_dim = 64
            # Required by test_completion.Model
            self.sampling_method = args.sampling_method
            self.sampling_steps = args.sampling_steps
    
    model_args = ModelArgs()
    
    # Define betas for diffusion (must be numpy array)
    def get_betas(schedule_type, beta_start, beta_end, time_num):
        if schedule_type == 'linear':
            betas = np.linspace(beta_start, beta_end, time_num, dtype=np.float64)
        else:
            raise NotImplementedError(schedule_type)
        return betas
    
    betas = get_betas('linear', 0.0001, 0.02, 1000)
    model_args.time_num = len(betas)
    
    model = Model(model_args, betas, 'mse', 'eps', 'fixedsmall',
                  width_mult=1.0, vox_res_mult=1.0)
    
    model = model.to(device)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(args.pcdiff_model, map_location=device)
    state_dict = checkpoint['model_state']
    
    # Handle DDP checkpoints
    if list(state_dict.keys())[0].startswith('model.module.'):
        state_dict = {k.replace('model.module.', 'model.'): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print("✓ Model loaded successfully")
    
    # Match training setup: defective skull contributes (num_points - num_nn) points
    sv_points = args.num_points - args.num_nn
    if input_points.shape[0] < sv_points:
        raise ValueError(
            f"Input point cloud has {input_points.shape[0]} points, "
            f"but the model expects at least {sv_points} defective points."
        )

    idx = np.random.choice(input_points.shape[0], sv_points, replace=False if input_points.shape[0] >= sv_points else True)
    partial_points_raw = input_points[idx]

    pc_min = partial_points_raw.min(axis=0)
    pc_max = partial_points_raw.max(axis=0)
    shift = (pc_min + pc_max) / 2.0
    scale = (pc_max - pc_min).max() / 2.0
    if scale <= 0:
        raise ValueError("Invalid scale computed from input point cloud bounding box.")
    scale = scale / 3.0

    partial_points = (partial_points_raw - shift) / scale
    input_points_normalized = (input_points - shift) / scale

    # Convert to torch
    pc_input = torch.from_numpy(partial_points).float().unsqueeze(0)  # (1, sv_points, 3)
    pc_input = pc_input.transpose(1, 2).to(device)  # (1, 3, sv_points)
    
    # Repeat for ensemble
    pc_input = pc_input.repeat(args.num_ens, 1, 1)  # (num_ens, 3, N)
    noise_shape = torch.Size([args.num_ens, 3, args.num_nn])
    
    print(f"Input shape: {pc_input.shape}")
    print(f"Generating {args.num_ens} ensemble sample(s) with {args.sampling_method} ({args.sampling_steps} steps)...")
    
    # Generate implant
    with torch.no_grad():
        sample = model.gen_samples(
            pc_input, noise_shape, device,
            clip_denoised=False,
            sampling_method=args.sampling_method,
            sampling_steps=args.sampling_steps
        )
        sample = sample.detach().cpu().numpy()
    
    # Denormalize
    sample = sample.transpose(0, 2, 1)  # (num_ens, num_nn, 3)
    sample_normalized = sample.copy()
    sample = sample * scale + shift
    print(f"✓ Generated implant point cloud: {sample.shape}")

    return sample, input_points, shift, scale, sample_normalized


def run_voxelization(args, implant_points, defective_points, output_dir):
    """Run voxelization to convert point cloud to mesh."""
    print("\n" + "="*60)
    print("Step 2: Voxelization - Converting to Volumetric Mesh")
    print("="*60)
    
    # Import voxelization modules
    from voxelization.src.model import Encode2Points
    from voxelization.src.utils import load_config, load_model_manual
    from voxelization.src import config as vox_config
    
    # Load config
    config_path = Path(args.vox_config).absolute()
    default_config = Path('voxelization/configs/default.yaml').absolute()
    cfg = load_config(str(config_path), str(default_config))
    
    # Override config settings
    cfg['test']['model_file'] = args.vox_model
    
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load model
    print(f"Loading voxelization model from: {args.vox_model}")
    model = Encode2Points(cfg).to(device)
    
    state_dict = torch.load(args.vox_model, map_location='cpu')
    load_model_manual(state_dict['state_dict'], model)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Get generator
    generator = vox_config.get_generator(model, cfg, device=device)
    
    # Combine defective skull + implant (take first ensemble sample)
    implant_pc = implant_points[0]  # (num_nn, 3)
    
    # Normalize to [0, 1]
    combined_points = np.concatenate([defective_points, implant_pc], axis=0)
    combined_points = combined_points / 512.0  # Normalize
    
    print(f"Combined point cloud: {combined_points.shape[0]} points")
    print(f"  - Defective skull: {defective_points.shape[0]} points")
    print(f"  - Implant: {implant_pc.shape[0]} points")
    
    # Convert to torch
    inputs = torch.from_numpy(combined_points).float().unsqueeze(0).to(device)
    
    print("Generating mesh...")
    with torch.no_grad():
        vertices, faces, points, normals, psr_grid = generator.generate_mesh(inputs)
    
    print(f"✓ Generated mesh:")
    print(f"  - Vertices: {len(vertices)}")
    print(f"  - Faces: {len(faces)}")
    
    # Convert PSR grid to binary volume
    psr_grid_np = psr_grid.detach().cpu().numpy()[0, :, :, :]
    volume_complete = np.zeros((512, 512, 512), dtype=np.float32)
    volume_complete[psr_grid_np <= 0] = 1
    
    return vertices, faces, volume_complete, implant_pc


def export_results(args, vertices, faces, volume_complete, implant_pc, 
                   defective_points, shift, scale, output_dir,
                   implant_points_normalized):
    """Export all output formats."""
    print("\n" + "="*60)
    print("Step 3: Exporting Results")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # 1. Export complete skull mesh (defective + implant)
    if args.export_ply or args.export_stl:
        print("Exporting complete skull (defective + implant)...")
        
        # Scale vertices back to original coordinate space (0-512)
        vertices_scaled = vertices * 512.0
        
        mesh_complete = trimesh.Trimesh(vertices=vertices_scaled, faces=faces)
        
        if args.export_ply:
            ply_path = output_dir / 'skull_complete.ply'
            mesh_complete.export(str(ply_path))
            results.append(('PLY (Complete)', ply_path, ply_path.stat().st_size))
            print(f"  ✓ {ply_path}")
        
        if args.export_stl:
            stl_path = output_dir / 'skull_complete.stl'
            mesh_complete.export(str(stl_path))
            results.append(('STL (Complete)', stl_path, stl_path.stat().st_size))
            print(f"  ✓ {stl_path}")
    
    # 2. Export implant-only mesh
    # Create mesh from implant point cloud
    if args.export_ply or args.export_stl:
        print("Exporting implant only...")
        
        # Create point cloud
        implant_pc_scaled = implant_pc  # Already in original space
        implant_cloud = trimesh.PointCloud(vertices=implant_pc_scaled)
        
        # Create convex hull for mesh
        try:
            implant_mesh = implant_cloud.convex_hull
            
            if args.export_ply:
                ply_path = output_dir / 'implant_only.ply'
                implant_mesh.export(str(ply_path))
                results.append(('PLY (Implant)', ply_path, ply_path.stat().st_size))
                print(f"  ✓ {ply_path}")
            
            if args.export_stl:
                stl_path = output_dir / 'implant_only.stl'
                implant_mesh.export(str(stl_path))
                results.append(('STL (Implant)', stl_path, stl_path.stat().st_size))
                print(f"  ✓ {stl_path}")
        except Exception as e:
            print(f"  ⚠ Could not create implant mesh: {e}")
    
    # 3. Export point clouds as PLY
    print("Exporting point clouds...")
    
    # Defective skull point cloud (original space)
    defective_cloud = trimesh.PointCloud(vertices=defective_points)
    defective_cloud.colors = np.tile([200, 200, 200, 255], (len(defective_points), 1))
    ply_path = output_dir / 'defective_skull_pc.ply'
    defective_cloud.export(str(ply_path))
    results.append(('PLY (Defective PC)', ply_path, ply_path.stat().st_size))
    print(f"  ✓ {ply_path}")
    
    # Implant point cloud (original space)
    implant_cloud = trimesh.PointCloud(vertices=implant_pc)
    implant_cloud.colors = np.tile([255, 100, 100, 255], (len(implant_pc), 1))
    ply_path = output_dir / 'implant_pc.ply'
    implant_cloud.export(str(ply_path))
    results.append(('PLY (Implant PC)', ply_path, ply_path.stat().st_size))
    print(f"  ✓ {ply_path}")

    # Defective skull (normalized space)
    defective_normalized = trimesh.PointCloud(vertices=(defective_points - shift) / scale)
    defective_normalized.colors = np.tile([200, 200, 200, 255], (len(defective_points), 1))
    ply_path = output_dir / 'defective_skull_pc_normalized.ply'
    defective_normalized.export(str(ply_path))
    results.append(('PLY (Defective PC Normalized)', ply_path, ply_path.stat().st_size))
    print(f"  ✓ {ply_path}")

    # Implant (normalized space as used by PCDiff)
    implant_normalized = trimesh.PointCloud(vertices=implant_points_normalized[0])
    implant_normalized.colors = np.tile([255, 100, 100, 255], (len(implant_points_normalized[0]), 1))
    ply_path = output_dir / 'implant_pc_normalized.ply'
    implant_normalized.export(str(ply_path))
    results.append(('PLY (Implant PC Normalized)', ply_path, ply_path.stat().st_size))
    print(f"  ✓ {ply_path}")
    
    # 4. Export NRRD volumes
    if args.export_nrrd:
        print("Exporting NRRD volumes...")
        
        nrrd_path = output_dir / 'skull_complete.nrrd'
        nrrd.write(str(nrrd_path), volume_complete.astype(np.float32))
        results.append(('NRRD (Complete)', nrrd_path, nrrd_path.stat().st_size))
        print(f"  ✓ {nrrd_path}")
    
    # 5. Save raw numpy arrays
    print("Saving numpy arrays...")
    
    np.save(output_dir / 'defective_skull.npy', defective_points)
    np.save(output_dir / 'implant.npy', implant_pc)
    np.save(output_dir / 'shift.npy', shift)
    np.save(output_dir / 'scale.npy', scale)
    np.save(output_dir / 'implant_normalized.npy', implant_points_normalized)
    print(f"  ✓ Saved numpy arrays")
    
    return results


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("  PCDiff + Voxelization Pipeline - Single Skull Inference")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"PCDiff Model: {args.pcdiff_model}")
    print(f"Voxelization Model: {args.vox_model}")
    print(f"Output: {args.output_dir}")
    print(f"GPU: {args.gpu}")
    
    # Determine output name
    if args.name:
        name = args.name
    else:
        name = Path(args.input).stem
    
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load input
        print("\n" + "="*60)
        print("Loading Input Skull")
        print("="*60)
        input_points, header = load_input_skull(args.input, args.num_points)
        print(f"✓ Loaded {input_points.shape[0]} points from {args.input}")
        
        # Step 2: Run PCDiff
        implant_points, defective_points, shift, scale, implant_points_normalized = run_pcdiff_inference(args, input_points)
        
        # Step 3: Run Voxelization
        vertices, faces, volume_complete, implant_pc = run_voxelization(
            args, implant_points, defective_points, output_dir
        )
        
        # Step 4: Export Results
        results = export_results(
            args, vertices, faces, volume_complete, implant_pc,
            defective_points, shift, scale, output_dir,
            implant_points_normalized
        )
        
        # Summary
        print("\n" + "="*70)
        print("  Pipeline Complete! ✓")
        print("="*70)
        print(f"\nOutput directory: {output_dir}")
        print("\nGenerated files:")
        for file_type, path, size in results:
            size_mb = size / (1024 * 1024)
            print(f"  {file_type:20s}: {path.name:30s} ({size_mb:.2f} MB)")
        
        print("\nVisualization:")
        print(f"  - Use skull_complete.ply/stl to view the complete skull with implant")
        print(f"  - Use implant_only.ply/stl to view just the generated implant")
        print(f"  - Use *_pc.ply files for point cloud visualization")
        
        print("\n3D Printing:")
        print(f"  - Use skull_complete.stl or implant_only.stl in your slicer")
        
        print("\nWeb Viewer:")
        print(f"  - Copy {output_dir} to inference_results_ddim50/syn/")
        print(f"  - Run: python3 pcdiff/utils/convert_to_web.py {output_dir}")
        print(f"  - Start viewer: cd web_viewer && ./start_dev.sh")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ Error during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

