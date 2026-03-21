"""
Convert PCDiff inference results (NPY format) to web-friendly formats (PLY/STL) for 3D visualization and printing.

This module handles:
- Loading .npy inference results (input defective skull, generated implant samples)
- Converting to PLY format with customizable colors for web visualization
- Converting to STL format for 3D printing
- Batch processing with skip logic to avoid re-conversion
- Handling ensemble outputs (multiple samples per input)
"""

import numpy as np
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import trimesh
from tqdm import tqdm


def load_npy_with_transform(npy_file: Path) -> np.ndarray:
    """Load numpy point cloud and handle different shapes."""
    points = np.load(npy_file)
    
    # Handle different shapes: (batch, num_points, 3) or (num_points, 3) or (1, num_points, 3)
    if len(points.shape) == 3:
        if points.shape[0] == 1:
            points = points[0]  # Remove batch dimension if batch=1
    elif len(points.shape) == 2:
        pass  # Already (num_points, 3)
    else:
        raise ValueError(f"Unexpected point cloud shape: {points.shape}")
    
    return points


def apply_inverse_transform(points: np.ndarray, shift: Optional[np.ndarray] = None, 
                           scale: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply inverse transformation to bring points back to original coordinate space."""
    if scale is not None:
        if len(scale.shape) > 1:
            scale = scale.flatten()
        points = points * scale
    
    if shift is not None:
        if len(shift.shape) > 1:
            shift = shift.flatten()
        points = points + shift
    
    return points


def npy_to_ply(npy_file: Path, output_file: Path, color: Tuple[int, int, int] = (128, 128, 128),
               shift: Optional[np.ndarray] = None, scale: Optional[np.ndarray] = None) -> int:
    """
    Convert numpy point cloud to PLY format with color.
    
    Args:
        npy_file: Path to .npy file containing point cloud
        output_file: Path to output .ply file
        color: RGB color tuple (0-255)
        shift: Optional shift vector to apply (inverse normalization)
        scale: Optional scale factor to apply (inverse normalization)
    
    Returns:
        Number of points in the point cloud
    """
    points = load_npy_with_transform(npy_file)
    points = points.reshape(-1, 3)
    
    # Apply inverse transformation if provided
    points = apply_inverse_transform(points, shift, scale)
    
    # Create point cloud with trimesh
    point_cloud = trimesh.PointCloud(vertices=points)
    
    # Add colors (trimesh expects colors in 0-255 range)
    colors = np.tile(color, (len(points), 1)).astype(np.uint8)
    point_cloud.colors = colors
    
    # Export to PLY
    output_file.parent.mkdir(parents=True, exist_ok=True)
    point_cloud.export(str(output_file))
    
    return len(points)


def npy_to_stl(npy_file: Path, output_file: Path, 
               shift: Optional[np.ndarray] = None, scale: Optional[np.ndarray] = None,
               method: str = 'ball_pivoting') -> int:
    """
    Convert numpy point cloud to STL format for 3D printing.
    Creates a mesh from the point cloud using surface reconstruction.
    
    Args:
        npy_file: Path to .npy file containing point cloud
        output_file: Path to output .stl file
        shift: Optional shift vector to apply (inverse normalization)
        scale: Optional scale factor to apply (inverse normalization)
        method: Surface reconstruction method ('ball_pivoting', 'alpha_shape', or 'convex_hull')
    
    Returns:
        Number of faces in the mesh
    """
    points = load_npy_with_transform(npy_file)
    points = points.reshape(-1, 3)
    
    # Apply inverse transformation if provided
    points = apply_inverse_transform(points, shift, scale)
    
    # Create point cloud
    point_cloud = trimesh.PointCloud(vertices=points)
    
    try:
        if method == 'convex_hull':
            # Fast but loses detail - creates convex hull
            mesh = point_cloud.convex_hull
        elif method == 'alpha_shape':
            # Better detail preservation with alpha shapes
            mesh = trimesh.creation.alpha_shape(points, alpha=2.0)
        else:  # ball_pivoting (default)
            # Best for smooth surfaces, requires normals estimation
            # Estimate normals first (simplified - just use convex hull for now)
            mesh = point_cloud.convex_hull
    except Exception as e:
        print(f"Warning: Surface reconstruction failed ({e}), using convex hull")
        mesh = point_cloud.convex_hull
    
    # Export to binary STL (more efficient than ASCII)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_file), file_type='stl')
    
    return len(mesh.faces)


def convert_inference_result(result_dir: Path, output_dir: Optional[Path] = None, 
                            force: bool = False, export_stl: bool = True) -> dict:
    """
    Convert a single inference result directory to PLY and STL formats.
    
    Args:
        result_dir: Directory containing input.npy, sample.npy, shift.npy, scale.npy
        output_dir: Output directory (defaults to result_dir/web)
        force: If True, reconvert even if output files exist
        export_stl: If True, also export STL files (slower)
    
    Returns:
        Dictionary with conversion statistics
    """
    if output_dir is None:
        output_dir = result_dir / 'web'
    
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    
    stats = {
        'result_name': result_dir.name,
        'converted': [],
        'skipped': [],
        'errors': []
    }
    
    # Load transformation parameters
    shift_file = result_dir / 'shift.npy'
    scale_file = result_dir / 'scale.npy'
    
    shift = np.load(shift_file) if shift_file.exists() else None
    scale = np.load(scale_file) if scale_file.exists() else None
    
    # Convert input (defective skull)
    input_npy = result_dir / 'input.npy'
    if input_npy.exists():
        input_ply = output_dir / 'input.ply'
        input_stl = output_dir / 'input.stl'
        
        if not input_ply.exists() or force:
            try:
                count = npy_to_ply(input_npy, input_ply, color=(200, 200, 200), 
                                  shift=shift, scale=scale)
                stats['converted'].append(f"input.ply ({count} points)")
            except Exception as e:
                stats['errors'].append(f"input.ply: {str(e)}")
        else:
            stats['skipped'].append('input.ply')
        
        if export_stl and (not input_stl.exists() or force):
            try:
                faces = npy_to_stl(input_npy, input_stl, shift=shift, scale=scale)
                stats['converted'].append(f"input.stl ({faces} faces)")
            except Exception as e:
                stats['errors'].append(f"input.stl: {str(e)}")
        elif export_stl:
            stats['skipped'].append('input.stl')
    
    # Convert sample (generated implant)
    sample_npy = result_dir / 'sample.npy'
    if sample_npy.exists():
        sample_data = np.load(sample_npy)
        
        # Check if we have ensemble outputs (multiple samples)
        if len(sample_data.shape) == 3 and sample_data.shape[0] > 1:
            # Multiple samples - save each separately
            for i in range(sample_data.shape[0]):
                sample_ply = output_dir / f'sample_{i}.ply'
                sample_stl = output_dir / f'sample_{i}.stl'
                
                # Create temporary file for this sample
                temp_npy = result_dir / f'_temp_sample_{i}.npy'
                np.save(temp_npy, sample_data[i])
                
                try:
                    if not sample_ply.exists() or force:
                        count = npy_to_ply(temp_npy, sample_ply, color=(255, 100, 100),
                                          shift=shift, scale=scale)
                        stats['converted'].append(f"sample_{i}.ply ({count} points)")
                    else:
                        stats['skipped'].append(f'sample_{i}.ply')
                    
                    if export_stl and (not sample_stl.exists() or force):
                        faces = npy_to_stl(temp_npy, sample_stl, shift=shift, scale=scale)
                        stats['converted'].append(f"sample_{i}.stl ({faces} faces)")
                    elif export_stl:
                        stats['skipped'].append(f'sample_{i}.stl')
                except Exception as e:
                    stats['errors'].append(f"sample_{i}: {str(e)}")
                finally:
                    # Clean up temp file
                    if temp_npy.exists():
                        temp_npy.unlink()
        else:
            # Single sample
            sample_ply = output_dir / 'sample.ply'
            sample_stl = output_dir / 'sample.stl'
            
            if not sample_ply.exists() or force:
                try:
                    count = npy_to_ply(sample_npy, sample_ply, color=(255, 100, 100),
                                      shift=shift, scale=scale)
                    stats['converted'].append(f"sample.ply ({count} points)")
                except Exception as e:
                    stats['errors'].append(f"sample.ply: {str(e)}")
            else:
                stats['skipped'].append('sample.ply')
            
            if export_stl and (not sample_stl.exists() or force):
                try:
                    faces = npy_to_stl(sample_npy, sample_stl, shift=shift, scale=scale)
                    stats['converted'].append(f"sample.stl ({faces} faces)")
                except Exception as e:
                    stats['errors'].append(f"sample.stl: {str(e)}")
            elif export_stl:
                stats['skipped'].append('sample.stl')
    
    return stats


def batch_convert_all(base_dir: Path, force: bool = False, export_stl: bool = True) -> dict:
    """
    Batch convert all inference results in a directory.
    
    Args:
        base_dir: Base directory containing inference results (e.g., inference_results_ddim50)
        force: If True, reconvert even if output files exist
        export_stl: If True, also export STL files (slower)
    
    Returns:
        Dictionary with overall statistics
    """
    base_dir = Path(base_dir)
    syn_dir = base_dir / 'syn'
    
    if not syn_dir.exists():
        raise FileNotFoundError(f"Synthetic results directory not found: {syn_dir}")
    
    # Find all result directories
    result_dirs = [d for d in syn_dir.iterdir() if d.is_dir() and (d / 'input.npy').exists()]
    
    if not result_dirs:
        print(f"No inference results found in {syn_dir}")
        return {'total': 0, 'results': []}
    
    print(f"Found {len(result_dirs)} inference results to convert")
    
    overall_stats = {
        'total': len(result_dirs),
        'results': []
    }
    
    for result_dir in tqdm(result_dirs, desc="Converting results"):
        stats = convert_inference_result(result_dir, force=force, export_stl=export_stl)
        overall_stats['results'].append(stats)
    
    return overall_stats


def print_stats(stats: dict):
    """Print conversion statistics in a readable format."""
    if 'results' in stats:
        # Batch stats
        total_converted = sum(len(r['converted']) for r in stats['results'])
        total_skipped = sum(len(r['skipped']) for r in stats['results'])
        total_errors = sum(len(r['errors']) for r in stats['results'])
        
        print(f"\n{'='*60}")
        print(f"Batch Conversion Complete")
        print(f"{'='*60}")
        print(f"Total results processed: {stats['total']}")
        print(f"Files converted: {total_converted}")
        print(f"Files skipped: {total_skipped}")
        print(f"Errors: {total_errors}")
        
        if total_errors > 0:
            print(f"\nErrors encountered:")
            for result in stats['results']:
                if result['errors']:
                    print(f"  {result['result_name']}:")
                    for error in result['errors']:
                        print(f"    - {error}")
    else:
        # Single result stats
        print(f"\n{'='*60}")
        print(f"Conversion Complete: {stats['result_name']}")
        print(f"{'='*60}")
        if stats['converted']:
            print(f"Converted:")
            for item in stats['converted']:
                print(f"  ✓ {item}")
        if stats['skipped']:
            print(f"Skipped (already exists):")
            for item in stats['skipped']:
                print(f"  - {item}")
        if stats['errors']:
            print(f"Errors:")
            for error in stats['errors']:
                print(f"  ✗ {error}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PCDiff inference results to PLY/STL formats for web viewing and 3D printing'
    )
    parser.add_argument('input_dir', type=str, 
                       help='Directory containing inference results or a single result directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (defaults to input_dir/web for single, or each result/web for batch)')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process all results in the input directory')
    parser.add_argument('--force', action='store_true',
                       help='Force reconversion even if output files exist')
    parser.add_argument('--no-stl', action='store_true',
                       help='Skip STL export (only generate PLY files)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        return 1
    
    export_stl = not args.no_stl
    
    try:
        if args.batch:
            stats = batch_convert_all(input_path, force=args.force, export_stl=export_stl)
        else:
            output_path = Path(args.output_dir) if args.output_dir else None
            stats = convert_inference_result(input_path, output_dir=output_path, 
                                           force=args.force, export_stl=export_stl)
        
        print_stats(stats)
        return 0
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

