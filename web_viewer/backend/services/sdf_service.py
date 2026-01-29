"""
Signed Distance Field (SDF) computation service.

Computes distance from each point in a point cloud to the nearest surface
of a reference volume. Positive = outside, negative = inside.

SDF is computed server-side (CPU-intensive EDT on 512^3 volumes), while
color mapping is done client-side for real-time profile switching.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_sdf_from_volume(
    points: np.ndarray,
    nrrd_path: str | Path,
    surface_threshold: float = 0.5,
) -> np.ndarray:
    """Compute approximate SDF by sampling the volume at point locations.

    1. Load the NRRD binary volume.
    2. Compute Euclidean Distance Transform for exterior and interior.
    3. Combine: SDF = exterior_distance - interior_distance.
    4. Interpolate SDF at point cloud positions.

    Args:
        points: (N, 3) array of point positions in world/voxel coordinates.
        nrrd_path: Path to the NRRD binary volume.
        surface_threshold: Binarization threshold for the volume.

    Returns:
        (N,) float32 array of SDF values.
    """
    import nrrd
    from scipy.ndimage import distance_transform_edt, map_coordinates

    nrrd_path = Path(nrrd_path)
    volume, header = nrrd.read(str(nrrd_path))

    # Binarize
    mask = (volume > surface_threshold).astype(np.float32)

    # Compute spacing from header
    spacing = _extract_spacing(header)

    # Euclidean distance transforms
    dist_outside = distance_transform_edt(1 - mask, sampling=spacing)
    dist_inside = distance_transform_edt(mask, sampling=spacing)

    # SDF: positive outside, negative inside
    sdf_volume = dist_outside - dist_inside

    # Map world coordinates to voxel indices
    voxel_coords = _world_to_voxel(points, header)

    # Trilinear interpolation
    sdf_values = map_coordinates(
        sdf_volume,
        voxel_coords.T,
        order=1,
        mode="nearest",
    )

    return sdf_values.astype(np.float32)


def compute_sdf_between_point_clouds(
    query_points: np.ndarray,
    reference_points: np.ndarray,
) -> np.ndarray:
    """Compute approximate SDF as distance from query points to nearest
    reference point. Sign is not available without a surface orientation,
    so this returns unsigned distances.

    Args:
        query_points: (N, 3) points to compute SDF for.
        reference_points: (M, 3) reference surface points.

    Returns:
        (N,) float32 array of unsigned distances.
    """
    from scipy.spatial import KDTree

    tree = KDTree(reference_points)
    distances, _ = tree.query(query_points)
    return distances.astype(np.float32)


def _extract_spacing(header: dict) -> tuple[float, ...]:
    """Extract voxel spacing from NRRD header."""
    spacings = header.get("spacings")
    if spacings is not None:
        return tuple(float(s) for s in spacings)

    space_dirs = header.get("space directions")
    if space_dirs is not None:
        spacing = []
        for row in space_dirs:
            if row is not None and hasattr(row, "__len__"):
                spacing.append(float(sum(x**2 for x in row) ** 0.5))
        if spacing:
            return tuple(spacing)

    # Fallback: assume isotropic 1.0
    sizes = header.get("sizes", [1, 1, 1])
    return tuple(1.0 for _ in sizes)


def _world_to_voxel(points: np.ndarray, header: dict) -> np.ndarray:
    """Convert world-coordinate points to voxel indices.

    If the NRRD has a 'space origin' and 'space directions', use them.
    Otherwise assume identity transform.
    """
    pts = np.asarray(points, dtype=np.float64)

    space_origin = header.get("space origin")
    space_dirs = header.get("space directions")

    if space_origin is not None and space_dirs is not None:
        origin = np.array(space_origin, dtype=np.float64)
        dirs = np.array(
            [d for d in space_dirs if d is not None and hasattr(d, "__len__")],
            dtype=np.float64,
        )
        if dirs.shape == (3, 3):
            # voxel = inv(dirs) @ (world - origin)
            inv_dirs = np.linalg.inv(dirs)
            voxel_coords = (pts - origin) @ inv_dirs.T
            return voxel_coords

    # Fallback: use spacing only
    spacing = _extract_spacing(header)
    voxel_coords = pts / np.array(spacing)
    return voxel_coords
