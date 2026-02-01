"""
Fit metrics computation service for the Implant Checker.

Provides:
- Point cloud voxelization into a shared binary grid
- Dice, Hausdorff (HD / HD95), and Border Dice metrics
- SDF heatmap (per-point distance) between two point clouds
- Auto-matching skull + implant pairs by skull_id

Metric functions (dc, hd, hd95, bdc) are inlined from
voxelization/eval_metrics.py to avoid cross-package import issues.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, generate_binary_structure, binary_erosion
from scipy.spatial import KDTree
from sqlalchemy.orm import Session

from web_viewer.backend.models.fit_metrics_result import FitMetricsResult
from web_viewer.backend.models.point_cloud import PointCloud
from web_viewer.backend.services.audit_service import AuditService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inlined metric functions (from voxelization/eval_metrics.py)
# ---------------------------------------------------------------------------

def _surface_distances(
    result: np.ndarray,
    reference: np.ndarray,
    voxelspacing=None,
    connectivity: int = 1,
) -> np.ndarray:
    """Surface distances from result to reference."""
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))

    if voxelspacing is not None:
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)

    footprint = generate_binary_structure(result.ndim, connectivity)

    if np.count_nonzero(result) == 0:
        raise RuntimeError("First array contains no binary object.")
    if np.count_nonzero(reference) == 0:
        raise RuntimeError("Second array contains no binary object.")

    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    return dt[result_border]


def _dice_coefficient(input1: np.ndarray, input2: np.ndarray) -> float:
    """Dice coefficient between two binary arrays."""
    a = np.atleast_1d(input1.astype(bool))
    b = np.atleast_1d(input2.astype(bool))
    intersection = np.count_nonzero(a & b)
    total = np.count_nonzero(a) + np.count_nonzero(b)
    if total == 0:
        return 0.0
    return 2.0 * intersection / float(total)


def _hausdorff_distance(
    result: np.ndarray,
    reference: np.ndarray,
    voxelspacing=None,
    connectivity: int = 1,
) -> float:
    """Symmetric Hausdorff Distance."""
    hd1 = _surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = _surface_distances(reference, result, voxelspacing, connectivity).max()
    return float(max(hd1, hd2))


def _hausdorff_distance_95(
    result: np.ndarray,
    reference: np.ndarray,
    voxelspacing=None,
    connectivity: int = 1,
) -> float:
    """95th percentile Hausdorff Distance."""
    hd1 = _surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = _surface_distances(reference, result, voxelspacing, connectivity)
    return float(np.percentile(np.hstack((hd1, hd2)), 95))


def _border_dice(
    implant_1: np.ndarray,
    implant_2: np.ndarray,
    defective_skull: np.ndarray,
    voxelspacing=None,
    distance: int = 10,
) -> float:
    """Border Dice coefficient near the defective skull boundary."""
    dt = distance_transform_edt(~(defective_skull > 0), sampling=voxelspacing)
    i1 = implant_1.copy()
    i2 = implant_2.copy()
    i1[dt > distance] = 0
    i2[dt > distance] = 0
    return _dice_coefficient(i1, i2)


# ---------------------------------------------------------------------------
# Voxelization helpers
# ---------------------------------------------------------------------------

def _compute_shared_bbox(
    *point_arrays: np.ndarray,
    padding_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a shared axis-aligned bounding box across multiple point clouds."""
    all_mins = []
    all_maxs = []
    for pts in point_arrays:
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        all_mins.append(pts.min(axis=0))
        all_maxs.append(pts.max(axis=0))

    bbox_min = np.min(all_mins, axis=0)
    bbox_max = np.max(all_maxs, axis=0)

    # Add padding
    extent = bbox_max - bbox_min
    pad = extent * padding_fraction
    bbox_min -= pad
    bbox_max += pad

    return bbox_min, bbox_max


def _voxelize_point_cloud(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Rasterize (N, 3) points into a binary (R, R, R) occupancy volume."""
    if points.ndim == 1:
        points = points.reshape(-1, 3)

    extent = bbox_max - bbox_min
    # Map points to [0, resolution-1] voxel indices
    normalized = (points - bbox_min) / extent  # [0, 1]
    indices = (normalized * (resolution - 1)).astype(np.int32)

    # Clamp to grid
    indices = np.clip(indices, 0, resolution - 1)

    volume = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    volume[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    return volume


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class FitMetricsService:
    def __init__(self, db: Session, audit: AuditService):
        self.db = db
        self.audit = audit

    def _load_points(self, pc_id: str) -> np.ndarray:
        """Load point cloud data from disk. Returns (N, 3) array.
        
        Supports .npy, .stl, and .ply file formats.
        """
        pc = self.db.query(PointCloud).filter(PointCloud.id == pc_id).first()
        if not pc:
            raise ValueError(f"PointCloud not found: {pc_id}")
        p = Path(pc.file_path)
        if not p.exists():
            raise ValueError(f"File not found: {pc.file_path}")
        
        suffix = p.suffix.lower()
        
        if suffix == '.npy':
            # allow_pickle=True is needed for .npy files that contain object arrays
            data = np.load(str(p), allow_pickle=True)
        elif suffix == '.stl':
            # Load STL mesh and extract vertices
            import trimesh
            mesh = trimesh.load(str(p))
            data = np.array(mesh.vertices)
        elif suffix == '.ply':
            # Load PLY point cloud
            import trimesh
            cloud = trimesh.load(str(p))
            if hasattr(cloud, 'vertices'):
                data = np.array(cloud.vertices)
            else:
                data = np.array(cloud)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Ensure (N, 3) shape
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        elif data.ndim == 2 and data.shape[1] > 3:
            data = data[:, :3]
        return data.astype(np.float64)

    def compute_fit_metrics(
        self,
        implant_pc_id: str,
        reference_pc_id: str,
        defective_skull_pc_id: str | None = None,
        resolution: int = 256,
    ) -> FitMetricsResult:
        """Compute all fit metrics between two point clouds.

        Voxelizes point clouds into a shared grid and computes:
        - Dice coefficient
        - Hausdorff Distance
        - 95th percentile Hausdorff Distance
        - Border Dice (if defective skull is provided)
        """
        t0 = time.time()

        # Create result record
        result = FitMetricsResult(
            implant_pc_id=implant_pc_id,
            reference_pc_id=reference_pc_id,
            defective_skull_pc_id=defective_skull_pc_id,
            resolution=resolution,
            computation_mode="voxel",
            status="computing",
        )
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)

        try:
            # Load point clouds
            implant_pts = self._load_points(implant_pc_id)
            reference_pts = self._load_points(reference_pc_id)

            skull_pts = None
            if defective_skull_pc_id:
                skull_pts = self._load_points(defective_skull_pc_id)

            # Compute shared bounding box
            all_clouds = [implant_pts, reference_pts]
            if skull_pts is not None:
                all_clouds.append(skull_pts)
            bbox_min, bbox_max = _compute_shared_bbox(*all_clouds)

            # Voxelize
            implant_vol = _voxelize_point_cloud(implant_pts, bbox_min, bbox_max, resolution)
            reference_vol = _voxelize_point_cloud(reference_pts, bbox_min, bbox_max, resolution)

            # Compute voxel spacing in world units
            extent = bbox_max - bbox_min
            voxel_spacing = extent / resolution

            # Dice
            result.dice_coefficient = _dice_coefficient(implant_vol, reference_vol)

            # Hausdorff distances (only if both volumes have voxels)
            try:
                result.hausdorff_distance = _hausdorff_distance(
                    implant_vol, reference_vol, voxelspacing=voxel_spacing
                )
                result.hausdorff_distance_95 = _hausdorff_distance_95(
                    implant_vol, reference_vol, voxelspacing=voxel_spacing
                )
            except RuntimeError as e:
                logger.warning("Hausdorff computation failed: %s", e)

            # Border Dice (requires defective skull)
            if skull_pts is not None:
                try:
                    skull_vol = _voxelize_point_cloud(skull_pts, bbox_min, bbox_max, resolution)
                    result.boundary_dice = _border_dice(
                        implant_vol, reference_vol, skull_vol, voxelspacing=voxel_spacing
                    )
                except RuntimeError as e:
                    logger.warning("Border Dice computation failed: %s", e)

            elapsed_ms = int((time.time() - t0) * 1000)
            result.computation_time_ms = elapsed_ms
            result.status = "completed"

            logger.info(
                "Fit metrics computed: dice=%.4f hd95=%.2f bdc=%s in %dms",
                result.dice_coefficient or 0,
                result.hausdorff_distance_95 or 0,
                f"{result.boundary_dice:.4f}" if result.boundary_dice is not None else "N/A",
                elapsed_ms,
            )

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            result.computation_time_ms = int((time.time() - t0) * 1000)
            logger.error("Fit metrics computation failed: %s", e, exc_info=True)

        self.db.commit()
        self.db.refresh(result)

        self.audit.log(
            action="fit_metrics.compute",
            entity_type="fit_metrics_result",
            entity_id=result.id,
            details={
                "implant_pc_id": implant_pc_id,
                "reference_pc_id": reference_pc_id,
                "status": result.status,
                "dice": result.dice_coefficient,
            },
        )
        return result

    def compute_sdf_heatmap(
        self,
        query_pc_id: str,
        reference_pc_id: str,
    ) -> np.ndarray:
        """Compute per-point distances from query to reference point cloud.

        Returns (N,) float32 array of unsigned distances.
        """
        query_pts = self._load_points(query_pc_id)
        reference_pts = self._load_points(reference_pc_id)

        tree = KDTree(reference_pts)
        distances, _ = tree.query(query_pts)

        self.audit.log(
            action="fit_metrics.sdf_heatmap",
            entity_type="point_cloud",
            entity_id=query_pc_id,
            details={
                "reference_pc_id": reference_pc_id,
                "num_query_points": len(query_pts),
            },
        )
        return distances.astype(np.float32)

    def get_result(self, result_id: str) -> FitMetricsResult | None:
        """Get a cached metrics result by ID."""
        return (
            self.db.query(FitMetricsResult)
            .filter(FitMetricsResult.id == result_id)
            .first()
        )

    def list_results(
        self,
        implant_pc_id: str | None = None,
        reference_pc_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[FitMetricsResult]:
        """List cached metrics results with optional filters."""
        q = self.db.query(FitMetricsResult)
        if implant_pc_id:
            q = q.filter(FitMetricsResult.implant_pc_id == implant_pc_id)
        if reference_pc_id:
            q = q.filter(FitMetricsResult.reference_pc_id == reference_pc_id)
        return (
            q.order_by(FitMetricsResult.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def auto_match_by_skull_id(
        self,
        project_id: str,
    ) -> list[dict]:
        """Find defective_skull + implant/generated_implant pairs in a project.

        Groups point clouds by skull_id, returning pairs where at least one
        defective_skull and one implant/generated_implant exist.

        Returns list of dicts:
            [{"skull_id": "...", "defective_skull": PointCloud, "implants": [PointCloud, ...]}]
        """
        pcs = (
            self.db.query(PointCloud)
            .filter(
                PointCloud.project_id == project_id,
                PointCloud.skull_id.isnot(None),
            )
            .all()
        )

        # Group by skull_id
        by_skull: dict[str, dict[str, list]] = defaultdict(
            lambda: {"defective_skulls": [], "implants": []}
        )
        for pc in pcs:
            if pc.scan_category == "defective_skull":
                by_skull[pc.skull_id]["defective_skulls"].append(pc)
            elif pc.scan_category in ("implant", "generated_implant"):
                by_skull[pc.skull_id]["implants"].append(pc)

        # Build pairs (need at least one of each)
        pairs = []
        for skull_id, group in by_skull.items():
            if group["defective_skulls"] and group["implants"]:
                pairs.append({
                    "skull_id": skull_id,
                    "defective_skull": group["defective_skulls"][0],
                    "implants": group["implants"],
                })

        return pairs
