"""
Service for generating watertight STL meshes from point clouds.

Uses Open3D Poisson surface reconstruction as the primary method,
producing 3D-print-ready meshes from cranial implant point clouds.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from sqlalchemy.orm import Session

from web_viewer.backend.models.point_cloud import PointCloud
from web_viewer.backend.services.audit_service import AuditService
from web_viewer.backend.services.file_service import get_file_size

logger = logging.getLogger(__name__)


class MeshService:
    def __init__(self, db: Session, audit: AuditService):
        self.db = db
        self.audit = audit

    def generate_stl(
        self,
        source_pc_id: str,
        method: str = "poisson",
        poisson_depth: int = 9,
        project_id: str | None = None,
    ) -> PointCloud:
        """Generate a watertight STL mesh from an NPY point cloud.

        Args:
            source_pc_id: ID of the source point cloud (must be NPY format).
            method: Reconstruction method ('poisson', 'ball_pivoting', 'convex_hull').
            poisson_depth: Octree depth for Poisson reconstruction (higher = more detail).
            project_id: Optional project to associate the STL with.

        Returns:
            New PointCloud record for the generated STL file.

        Raises:
            ValueError: If the source point cloud is not found or invalid.
        """
        # Check if STL already exists
        existing = self.get_stl_for_source(source_pc_id)
        if existing:
            return existing

        # Load source point cloud
        source_pc = self.db.query(PointCloud).filter(PointCloud.id == source_pc_id).first()
        if not source_pc:
            raise ValueError(f"Source point cloud not found: {source_pc_id}")

        source_path = Path(source_pc.file_path)
        if not source_path.exists():
            raise ValueError(f"Source file not found on disk: {source_pc.file_path}")

        if source_pc.file_format != "npy":
            raise ValueError(f"STL generation requires NPY format, got: {source_pc.file_format}")

        # Load points
        points = np.load(str(source_path))
        if points.ndim == 3:
            points = points[0] if points.shape[0] == 1 else points.reshape(-1, 3)
        points = points.reshape(-1, 3).astype(np.float64)

        logger.info(
            "Generating STL mesh from %s (%d points, method=%s)",
            source_pc.name,
            len(points),
            method,
        )

        t0 = time.time()
        mesh = self._reconstruct(points, method, poisson_depth)
        elapsed_ms = int((time.time() - t0) * 1000)

        # Determine output path
        stl_path = source_path.with_name(f"{source_path.stem}_mesh.stl")

        # Export binary STL
        mesh.export(str(stl_path), file_type="stl")
        logger.info(
            "STL exported: %s (%d faces, watertight=%s, %dms)",
            stl_path.name,
            len(mesh.faces),
            mesh.is_watertight,
            elapsed_ms,
        )

        # Register as PointCloud
        metadata = {
            "source_pc_id": source_pc_id,
            "method": method,
            "num_faces": len(mesh.faces),
            "num_vertices": len(mesh.vertices),
            "is_watertight": mesh.is_watertight,
            "generation_time_ms": elapsed_ms,
        }
        if method == "poisson":
            metadata["poisson_depth"] = poisson_depth

        stl_pc = PointCloud(
            name=f"{source_pc.name} (STL Mesh)",
            description=f"3D-printable mesh generated from {source_pc.name}",
            file_path=str(stl_path),
            file_format="stl",
            file_size_bytes=get_file_size(stl_path),
            num_points=None,
            point_dims=3,
            scan_category=source_pc.scan_category,
            defect_type=source_pc.defect_type,
            skull_id=source_pc.skull_id,
            project_id=project_id or source_pc.project_id,
            metadata_json=json.dumps(metadata),
        )
        self.db.add(stl_pc)
        self.db.commit()
        self.db.refresh(stl_pc)

        self.audit.log(
            action="mesh.generate_stl",
            entity_type="point_cloud",
            entity_id=stl_pc.id,
            details={
                "source_pc_id": source_pc_id,
                "method": method,
                "num_faces": len(mesh.faces),
                "is_watertight": mesh.is_watertight,
                "generation_time_ms": elapsed_ms,
            },
        )

        return stl_pc

    def get_stl_for_source(self, source_pc_id: str) -> PointCloud | None:
        """Check if an STL mesh has already been generated for a source point cloud."""
        all_stl = self.db.query(PointCloud).filter(PointCloud.file_format == "stl").all()
        for pc in all_stl:
            if pc.metadata_json:
                try:
                    meta = json.loads(pc.metadata_json)
                    if meta.get("source_pc_id") == source_pc_id:
                        # Verify file still exists
                        if Path(pc.file_path).exists():
                            return pc
                except (json.JSONDecodeError, TypeError):
                    pass
        return None

    def load_stl_binary(self, stl_pc_id: str) -> bytes:
        """Load the raw STL binary data from disk."""
        pc = self.db.query(PointCloud).filter(PointCloud.id == stl_pc_id).first()
        if not pc:
            raise ValueError(f"PointCloud not found: {stl_pc_id}")

        p = Path(pc.file_path)
        if not p.exists():
            raise ValueError(f"STL file not found on disk: {pc.file_path}")

        return p.read_bytes()

    # ------------------------------------------------------------------
    # Reconstruction methods
    # ------------------------------------------------------------------

    # Max points before voxel downsampling is applied
    _MAX_POINTS_FOR_POISSON = 80_000

    def _reconstruct(
        self,
        points: np.ndarray,
        method: str,
        poisson_depth: int,
    ) -> trimesh.Trimesh:
        """Run surface reconstruction with fallback chain."""
        if method == "poisson":
            try:
                return self._poisson_reconstruct(points, poisson_depth)
            except Exception as e:
                logger.warning("Poisson reconstruction failed (%s), trying ball_pivoting", e)
                method = "ball_pivoting"

        if method == "ball_pivoting":
            try:
                return self._ball_pivoting_reconstruct(points)
            except Exception as e:
                logger.warning("Ball pivoting failed (%s), falling back to convex_hull", e)

        # Final fallback
        return self._convex_hull_reconstruct(points)

    def _prepare_point_cloud(self, points: np.ndarray) -> tuple[o3d.geometry.PointCloud, float]:
        """Create Open3D point cloud, downsample if needed, estimate normals.

        Returns the prepared point cloud and the auto-computed normal radius.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        n_original = len(points)

        # Uniform downsample: take every k-th point.
        # This is spatial-distribution-agnostic (works for surfaces, not just
        # volume-filling clouds where voxel downsampling would be appropriate).
        if n_original > self._MAX_POINTS_FOR_POISSON:
            every_k = max(2, n_original // self._MAX_POINTS_FOR_POISSON)
            pcd = pcd.uniform_down_sample(every_k)
            logger.info(
                "Downsampled %d -> %d points (every_k=%d)",
                n_original,
                len(pcd.points),
                every_k,
            )

        # Normal estimation radius from actual nearest-neighbor distances
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = float(np.mean(distances))
        normal_radius = avg_dist * 6.0  # cover ~6 nearest neighbors

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=30,
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        logger.info(
            "Normals estimated (radius=%.4f, avg_nn_dist=%.4f, %d points)",
            normal_radius,
            avg_dist,
            len(pcd.points),
        )

        return pcd, normal_radius

    def _poisson_reconstruct(self, points: np.ndarray, depth: int = 9) -> trimesh.Trimesh:
        """Open3D Poisson surface reconstruction."""
        pcd, _nr = self._prepare_point_cloud(points)

        # Scale depth with point count — fewer points need lower depth
        n_pts = len(pcd.points)
        effective_depth = depth
        if n_pts < 10_000:
            effective_depth = min(depth, 7)
        elif n_pts < 50_000:
            effective_depth = min(depth, 8)

        logger.info("Running Poisson reconstruction (depth=%d, %d points)", effective_depth, n_pts)

        # Poisson reconstruction
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=effective_depth, scale=1.1, linear_fit=False
        )

        # Check that we got geometry
        if len(mesh_o3d.vertices) == 0 or len(mesh_o3d.triangles) == 0:
            raise ValueError("Poisson produced empty mesh")

        # Remove low-density vertices (Poisson "skirt" artifacts)
        densities_np = np.asarray(densities)
        if len(densities_np) > 0:
            threshold = np.quantile(densities_np, 0.01)
            vertices_to_remove = densities_np < threshold
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

        # Convert to trimesh
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        if len(faces) == 0:
            raise ValueError("Poisson mesh empty after density cleanup")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Repair if needed
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)

        return mesh

    def _ball_pivoting_reconstruct(self, points: np.ndarray) -> trimesh.Trimesh:
        """Open3D Ball Pivoting Algorithm reconstruction."""
        pcd, _nr = self._prepare_point_cloud(points)

        # Estimate ball radius from point spacing
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * f for f in [1.0, 1.5, 2.0, 3.0]]

        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        if len(faces) == 0:
            raise ValueError("Ball pivoting produced no faces")

        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def _convex_hull_reconstruct(self, points: np.ndarray) -> trimesh.Trimesh:
        """Trimesh convex hull fallback."""
        pc = trimesh.PointCloud(vertices=points)
        return pc.convex_hull
