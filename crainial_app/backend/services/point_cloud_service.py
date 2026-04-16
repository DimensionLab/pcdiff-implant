"""
Service for managing PointCloud records (NPY / PLY / STL surface files).
"""

import json
import logging
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from crainial_app.backend.models.point_cloud import PointCloud
from crainial_app.backend.services.audit_service import AuditService
from crainial_app.backend.services.file_service import (
    ALLOWED_POINT_CLOUD_EXTENSIONS,
    compute_sha256,
    get_file_size,
    read_npy_metadata,
    validate_file_path,
)

logger = logging.getLogger(__name__)


class PointCloudService:
    def __init__(self, db: Session, audit: AuditService):
        self.db = db
        self.audit = audit

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_point_clouds(
        self,
        project_id: str | None = None,
        scan_id: str | None = None,
        scan_category: str | None = None,
        skull_id: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[PointCloud]:
        q = self.db.query(PointCloud)
        if project_id:
            q = q.filter(PointCloud.project_id == project_id)
        if scan_id:
            q = q.filter(PointCloud.scan_id == scan_id)
        if scan_category:
            q = q.filter(PointCloud.scan_category == scan_category)
        if skull_id:
            q = q.filter(PointCloud.skull_id == skull_id)
        return q.order_by(PointCloud.name).offset(offset).limit(limit).all()

    def get_point_cloud(self, pc_id: str) -> PointCloud | None:
        return self.db.query(PointCloud).filter(PointCloud.id == pc_id).first()

    def register_point_cloud(
        self,
        file_path: str,
        name: str | None = None,
        scan_category: str | None = None,
        defect_type: str | None = None,
        skull_id: str | None = None,
        project_id: str | None = None,
        scan_id: str | None = None,
        description: str | None = None,
        compute_checksum: bool = True,
    ) -> PointCloud:
        """Register a filesystem path as a point cloud."""
        p = validate_file_path(file_path, ALLOWED_POINT_CLOUD_EXTENSIONS)
        fmt = p.suffix.lower().lstrip(".")

        meta = {}
        num_points = None
        point_dims = 3
        if fmt == "npy":
            meta = read_npy_metadata(p)
            num_points = meta.get("num_points")
            point_dims = meta.get("point_dims", 3)

        pc = PointCloud(
            name=name or p.stem,
            description=description,
            file_path=str(p),
            file_format=fmt,
            file_size_bytes=get_file_size(p),
            num_points=num_points,
            point_dims=point_dims,
            scan_category=scan_category,
            defect_type=defect_type,
            skull_id=skull_id,
            project_id=project_id,
            scan_id=scan_id,
            metadata_json=json.dumps(meta, default=str) if meta else None,
            checksum_sha256=compute_sha256(p) if compute_checksum else None,
        )
        self.db.add(pc)
        self.db.commit()
        self.db.refresh(pc)

        self.audit.log(
            action="point_cloud.register",
            entity_type="point_cloud",
            entity_id=pc.id,
            details={"file_path": str(p), "scan_category": scan_category},
        )
        return pc

    def delete_point_cloud(self, pc_id: str) -> bool:
        pc = self.get_point_cloud(pc_id)
        if not pc:
            return False
        self.db.delete(pc)
        self.db.commit()
        self.audit.log(
            action="point_cloud.delete",
            entity_type="point_cloud",
            entity_id=pc_id,
        )
        return True

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_point_cloud_data(self, pc_id: str) -> np.ndarray:
        """Load the raw point cloud data from disk.

        Returns:
            numpy array of shape (N, 3) for NPY files.

        Raises:
            ValueError: If the point cloud or file is not found.
        """
        pc = self.get_point_cloud(pc_id)
        if not pc:
            raise ValueError(f"PointCloud not found: {pc_id}")

        p = Path(pc.file_path)
        if not p.exists():
            raise ValueError(f"File not found on disk: {pc.file_path}")

        if pc.file_format == "npy":
            data = np.load(str(p))
            self.audit.log(
                action="point_cloud.load_data",
                entity_type="point_cloud",
                entity_id=pc_id,
                details={"shape": list(data.shape)},
            )
            return data

        raise ValueError(f"Unsupported format for data loading: {pc.file_format}")
