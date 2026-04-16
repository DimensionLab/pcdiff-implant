"""
Service for managing Scan records (NRRD / DICOM volumes).

Handles registration, metadata extraction, and bulk import of
the SkullBreak dataset.
"""

import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from crainial_app.backend.models.point_cloud import PointCloud
from crainial_app.backend.models.scan import Scan
from crainial_app.backend.services.audit_service import AuditService
from crainial_app.backend.services.file_service import (
    ALLOWED_POINT_CLOUD_EXTENSIONS,
    ALLOWED_SCAN_EXTENSIONS,
    compute_sha256,
    get_file_size,
    read_npy_metadata,
    read_nrrd_metadata,
    validate_file_path,
)

logger = logging.getLogger(__name__)


class ScanService:
    def __init__(self, db: Session, audit: AuditService):
        self.db = db
        self.audit = audit

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_scans(
        self,
        project_id: str | None = None,
        scan_category: str | None = None,
        skull_id: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Scan]:
        q = self.db.query(Scan)
        if project_id:
            q = q.filter(Scan.project_id == project_id)
        if scan_category:
            q = q.filter(Scan.scan_category == scan_category)
        if skull_id:
            q = q.filter(Scan.skull_id == skull_id)
        return q.order_by(Scan.name).offset(offset).limit(limit).all()

    def get_scan(self, scan_id: str) -> Scan | None:
        return self.db.query(Scan).filter(Scan.id == scan_id).first()

    def register_scan(
        self,
        file_path: str,
        name: str | None = None,
        scan_category: str | None = None,
        defect_type: str | None = None,
        skull_id: str | None = None,
        project_id: str | None = None,
        description: str | None = None,
        compute_checksum: bool = True,
    ) -> Scan:
        """Register a filesystem path as a scan.

        Validates the file exists, reads NRRD metadata, computes checksum.
        """
        p = validate_file_path(file_path, ALLOWED_SCAN_EXTENSIONS)

        # Read metadata
        meta = read_nrrd_metadata(p)
        dims = meta.get("dims", [])
        spacing = meta.get("spacing", [])

        scan = Scan(
            name=name or p.stem,
            description=description,
            file_path=str(p),
            file_format="nrrd",
            file_size_bytes=get_file_size(p),
            volume_dims_x=dims[0] if len(dims) > 0 else None,
            volume_dims_y=dims[1] if len(dims) > 1 else None,
            volume_dims_z=dims[2] if len(dims) > 2 else None,
            voxel_spacing_x=spacing[0] if len(spacing) > 0 else None,
            voxel_spacing_y=spacing[1] if len(spacing) > 1 else None,
            voxel_spacing_z=spacing[2] if len(spacing) > 2 else None,
            scan_category=scan_category,
            defect_type=defect_type,
            skull_id=skull_id,
            project_id=project_id,
            metadata_json=json.dumps(meta, default=str) if meta else None,
            checksum_sha256=compute_sha256(p) if compute_checksum else None,
        )
        self.db.add(scan)
        self.db.commit()
        self.db.refresh(scan)

        self.audit.log(
            action="scan.register",
            entity_type="scan",
            entity_id=scan.id,
            details={"file_path": str(p), "scan_category": scan_category},
        )
        return scan

    def delete_scan(self, scan_id: str) -> bool:
        scan = self.get_scan(scan_id)
        if not scan:
            return False
        self.db.delete(scan)
        self.db.commit()
        self.audit.log(
            action="scan.delete",
            entity_type="scan",
            entity_id=scan_id,
        )
        return True

    # ------------------------------------------------------------------
    # Bulk import
    # ------------------------------------------------------------------

    def import_skullbreak(
        self,
        base_dir: str | Path,
        project_id: str | None = None,
        compute_checksums: bool = False,
    ) -> dict:
        """Bulk-import the SkullBreak dataset.

        Scans the directory structure:
          complete_skull/*.nrrd + *_surf.npy
          defective_skull/{type}/*.nrrd + *_surf.npy
          implant/{type}/*.nrrd + *_surf.npy

        Returns import statistics.
        """
        base = Path(base_dir)
        if not base.is_dir():
            raise ValueError(f"SkullBreak directory not found: {base}")

        stats = {"scans_created": 0, "point_clouds_created": 0, "skipped": 0, "errors": []}

        # -- complete skulls --
        self._import_category(
            base / "complete_skull",
            scan_category="complete_skull",
            defect_type=None,
            project_id=project_id,
            compute_checksums=compute_checksums,
            stats=stats,
        )

        # -- defective skulls --
        defective_dir = base / "defective_skull"
        if defective_dir.is_dir():
            for type_dir in sorted(defective_dir.iterdir()):
                if type_dir.is_dir():
                    self._import_category(
                        type_dir,
                        scan_category="defective_skull",
                        defect_type=type_dir.name,
                        project_id=project_id,
                        compute_checksums=compute_checksums,
                        stats=stats,
                    )

        # -- implants --
        implant_dir = base / "implant"
        if implant_dir.is_dir():
            for type_dir in sorted(implant_dir.iterdir()):
                if type_dir.is_dir():
                    self._import_category(
                        type_dir,
                        scan_category="implant",
                        defect_type=type_dir.name,
                        project_id=project_id,
                        compute_checksums=compute_checksums,
                        stats=stats,
                    )

        self.audit.log(
            action="scan.import_skullbreak",
            details={"base_dir": str(base), **stats},
        )
        return stats

    def _import_category(
        self,
        directory: Path,
        scan_category: str,
        defect_type: str | None,
        project_id: str | None,
        compute_checksums: bool,
        stats: dict,
    ) -> None:
        if not directory.is_dir():
            return

        nrrd_files = sorted(directory.glob("*.nrrd"))
        for nrrd_path in nrrd_files:
            skull_id = nrrd_path.stem  # e.g. '059'

            # Skip if already registered
            existing = self.db.query(Scan).filter(Scan.file_path == str(nrrd_path)).first()
            if existing:
                stats["skipped"] += 1
                continue

            try:
                scan = self.register_scan(
                    file_path=str(nrrd_path),
                    name=f"{scan_category}/{defect_type or ''}/{skull_id}".strip("/"),
                    scan_category=scan_category,
                    defect_type=defect_type,
                    skull_id=skull_id,
                    project_id=project_id,
                    compute_checksum=compute_checksums,
                )
                stats["scans_created"] += 1

                # Look for matching surface point cloud
                npy_path = nrrd_path.with_name(f"{skull_id}_surf.npy")
                if npy_path.exists():
                    self._register_point_cloud_for_scan(
                        scan=scan,
                        npy_path=npy_path,
                        scan_category=scan_category,
                        defect_type=defect_type,
                        skull_id=skull_id,
                        project_id=project_id,
                        compute_checksums=compute_checksums,
                    )
                    stats["point_clouds_created"] += 1

            except Exception as e:
                logger.error("Failed to import %s: %s", nrrd_path, e)
                stats["errors"].append({"file": str(nrrd_path), "error": str(e)})

    def _register_point_cloud_for_scan(
        self,
        scan: Scan,
        npy_path: Path,
        scan_category: str,
        defect_type: str | None,
        skull_id: str,
        project_id: str | None,
        compute_checksums: bool,
    ) -> PointCloud:
        npy_meta = read_npy_metadata(npy_path)
        pc = PointCloud(
            name=f"{scan.name}_surf",
            file_path=str(npy_path),
            file_format="npy",
            file_size_bytes=get_file_size(npy_path),
            num_points=npy_meta.get("num_points"),
            point_dims=npy_meta.get("point_dims", 3),
            scan_category=scan_category,
            defect_type=defect_type,
            skull_id=skull_id,
            project_id=project_id,
            scan_id=scan.id,
            metadata_json=json.dumps(npy_meta, default=str) if npy_meta else None,
            checksum_sha256=compute_sha256(npy_path) if compute_checksums else None,
        )
        self.db.add(pc)
        self.db.commit()
        self.db.refresh(pc)

        self.audit.log(
            action="point_cloud.register",
            entity_type="point_cloud",
            entity_id=pc.id,
            details={"file_path": str(npy_path), "linked_scan_id": scan.id},
        )
        return pc
