"""Scan CRUD + SkullBreak import endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from crainial_app.backend.database import get_db
from crainial_app.backend.schemas.point_cloud import PointCloudRead
from crainial_app.backend.schemas.scan import (
    ImportResult,
    ScanCreate,
    ScanRead,
    ScanUpdate,
    SkullBreakImportRequest,
)
from crainial_app.backend.services.audit_service import AuditService
from crainial_app.backend.services.point_cloud_service import PointCloudService
from crainial_app.backend.services.scan_service import ScanService

router = APIRouter(prefix="/api/v1/scans", tags=["scans"])


def _get_services(db: Session = Depends(get_db)):
    audit = AuditService(db)
    return {
        "scan": ScanService(db, audit),
        "point_cloud": PointCloudService(db, audit),
    }


@router.get("/", response_model=list[ScanRead])
def list_scans(
    project_id: str | None = Query(None),
    scan_category: str | None = Query(None),
    skull_id: str | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    services=Depends(_get_services),
):
    return services["scan"].list_scans(
        project_id=project_id,
        scan_category=scan_category,
        skull_id=skull_id,
        limit=limit,
        offset=offset,
    )


@router.post("/", response_model=ScanRead, status_code=201)
def create_scan(body: ScanCreate, services=Depends(_get_services)):
    try:
        return services["scan"].register_scan(
            file_path=body.file_path,
            name=body.name,
            scan_category=body.scan_category,
            defect_type=body.defect_type,
            skull_id=body.skull_id,
            project_id=body.project_id,
            description=body.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/import-skullbreak", response_model=ImportResult)
def import_skullbreak(body: SkullBreakImportRequest, services=Depends(_get_services)):
    try:
        stats = services["scan"].import_skullbreak(
            base_dir=body.base_dir,
            project_id=body.project_id,
            compute_checksums=body.compute_checksums,
        )
        return ImportResult(**stats)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{scan_id}", response_model=ScanRead)
def get_scan(scan_id: str, services=Depends(_get_services)):
    scan = services["scan"].get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


@router.put("/{scan_id}", response_model=ScanRead)
def update_scan(scan_id: str, body: ScanUpdate, services=Depends(_get_services)):
    scan = services["scan"].get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    update_data = body.model_dump(exclude_none=True)
    for key, value in update_data.items():
        setattr(scan, key, value)
    services["scan"].db.commit()
    services["scan"].db.refresh(scan)
    return scan


@router.delete("/{scan_id}", status_code=204)
def delete_scan(scan_id: str, services=Depends(_get_services)):
    if not services["scan"].delete_scan(scan_id):
        raise HTTPException(status_code=404, detail="Scan not found")


@router.get("/{scan_id}/point-clouds", response_model=list[PointCloudRead])
def list_scan_point_clouds(scan_id: str, services=Depends(_get_services)):
    return services["point_cloud"].list_point_clouds(scan_id=scan_id)
