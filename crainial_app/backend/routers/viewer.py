"""Binary data serving endpoints for 3D viewers (vtk.js / Three.js)."""

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from crainial_app.backend.database import get_db
from crainial_app.backend.services.audit_service import AuditService
from crainial_app.backend.services.point_cloud_service import PointCloudService
from crainial_app.backend.services.scan_service import ScanService

router = APIRouter(prefix="/api/v1/viewer", tags=["viewer"])


def _get_services(db: Session = Depends(get_db)):
    audit = AuditService(db)
    return {
        "scan": ScanService(db, audit),
        "point_cloud": PointCloudService(db, audit),
        "audit": audit,
    }


@router.get("/scans/{scan_id}/nrrd")
def serve_nrrd(scan_id: str, services=Depends(_get_services)):
    """Serve raw NRRD file for vtk.js NrrdReader."""
    scan = services["scan"].get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    file_path = Path(scan.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="NRRD file not found on disk")

    services["audit"].log(
        action="scan.view",
        entity_type="scan",
        entity_id=scan_id,
        details={"served_format": "nrrd"},
    )

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=file_path.name,
        headers={
            "Access-Control-Expose-Headers": "Content-Length",
        },
    )


@router.get("/scans/{scan_id}/metadata")
def get_scan_metadata(scan_id: str, services=Depends(_get_services)):
    """Return volume metadata (dims, spacing, etc.) as JSON."""
    scan = services["scan"].get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    return {
        "id": scan.id,
        "name": scan.name,
        "file_format": scan.file_format,
        "dims": [scan.volume_dims_x, scan.volume_dims_y, scan.volume_dims_z],
        "spacing": [scan.voxel_spacing_x, scan.voxel_spacing_y, scan.voxel_spacing_z],
        "scan_category": scan.scan_category,
        "defect_type": scan.defect_type,
        "skull_id": scan.skull_id,
    }


@router.get("/scans/{scan_id}/volume-data")
def serve_volume_data(scan_id: str, services=Depends(_get_services)):
    """Parse NRRD and return raw voxel data as binary with metadata in headers.

    vtk.js does not include an NRRDReader. This endpoint reads the NRRD
    server-side and returns raw uint8 voxel data that the frontend can
    directly load into a vtkImageData object.
    """
    import nrrd

    scan = services["scan"].get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    file_path = Path(scan.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="NRRD file not found on disk")

    try:
        data, header = nrrd.read(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read NRRD: {e}")

    # Normalize to uint8 for volume rendering
    data_f = data.astype(np.float64)
    d_min, d_max = data_f.min(), data_f.max()
    if d_max > d_min:
        data_u8 = ((data_f - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        data_u8 = np.zeros_like(data, dtype=np.uint8)

    # Ensure C-contiguous / Fortran order matching vtk.js expectations
    data_u8 = np.ascontiguousarray(data_u8)

    dims = list(data_u8.shape)
    spacing = header.get("space directions", None)
    origin = header.get("space origin", None)

    # Parse spacing from space directions (diagonal of 3x3 matrix)
    if spacing is not None:
        try:
            spacing = [abs(float(spacing[i][i])) for i in range(3)]
        except (IndexError, TypeError):
            spacing = [1.0, 1.0, 1.0]
    else:
        spacings_field = header.get("spacings", None)
        if spacings_field is not None:
            spacing = [float(s) for s in spacings_field]
        else:
            spacing = [1.0, 1.0, 1.0]

    if origin is not None:
        origin = [float(o) for o in origin]
    else:
        origin = [0.0, 0.0, 0.0]

    metadata = {
        "dims": dims,
        "spacing": spacing,
        "origin": origin,
        "dtype": "uint8",
        "scalar_range": [float(d_min), float(d_max)],
    }

    services["audit"].log(
        action="scan.view",
        entity_type="scan",
        entity_id=scan_id,
        details={"served_format": "raw_volume"},
    )

    return Response(
        content=data_u8.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Volume-Metadata": json.dumps(metadata),
            "Access-Control-Expose-Headers": "X-Volume-Metadata, Content-Length",
        },
    )


@router.get("/point-clouds/{pc_id}/npy")
def serve_npy(pc_id: str, services=Depends(_get_services)):
    """Serve raw NPY file for frontend loading."""
    pc = services["point_cloud"].get_point_cloud(pc_id)
    if not pc:
        raise HTTPException(status_code=404, detail="PointCloud not found")

    file_path = Path(pc.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="NPY file not found on disk")

    services["audit"].log(
        action="point_cloud.view",
        entity_type="point_cloud",
        entity_id=pc_id,
        details={"served_format": "npy"},
    )

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=file_path.name,
    )
