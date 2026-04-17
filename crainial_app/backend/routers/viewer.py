"""Binary data serving endpoints for 3D viewers (vtk.js / Three.js)."""

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
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


def _parse_nrrd_spatial(header: dict) -> tuple[list[float], list[float]]:
    """Extract [spacing_x, spacing_y, spacing_z] and [origin_x, origin_y, origin_z] from an NRRD header."""
    spacing = header.get("space directions", None)
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

    origin = header.get("space origin", None)
    if origin is not None:
        origin = [float(o) for o in origin]
    else:
        origin = [0.0, 0.0, 0.0]

    return spacing, origin


@router.get("/scans/{scan_id}/volume-data")
def serve_volume_data(
    scan_id: str,
    align_to: str | None = Query(None, description="Resample this volume to match another scan's grid"),
    services=Depends(_get_services),
):
    """Parse NRRD and return raw voxel data as binary with metadata in headers.

    vtk.js does not include an NRRDReader. This endpoint reads the NRRD
    server-side and returns raw uint8 voxel data that the frontend can
    directly load into a vtkImageData object.

    When ``align_to`` is provided, the volume is resampled (nearest-neighbor
    for masks, linear for general data) to match the reference scan's grid
    so both volumes occupy the same world-space bounding box.
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

    spacing, origin = _parse_nrrd_spatial(header)

    # If align_to is set, resample this volume to match the reference scan's grid.
    if align_to:
        ref_scan = services["scan"].get_scan(align_to)
        if ref_scan:
            ref_path = Path(ref_scan.file_path)
            if ref_path.exists():
                try:
                    ref_data, ref_header = nrrd.read(str(ref_path))
                    ref_spacing, ref_origin = _parse_nrrd_spatial(ref_header)
                    ref_dims = list(ref_data.shape)

                    from scipy.ndimage import zoom as ndizoom

                    src_shape = np.array(data.shape, dtype=np.float64)
                    dst_shape = np.array(ref_dims, dtype=np.float64)
                    factors = dst_shape / src_shape

                    is_binary = scan.scan_category in ("implant", "generated_implant") or np.unique(data).size <= 3
                    order = 0 if is_binary else 1
                    data = ndizoom(data.astype(np.float32), factors, order=order)
                    if is_binary:
                        data = (data > 0.5).astype(np.float32)

                    spacing = ref_spacing
                    origin = ref_origin
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning("align_to resample failed: %s", e)

    # Normalize to uint8 for volume rendering
    data_f = data.astype(np.float64)
    d_min, d_max = data_f.min(), data_f.max()
    if d_max > d_min:
        data_u8 = ((data_f - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        data_u8 = np.zeros_like(data, dtype=np.uint8)

    data_u8 = np.ascontiguousarray(data_u8)

    metadata = {
        "dims": list(data_u8.shape),
        "spacing": spacing,
        "origin": origin,
        "dtype": "uint8",
        "scalar_range": [float(d_min), float(d_max)],
    }

    services["audit"].log(
        action="scan.view",
        entity_type="scan",
        entity_id=scan_id,
        details={"served_format": "raw_volume", "aligned_to": align_to},
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
