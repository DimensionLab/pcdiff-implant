"""PointCloud CRUD + data/SDF serving endpoints."""

import io
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from web_viewer.backend.database import get_db
from web_viewer.backend.schemas.point_cloud import (
    PointCloudCreate,
    PointCloudRead,
    PointCloudUpdate,
    SDFRequest,
    STLStatusResponse,
)
from web_viewer.backend.services.audit_service import AuditService
from web_viewer.backend.services.mesh_service import MeshService
from web_viewer.backend.services.point_cloud_service import PointCloudService
from web_viewer.backend.services.scan_service import ScanService
from web_viewer.backend.services.sdf_service import compute_sdf_from_volume

router = APIRouter(prefix="/api/v1/point-clouds", tags=["point-clouds"])


def _get_services(db: Session = Depends(get_db)):
    audit = AuditService(db)
    return {
        "point_cloud": PointCloudService(db, audit),
        "scan": ScanService(db, audit),
        "mesh": MeshService(db, audit),
    }


@router.get("/", response_model=list[PointCloudRead])
def list_point_clouds(
    project_id: str | None = Query(None),
    scan_id: str | None = Query(None),
    scan_category: str | None = Query(None),
    skull_id: str | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    services=Depends(_get_services),
):
    return services["point_cloud"].list_point_clouds(
        project_id=project_id,
        scan_id=scan_id,
        scan_category=scan_category,
        skull_id=skull_id,
        limit=limit,
        offset=offset,
    )


@router.post("/", response_model=PointCloudRead, status_code=201)
def create_point_cloud(body: PointCloudCreate, services=Depends(_get_services)):
    try:
        return services["point_cloud"].register_point_cloud(
            file_path=body.file_path,
            name=body.name,
            scan_category=body.scan_category,
            defect_type=body.defect_type,
            skull_id=body.skull_id,
            project_id=body.project_id,
            scan_id=body.scan_id,
            description=body.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{pc_id}", response_model=PointCloudRead)
def get_point_cloud(pc_id: str, services=Depends(_get_services)):
    pc = services["point_cloud"].get_point_cloud(pc_id)
    if not pc:
        raise HTTPException(status_code=404, detail="PointCloud not found")
    return pc


@router.put("/{pc_id}", response_model=PointCloudRead)
def update_point_cloud(pc_id: str, body: PointCloudUpdate, services=Depends(_get_services)):
    pc = services["point_cloud"].get_point_cloud(pc_id)
    if not pc:
        raise HTTPException(status_code=404, detail="PointCloud not found")

    update_data = body.model_dump(exclude_none=True)
    for key, value in update_data.items():
        setattr(pc, key, value)
    services["point_cloud"].db.commit()
    services["point_cloud"].db.refresh(pc)
    return pc


@router.delete("/{pc_id}", status_code=204)
def delete_point_cloud(pc_id: str, services=Depends(_get_services)):
    if not services["point_cloud"].delete_point_cloud(pc_id):
        raise HTTPException(status_code=404, detail="PointCloud not found")


@router.get("/{pc_id}/data")
def get_point_cloud_data(pc_id: str, services=Depends(_get_services)):
    """Stream the raw point cloud data as a binary NPY file."""
    try:
        data = services["point_cloud"].load_point_cloud_data(pc_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    buf = io.BytesIO()
    np.save(buf, data)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{pc_id}.npy"',
            "X-Num-Points": str(data.shape[0]),
            "X-Point-Dims": str(data.shape[1] if data.ndim > 1 else 1),
        },
    )


@router.get("/{pc_id}/sdf")
def get_sdf_values(pc_id: str, services=Depends(_get_services)):
    """Compute SDF for this point cloud using its linked scan volume.

    Returns raw float32 binary data (N floats).
    """
    pc = services["point_cloud"].get_point_cloud(pc_id)
    if not pc:
        raise HTTPException(status_code=404, detail="PointCloud not found")

    if not pc.scan_id:
        raise HTTPException(
            status_code=400,
            detail="No linked scan for SDF computation. Use POST with reference_scan_id.",
        )

    scan = services["scan"].get_scan(pc.scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Linked scan not found")

    try:
        points = services["point_cloud"].load_point_cloud_data(pc_id)
        sdf = compute_sdf_from_volume(points, scan.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SDF computation failed: {e}")

    buf = io.BytesIO(sdf.tobytes())
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Dtype": "float32",
            "X-Count": str(len(sdf)),
        },
    )


@router.post("/{pc_id}/sdf")
def compute_sdf_custom(pc_id: str, body: SDFRequest, services=Depends(_get_services)):
    """Compute SDF with custom parameters (e.g. a different reference scan)."""
    pc = services["point_cloud"].get_point_cloud(pc_id)
    if not pc:
        raise HTTPException(status_code=404, detail="PointCloud not found")

    ref_scan_id = body.reference_scan_id or pc.scan_id
    if not ref_scan_id:
        raise HTTPException(status_code=400, detail="No reference scan provided")

    scan = services["scan"].get_scan(ref_scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Reference scan not found")

    try:
        points = services["point_cloud"].load_point_cloud_data(pc_id)
        sdf = compute_sdf_from_volume(points, scan.file_path, surface_threshold=body.surface_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SDF computation failed: {e}")

    buf = io.BytesIO(sdf.tobytes())
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={
            "X-Dtype": "float32",
            "X-Count": str(len(sdf)),
        },
    )


# ------------------------------------------------------------------
# STL mesh generation
# ------------------------------------------------------------------


@router.post("/{pc_id}/generate-stl", response_model=PointCloudRead, status_code=201)
def generate_stl(
    pc_id: str,
    method: str = Query("poisson"),
    depth: int = Query(8, ge=5, le=10),
    services=Depends(_get_services),
):
    """Generate a watertight STL mesh from a point cloud.

    Uses Open3D Poisson surface reconstruction by default.
    Returns the existing STL if one was already generated for this source.
    """
    try:
        return services["mesh"].generate_stl(
            source_pc_id=pc_id,
            method=method,
            poisson_depth=depth,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Mesh generation failed: {e}",
        )


@router.get("/{pc_id}/stl-status", response_model=STLStatusResponse)
def get_stl_status(pc_id: str, services=Depends(_get_services)):
    """Check if an STL mesh has been generated for this point cloud."""
    pc = services["point_cloud"].get_point_cloud(pc_id)
    if not pc:
        raise HTTPException(status_code=404, detail="PointCloud not found")

    existing = services["mesh"].get_stl_for_source(pc_id)
    if existing and existing.metadata_json:
        import json

        meta = json.loads(existing.metadata_json)
    else:
        meta = None

    return STLStatusResponse(
        has_stl=existing is not None,
        stl_pc_id=existing.id if existing else None,
        stl_metadata=meta,
        source_pc_id=pc_id,
    )


@router.get("/{pc_id}/stl")
def download_stl(pc_id: str, services=Depends(_get_services)):
    """Download the STL file for a point cloud.

    Works with both source point cloud IDs (finds the linked STL)
    and direct STL point cloud IDs.
    """
    pc = services["point_cloud"].get_point_cloud(pc_id)
    if not pc:
        raise HTTPException(status_code=404, detail="PointCloud not found")

    # If this is an STL record, serve it directly
    if pc.file_format == "stl":
        stl_pc = pc
    else:
        # Look for a generated STL for this source
        stl_pc = services["mesh"].get_stl_for_source(pc_id)
        if not stl_pc:
            raise HTTPException(
                status_code=404,
                detail="No STL mesh found. Generate one first via POST /generate-stl",
            )

    try:
        data = services["mesh"].load_stl_binary(stl_pc.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    filename = Path(stl_pc.file_path).name
    buf = io.BytesIO(data)

    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(data)),
            "X-Num-Faces": str(_get_meta_field(stl_pc, "num_faces", "unknown")),
            "X-Is-Watertight": str(_get_meta_field(stl_pc, "is_watertight", "unknown")),
        },
    )


def _get_meta_field(pc: "PointCloud", field: str, default: str = "") -> str:
    """Extract a field from a PointCloud's metadata_json."""
    if pc.metadata_json:
        import json

        try:
            meta = json.loads(pc.metadata_json)
            return str(meta.get(field, default))
        except (json.JSONDecodeError, TypeError):
            pass
    return default
