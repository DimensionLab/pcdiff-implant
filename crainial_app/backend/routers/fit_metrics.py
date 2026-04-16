"""Fit metrics computation endpoints for the Implant Checker."""

import io

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from crainial_app.backend.database import get_db
from crainial_app.backend.schemas.fit_metrics import (
    FitMetricsRead,
    FitMetricsRequest,
    SDFHeatmapRequest,
)
from crainial_app.backend.services.audit_service import AuditService
from crainial_app.backend.services.fit_metrics_service import FitMetricsService

router = APIRouter(prefix="/api/v1/fit-metrics", tags=["fit-metrics"])


def _get_service(db: Session = Depends(get_db)) -> FitMetricsService:
    audit = AuditService(db)
    return FitMetricsService(db, audit)


@router.post("/compute", response_model=FitMetricsRead, status_code=201)
def compute_fit_metrics(body: FitMetricsRequest, svc: FitMetricsService = Depends(_get_service)):
    """Compute fit metrics between two point clouds.

    Voxelizes both point clouds into a shared grid and computes Dice, HD, HD95,
    and optionally Border Dice (if defective skull is provided).
    """
    try:
        result = svc.compute_fit_metrics(
            implant_pc_id=body.implant_pc_id,
            reference_pc_id=body.reference_pc_id,
            defective_skull_pc_id=body.defective_skull_pc_id,
            resolution=body.resolution,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result.status == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Metric computation failed: {result.error_message}",
        )

    return result


@router.get("/{result_id}", response_model=FitMetricsRead)
def get_fit_metrics_result(result_id: str, svc: FitMetricsService = Depends(_get_service)):
    """Get a cached fit metrics result by ID."""
    result = svc.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@router.get("/", response_model=list[FitMetricsRead])
def list_fit_metrics_results(
    implant_pc_id: str | None = Query(None),
    reference_pc_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    svc: FitMetricsService = Depends(_get_service),
):
    """List cached fit metrics results with optional filters."""
    return svc.list_results(
        implant_pc_id=implant_pc_id,
        reference_pc_id=reference_pc_id,
        limit=limit,
        offset=offset,
    )


@router.post("/sdf-heatmap")
def compute_sdf_heatmap(body: SDFHeatmapRequest, svc: FitMetricsService = Depends(_get_service)):
    """Compute per-point distances from query to reference point cloud.

    Returns raw binary float32 data (N floats).
    """
    try:
        distances = svc.compute_sdf_heatmap(
            query_pc_id=body.query_pc_id,
            reference_pc_id=body.reference_pc_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    buf = io.BytesIO(distances.tobytes())
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={
            "X-Dtype": "float32",
            "X-Count": str(len(distances)),
        },
    )
