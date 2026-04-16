"""Pydantic schemas for fit metrics endpoints."""

from datetime import datetime

from pydantic import BaseModel

from crainial_app.backend.schemas.point_cloud import PointCloudRead


class FitMetricsRequest(BaseModel):
    implant_pc_id: str
    reference_pc_id: str
    defective_skull_pc_id: str | None = None
    resolution: int = 256


class FitMetricsRead(BaseModel):
    id: str
    implant_pc_id: str
    reference_pc_id: str
    defective_skull_pc_id: str | None = None
    dice_coefficient: float | None = None
    hausdorff_distance: float | None = None
    hausdorff_distance_95: float | None = None
    boundary_dice: float | None = None
    resolution: int
    voxel_spacing: float | None = None
    computation_mode: str
    computation_time_ms: int | None = None
    status: str
    error_message: str | None = None
    created_at: datetime
    created_by: str

    model_config = {"from_attributes": True}


class SDFHeatmapRequest(BaseModel):
    query_pc_id: str
    reference_pc_id: str


class SkullImplantPair(BaseModel):
    skull_id: str
    defective_skull: PointCloudRead
    implants: list[PointCloudRead]
