"""Pydantic schemas for PointCloud endpoints."""

import struct
from datetime import datetime

from pydantic import BaseModel, field_validator


class PointCloudCreate(BaseModel):
    file_path: str
    name: str | None = None
    scan_category: str | None = None
    defect_type: str | None = None
    skull_id: str | None = None
    project_id: str | None = None
    scan_id: str | None = None
    description: str | None = None


class PointCloudUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    scan_category: str | None = None
    defect_type: str | None = None
    skull_id: str | None = None
    project_id: str | None = None
    scan_id: str | None = None


class PointCloudRead(BaseModel):
    id: str
    project_id: str | None = None
    scan_id: str | None = None
    name: str
    description: str | None = None
    file_path: str
    file_format: str
    file_size_bytes: int | None = None
    num_points: int | None = None
    point_dims: int
    scan_category: str | None = None
    defect_type: str | None = None
    skull_id: str | None = None
    metadata_json: str | None = None
    checksum_sha256: str | None = None
    created_at: datetime
    updated_at: datetime
    created_by: str

    model_config = {"from_attributes": True}

    @field_validator(
        "file_size_bytes", "num_points", "point_dims",
        mode="before",
    )
    @classmethod
    def _coerce_int_from_bytes(cls, v: object) -> object:
        """Handle numpy int64 values stored as raw 8-byte blobs in SQLite."""
        if isinstance(v, (bytes, bytearray)) and len(v) == 8:
            return struct.unpack("<q", v)[0]
        return v


class SDFRequest(BaseModel):
    """Custom SDF computation parameters."""
    reference_scan_id: str | None = None
    surface_threshold: float = 0.5


class STLStatusResponse(BaseModel):
    """Response for STL generation status check."""
    has_stl: bool
    stl_pc_id: str | None = None
    stl_metadata: dict | None = None
    source_pc_id: str
