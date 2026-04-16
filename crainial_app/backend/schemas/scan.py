"""Pydantic schemas for Scan endpoints."""

import struct
from datetime import datetime

from pydantic import BaseModel, field_validator


class ScanCreate(BaseModel):
    file_path: str
    name: str | None = None
    scan_category: str | None = None
    defect_type: str | None = None
    skull_id: str | None = None
    project_id: str | None = None
    description: str | None = None


class ScanUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    scan_category: str | None = None
    defect_type: str | None = None
    skull_id: str | None = None
    project_id: str | None = None


class ScanRead(BaseModel):
    id: str
    project_id: str | None = None
    name: str
    description: str | None = None
    file_path: str
    file_format: str
    file_size_bytes: int | None = None
    volume_dims_x: int | None = None
    volume_dims_y: int | None = None
    volume_dims_z: int | None = None
    voxel_spacing_x: float | None = None
    voxel_spacing_y: float | None = None
    voxel_spacing_z: float | None = None
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
        "volume_dims_x",
        "volume_dims_y",
        "volume_dims_z",
        "file_size_bytes",
        mode="before",
    )
    @classmethod
    def _coerce_int_from_bytes(cls, v: object) -> object:
        """Handle numpy int64 values stored as raw 8-byte blobs in SQLite."""
        if isinstance(v, (bytes, bytearray)) and len(v) == 8:
            return struct.unpack("<q", v)[0]
        return v

    @field_validator(
        "voxel_spacing_x",
        "voxel_spacing_y",
        "voxel_spacing_z",
        mode="before",
    )
    @classmethod
    def _coerce_float_from_bytes(cls, v: object) -> object:
        """Handle numpy float64 values stored as raw 8-byte blobs in SQLite."""
        if isinstance(v, (bytes, bytearray)) and len(v) == 8:
            return struct.unpack("<d", v)[0]
        return v


class SkullBreakImportRequest(BaseModel):
    base_dir: str
    project_id: str | None = None
    compute_checksums: bool = False


class ImportResult(BaseModel):
    scans_created: int
    point_clouds_created: int
    skipped: int
    errors: list[dict]
