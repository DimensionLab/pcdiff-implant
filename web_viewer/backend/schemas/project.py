"""Pydantic schemas for Project endpoints."""

from datetime import datetime

from pydantic import BaseModel


class ProjectCreate(BaseModel):
    """Request to create a new project/case."""

    name: str
    description: str | None = None
    patient_id: str | None = None
    reconstruction_type: str | None = None
    implant_material: str | None = None
    notes: str | None = None
    region_code: str | None = None
    metadata_json: str | None = None


class ProjectUpdate(BaseModel):
    """Request to update a project/case."""

    name: str | None = None
    description: str | None = None
    patient_id: str | None = None
    reconstruction_type: str | None = None
    implant_material: str | None = None
    notes: str | None = None
    region_code: str | None = None
    metadata_json: str | None = None


class ProjectRead(BaseModel):
    """Response schema for project/case."""

    id: str
    name: str
    description: str | None = None
    patient_id: str | None = None
    reconstruction_type: str | None = None
    implant_material: str | None = None
    notes: str | None = None
    region_code: str | None = None
    metadata_json: str | None = None
    created_at: datetime
    updated_at: datetime
    created_by: str

    model_config = {"from_attributes": True}
