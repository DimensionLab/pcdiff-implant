"""Pydantic schemas for CaseReport endpoints."""

from datetime import datetime

from pydantic import BaseModel


class CaseReportCreate(BaseModel):
    """Request to generate a new case report."""

    project_id: str
    title: str | None = None  # Auto-generated if not provided
    region_code: str | None = None  # Will use project's region if not provided


class CaseReportRead(BaseModel):
    """Response schema for case report."""

    id: str
    project_id: str
    title: str
    html_content: str
    pdf_path: str | None = None
    template_version: str
    prompt_version: str | None = None
    ai_model: str | None = None
    ai_provider: str | None = None
    ai_request_id: str | None = None
    region_code: str | None = None
    generated_at: datetime
    metadata_json: str | None = None
    created_at: datetime
    updated_at: datetime
    created_by: str

    model_config = {"from_attributes": True}


class CaseReportSummary(BaseModel):
    """Summary schema for listing reports (without full HTML content)."""

    id: str
    project_id: str
    title: str
    template_version: str
    ai_model: str | None = None
    region_code: str | None = None
    generated_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}
