"""Pydantic schemas for cran-2 GenerationJob endpoints."""

import json
from datetime import datetime

from pydantic import BaseModel, Field, computed_field


class GenerationJobCreate(BaseModel):
    """Request to create a cran-2 generation job."""

    project_id: str | None = None
    input_scan_id: str
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    name: str | None = None
    description: str | None = None


class GenerationJobUpdate(BaseModel):
    """Request to update generation job metadata."""

    name: str | None = None
    description: str | None = None


class GenerationJobRead(BaseModel):
    """Response schema for a cran-2 generation job."""

    id: str
    name: str
    description: str | None = None
    project_id: str | None = None
    input_scan_id: str
    output_scan_id: str | None = None

    status: str
    progress_percent: int
    current_step: str | None = None
    error_message: str | None = None

    threshold: float
    runpod_job_id: str | None = None

    metrics_json: str | None = None
    queued_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    generation_time_ms: int | None = None
    inference_time_ms: int | None = None
    created_at: datetime
    updated_at: datetime
    created_by: str

    model_config = {"from_attributes": True}

    @computed_field
    @property
    def metrics(self) -> dict | None:
        if self.metrics_json:
            try:
                return json.loads(self.metrics_json)
            except (json.JSONDecodeError, TypeError):
                return None
        return None
