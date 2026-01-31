"""Pydantic schemas for GenerationJob endpoints."""

import json
from datetime import datetime

from pydantic import BaseModel, computed_field


class GenerationJobCreate(BaseModel):
    """Request to create a new generation job."""

    project_id: str
    input_pc_id: str
    sampling_method: str = "ddim"  # "ddim" or "ddpm"
    sampling_steps: int = 50  # Default for DDIM
    num_ensemble: int = 5  # Default ensemble count
    name: str | None = None
    description: str | None = None


class GenerationJobUpdate(BaseModel):
    """Request to update generation job metadata."""

    name: str | None = None
    description: str | None = None
    selected_output_id: str | None = None


class GenerationJobRead(BaseModel):
    """Response schema for generation job."""

    id: str
    name: str
    description: str | None = None
    project_id: str | None = None
    input_pc_id: str
    status: str
    progress_percent: int
    current_step: str | None = None
    error_message: str | None = None
    sampling_method: str
    sampling_steps: int
    num_ensemble: int
    output_pc_ids_json: str | None = None
    output_stl_ids_json: str | None = None
    selected_output_id: str | None = None
    metrics_json: str | None = None
    queued_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    generation_time_ms: int | None = None
    created_at: datetime
    updated_at: datetime
    created_by: str

    model_config = {"from_attributes": True}

    @computed_field
    @property
    def output_pc_ids(self) -> list[str]:
        """Parse output point cloud IDs from JSON."""
        if self.output_pc_ids_json:
            try:
                return json.loads(self.output_pc_ids_json)
            except (json.JSONDecodeError, TypeError):
                pass
        return []

    @computed_field
    @property
    def output_stl_ids(self) -> list[str]:
        """Parse output STL IDs from JSON."""
        if self.output_stl_ids_json:
            try:
                return json.loads(self.output_stl_ids_json)
            except (json.JSONDecodeError, TypeError):
                pass
        return []

    @computed_field
    @property
    def metrics(self) -> dict | None:
        """Parse metrics from JSON."""
        if self.metrics_json:
            try:
                return json.loads(self.metrics_json)
            except (json.JSONDecodeError, TypeError):
                pass
        return None


class SelectOutputRequest(BaseModel):
    """Request to select a specific output from ensemble results."""

    output_id: str
