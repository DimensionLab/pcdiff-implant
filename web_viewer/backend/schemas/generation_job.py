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
    use_cloud: bool | None = None  # None = use default from settings, True = force cloud, False = force local
    pcdiff_model: str = "best"  # "best" or "latest" - which PCDiff model checkpoint to use
    voxelization_resolution: int = 512  # PSR grid resolution: 128, 256, 512 (default), 1024
    smoothing_iterations: int = 0  # Laplacian smoothing: 0 = disabled, 1-100
    close_holes: bool = False  # Fill holes in the generated mesh


class RevoxelizeJobCreate(BaseModel):
    """Request to create a re-voxelization job (mesh generation only, no diffusion)."""

    project_id: str
    source_implant_pc_id: str  # Existing implant point cloud to re-voxelize
    input_pc_id: str  # Defective skull point cloud (needed for combined voxelization)
    voxelization_resolution: int = 512  # PSR grid resolution: 128, 256, 512, 1024
    name: str | None = None
    description: str | None = None
    use_cloud: bool | None = None
    smoothing_iterations: int = 0  # Laplacian smoothing: 0 = disabled, 1-100
    close_holes: bool = False  # Fill holes in the generated mesh


class GenerationJobUpdate(BaseModel):
    """Request to update generation job metadata."""

    name: str | None = None
    description: str | None = None
    selected_output_id: str | None = None


class GenerationJobRead(BaseModel):
    """Response schema for generation job.

    Supports parent-child hierarchy for parallel ensemble generation:
    - Parent jobs have child_jobs populated and parent_job_id is None
    - Child jobs have parent_job_id set and ensemble_index indicating their position
    """

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
    pcdiff_model: str | None = None  # "best" or "latest"
    voxelization_resolution: int = 512  # PSR grid resolution
    smoothing_iterations: int = 0  # Laplacian smoothing iterations
    close_holes: bool = False  # Fill holes in mesh
    source_implant_pc_id: str | None = None  # Set for re-voxelization jobs

    # Parent-child hierarchy fields
    parent_job_id: str | None = None
    ensemble_index: int | None = None  # 0-based index for child jobs

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


class GenerationJobWithChildren(GenerationJobRead):
    """Response schema for a parent job including its child jobs.

    Used when fetching a parent job to show all ensemble progress.
    """

    child_jobs: list["GenerationJobRead"] = []

    @computed_field
    @property
    def is_parent_job(self) -> bool:
        """True if this is a parent job with child ensemble jobs."""
        return self.parent_job_id is None and self.num_ensemble > 1

    @computed_field
    @property
    def overall_progress(self) -> int:
        """Calculate overall progress from child jobs."""
        if not self.child_jobs:
            return self.progress_percent
        total = sum(child.progress_percent for child in self.child_jobs)
        return total // len(self.child_jobs)

    @computed_field
    @property
    def completed_children(self) -> int:
        """Count of completed child jobs."""
        return sum(1 for child in self.child_jobs if child.status == "completed")

    @computed_field
    @property
    def failed_children(self) -> int:
        """Count of failed child jobs."""
        return sum(1 for child in self.child_jobs if child.status == "failed")


class SelectOutputRequest(BaseModel):
    """Request to select a specific output from ensemble results."""

    output_id: str
