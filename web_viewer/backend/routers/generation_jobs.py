"""Generation Job endpoints for implant generation pipeline."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from web_viewer.backend.database import get_db
from web_viewer.backend.schemas.generation_job import (
    GenerationJobCreate,
    GenerationJobRead,
    SelectOutputRequest,
)
from web_viewer.backend.services.audit_service import AuditService
from web_viewer.backend.services.generation_service import GenerationService

router = APIRouter(prefix="/api/v1/generation-jobs", tags=["generation-jobs"])


def _get_service(db: Session = Depends(get_db)) -> GenerationService:
    audit = AuditService(db)
    return GenerationService(db, audit)


@router.post("/", response_model=GenerationJobRead, status_code=201)
def create_job(
    body: GenerationJobCreate,
    background_tasks: BackgroundTasks,
    service: GenerationService = Depends(_get_service),
):
    """Create a new generation job and queue it for execution.

    The job will be executed in the background. Poll the job status
    endpoint to track progress.
    """
    try:
        job = service.create_job(
            project_id=body.project_id,
            input_pc_id=body.input_pc_id,
            sampling_method=body.sampling_method,
            sampling_steps=body.sampling_steps,
            num_ensemble=body.num_ensemble,
            name=body.name,
            description=body.description,
        )

        # Queue background execution
        background_tasks.add_task(service.execute_generation, job.id)

        return job
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=list[GenerationJobRead])
def list_jobs(
    project_id: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    service: GenerationService = Depends(_get_service),
):
    """List generation jobs with optional filtering."""
    return service.list_jobs(
        project_id=project_id,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=GenerationJobRead)
def get_job(job_id: str, service: GenerationService = Depends(_get_service)):
    """Get a specific generation job by ID."""
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")
    return job


@router.post("/{job_id}/cancel", status_code=204)
def cancel_job(job_id: str, service: GenerationService = Depends(_get_service)):
    """Cancel a pending or running generation job."""
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")

    if job.status in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}",
        )

    service.cancel_job(job_id)


@router.post("/{job_id}/select-output", response_model=GenerationJobRead)
def select_output(
    job_id: str,
    body: SelectOutputRequest,
    service: GenerationService = Depends(_get_service),
):
    """Select a specific output from ensemble results as the preferred one."""
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Can only select output from completed jobs",
        )

    try:
        return service.select_output(job_id, body.output_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{job_id}/unselected-outputs", status_code=204)
def delete_unselected_outputs(
    job_id: str, service: GenerationService = Depends(_get_service)
):
    """Delete all unselected outputs from a completed job.

    Requires that an output has been selected first.
    """
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Can only delete outputs from completed jobs",
        )

    if not job.selected_output_id:
        raise HTTPException(
            status_code=400,
            detail="Must select an output before deleting unselected ones",
        )

    service.delete_unselected_outputs(job_id)
