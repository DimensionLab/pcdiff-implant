"""Generation Job endpoints for implant generation pipeline.

Supports parallel ensemble generation:
- When num_ensemble > 1 and using cloud, creates a parent job with N child jobs
- Each child job runs on a separate GPU worker in parallel
- Parent job aggregates status from all children
"""

import logging
import threading

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from web_viewer.backend.database import SessionLocal, get_db
from web_viewer.backend.schemas.generation_job import (
    GenerationJobCreate,
    GenerationJobRead,
    GenerationJobWithChildren,
    RevoxelizeJobCreate,
    SelectOutputRequest,
)
from web_viewer.backend.services.audit_service import AuditService
from web_viewer.backend.services.generation_service import GenerationService
from web_viewer.backend.services.settings_service import SettingsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/generation-jobs", tags=["generation-jobs"])


def _execute_child_job_in_thread(child_job_id: str):
    """Execute a child job in its own thread with its own database session."""
    # Create a new database session for this thread
    db = SessionLocal()
    try:
        audit = AuditService(db)
        service = GenerationService(db, audit)
        service.execute_child_cloud_generation(child_job_id)
    except Exception as e:
        logger.exception(f"Child job {child_job_id} failed: {e}")
    finally:
        db.close()


def _start_parallel_child_jobs(child_job_ids: list[str]):
    """Start all child jobs in parallel using threads."""
    logger.info(f"Starting {len(child_job_ids)} child jobs in parallel")

    # Start each child job in its own thread
    # Threads run independently and don't block each other
    threads = []
    for job_id in child_job_ids:
        thread = threading.Thread(
            target=_execute_child_job_in_thread,
            args=(job_id,),
            name=f"child-job-{job_id[:8]}",
            daemon=True,  # Daemon threads don't block process exit
        )
        thread.start()
        threads.append(thread)
        logger.info(f"Started thread for child job {job_id}")

    logger.info(f"All {len(child_job_ids)} child job threads started")


def _get_service(db: Session = Depends(get_db)) -> GenerationService:
    audit = AuditService(db)
    return GenerationService(db, audit)


def _get_settings_service(db: Session = Depends(get_db)) -> SettingsService:
    return SettingsService(db)


@router.post("/", response_model=GenerationJobRead, status_code=201)
def create_job(
    body: GenerationJobCreate,
    background_tasks: BackgroundTasks,
    service: GenerationService = Depends(_get_service),
    settings_service: SettingsService = Depends(_get_settings_service),
):
    """Create a new generation job and queue it for execution.

    The job will be executed in the background. Poll the job status
    endpoint to track progress.

    For cloud generation with num_ensemble > 1:
    - Creates a parent job to track overall progress
    - Creates N child jobs (one per ensemble) that run in parallel on separate GPU workers
    - Each child job generates exactly 1 implant
    - Parent job aggregates results from all children

    For local generation or single ensemble:
    - Creates a single job that runs all ensembles sequentially
    """
    try:
        # Determine whether to use cloud or local execution
        cloud_enabled = settings_service.get_value("cloud_generation_enabled", "false").lower() == "true"
        use_cloud = body.use_cloud if body.use_cloud is not None else cloud_enabled

        if use_cloud:
            # Verify cloud is configured
            endpoint_id = settings_service.get_value("runpod_endpoint_id", "")
            api_key = settings_service.get_value("runpod_api_key", "")
            if not endpoint_id or not api_key:
                raise ValueError(
                    "Cloud generation enabled but not configured. Set Runpod endpoint ID and API key in settings."
                )

            if body.num_ensemble > 1:
                # PARALLEL ENSEMBLE: Create parent job + N child jobs
                parent_job = service.create_job(
                    project_id=body.project_id,
                    input_pc_id=body.input_pc_id,
                    sampling_method=body.sampling_method,
                    sampling_steps=body.sampling_steps,
                    num_ensemble=body.num_ensemble,
                    name=body.name,
                    description=body.description,
                    pcdiff_model=body.pcdiff_model,
                    voxelization_resolution=body.voxelization_resolution,
                    smoothing_iterations=body.smoothing_iterations,
                    close_holes=body.close_holes,
                )

                # Mark parent as running (children will update aggregate status)
                parent_job.status = "running"
                parent_job.started_at = parent_job.queued_at
                parent_job.current_step = f"Starting {body.num_ensemble} parallel jobs..."
                service.db.commit()
                service.db.refresh(parent_job)

                # Create child jobs and collect their IDs
                child_job_ids = []
                for i in range(body.num_ensemble):
                    child_job = service.create_child_job(parent_job, ensemble_index=i)
                    child_job_ids.append(child_job.id)

                # Start all child jobs in parallel using background thread
                # Each child job will have its own database session
                background_tasks.add_task(_start_parallel_child_jobs, child_job_ids)

                return parent_job
            else:
                # Single ensemble - use original flow
                job = service.create_job(
                    project_id=body.project_id,
                    input_pc_id=body.input_pc_id,
                    sampling_method=body.sampling_method,
                    sampling_steps=body.sampling_steps,
                    num_ensemble=1,
                    name=body.name,
                    description=body.description,
                    pcdiff_model=body.pcdiff_model,
                    voxelization_resolution=body.voxelization_resolution,
                    smoothing_iterations=body.smoothing_iterations,
                    close_holes=body.close_holes,
                )
                background_tasks.add_task(service.execute_cloud_generation, job.id)
                return job
        else:
            # Local execution - sequential (original behavior)
            job = service.create_job(
                project_id=body.project_id,
                input_pc_id=body.input_pc_id,
                sampling_method=body.sampling_method,
                sampling_steps=body.sampling_steps,
                num_ensemble=body.num_ensemble,
                name=body.name,
                description=body.description,
                pcdiff_model=body.pcdiff_model,
                voxelization_resolution=body.voxelization_resolution,
                smoothing_iterations=body.smoothing_iterations,
                close_holes=body.close_holes,
            )
            background_tasks.add_task(service.execute_generation, job.id)
            return job

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/revoxelize", response_model=GenerationJobRead, status_code=201)
def create_revoxelization_job(
    body: RevoxelizeJobCreate,
    background_tasks: BackgroundTasks,
    service: GenerationService = Depends(_get_service),
    settings_service: SettingsService = Depends(_get_settings_service),
):
    """Create a re-voxelization job to regenerate mesh from existing implant point cloud.

    Use this to generate a new STL mesh with a different level of detail (resolution)
    from an already-generated implant point cloud. This skips the diffusion step
    and only runs the voxelization/mesh generation part.

    Resolution options:
    - 128: Fast, low detail (for previews)
    - 256: Medium detail
    - 512: High detail (default, balanced)
    - 1024: Ultra detail (slower, for final production)

    Can run locally or in the cloud depending on settings and use_cloud flag.
    """
    try:
        # Create re-voxelization job
        job = service.create_revoxelization_job(
            project_id=body.project_id,
            source_implant_pc_id=body.source_implant_pc_id,
            input_pc_id=body.input_pc_id,
            voxelization_resolution=body.voxelization_resolution,
            name=body.name,
            description=body.description,
            smoothing_iterations=body.smoothing_iterations,
            close_holes=body.close_holes,
        )

        # Check if we should use cloud execution
        cloud_enabled = settings_service.get_value("cloud_generation_enabled", "false").lower() == "true"
        use_cloud = body.use_cloud if body.use_cloud is not None else cloud_enabled

        if use_cloud:
            # Verify cloud is configured
            endpoint_id = settings_service.get_value("runpod_endpoint_id", "")
            api_key = settings_service.get_value("runpod_api_key", "")
            if not endpoint_id or not api_key:
                logger.warning("Cloud requested but not configured, falling back to local")
                background_tasks.add_task(service.execute_generation, job.id)
            else:
                # Run re-voxelization in cloud
                logger.info(f"Running re-voxelization in cloud (resolution: {body.voxelization_resolution}³)")
                background_tasks.add_task(service.execute_cloud_revoxelization, job.id)
        else:
            # Run locally
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


@router.get("/{job_id}", response_model=GenerationJobWithChildren)
def get_job(job_id: str, service: GenerationService = Depends(_get_service)):
    """Get a specific generation job by ID.

    For parent jobs (num_ensemble > 1 with cloud), includes child_jobs array
    with the status of each parallel ensemble job.
    """
    job = service.get_job_with_children(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")

    # Convert to response with children
    response = GenerationJobWithChildren.model_validate(job)
    response.child_jobs = [GenerationJobRead.model_validate(child) for child in job.child_jobs]
    return response


@router.get("/{job_id}/children", response_model=list[GenerationJobRead])
def get_child_jobs(job_id: str, service: GenerationService = Depends(_get_service)):
    """Get all child jobs for a parent job.

    Returns empty list if the job has no children (single ensemble or local execution).
    """
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")

    return service.get_child_jobs(job_id)


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
def delete_unselected_outputs(job_id: str, service: GenerationService = Depends(_get_service)):
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
