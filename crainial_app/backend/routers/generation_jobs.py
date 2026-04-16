"""Generation Job endpoints for the cran-2 implant pipeline."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from crainial_app.backend.database import SessionLocal, get_db
from crainial_app.backend.schemas.generation_job import (
    GenerationJobCreate,
    GenerationJobRead,
)
from crainial_app.backend.services.audit_service import AuditService
from crainial_app.backend.services.generation_service import GenerationService
from crainial_app.backend.services.settings_service import SettingsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/generation-jobs", tags=["generation-jobs"])


def _get_service(db: Session = Depends(get_db)) -> GenerationService:
    audit = AuditService(db)
    return GenerationService(db, audit)


def _get_settings_service(db: Session = Depends(get_db)) -> SettingsService:
    return SettingsService(db)


def _execute_in_thread(job_id: str) -> None:
    """Run a cran-2 job on a fresh DB session inside a worker thread."""
    db = SessionLocal()
    try:
        audit = AuditService(db)
        service = GenerationService(db, audit)
        service.execute_cloud_generation(job_id)
    except Exception:
        logger.exception("cran-2 job %s crashed", job_id)
    finally:
        db.close()


@router.post("/", response_model=GenerationJobRead, status_code=201)
def create_job(
    body: GenerationJobCreate,
    background_tasks: BackgroundTasks,
    service: GenerationService = Depends(_get_service),
    settings_service: SettingsService = Depends(_get_settings_service),
):
    """Create a cran-2 generation job and queue it for execution."""
    cloud_enabled = (
        (settings_service.get_value("cloud_generation_enabled", "true") or "true").lower() == "true"
    )
    if not cloud_enabled:
        raise HTTPException(
            status_code=400,
            detail="Cloud generation is disabled. Enable it in Settings.",
        )

    endpoint_id = settings_service.get_value("runpod_endpoint_id", "") or ""
    api_key = settings_service.get_value("runpod_api_key", "") or ""
    if not endpoint_id or not api_key:
        raise HTTPException(
            status_code=400,
            detail="cran-2 endpoint not configured. Set Runpod endpoint ID and API key in Settings.",
        )

    try:
        job = service.create_job(
            input_scan_id=body.input_scan_id,
            threshold=body.threshold,
            project_id=body.project_id,
            name=body.name,
            description=body.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    background_tasks.add_task(_execute_in_thread, job.id)
    return job


@router.get("/", response_model=list[GenerationJobRead])
def list_jobs(
    project_id: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    service: GenerationService = Depends(_get_service),
):
    return service.list_jobs(
        project_id=project_id,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=GenerationJobRead)
def get_job(job_id: str, service: GenerationService = Depends(_get_service)):
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")
    return job


@router.post("/{job_id}/cancel", status_code=204)
def cancel_job(job_id: str, service: GenerationService = Depends(_get_service)):
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")
    if job.status in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}",
        )
    service.cancel_job(job_id)
