"""Generation Job endpoints for the cran-2 implant pipeline."""

import json
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from crainial_app.backend.database import SessionLocal, get_db
from crainial_app.backend.schemas.generation_job import (
    GenerationJobCreate,
    GenerationJobRead,
)
from crainial_app.backend.services.audit_service import AuditService
from crainial_app.backend.services.generation_service import GenerationService
from crainial_app.backend.services.runpod_service import download_from_s3_url
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


ARTIFACT_KEY_MAP = {
    "nrrd": "implant_volume_nrrd",
    "stl": "implant_stl",
    "ply": "implant_ply",
    "npy": "implant_volume_npy",
}
ARTIFACT_MEDIA = {
    "nrrd": "application/octet-stream",
    "stl": "model/stl",
    "ply": "application/x-ply",
    "npy": "application/octet-stream",
}
ARTIFACT_EXT = {
    "nrrd": ".nrrd",
    "stl": ".stl",
    "ply": ".ply",
    "npy": ".npy",
}


@router.get("/{job_id}/artifacts")
def list_artifacts(job_id: str, service: GenerationService = Depends(_get_service)):
    """List available downloadable artifacts for a completed job."""
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    metrics = {}
    if job.metrics_json:
        try:
            metrics = json.loads(job.metrics_json)
        except (json.JSONDecodeError, TypeError):
            pass

    s3_urls = metrics.get("s3_urls", {})
    available = []
    for fmt, key in ARTIFACT_KEY_MAP.items():
        if key in s3_urls:
            available.append({"format": fmt, "key": key})
    return {"job_id": job_id, "artifacts": available}


@router.get("/{job_id}/download/{fmt}")
def download_artifact(
    job_id: str,
    fmt: str,
    service: GenerationService = Depends(_get_service),
    settings_service: SettingsService = Depends(_get_settings_service),
):
    """Download a specific artifact file from a completed job.

    For NRRD, serves the local file if available. For STL/PLY/NPY,
    proxies the download from S3.
    """
    if fmt not in ARTIFACT_KEY_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown format: {fmt}. Use one of: {list(ARTIFACT_KEY_MAP)}")

    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    filename = f"{job.name or job_id}{ARTIFACT_EXT[fmt]}"

    # For NRRD, try the local registered scan file first
    if fmt == "nrrd" and job.output_scan_id:
        scan = service.db.query(service.db.registry.mappers[0].class_).get(job.output_scan_id) if hasattr(service.db, 'registry') else None
        # Simpler: check the generated implants dir
        from crainial_app.backend.services.generation_service import _generated_implants_dir
        local = _generated_implants_dir() / f"{job.id}_implant.nrrd"
        if local.is_file():
            return FileResponse(
                path=str(local),
                media_type=ARTIFACT_MEDIA[fmt],
                filename=filename,
            )

    # Otherwise, download from S3 URL stored in metrics_json
    metrics = {}
    if job.metrics_json:
        try:
            metrics = json.loads(job.metrics_json)
        except (json.JSONDecodeError, TypeError):
            pass

    s3_urls = metrics.get("s3_urls", {})
    s3_key = ARTIFACT_KEY_MAP[fmt]
    s3_url = s3_urls.get(s3_key)
    if not s3_url:
        raise HTTPException(status_code=404, detail=f"Artifact '{fmt}' not available for this job")

    # Download to a temp file and serve
    tmp = Path(tempfile.mkdtemp()) / f"{job_id}{ARTIFACT_EXT[fmt]}"
    try:
        download_from_s3_url(
            s3_url=s3_url,
            local_path=tmp,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            s3_region=settings_service.get_value("aws_s3_region", "eu-central-1"),
        )
    except Exception as e:
        logger.exception("Failed to download artifact %s for job %s", fmt, job_id)
        raise HTTPException(status_code=502, detail=f"Failed to download from S3: {e}")

    return FileResponse(
        path=str(tmp),
        media_type=ARTIFACT_MEDIA[fmt],
        filename=filename,
    )
