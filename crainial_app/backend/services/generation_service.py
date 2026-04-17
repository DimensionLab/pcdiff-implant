"""
Service for generating cranial implants via the cran-2 RunPod endpoint.

A job:
1. Reads a defective-skull NRRD from a Scan record on disk.
2. Submits its bytes to the cran-2 RunPod serverless endpoint.
3. Polls until the endpoint returns an implant-mask NRRD S3 URL.
4. Downloads the implant NRRD locally and registers it as a new Scan
   (scan_category='implant'), linked to the source via the parent project.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from crainial_app.backend.config import settings as app_settings
from crainial_app.backend.models.generation_job import GenerationJob
from crainial_app.backend.models.notification import Notification
from crainial_app.backend.models.scan import Scan
from crainial_app.backend.services.audit_service import AuditService
from crainial_app.backend.services.runpod_service import (
    RunpodError,
    RunpodService,
    download_from_s3_url,
    parse_runpod_results,
)
from crainial_app.backend.services.scan_service import ScanService
from crainial_app.backend.services.settings_service import SettingsService

logger = logging.getLogger(__name__)


def _generated_implants_dir() -> Path:
    """Directory where generated implant NRRDs are stored locally."""
    d = app_settings.project_root / "crainial_app" / "data" / "generated_implants"
    d.mkdir(parents=True, exist_ok=True)
    return d


class GenerationService:
    """Service for managing cran-2 implant generation jobs."""

    def __init__(self, db: Session, audit: AuditService):
        self.db = db
        self.audit = audit

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    def create_job(
        self,
        input_scan_id: str,
        threshold: float = 0.5,
        project_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> GenerationJob:
        scan = self.db.query(Scan).filter(Scan.id == input_scan_id).first()
        if not scan:
            raise ValueError(f"Input scan {input_scan_id} not found")
        if scan.file_format != "nrrd":
            raise ValueError(f"Input scan must be NRRD (got {scan.file_format})")

        job = GenerationJob(
            name=name or f"cran-2: {scan.name}",
            description=description,
            project_id=project_id or scan.project_id,
            input_scan_id=input_scan_id,
            threshold=float(threshold),
            status="pending",
            progress_percent=0,
            current_step="Queued",
            queued_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        self.audit.log(
            action="generation_job.create",
            entity_type="generation_job",
            entity_id=job.id,
            details={"input_scan_id": input_scan_id, "threshold": threshold},
        )
        return job

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        return self.db.query(GenerationJob).filter(GenerationJob.id == job_id).first()

    def list_jobs(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GenerationJob]:
        q = self.db.query(GenerationJob)
        if project_id:
            q = q.filter(GenerationJob.project_id == project_id)
        if status:
            q = q.filter(GenerationJob.status == status)
        return q.order_by(GenerationJob.created_at.desc()).offset(offset).limit(limit).all()

    def cancel_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        job.status = "cancelled"
        job.current_step = "Cancelled by user"
        job.completed_at = datetime.now(timezone.utc)
        self.db.commit()
        self.audit.log(
            action="generation_job.cancel",
            entity_type="generation_job",
            entity_id=job_id,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_cloud_generation(self, job_id: str) -> None:
        """Run a cran-2 generation job against the configured RunPod endpoint."""
        job = self.get_job(job_id)
        if not job:
            logger.error("Job %s not found", job_id)
            return
        if job.status == "cancelled":
            return

        settings_service = SettingsService(self.db)
        endpoint_id = settings_service.get_value("runpod_endpoint_id", "") or ""
        api_key = settings_service.get_value("runpod_api_key", "") or ""
        s3_bucket = settings_service.get_value("aws_s3_bucket", "") or ""
        s3_region = settings_service.get_value("aws_s3_region", "eu-central-1") or "eu-central-1"

        if not endpoint_id or not api_key:
            self._fail(job, "Cloud generation is not configured (Runpod endpoint ID or API key missing).")
            return

        scan = job.input_scan
        if not scan:
            self._fail(job, "Input scan not found")
            return

        scan_path = Path(scan.file_path)
        if not scan_path.is_file():
            self._fail(job, f"Input NRRD missing on disk: {scan_path}")
            return

        if not scan.defect_type:
            self._fail(
                job,
                "Scan is missing 'defect_type' — cran-2 v3 requires one of "
                "bilateral / frontoorbital / parietotemporal / random_1 / random_2. "
                "Edit the scan and set its defect type before generating.",
            )
            return

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.current_step = "Uploading defective skull to cran-2"
        job.progress_percent = 5
        self.db.commit()

        runpod = RunpodService(
            endpoint_id=endpoint_id,
            api_key=api_key,
            s3_bucket=s3_bucket or None,
            s3_region=s3_region,
        )

        cancel_event = threading.Event()

        def should_cancel() -> bool:
            self.db.refresh(job)
            if job.status == "cancelled":
                cancel_event.set()
                return True
            return False

        def progress_callback(status: str, percent: int) -> None:
            job.current_step = f"cran-2 {status.lower()}"
            # Reserve top of bar for download/registration after RunPod completes.
            job.progress_percent = max(job.progress_percent, min(percent * 80 // 100, 80))
            self.db.commit()

        try:
            nrrd_bytes = scan_path.read_bytes()
            output_prefix = f"cran2/{job.id}"

            runpod_job_id = runpod.submit_cran2_job_sync(
                defective_skull_nrrd=nrrd_bytes,
                threshold=job.threshold,
                output_prefix=output_prefix,
                defect_type=scan.defect_type,
            )
            job.runpod_job_id = runpod_job_id
            job.current_step = "cran-2 job submitted, waiting for GPU"
            self.db.commit()

            status_response = runpod.wait_for_completion_sync(
                job_id=runpod_job_id,
                progress_callback=progress_callback,
                poll_interval=3.0,
                timeout=600.0,
                should_cancel_callback=should_cancel,
            )
        except RunpodError as e:
            self._fail(job, str(e))
            return
        except Exception as e:
            logger.exception("Unexpected error running cran-2 job %s", job.id)
            self._fail(job, f"Unexpected error: {e}")
            return

        output = status_response.get("output") or {}
        try:
            parsed = parse_runpod_results(output)
        except RunpodError as e:
            self._fail(job, str(e))
            return

        nrrd_url = parsed.get("implant_nrrd_url")
        if not nrrd_url:
            self._fail(job, f"cran-2 returned no implant_volume_nrrd URL: {output}")
            return

        job.current_step = "Downloading implant NRRD"
        job.progress_percent = 85
        self.db.commit()

        local_path = _generated_implants_dir() / f"{job.id}_implant.nrrd"
        try:
            download_from_s3_url(
                s3_url=nrrd_url,
                local_path=local_path,
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                s3_region=s3_region,
            )
        except Exception as e:
            logger.exception("Failed to download implant NRRD for job %s", job.id)
            self._fail(job, f"Failed to download implant NRRD: {e}")
            return

        job.current_step = "Registering implant scan"
        job.progress_percent = 95
        self.db.commit()

        try:
            scan_service = ScanService(self.db, self.audit)
            implant_scan = scan_service.register_scan(
                file_path=str(local_path),
                name=f"{scan.name}_implant",
                scan_category="implant",
                defect_type=scan.defect_type,
                skull_id=scan.skull_id,
                project_id=job.project_id,
                description=f"cran-2 implant for scan {scan.id} (job {job.id})",
                compute_checksum=True,
            )
        except Exception as e:
            logger.exception("Failed to register implant scan for job %s", job.id)
            self._fail(job, f"Failed to register implant scan: {e}")
            return

        inference_seconds = parsed.get("inference_time_seconds")
        processing_seconds = parsed.get("processing_time_seconds")
        completed_at = datetime.now(timezone.utc)

        job.output_scan_id = implant_scan.id
        job.status = "completed"
        job.progress_percent = 100
        job.current_step = "Done"
        job.completed_at = completed_at
        if job.started_at:
            started = job.started_at.replace(tzinfo=timezone.utc) if job.started_at.tzinfo is None else job.started_at
            job.generation_time_ms = int((completed_at - started).total_seconds() * 1000)
        if inference_seconds is not None:
            job.inference_time_ms = int(float(inference_seconds) * 1000)
        job.metrics_json = json.dumps(
            {
                "implant_nrrd_url": nrrd_url,
                "s3_urls": parsed.get("s3_urls"),
                "inference_time_seconds": inference_seconds,
                "processing_time_seconds": processing_seconds,
                "model_source": parsed.get("model_source"),
            },
            default=str,
        )
        self.db.commit()

        self._notify(
            type_="generation_completed",
            title="cran-2 generation complete",
            message=f"Generated implant for {scan.name} in {job.generation_time_ms or 0} ms",
            entity_type="generation_job",
            entity_id=job.id,
        )

        self.audit.log(
            action="generation_job.complete",
            entity_type="generation_job",
            entity_id=job.id,
            details={
                "output_scan_id": implant_scan.id,
                "implant_nrrd_url": nrrd_url,
                "inference_time_seconds": inference_seconds,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fail(self, job: GenerationJob, message: str) -> None:
        logger.error("Job %s failed: %s", job.id, message)
        job.status = "failed"
        job.error_message = message
        job.current_step = "Failed"
        job.completed_at = datetime.now(timezone.utc)
        self.db.commit()
        self._notify(
            type_="generation_failed",
            title="cran-2 generation failed",
            message=message,
            entity_type="generation_job",
            entity_id=job.id,
        )
        self.audit.log(
            action="generation_job.fail",
            entity_type="generation_job",
            entity_id=job.id,
            details={"error": message},
        )

    def _notify(
        self,
        type_: str,
        title: str,
        message: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> None:
        try:
            notification = Notification(
                type=type_,
                title=title,
                message=message,
                entity_type=entity_type,
                entity_id=entity_id,
            )
            self.db.add(notification)
            self.db.commit()
        except Exception:
            logger.warning("Failed to create notification", exc_info=True)
            self.db.rollback()
