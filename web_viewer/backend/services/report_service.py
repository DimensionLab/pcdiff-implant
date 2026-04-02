"""Service for generating case reports using OpenRouter AI."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy.orm import Session

from web_viewer.backend.config import settings
from web_viewer.backend.models.case_report import CaseReport
from web_viewer.backend.models.generation_job import GenerationJob
from web_viewer.backend.models.patient import Patient
from web_viewer.backend.models.point_cloud import PointCloud
from web_viewer.backend.models.project import Project
from web_viewer.backend.models.scan import Scan
from web_viewer.backend.services.audit_service import AuditService
from web_viewer.backend.services.report_prompts.v1_0 import (
    CASE_REPORT_TEMPLATE,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    get_compliance_section,
)

logger = logging.getLogger(__name__)

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-4.5-sonnet"


class ReportService:
    """Service for generating and managing case reports."""

    def __init__(self, db: Session, audit: AuditService):
        self.db = db
        self.audit = audit
        self.reports_dir = Path(settings.project_root) / "web_viewer" / "data" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def list_reports(
        self,
        project_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CaseReport]:
        """List case reports with optional filtering."""
        query = self.db.query(CaseReport)

        if project_id:
            query = query.filter(CaseReport.project_id == project_id)

        return query.order_by(CaseReport.generated_at.desc()).offset(offset).limit(limit).all()

    def get_report(self, report_id: str) -> CaseReport | None:
        """Get a specific report by ID."""
        return self.db.query(CaseReport).filter(CaseReport.id == report_id).first()

    def generate_report(
        self,
        project_id: str,
        title: str | None = None,
        region_code: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> CaseReport:
        """Generate a new case report using OpenRouter AI.

        Args:
            project_id: The project to generate a report for
            title: Optional custom title (auto-generated if not provided)
            region_code: Optional region code override (uses project's if not provided)
            model: OpenRouter model to use for generation

        Returns:
            The generated CaseReport record

        Raises:
            ValueError: If project not found or API call fails
        """
        # Get project and related data
        project = self.db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        # Get patient if linked
        patient: Patient | None = None
        if project.patient_id:
            patient = self.db.query(Patient).filter(Patient.id == project.patient_id).first()

        # Use provided region_code or fall back to project's region
        effective_region = region_code or project.region_code

        # Gather related data
        scans = self.db.query(Scan).filter(Scan.project_id == project_id).all()
        point_clouds = self.db.query(PointCloud).filter(PointCloud.project_id == project_id).all()
        generation_jobs = (
            self.db.query(GenerationJob)
            .filter(GenerationJob.project_id == project_id)
            .filter(GenerationJob.status == "completed")
            .all()
        )

        # Build context for the prompt
        context = self._build_report_context(project, patient, scans, point_clouds, generation_jobs, effective_region)

        # Generate title if not provided
        if not title:
            patient_name = f"{patient.first_name or ''} {patient.last_name or ''}".strip() if patient else "Unknown"
            title = f"Cranioplasty Report - {patient_name} - {datetime.now().strftime('%Y-%m-%d')}"

        # Call OpenRouter API
        html_content, ai_request_id = self._call_openrouter(context, model)

        # Create the report record
        report = CaseReport(
            project_id=project_id,
            title=title,
            html_content=html_content,
            template_version="v1.0",
            prompt_version=PROMPT_VERSION,
            ai_model=model,
            ai_provider="openrouter",
            ai_request_id=ai_request_id,
            region_code=effective_region,
            generated_at=datetime.now(timezone.utc),
            metadata_json=json.dumps(
                {
                    "scans_count": len(scans),
                    "point_clouds_count": len(point_clouds),
                    "generation_jobs_count": len(generation_jobs),
                    "patient_id": project.patient_id,
                }
            ),
        )

        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)

        self.audit.log(
            action="case_report.generate",
            entity_type="case_report",
            entity_id=report.id,
            details={
                "project_id": project_id,
                "ai_model": model,
                "region_code": effective_region,
            },
        )

        logger.info(f"Generated case report: {report.id} for project {project_id}")
        return report

    def _build_report_context(
        self,
        project: Project,
        patient: Patient | None,
        scans: list[Scan],
        point_clouds: list[PointCloud],
        generation_jobs: list[GenerationJob],
        region_code: str | None,
    ) -> str:
        """Build the context string for the AI prompt."""

        # Patient info
        if patient:
            patient_name = f"{patient.first_name or ''} {patient.last_name or ''}".strip() or "N/A"
            patient_code = patient.patient_code
            dob = str(patient.date_of_birth) if patient.date_of_birth else "N/A"
            sex = patient.sex or "N/A"
            mrn = patient.medical_record_number or "N/A"
        else:
            patient_name = "N/A"
            patient_code = "N/A"
            dob = "N/A"
            sex = "N/A"
            mrn = "N/A"

        # Scan info
        scan_lines = []
        for scan in scans:
            dims = f"{scan.volume_dims_x}x{scan.volume_dims_y}x{scan.volume_dims_z}" if scan.volume_dims_x else "N/A"
            spacing = (
                f"{scan.voxel_spacing_x:.3f}x{scan.voxel_spacing_y:.3f}x{scan.voxel_spacing_z:.3f}"
                if scan.voxel_spacing_x
                else "N/A"
            )
            scan_lines.append(
                f"- {scan.name}: Category={scan.scan_category or 'N/A'}, Dimensions={dims}, Spacing={spacing}mm"
            )
        scan_info = "\n".join(scan_lines) if scan_lines else "No scans available"

        # Implant info (generated point clouds)
        implant_pcs = [pc for pc in point_clouds if pc.scan_category in ("implant", "generated_implant")]
        implant_lines = []
        for pc in implant_pcs:
            implant_lines.append(f"- {pc.name}: Points={pc.num_points or 'N/A'}, Format={pc.file_format}")
        implant_info = "\n".join(implant_lines) if implant_lines else "No implants generated"

        # Generation job details
        job_lines = []
        for job in generation_jobs:
            duration = f"{job.generation_time_ms / 1000:.1f}s" if job.generation_time_ms else "N/A"
            job_lines.append(
                f"- {job.name}: Method={job.sampling_method.upper()}, "
                f"Steps={job.sampling_steps}, Ensemble={job.num_ensemble}, "
                f"Duration={duration}"
            )
        generation_details = "\n".join(job_lines) if job_lines else "No generation jobs completed"

        # Quality metrics
        metrics_lines = []
        for job in generation_jobs:
            if job.metrics_json:
                try:
                    metrics = json.loads(job.metrics_json)
                    for key, val in metrics.items():
                        if isinstance(val, dict):
                            dsc = val.get("dsc", "N/A")
                            bdsc = val.get("bdsc", "N/A")
                            hd95 = val.get("hd95", "N/A")
                            metrics_lines.append(f"- Ensemble {key}: DSC={dsc}, bDSC={bdsc}, HD95={hd95}mm")
                except json.JSONDecodeError:
                    pass
        quality_metrics = "\n".join(metrics_lines) if metrics_lines else "No metrics available"

        # Add compliance section
        compliance = get_compliance_section(region_code)

        # Build the full prompt
        prompt = CASE_REPORT_TEMPLATE.format(
            patient_code=patient_code,
            patient_name=patient_name,
            date_of_birth=dob,
            sex=sex,
            medical_record_number=mrn,
            project_name=project.name,
            project_description=project.description or "N/A",
            reconstruction_type=project.reconstruction_type or "Cranioplasty",
            implant_material=project.implant_material or "Not specified",
            region_code=region_code or "Not specified",
            case_notes=project.notes or "N/A",
            scan_info=scan_info,
            implant_info=implant_info,
            generation_details=generation_details,
            quality_metrics=quality_metrics,
        )

        # Append compliance section to the prompt
        prompt += f"\n\n{compliance}"

        return prompt

    def _call_openrouter(self, prompt: str, model: str) -> tuple[str, str | None]:
        """Call the OpenRouter API to generate report content.

        Returns:
            Tuple of (generated_html, request_id)
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Please configure it in web_viewer/.env")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://dimensionlab.ai/crainial",
            "X-Title": "DimensionLab CrAInial",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,  # Lower temperature for more consistent output
            "max_tokens": 8000,
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
                response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            request_id = data.get("id")

            # Extract HTML from the response (in case it's wrapped in markdown)
            html_content = self._extract_html(content)

            logger.info(f"OpenRouter API call successful, request_id={request_id}")
            return html_content, request_id

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error: {e.response.status_code} - {e.response.text}")
            raise ValueError(f"AI generation failed: {e.response.text}")
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise ValueError(f"AI generation failed: {str(e)}")

    def _extract_html(self, content: str) -> str:
        """Extract HTML content from AI response, handling markdown code blocks."""
        # Check if content is wrapped in markdown code block
        if "```html" in content:
            start = content.find("```html") + 7
            end = content.rfind("```")
            if end > start:
                return content[start:end].strip()

        # Check for generic code block
        if content.startswith("```") and content.endswith("```"):
            return content[3:-3].strip()

        return content

    def delete_report(self, report_id: str) -> bool:
        """Delete a case report."""
        report = self.get_report(report_id)
        if not report:
            return False

        # Delete PDF file if exists
        if report.pdf_path:
            try:
                Path(report.pdf_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete PDF file: {e}")

        self.db.delete(report)
        self.db.commit()

        self.audit.log(
            action="case_report.delete",
            entity_type="case_report",
            entity_id=report_id,
        )

        logger.info(f"Deleted case report: {report_id}")
        return True
