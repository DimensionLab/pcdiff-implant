"""Case Report endpoints for generating and managing cranioplasty reports."""

import io
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session

from crainial_app.backend.database import get_db
from crainial_app.backend.schemas.case_report import (
    CaseReportCreate,
    CaseReportRead,
    CaseReportSummary,
)
from crainial_app.backend.services.audit_service import AuditService
from crainial_app.backend.services.report_service import ReportService

router = APIRouter(prefix="/api/v1/case-reports", tags=["case-reports"])


def _get_service(db: Session = Depends(get_db)) -> ReportService:
    audit = AuditService(db)
    return ReportService(db, audit)


@router.get("/", response_model=list[CaseReportSummary])
def list_reports(
    project_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    service: ReportService = Depends(_get_service),
):
    """List case reports with optional project filtering.

    Returns summaries without full HTML content for efficiency.
    """
    return service.list_reports(project_id=project_id, limit=limit, offset=offset)


@router.post("/", response_model=CaseReportRead, status_code=201)
def generate_report(
    body: CaseReportCreate,
    model: str = Query("anthropic/claude-4.5-sonnet", description="OpenRouter model to use"),
    service: ReportService = Depends(_get_service),
):
    """Generate a new case report using AI.

    The report is generated using OpenRouter's API with the specified model.
    The default model is Claude 4.5 Sonnet, which provides high-quality output.

    Available models include:
    - anthropic/claude-4.5-sonnet (default, best quality)
    - anthropic/claude-4.5-haiku (faster, lower cost)
    - openai/gpt-4o (alternative)
    - openai/gpt-4o-mini (faster, lower cost)
    """
    try:
        return service.generate_report(
            project_id=body.project_id,
            title=body.title,
            region_code=body.region_code,
            model=model,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{report_id}", response_model=CaseReportRead)
def get_report(report_id: str, service: ReportService = Depends(_get_service)):
    """Get a specific case report by ID."""
    report = service.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.get("/{report_id}/html")
def get_report_html(report_id: str, service: ReportService = Depends(_get_service)):
    """Get the HTML content of a report for display.

    Returns raw HTML content suitable for rendering in an iframe or directly.
    """
    report = service.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Wrap in a full HTML document if it's just a fragment
    html = report.html_content
    if not html.strip().lower().startswith("<!doctype") and not html.strip().lower().startswith("<html"):
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }}
        h1, h2, h3 {{ color: #1a1a2e; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{ background-color: #f4f4f4; }}
        .metadata {{
            font-size: 0.85rem;
            color: #666;
            border-top: 1px solid #ddd;
            padding-top: 1rem;
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
{html}
<div class="metadata">
    <p>Report ID: {report.id}</p>
    <p>Generated: {report.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
    <p>Model: {report.ai_model or "N/A"}</p>
    <p>Template Version: {report.template_version}</p>
</div>
</body>
</html>"""

    return HTMLResponse(content=html)


@router.get("/{report_id}/pdf")
def download_report_pdf(report_id: str, service: ReportService = Depends(_get_service)):
    """Download the report as a PDF.

    If a PDF hasn't been generated yet, it will be created on-demand.
    """
    report = service.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Check if PDF already exists
    if report.pdf_path and Path(report.pdf_path).exists():
        pdf_data = Path(report.pdf_path).read_bytes()
    else:
        # Generate PDF on-demand
        try:
            pdf_data = _generate_pdf(report.html_content, report.title)
            # Optionally save it for future use
            pdf_path = service.reports_dir / f"{report.id}.pdf"
            pdf_path.write_bytes(pdf_data)
            report.pdf_path = str(pdf_path)
            service.db.commit()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"PDF generation failed: {str(e)}. Please ensure weasyprint or reportlab is installed.",
            )

    filename = f"report_{report.id[:8]}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_data),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{report_id}", status_code=204)
def delete_report(report_id: str, service: ReportService = Depends(_get_service)):
    """Delete a case report."""
    if not service.delete_report(report_id):
        raise HTTPException(status_code=404, detail="Report not found")


def _generate_pdf(html_content: str, title: str) -> bytes:
    """Generate a PDF from HTML content.

    Tries weasyprint first (better quality), falls back to basic approach.
    """
    # Wrap HTML in a full document if needed
    if not html_content.strip().lower().startswith("<!doctype"):
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 0 auto;
            padding: 20px;
            font-size: 11pt;
            line-height: 1.5;
        }}
        h1 {{ font-size: 18pt; }}
        h2 {{ font-size: 14pt; }}
        h3 {{ font-size: 12pt; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; font-size: 10pt; }}
        th {{ background-color: #f0f0f0; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

    # Try weasyprint first (best quality)
    try:
        from weasyprint import HTML

        return HTML(string=html_content).write_pdf()
    except ImportError:
        pass

    # Fallback: Try xhtml2pdf
    try:
        from xhtml2pdf import pisa

        output = io.BytesIO()
        pisa_status = pisa.CreatePDF(html_content, dest=output)
        if pisa_status.err:
            raise ValueError("xhtml2pdf conversion failed")
        return output.getvalue()
    except ImportError:
        pass

    # Final fallback: raise error with installation instructions
    raise ImportError("No PDF library available. Install one with: pip install weasyprint or pip install xhtml2pdf")
