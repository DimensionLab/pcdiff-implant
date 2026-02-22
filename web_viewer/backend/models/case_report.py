"""CaseReport model -- stores generated report snapshots for regulatory traceability."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from web_viewer.backend.database import Base
from web_viewer.backend.models.base import AuditMixin, UUIDMixin

if TYPE_CHECKING:
    from web_viewer.backend.models.project import Project


class CaseReport(UUIDMixin, AuditMixin, Base):
    """Report snapshot for a cranioplasty case.

    Stores the generated HTML report and PDF path along with metadata
    for regulatory compliance and auditability (ASME V&V 40, MDR).
    """

    __tablename__ = "case_reports"

    # Link to project
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )

    # Report content
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    html_content: Mapped[str] = mapped_column(Text, nullable=False)
    pdf_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Template/prompt versioning for regulatory traceability
    template_version: Mapped[str] = mapped_column(
        String(50), nullable=False, default="v1.0"
    )
    prompt_version: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # Version of AI prompt used

    # AI generation metadata (for audit trail)
    ai_model: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )  # e.g., "anthropic/claude-4.5-sonnet"
    ai_provider: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # e.g., "openrouter"
    ai_request_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )  # Provider request ID for tracing

    # Region-specific compliance
    region_code: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True
    )  # ISO 3166-1 alpha-2 code

    # Generation timestamp
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Extended metadata (input data snapshot, metrics, etc.)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="case_reports")

    def __repr__(self) -> str:
        return f"<CaseReport id={self.id!r} project_id={self.project_id!r} title={self.title!r}>"
