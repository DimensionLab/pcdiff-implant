"""GenerationJob model -- tracks cran-2 implant generation jobs.

A job submits a defective-skull NRRD volume to the cran-2 RunPod endpoint
and stores the resulting implant-mask NRRD as a new Scan.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from crainial_app.backend.database import Base
from crainial_app.backend.models.base import AuditMixin, UUIDMixin

if TYPE_CHECKING:
    from crainial_app.backend.models.project import Project
    from crainial_app.backend.models.scan import Scan


class GenerationJob(UUIDMixin, AuditMixin, Base):
    __tablename__ = "generation_jobs"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    input_scan_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False
    )
    output_scan_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("scans.id", ondelete="SET NULL"), nullable=True
    )

    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )  # pending, running, completed, failed, cancelled
    progress_percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_step: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    threshold: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)

    runpod_job_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    queued_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    generation_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    inference_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    project: Mapped[Optional["Project"]] = relationship(back_populates="generation_jobs")
    input_scan: Mapped["Scan"] = relationship(foreign_keys=[input_scan_id])
    output_scan: Mapped[Optional["Scan"]] = relationship(foreign_keys=[output_scan_id])

    def __repr__(self) -> str:
        return f"<GenerationJob id={self.id!r} name={self.name!r} status={self.status!r}>"
