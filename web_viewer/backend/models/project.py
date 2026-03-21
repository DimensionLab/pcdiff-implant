"""Project model -- groups related scans, point clouds, and case metadata."""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from web_viewer.backend.database import Base
from web_viewer.backend.models.base import AuditMixin, UUIDMixin

if TYPE_CHECKING:
    from web_viewer.backend.models.case_report import CaseReport
    from web_viewer.backend.models.generation_job import GenerationJob
    from web_viewer.backend.models.patient import Patient
    from web_viewer.backend.models.point_cloud import PointCloud
    from web_viewer.backend.models.scan import Scan


class Project(UUIDMixin, AuditMixin, Base):
    """Project/case representing a cranioplasty procedure for a patient."""

    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Patient link (optional for backward compatibility)
    patient_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("patients.id", ondelete="SET NULL"), nullable=True
    )

    # Case metadata for doctor portal
    reconstruction_type: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )  # e.g., "cranioplasty", "maxillofacial", "orbital"

    implant_material: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )  # e.g., "PEEK", "Titanium", "MEDPOR", "PMMA"

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Region code for regulatory customization (ISO 3166-1 alpha-2)
    region_code: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True
    )  # e.g., "US", "EU", "SK"

    # Extended metadata for future dynamic forms and regional compliance
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    patient: Mapped[Optional["Patient"]] = relationship(back_populates="projects")
    scans: Mapped[list["Scan"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    point_clouds: Mapped[list["PointCloud"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    generation_jobs: Mapped[list["GenerationJob"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    case_reports: Mapped[list["CaseReport"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Project id={self.id!r} name={self.name!r}>"
