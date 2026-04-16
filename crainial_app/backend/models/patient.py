"""Patient model -- stores basic patient profile for cranioplasty cases."""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Date, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from crainial_app.backend.database import Base
from crainial_app.backend.models.base import AuditMixin, UUIDMixin

if TYPE_CHECKING:
    from crainial_app.backend.models.project import Project


class Patient(UUIDMixin, AuditMixin, Base):
    """Patient record for tracking cranioplasty cases.

    Note: In production, PHI fields should be encrypted and access-controlled
    per HIPAA/GDPR requirements. This MVP stores minimal identifiers.
    """

    __tablename__ = "patients"

    # Basic identifiers (anonymized/pseudonymized in production)
    patient_code: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )  # Internal reference code (e.g., "PAT-2026-001")

    # Demographics (optional, for clinical context)
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    date_of_birth: Mapped[Optional[str]] = mapped_column(Date, nullable=True)
    sex: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # male, female, other, unknown

    # Contact (optional)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Medical identifiers (for integration with hospital systems)
    medical_record_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # MRN from hospital system

    # Insurance (placeholder for future regional customization)
    insurance_provider: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    insurance_policy_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Notes
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Extended metadata for future dynamic forms
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    projects: Mapped[list["Project"]] = relationship(back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Patient id={self.id!r} code={self.patient_code!r}>"
