"""Scan model -- NRRD / DICOM volume files registered from the local filesystem."""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from web_viewer.backend.database import Base
from web_viewer.backend.models.base import AuditMixin, UUIDMixin

if TYPE_CHECKING:
    from web_viewer.backend.models.point_cloud import PointCloud
    from web_viewer.backend.models.project import Project


class Scan(UUIDMixin, AuditMixin, Base):
    __tablename__ = "scans"

    # Ownership
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )

    # Display
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # File reference (absolute path on disk)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_format: Mapped[str] = mapped_column(String(20), nullable=False)  # 'nrrd' | 'dicom_series'
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Volume metadata (extracted from NRRD header)
    volume_dims_x: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    volume_dims_y: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    volume_dims_z: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    voxel_spacing_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    voxel_spacing_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    voxel_spacing_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Classification
    scan_category: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # complete_skull, defective_skull, implant, other
    defect_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # bilateral, frontoorbital, parietotemporal, random_1, random_2
    skull_id: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # e.g. '059'

    # Extended metadata + provenance
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    checksum_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Relationships
    project: Mapped[Optional["Project"]] = relationship(back_populates="scans")
    point_clouds: Mapped[list["PointCloud"]] = relationship(back_populates="scan", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Scan id={self.id!r} name={self.name!r} format={self.file_format!r}>"
