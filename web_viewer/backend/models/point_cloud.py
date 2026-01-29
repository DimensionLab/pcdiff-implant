"""PointCloud model -- NPY / PLY / STL surface files registered from the local filesystem."""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from web_viewer.backend.database import Base
from web_viewer.backend.models.base import AuditMixin, UUIDMixin

if TYPE_CHECKING:
    from web_viewer.backend.models.project import Project
    from web_viewer.backend.models.scan import Scan


class PointCloud(UUIDMixin, AuditMixin, Base):
    __tablename__ = "point_clouds"

    # Ownership
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    scan_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("scans.id", ondelete="SET NULL"), nullable=True
    )

    # Display
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # File reference
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_format: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # 'npy' | 'ply' | 'stl'
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Point cloud metadata
    num_points: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    point_dims: Mapped[int] = mapped_column(Integer, default=3, nullable=False)

    # Classification
    scan_category: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # complete_skull, defective_skull, implant, generated_implant, other
    defect_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    skull_id: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Extended metadata + provenance
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    checksum_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Relationships
    project: Mapped[Optional["Project"]] = relationship(back_populates="point_clouds")
    scan: Mapped[Optional["Scan"]] = relationship(back_populates="point_clouds")

    def __repr__(self) -> str:
        return f"<PointCloud id={self.id!r} name={self.name!r} points={self.num_points}>"
