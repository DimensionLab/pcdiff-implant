"""GenerationJob model -- tracks implant generation jobs using PCDiff + Voxelization pipeline."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from web_viewer.backend.database import Base
from web_viewer.backend.models.base import AuditMixin, UUIDMixin

if TYPE_CHECKING:
    from web_viewer.backend.models.point_cloud import PointCloud
    from web_viewer.backend.models.project import Project


class GenerationJob(UUIDMixin, AuditMixin, Base):
    __tablename__ = "generation_jobs"

    # Metadata
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # References
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    input_pc_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("point_clouds.id", ondelete="CASCADE"), nullable=False
    )

    # Status tracking
    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )  # pending, running, completed, failed, cancelled
    progress_percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_step: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Generation parameters
    sampling_method: Mapped[str] = mapped_column(
        String(20), default="ddim", nullable=False
    )  # "ddim" or "ddpm"
    sampling_steps: Mapped[int] = mapped_column(Integer, default=50, nullable=False)
    num_ensemble: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    pcdiff_model: Mapped[Optional[str]] = mapped_column(
        String(20), default="best", nullable=True
    )  # "best" or "latest" - which PCDiff model checkpoint to use

    # Timing
    queued_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    generation_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Results (JSON arrays of output IDs)
    output_pc_ids_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # ["uuid1", "uuid2", ...] for ensemble point clouds
    output_stl_ids_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # ["uuid1", "uuid2", ...] for voxelized meshes
    selected_output_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )  # User's chosen best result

    # Evaluation metrics (JSON)
    metrics_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # {ensemble_idx: {dsc, bdsc, hd95}, ...}

    # Relationships
    project: Mapped[Optional["Project"]] = relationship(back_populates="generation_jobs")
    input_point_cloud: Mapped["PointCloud"] = relationship(foreign_keys=[input_pc_id])

    def __repr__(self) -> str:
        return f"<GenerationJob id={self.id!r} name={self.name!r} status={self.status!r}>"
