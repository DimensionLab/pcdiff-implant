"""FitMetricsResult model -- cached fit metric computations between point clouds."""

from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from web_viewer.backend.database import Base
from web_viewer.backend.models.base import AuditMixin, UUIDMixin


class FitMetricsResult(UUIDMixin, AuditMixin, Base):
    __tablename__ = "fit_metrics_results"

    # Point cloud references
    implant_pc_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("point_clouds.id", ondelete="CASCADE"), nullable=False
    )
    reference_pc_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("point_clouds.id", ondelete="CASCADE"), nullable=False
    )
    defective_skull_pc_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("point_clouds.id", ondelete="SET NULL"), nullable=True
    )

    # Metric values
    dice_coefficient: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hausdorff_distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hausdorff_distance_95: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    boundary_dice: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Computation parameters
    resolution: Mapped[int] = mapped_column(Integer, default=256, nullable=False)
    voxel_spacing: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    computation_mode: Mapped[str] = mapped_column(
        String(20), default="voxel", nullable=False
    )  # "voxel" or "point_cloud"
    computation_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )  # pending, computing, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<FitMetricsResult id={self.id!r} "
            f"dice={self.dice_coefficient} hd95={self.hausdorff_distance_95} "
            f"status={self.status!r}>"
        )
