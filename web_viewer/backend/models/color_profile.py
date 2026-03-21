"""Color profile model -- SDF colorization presets for point cloud visualization."""

from typing import Optional

from sqlalchemy import Boolean, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from web_viewer.backend.database import Base
from web_viewer.backend.models.base import AuditMixin, UUIDMixin


class ColorProfile(UUIDMixin, AuditMixin, Base):
    __tablename__ = "color_profiles"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Color map configuration
    color_map_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # 'diverging' | 'sequential' | 'categorical'

    # JSON array of {"value": float, "color": "#rrggbb"} objects
    color_stops: Mapped[str] = mapped_column(Text, nullable=False)

    # SDF value range this profile is designed for
    sdf_range_min: Mapped[float] = mapped_column(Float, default=-1.0, nullable=False)
    sdf_range_max: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        return f"<ColorProfile id={self.id!r} name={self.name!r}>"
