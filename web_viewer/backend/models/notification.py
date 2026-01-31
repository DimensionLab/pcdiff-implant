"""Notification model -- tracks user notifications for async operations."""

from typing import Optional

from sqlalchemy import Boolean, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from web_viewer.backend.database import Base
from web_viewer.backend.models.base import AuditMixin, UUIDMixin


class Notification(UUIDMixin, AuditMixin, Base):
    __tablename__ = "notifications"

    # Notification content
    type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # generation_started, generation_completed, generation_failed
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    # Entity reference for navigation
    entity_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # generation_job, point_cloud, etc.
    entity_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Read status
    read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        return f"<Notification id={self.id!r} type={self.type!r} read={self.read!r}>"
