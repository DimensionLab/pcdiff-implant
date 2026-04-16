"""
Audit log model -- append-only regulatory trail.

This table must NEVER have rows updated or deleted. All entries are
immutable once written. This is a requirement for MDR / IEC 62304 /
ISO 13485 compliance.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from crainial_app.backend.database import Base


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # What happened
    action: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # e.g. 'scan.create', 'scan.view', 'inference.start'

    # What was affected (polymorphic reference)
    entity_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # 'scan', 'point_cloud', 'project', ...
    entity_id: Mapped[str | None] = mapped_column(String(36), nullable=True)  # UUID of the affected entity

    # Who did it
    user_id: Mapped[str] = mapped_column(String(100), default="system", nullable=False)

    # Extra context
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # Traceability
    software_version: Mapped[str] = mapped_column(String(20), nullable=False)

    def __repr__(self) -> str:
        return f"<AuditLog id={self.id} action={self.action!r} entity={self.entity_type}:{self.entity_id}>"
