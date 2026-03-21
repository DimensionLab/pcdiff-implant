"""
Shared mixins for SQLAlchemy models.

UUIDMixin  -- Provides a UUID primary key.
AuditMixin -- Adds created_at, updated_at, created_by columns for regulatory
              traceability (ISO 13485 / IEC 62304).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column


class UUIDMixin:
    """Provides a UUID primary key stored as TEXT (SQLite-compatible)."""

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )


class AuditMixin:
    """Regulatory audit columns present on every data table."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    created_by: Mapped[str] = mapped_column(
        String(100),
        default="system",
        nullable=False,
    )
