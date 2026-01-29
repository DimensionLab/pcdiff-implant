"""
Append-only audit logging service.

Every mutating operation in the application must be logged here.
This is a regulatory requirement for MDR / IEC 62304 / ISO 13485.
The audit_log table must never have rows updated or deleted.
"""

import json
import logging

from sqlalchemy.orm import Session

from web_viewer.backend.config import settings
from web_viewer.backend.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


class AuditService:
    def __init__(self, db: Session):
        self.db = db

    def log(
        self,
        action: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        user_id: str = "system",
        details: dict | None = None,
        ip_address: str | None = None,
        session_id: str | None = None,
    ) -> AuditLog:
        """Append an immutable audit log entry.

        Args:
            action: Dot-separated action string, e.g. 'scan.create', 'inference.start'.
            entity_type: The type of entity affected ('scan', 'point_cloud', etc.).
            entity_id: UUID of the affected entity.
            user_id: Identifier of the acting user (defaults to 'system').
            details: Arbitrary JSON-serialisable dict with action-specific context.
            ip_address: Client IP address from the request.
            session_id: Session identifier for request correlation.

        Returns:
            The created AuditLog record.
        """
        entry = AuditLog(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            details_json=json.dumps(details, default=str) if details else None,
            ip_address=ip_address,
            session_id=session_id,
            software_version=settings.software_version,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)

        logger.info(
            "Audit: %s %s:%s by %s",
            action,
            entity_type or "-",
            entity_id or "-",
            user_id,
        )
        return entry

    def query(
        self,
        entity_type: str | None = None,
        entity_id: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        """Query audit log entries with optional filters."""
        q = self.db.query(AuditLog)
        if entity_type:
            q = q.filter(AuditLog.entity_type == entity_type)
        if entity_id:
            q = q.filter(AuditLog.entity_id == entity_id)
        if action:
            q = q.filter(AuditLog.action == action)
        return q.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit).all()

    def count(
        self,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> int:
        q = self.db.query(AuditLog)
        if entity_type:
            q = q.filter(AuditLog.entity_type == entity_type)
        if entity_id:
            q = q.filter(AuditLog.entity_id == entity_id)
        return q.count()
