"""
Service for managing user notifications.
"""

import logging

from sqlalchemy.orm import Session

from crainial_app.backend.models.notification import Notification
from crainial_app.backend.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for managing notifications."""

    def __init__(self, db: Session, audit: AuditService):
        self.db = db
        self.audit = audit

    def create(
        self,
        type: str,
        title: str,
        message: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> Notification:
        """Create a new notification."""
        notification = Notification(
            type=type,
            title=title,
            message=message,
            entity_type=entity_type,
            entity_id=entity_id,
            read=False,
        )
        self.db.add(notification)
        self.db.commit()
        self.db.refresh(notification)
        return notification

    def get(self, notification_id: str) -> Notification | None:
        """Get a notification by ID."""
        return self.db.query(Notification).filter(Notification.id == notification_id).first()

    def list_notifications(
        self,
        limit: int = 50,
        offset: int = 0,
        unread_only: bool = False,
    ) -> list[Notification]:
        """List notifications, most recent first, unread prioritized."""
        q = self.db.query(Notification)
        if unread_only:
            q = q.filter(Notification.read == False)
        # Order by read status (unread first), then by creation time (newest first)
        return q.order_by(Notification.read.asc(), Notification.created_at.desc()).offset(offset).limit(limit).all()

    def count_unread(self) -> int:
        """Count unread notifications."""
        return self.db.query(Notification).filter(Notification.read == False).count()

    def mark_read(self, notification_id: str) -> Notification | None:
        """Mark a notification as read."""
        notification = self.get(notification_id)
        if not notification:
            return None

        notification.read = True
        self.db.commit()
        self.db.refresh(notification)
        return notification

    def mark_all_read(self) -> int:
        """Mark all notifications as read. Returns count of updated records."""
        count = self.db.query(Notification).filter(Notification.read == False).update({"read": True})
        self.db.commit()
        return count

    def delete(self, notification_id: str) -> bool:
        """Delete a notification."""
        notification = self.get(notification_id)
        if not notification:
            return False

        self.db.delete(notification)
        self.db.commit()
        return True
