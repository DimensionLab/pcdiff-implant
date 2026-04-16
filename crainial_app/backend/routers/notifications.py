"""Notification endpoints for async operation updates."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from crainial_app.backend.database import get_db
from crainial_app.backend.schemas.notification import NotificationRead
from crainial_app.backend.services.audit_service import AuditService
from crainial_app.backend.services.notification_service import NotificationService

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


def _get_service(db: Session = Depends(get_db)) -> NotificationService:
    audit = AuditService(db)
    return NotificationService(db, audit)


@router.get("/", response_model=list[NotificationRead])
def list_notifications(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    unread_only: bool = Query(False),
    service: NotificationService = Depends(_get_service),
):
    """List notifications, most recent first, unread prioritized."""
    return service.list_notifications(
        limit=limit,
        offset=offset,
        unread_only=unread_only,
    )


@router.get("/unread-count")
def get_unread_count(service: NotificationService = Depends(_get_service)):
    """Get the count of unread notifications."""
    return {"count": service.count_unread()}


@router.get("/{notification_id}", response_model=NotificationRead)
def get_notification(notification_id: str, service: NotificationService = Depends(_get_service)):
    """Get a specific notification by ID."""
    notification = service.get(notification_id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    return notification


@router.post("/{notification_id}/read", response_model=NotificationRead)
def mark_read(notification_id: str, service: NotificationService = Depends(_get_service)):
    """Mark a notification as read."""
    notification = service.mark_read(notification_id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    return notification


@router.post("/read-all")
def mark_all_read(service: NotificationService = Depends(_get_service)):
    """Mark all notifications as read."""
    count = service.mark_all_read()
    return {"updated": count}


@router.delete("/{notification_id}", status_code=204)
def delete_notification(notification_id: str, service: NotificationService = Depends(_get_service)):
    """Delete a notification."""
    if not service.delete(notification_id):
        raise HTTPException(status_code=404, detail="Notification not found")
