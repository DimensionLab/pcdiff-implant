"""Pydantic schemas for Notification endpoints."""

from datetime import datetime

from pydantic import BaseModel


class NotificationRead(BaseModel):
    """Response schema for notification."""

    id: str
    type: str
    title: str
    message: str
    entity_type: str | None = None
    entity_id: str | None = None
    read: bool
    created_at: datetime
    created_by: str

    model_config = {"from_attributes": True}


class NotificationCreate(BaseModel):
    """Request to create a notification (internal use)."""

    type: str
    title: str
    message: str
    entity_type: str | None = None
    entity_id: str | None = None
