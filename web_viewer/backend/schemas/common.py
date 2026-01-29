"""Shared Pydantic schemas for pagination, audit log, etc."""

from datetime import datetime

from pydantic import BaseModel


class PaginationParams(BaseModel):
    limit: int = 100
    offset: int = 0


class AuditLogRead(BaseModel):
    id: int
    timestamp: datetime
    action: str
    entity_type: str | None = None
    entity_id: str | None = None
    user_id: str
    details_json: str | None = None
    ip_address: str | None = None
    session_id: str | None = None
    software_version: str

    model_config = {"from_attributes": True}


class AuditLogList(BaseModel):
    items: list[AuditLogRead]
    total: int
