"""Audit log query + export endpoints."""

import csv
import io
import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from web_viewer.backend.database import get_db
from web_viewer.backend.schemas.common import AuditLogList, AuditLogRead
from web_viewer.backend.services.audit_service import AuditService

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


def _get_service(db: Session = Depends(get_db)):
    return AuditService(db)


@router.get("/", response_model=AuditLogList)
def list_audit_logs(
    entity_type: str | None = Query(None),
    entity_id: str | None = Query(None),
    action: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    service: AuditService = Depends(_get_service),
):
    items = service.query(
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        limit=limit,
        offset=offset,
    )
    total = service.count(entity_type=entity_type, entity_id=entity_id)
    return AuditLogList(
        items=[AuditLogRead.model_validate(item) for item in items],
        total=total,
    )


@router.get("/entity/{entity_type}/{entity_id}", response_model=list[AuditLogRead])
def get_entity_audit_trail(
    entity_type: str,
    entity_id: str,
    service: AuditService = Depends(_get_service),
):
    items = service.query(entity_type=entity_type, entity_id=entity_id, limit=500)
    return [AuditLogRead.model_validate(item) for item in items]


@router.get("/export")
def export_audit_csv(
    entity_type: str | None = Query(None),
    service: AuditService = Depends(_get_service),
):
    """Export audit log as CSV for regulatory submissions."""
    items = service.query(entity_type=entity_type, limit=100_000)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "timestamp",
            "action",
            "entity_type",
            "entity_id",
            "user_id",
            "details",
            "ip_address",
            "session_id",
            "software_version",
        ]
    )

    for item in items:
        writer.writerow(
            [
                item.timestamp.isoformat() if item.timestamp else "",
                item.action,
                item.entity_type or "",
                item.entity_id or "",
                item.user_id,
                item.details_json or "",
                item.ip_address or "",
                item.session_id or "",
                item.software_version,
            ]
        )

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="audit_log.csv"'},
    )
