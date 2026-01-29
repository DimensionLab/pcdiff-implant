"""Request-level audit middleware for regulatory compliance logging."""

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("web_viewer.audit")


class AuditMiddleware(BaseHTTPMiddleware):
    """Logs every API request with timing, status, and correlation ID.

    All mutating operations are logged at INFO level. Reads are logged
    at DEBUG level to avoid excessive noise.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        response: Response = await call_next(request)

        duration_ms = (time.monotonic() - start_time) * 1000
        level = logging.INFO if request.method in ("POST", "PUT", "PATCH", "DELETE") else logging.DEBUG

        logger.log(
            level,
            "API %s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "client_ip": request.client.host if request.client else None,
            },
        )

        response.headers["X-Request-ID"] = request_id
        return response
