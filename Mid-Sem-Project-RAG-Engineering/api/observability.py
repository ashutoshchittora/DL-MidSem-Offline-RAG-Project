"""
C12 — FastAPI Observability

Structured request logs, request IDs, and latency measurements.

Does not record question text, generated answers, or retrieved
passages.

Uses Python's standard library.
"""

import json
import logging
import time
import uuid

from datetime import datetime, timezone

from fastapi import FastAPI, Request


class JsonFormatter(logging.Formatter):
    """Format log records as structured JSON."""

    def format(self, record):

        payload = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "level": record.levelname,
            "message": record.getMessage(),
        }

        if hasattr(record, "event_data"):
            payload.update(record.event_data)

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str
        )


def configure_logger():

    logger = logging.getLogger(
        "rag_engineering.observability"
    )

    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())

        logger.addHandler(handler)

    return logger


def install_observability(app: FastAPI):
    """
    Install request-level instrumentation.

    Call once immediately after creating the FastAPI app.
    """

    if getattr(
        app.state,
        "observability_installed",
        False
    ):
        return

    logger = configure_logger()

    @app.middleware("http")
    async def observe_request(request: Request, call_next):

        request_id = str(uuid.uuid4())

        request.state.request_id = request_id

        start = time.perf_counter()

        status_code = 500
        error_type = None

        try:

            response = await call_next(request)

            status_code = response.status_code

            elapsed = time.perf_counter() - start

            response.headers[
                "X-Request-ID"
            ] = request_id

            response.headers[
                "X-Process-Time"
            ] = f"{elapsed:.6f}"

            return response

        except Exception as exc:

            error_type = type(exc).__name__

            raise

        finally:

            elapsed = time.perf_counter() - start

            event = {
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "route": (
                    request.scope.get("route").path
                    if request.scope.get("route")
                    else request.url.path
                ),
                "status_code": status_code,
                "duration_seconds": round(elapsed, 6),
            }

            if error_type is not None:
                event["error_type"] = error_type

            logger.info(
                "HTTP request completed",
                extra={"event_data": event}
            )

    app.state.observability_installed = True