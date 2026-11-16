"""
C15 — Engineering API Security Baseline.

Optional API key for the answer endpoint and response headers.
Does not modify the academic RAG implementation.
"""

import hmac
import os

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse


def install_security(app: FastAPI, api_key=None):
    """
    Install security middleware once.

    RAG_API_KEY enables API-key protection when configured.
    Without it, the application remains in local development mode.
    """

    if getattr(app.state, "security_installed", False):
        return

    expected_key = (
        os.environ.get("RAG_API_KEY")
        if api_key is None
        else api_key
    )

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):

        protected = request.url.path == "/answer"

        if protected and expected_key:

            supplied_key = request.headers.get(
                "X-API-Key", ""
            )

            authorized = hmac.compare_digest(
                supplied_key,
                expected_key
            )

            if not authorized:
                response = JSONResponse(
                    status_code=401,
                    content={
                        "detail": "Unauthorized"
                    }
                )
            else:
                response = await call_next(request)

        else:
            response = await call_next(request)

        response.headers[
            "X-Content-Type-Options"
        ] = "nosniff"

        response.headers[
            "X-Frame-Options"
        ] = "DENY"

        response.headers[
            "Cache-Control"
        ] = "no-store"

        return response

    app.state.security_installed = True
    app.state.api_key_required = bool(expected_key)