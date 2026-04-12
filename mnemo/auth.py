from __future__ import annotations

from fastapi import HTTPException, Request


def require_api_key(request: Request) -> None:
    """
    When MNEMO_API_KEY is set in the environment, require either:
    - Authorization: Bearer <token>
    - X-API-Key: <token>
    """
    from mnemo import config

    expected = config.MNEMO_API_KEY
    if not expected:
        return
    auth = request.headers.get("Authorization") or ""
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token:
        token = (request.headers.get("X-API-Key") or "").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

