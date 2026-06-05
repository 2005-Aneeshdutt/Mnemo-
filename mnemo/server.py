from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from mnemo.client import create_client
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from mnemo import auth as auth_mod
from mnemo import config
from mnemo import metrics as metrics_mod
from mnemo.agent import ChatState, chat_turn, chat_turn_stream
from mnemo import store
from mnemo.tenancy import scope_session


def _rate_limit_key(request: Request) -> str:
    tenant = (request.headers.get("X-Tenant-ID") or "default").strip() or "default"
    return f"{tenant}:{get_remote_address(request)}"


limiter = Limiter(key_func=_rate_limit_key)


class ChatRequest(BaseModel):
    tenant_id: str = Field(default="default", min_length=1, max_length=64)
    session_id: str = Field(default="default", min_length=1, max_length=256)
    message: str = Field(..., min_length=1, max_length=32000)


class ChatResponse(BaseModel):
    reply: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.GROQ_API_KEY and not config.GEMINI_API_KEY:
        raise RuntimeError("Set GROQ_API_KEY or GEMINI_API_KEY in your .env file.")
    app.state.client = create_client()
    app.state.conn = store.connect(config.DB_PATH)
    store.init_schema(app.state.conn)
    app.state.chat_states: dict[str, ChatState] = {}
    yield
    app.state.conn.close()


app = FastAPI(title="Mnemo API", version="0.4.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def _track_request_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    metrics_mod.record_request(
        endpoint=request.url.path,
        status=response.status_code,
        latency_ms=latency_ms,
    )
    return response


def _resolve_tenant(x_tenant_id: str | None, body_tenant: str) -> str:
    t = (x_tenant_id or body_tenant or "default").strip() or "default"
    return t


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def prometheus_metrics() -> str:
    """Prometheus text-format metrics scrape endpoint."""
    return metrics_mod.metrics_prometheus()


@app.get("/v1/metrics", dependencies=[Depends(auth_mod.require_api_key)])
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def json_metrics(request: Request) -> dict[str, Any]:
    """JSON metrics snapshot: counters, latency histograms (p50/p95/p99), token savings."""
    return metrics_mod.metrics_json()


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    dependencies=[Depends(auth_mod.require_api_key)],
)
@limiter.limit(config.RATE_LIMIT_CHAT)
def post_chat(
    request: Request,
    req: ChatRequest,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> ChatResponse:
    tenant = _resolve_tenant(x_tenant_id, req.tenant_id)
    try:
        scoped = scope_session(tenant, req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    states: dict[str, ChatState] = request.app.state.chat_states
    if scoped not in states:
        states[scoped] = ChatState(session_id=scoped, tenant_id=tenant)
    state = states[scoped]
    try:
        reply = chat_turn(request.app.state.client, request.app.state.conn, state, req.message)
    except Exception as e:
        logging.exception("POST /v1/chat failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return ChatResponse(reply=reply)


@app.post(
    "/v1/chat/stream",
    dependencies=[Depends(auth_mod.require_api_key)],
    response_class=StreamingResponse,
)
@limiter.limit(config.RATE_LIMIT_CHAT)
def post_chat_stream(
    request: Request,
    req: ChatRequest,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> StreamingResponse:
    tenant = _resolve_tenant(x_tenant_id, req.tenant_id)
    try:
        scoped = scope_session(tenant, req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    states: dict[str, ChatState] = request.app.state.chat_states
    if scoped not in states:
        states[scoped] = ChatState(session_id=scoped, tenant_id=tenant)
    state = states[scoped]

    client = request.app.state.client
    conn = request.app.state.conn

    def _generate():
        try:
            for chunk in chat_turn_stream(client, conn, state, req.message):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logging.exception("POST /v1/chat/stream failed")
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.get(
    "/v1/sessions/{session_id}/memory",
    dependencies=[Depends(auth_mod.require_api_key)],
)
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def get_memory(
    request: Request,
    session_id: str,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict[str, Any]:
    tenant = _resolve_tenant(x_tenant_id, "default")
    try:
        scoped = scope_session(tenant, session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    rows = store.list_chunks_for_session(request.app.state.conn, scoped)
    items = []
    for r in rows:
        items.append(
            {
                "id": r["id"],
                "kind": r["kind"],
                "content": r["content"],
                "subject": r["subj"],
                "predicate": r["pred"],
                "object": r["obj"],
                "has_embedding": r["embedding"] is not None,
                "created_at": r["created_at"],
            }
        )
    return {"tenant_id": tenant, "session_id": session_id, "scoped_session": scoped, "count": len(items), "items": items}


@app.delete(
    "/v1/sessions/{session_id}/memory",
    dependencies=[Depends(auth_mod.require_api_key)],
)
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def delete_memory(
    request: Request,
    session_id: str,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict[str, Any]:
    tenant = _resolve_tenant(x_tenant_id, "default")
    try:
        scoped = scope_session(tenant, session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    n = store.clear_session(request.app.state.conn, scoped)
    if scoped in request.app.state.chat_states:
        request.app.state.chat_states[scoped].messages.clear()
    return {"tenant_id": tenant, "session_id": session_id, "deleted": n}


@app.get(
    "/v1/users/{tenant_id}/profile",
    dependencies=[Depends(auth_mod.require_api_key)],
)
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def get_profile(
    request: Request,
    tenant_id: str,
) -> dict[str, Any]:
    rows = store.list_profile_chunks(request.app.state.conn, tenant_id)
    items = [
        {
            "id": r["id"],
            "kind": r["kind"],
            "content": r["content"],
            "subject": r["subj"],
            "predicate": r["pred"],
            "object": r["obj"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    return {"tenant_id": tenant_id, "count": len(items), "items": items}


@app.delete(
    "/v1/users/{tenant_id}/profile",
    dependencies=[Depends(auth_mod.require_api_key)],
)
@limiter.limit(config.RATE_LIMIT_DEFAULT)
def delete_profile(
    request: Request,
    tenant_id: str,
) -> dict[str, Any]:
    n = store.clear_profile(request.app.state.conn, tenant_id)
    return {"tenant_id": tenant_id, "deleted": n}


def main() -> int:
    if not config.GROQ_API_KEY and not config.GEMINI_API_KEY:
        print("Set GROQ_API_KEY or GEMINI_API_KEY in environment or .env file.", file=sys.stderr)
        return 1
    uvicorn.run(
        "mnemo.server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

