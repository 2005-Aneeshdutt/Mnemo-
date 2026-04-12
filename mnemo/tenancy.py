from __future__ import annotations

import re

_TENANT_RE = re.compile(r"^[a-zA-Z0-9._-]{1,64}$")


def scope_session(tenant_id: str, session_id: str) -> str:
    """Isolate storage per tenant without DB migration (composite key)."""
    t = (tenant_id or "default").strip() or "default"
    s = (session_id or "default").strip() or "default"
    if not _TENANT_RE.match(t):
        raise ValueError("Invalid tenant_id (use 1–64 chars: [a-zA-Z0-9._-])")
    if not s or len(s) > 256:
        raise ValueError("session_id length must be 1–256")
    if "::" in s:
        raise ValueError("session_id must not contain '::'")
    return f"{t}::{s}"


def parse_scoped_session(scoped: str) -> tuple[str, str]:
    if "::" not in scoped:
        return "default", scoped
    t, _, rest = scoped.partition("::")
    return t or "default", rest

